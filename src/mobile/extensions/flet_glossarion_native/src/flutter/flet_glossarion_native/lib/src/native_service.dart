import 'dart:async';
import 'dart:io' show File;

import 'package:flet/flet.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:receive_sharing_intent/receive_sharing_intent.dart';

import 'task_handler.dart';

const String kGlossarionNativeVersion = '0.1.0';

/// Channel to GlossarionNativePlugin.kt / GlossarionNativePlugin.swift.
const MethodChannel _platformChannel =
    MethodChannel('glossarion_native/platform');

/// Must match JOB_SERVICE_NOTIFICATION_ID in the Python package.
const int _kJobServiceId = 41100;

/// Meta-data name declared in android/src/main/AndroidManifest.xml.
const String _kNotificationIconMetaData =
    'com.glossarion.native.notification_icon';

/// Flet service behind the Python `GlossarionNative` control.
///
/// Python -> Dart: `control.addInvokeMethodListener` (method names below).
/// Dart -> Python: `control.triggerEvent` with `share`, `foreground`,
/// `background_task` and `notification` (payload keys = Python dataclass
/// fields in flet_glossarion_native/types.py).
class GlossarionNativeService extends FletService {
  GlossarionNativeService({required super.control});

  static GlossarionNativeService? _active;

  final Completer<void> _ready = Completer<void>();
  final List<Map<String, dynamic>> _shared = <Map<String, dynamic>>[];
  final Set<String> _sharedIds = <String>{};
  Map<String, dynamic>? _launchNotification;
  StreamSubscription<List<SharedMediaFile>>? _shareSubscription;
  bool _taskCallbackAdded = false;
  String? _lastError;
  int _sequence = 0;

  bool get _isAndroid =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  bool get _isIOS => !kIsWeb && defaultTargetPlatform == TargetPlatform.iOS;

  bool get _isMobile => _isAndroid || _isIOS;

  @override
  void init() {
    super.init();
    debugPrint("GlossarionNativeService(${control.id}).init");
    control.addInvokeMethodListener(_invokeMethod);

    if (!_isMobile) {
      _ready.complete();
      return;
    }

    _active = this;
    _platformChannel.setMethodCallHandler(_onPlatformCall);

    if (_isAndroid) {
      FlutterForegroundTask.addTaskDataCallback(_onTaskData);
      _taskCallbackAdded = true;
    }

    try {
      _shareSubscription = ReceiveSharingIntent.instance.getMediaStream().listen(
        (List<SharedMediaFile> files) => _onSharingIntentMedia(files, initial: false),
        onError: (Object error) {
          debugPrint("GlossarionNative: share stream error: $error");
        },
      );
    } catch (e) {
      debugPrint("GlossarionNative: cannot listen to receive_sharing_intent: $e");
    }

    unawaited(_attach());
  }

  @override
  void update() {}

  @override
  void dispose() {
    debugPrint("GlossarionNativeService(${control.id}).dispose()");
    control.removeInvokeMethodListener(_invokeMethod);
    _shareSubscription?.cancel();
    _shareSubscription = null;
    if (_taskCallbackAdded) {
      FlutterForegroundTask.removeTaskDataCallback(_onTaskData);
      _taskCallbackAdded = false;
    }
    if (identical(_active, this)) {
      _platformChannel.setMethodCallHandler(null);
      _active = null;
    }
    super.dispose();
  }

  // ---------------------------------------------------------------- startup

  /// Collects everything the platform received before Dart was listening:
  /// cold-start Open-with/Share items, the launch notification and taps that
  /// arrived while the engine was starting.
  Future<void> _attach() async {
    try {
      final dynamic result = await _platformChannel.invokeMethod<dynamic>('attach');
      if (result is Map) {
        final dynamic shared = result['shared'];
        if (shared is List) {
          _addShared(shared, emit: false);
        }
        final dynamic launch = result['launch_notification'];
        if (launch is Map) {
          _launchNotification = _stringMap(launch);
        }
        final dynamic notifications = result['notifications'];
        if (notifications is List) {
          for (final dynamic n in notifications) {
            if (n is Map) {
              control.triggerEvent('notification', _stringMap(n));
            }
          }
        }
      }
    } catch (e) {
      _lastError = 'attach: $e';
      debugPrint("GlossarionNative: $_lastError");
    }

    try {
      final List<SharedMediaFile> initial =
          await ReceiveSharingIntent.instance.getInitialMedia();
      if (initial.isNotEmpty) {
        _onSharingIntentMedia(initial, initial: true);
        await ReceiveSharingIntent.instance.reset();
      }
    } catch (e) {
      debugPrint("GlossarionNative: receive_sharing_intent initial media: $e");
    }

    if (!_ready.isCompleted) {
      _ready.complete();
    }
  }

  Future<void> _waitReady() async {
    if (_ready.isCompleted) return;
    await _ready.future.timeout(const Duration(seconds: 5), onTimeout: () {});
  }

  // ------------------------------------------------------- platform -> Dart

  Future<dynamic> _onPlatformCall(MethodCall call) async {
    final dynamic args = call.arguments;
    switch (call.method) {
      case 'share':
        if (args is Map && args['items'] is List) {
          _addShared(args['items'] as List, emit: true);
        }
        return null;
      case 'notification':
        if (args is Map) {
          control.triggerEvent('notification', _stringMap(args));
        }
        return null;
      case 'background_task':
        if (args is Map) {
          control.triggerEvent('background_task', _stringMap(args));
        }
        return null;
      default:
        throw MissingPluginException(
            'GlossarionNative: unknown platform call ${call.method}');
    }
  }

  /// Data sent by GlossarionTaskHandler from the foreground-service isolate.
  void _onTaskData(Object data) {
    if (data is Map) {
      control.triggerEvent('foreground', _stringMap(data));
    }
  }

  void _addShared(List<dynamic> raw, {required bool emit}) {
    final List<Map<String, dynamic>> added = <Map<String, dynamic>>[];
    for (final dynamic item in raw) {
      if (item is! Map) continue;
      final Map<String, dynamic> map = _stringMap(item);
      String id = (map['id'] ?? '').toString();
      if (id.isEmpty) {
        id = _newId('item');
        map['id'] = id;
      }
      if (_sharedIds.contains(id)) continue;
      _sharedIds.add(id);
      _shared.add(map);
      added.add(map);
    }
    if (emit && added.isNotEmpty) {
      control.triggerEvent('share', <String, dynamic>{'items': added});
    }
  }

  /// receive_sharing_intent delivers SEND intents that reach MainActivity
  /// directly and (in a later milestone) the iOS Share Extension. Android
  /// "Open with"/"Share" normally go through ShareReceiverActivity and the
  /// Kotlin plugin instead.
  void _onSharingIntentMedia(List<SharedMediaFile> files, {required bool initial}) {
    final List<Map<String, dynamic>> items = <Map<String, dynamic>>[];
    for (final SharedMediaFile file in files) {
      final Map<String, dynamic>? mapped = _fromSharingIntent(file);
      if (mapped != null) items.add(mapped);
    }
    if (items.isNotEmpty) {
      _addShared(items, emit: !initial);
    }
  }

  Map<String, dynamic>? _fromSharingIntent(SharedMediaFile file) {
    final String path = file.path;
    final String id = _newId('rsi');
    switch (file.type) {
      case SharedMediaType.url:
        // Deep links into this app are routed by Flet (page.route); never
        // treat them as shared content.
        if (path.toLowerCase().startsWith('glossarion:')) return null;
        return <String, dynamic>{
          'id': id,
          'kind': 'url',
          'text': path,
          'source': 'share',
        };
      case SharedMediaType.text:
        // A text/* file shared as a stream is reported as "text" with a path.
        if (!_isExistingFile(path)) {
          return <String, dynamic>{
            'id': id,
            'kind': 'text',
            'text': path,
            'mime_type': file.mimeType,
            'source': 'share',
          };
        }
        break;
      default:
        break;
    }
    return <String, dynamic>{
      'id': id,
      'kind': 'file',
      'path': path,
      'name': _basename(path),
      'mime_type': file.mimeType,
      'source': 'share',
    };
  }

  // -------------------------------------------------------- Python -> Dart

  Future<dynamic> _invokeMethod(String name, dynamic args) async {
    try {
      switch (name) {
        case 'get_platform_info':
          return await _platformInfo();
        case 'get_initial_shared':
          await _waitReady();
          return List<Map<String, dynamic>>.from(_shared);
        case 'clear_shared':
          await _clearShared(args);
          return null;
        case 'get_launch_notification':
          await _waitReady();
          return _launchNotification;
        case 'init_notifications':
          return _isMobile
              ? (await _platform('init_notifications', args)) == true
              : false;
        case 'show_notification':
          return _isMobile
              ? (await _platform('show_notification', args)) == true
              : false;
        case 'cancel_notification':
          if (_isMobile) await _platform('cancel_notification', args);
          return null;
        case 'start_job_service':
          return await _startJobService(args);
        case 'update_job_service':
          return await _updateJobService(args);
        case 'stop_job_service':
          return await _stopJobService();
        case 'is_job_service_running':
          return _isAndroid ? await FlutterForegroundTask.isRunningService : false;
        case 'begin_background_task':
          if (!_isIOS) return -1;
          final dynamic taskId = await _platform('begin_background_task', args);
          return taskId is num ? taskId.toInt() : -1;
        case 'end_background_task':
          if (_isIOS) await _platform('end_background_task', args);
          return null;
        case 'background_time_remaining':
          return _isIOS ? await _platform('background_time_remaining', args) : null;
        case 'start_continued_processing':
          return _isIOS
              ? (await _platform('start_continued_processing', args)) == true
              : false;
        case 'update_continued_processing':
          return _isIOS
              ? (await _platform('update_continued_processing', args)) == true
              : false;
        case 'finish_continued_processing':
          if (_isIOS) await _platform('finish_continued_processing', args);
          return null;
        case 'save_to_downloads':
          return _isAndroid ? await _platform('save_to_downloads', args) : null;
        default:
          throw Exception('Unknown GlossarionNative method: $name');
      }
    } on MissingPluginException catch (e) {
      // The platform side does not implement this method on this OS.
      _lastError = '$name: $e';
      debugPrint("GlossarionNative: $_lastError");
      return _defaultResult(name);
    } on PlatformException catch (e) {
      _lastError = '$name: ${e.code}: ${e.message}';
      debugPrint("GlossarionNative: $_lastError");
      throw Exception('GlossarionNative.$name failed: ${e.code}: ${e.message}');
    }
  }

  dynamic _defaultResult(String name) {
    switch (name) {
      case 'begin_background_task':
        return -1;
      case 'init_notifications':
      case 'show_notification':
      case 'start_job_service':
      case 'update_job_service':
      case 'stop_job_service':
      case 'is_job_service_running':
      case 'start_continued_processing':
      case 'update_continued_processing':
        return false;
      default:
        return null;
    }
  }

  Future<dynamic> _platform(String method, dynamic args) async {
    final dynamic result = await _platformChannel.invokeMethod<dynamic>(
        method, args == null ? null : _normalize(args));
    return _normalize(result);
  }

  Future<Map<String, dynamic>> _platformInfo() async {
    final Map<String, dynamic> info = <String, dynamic>{
      'platform': _isAndroid
          ? 'android'
          : _isIOS
              ? 'ios'
              : defaultTargetPlatform.name,
      'extension_version': kGlossarionNativeVersion,
    };
    if (_isMobile) {
      try {
        final dynamic native = await _platform('get_platform_info', null);
        if (native is Map) {
          info.addAll(_stringMap(native));
        }
      } catch (e) {
        info['platform_error'] = e.toString();
      }
    }
    if (_isAndroid) {
      try {
        info['job_service_running'] = await FlutterForegroundTask.isRunningService;
      } catch (e) {
        info['job_service_error'] = e.toString();
      }
    }
    info['pending_shared'] = _shared.length;
    if (_lastError != null) {
      info['last_error'] = _lastError;
    }
    return info;
  }

  Future<void> _clearShared(dynamic args) async {
    final bool deleteFiles = args is Map && args['delete_files'] == true;
    _shared.clear();
    if (!_isMobile) return;
    try {
      await _platformChannel.invokeMethod<dynamic>(
          'clear_shared', <String, dynamic>{'delete_files': deleteFiles});
    } catch (e) {
      debugPrint("GlossarionNative: clear_shared: $e");
    }
    try {
      await ReceiveSharingIntent.instance.reset();
    } catch (_) {}
  }

  // ------------------------------------------ Android foreground service

  Future<bool> _startJobService(dynamic args) async {
    if (!_isAndroid) return false;
    final Map<dynamic, dynamic> a =
        args is Map ? args : const <dynamic, dynamic>{};
    final String title = (a['title'] ?? 'Glossarion').toString();
    final String text = (a['text'] ?? '').toString();
    final String channelId = (a['channel_id'] ?? 'jobs.progress').toString();
    final String channelName = (a['channel_name'] ?? 'Job progress').toString();
    final dynamic rawServiceId = a['service_id'];
    final int serviceId = rawServiceId is num ? rawServiceId.toInt() : _kJobServiceId;

    final List<NotificationButton> buttons = <NotificationButton>[];
    final dynamic rawButtons = a['buttons'];
    if (rawButtons is List) {
      for (final dynamic b in rawButtons) {
        if (b is Map) {
          final String id = (b['id'] ?? '').toString();
          final String label = (b['text'] ?? '').toString();
          if (id.isNotEmpty && label.isNotEmpty) {
            buttons.add(NotificationButton(id: id, text: label));
          }
        }
      }
    }

    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: channelId.isEmpty ? 'jobs.progress' : channelId,
        channelName: channelName.isEmpty ? 'Job progress' : channelName,
        channelDescription: 'Ongoing translation and tool jobs',
        channelImportance: NotificationChannelImportance.LOW,
        priority: NotificationPriority.LOW,
        onlyAlertOnce: true,
      ),
      iosNotificationOptions: const IOSNotificationOptions(
        showNotification: false,
        playSound: false,
      ),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.nothing(),
        autoRunOnBoot: false,
        autoRunOnMyPackageReplaced: false,
        allowWakeLock: a['wake_lock'] != false,
        allowWifiLock: a['wifi_lock'] != false,
        // A restarted service would have no Python job behind it.
        allowAutoRestart: false,
        // stopWithTask stays unset (null): android:stopWithTask="true" in the
        // merged manifest already stops the service when the app is swiped
        // away. Setting it here makes flutter_foreground_task 11.x stop the
        // service as soon as no activity of the app is resumed
        // (TrackVisibilityUtils), i.e. when Home is pressed or the sign-in
        // Custom Tab opens - which left the OAuth loopback unprotected.
      ),
    );

    if (await FlutterForegroundTask.isRunningService) {
      final ServiceRequestResult updated = await FlutterForegroundTask.updateService(
        notificationTitle: title,
        notificationText: text,
        notificationButtons: buttons,
      );
      return _serviceOk(updated, 'start_job_service(update)');
    }

    final ServiceRequestResult result = await FlutterForegroundTask.startService(
      serviceId: serviceId,
      serviceTypes: const <ForegroundServiceTypes>[ForegroundServiceTypes.dataSync],
      notificationTitle: title,
      notificationText: text,
      notificationIcon: const NotificationIcon(metaDataName: _kNotificationIconMetaData),
      notificationButtons: buttons,
      callback: glossarionStartCallback,
    );
    return _serviceOk(result, 'start_job_service');
  }

  Future<bool> _updateJobService(dynamic args) async {
    if (!_isAndroid) return false;
    if (!await FlutterForegroundTask.isRunningService) return false;
    final Map<dynamic, dynamic> a =
        args is Map ? args : const <dynamic, dynamic>{};
    final ServiceRequestResult result = await FlutterForegroundTask.updateService(
      notificationTitle: a['title']?.toString(),
      notificationText: a['text']?.toString(),
    );
    return _serviceOk(result, 'update_job_service');
  }

  Future<bool> _stopJobService() async {
    if (!_isAndroid) return false;
    if (!await FlutterForegroundTask.isRunningService) return true;
    final ServiceRequestResult result = await FlutterForegroundTask.stopService();
    return _serviceOk(result, 'stop_job_service');
  }

  bool _serviceOk(ServiceRequestResult result, String what) {
    if (result is ServiceRequestFailure) {
      _lastError = '$what: ${result.error}';
      debugPrint("GlossarionNative: $_lastError");
      return false;
    }
    return true;
  }

  // ---------------------------------------------------------------- helpers

  String _newId(String prefix) {
    _sequence += 1;
    return '$prefix-${DateTime.now().microsecondsSinceEpoch}-$_sequence';
  }

  static bool _isExistingFile(String path) {
    if (!path.startsWith('/')) return false;
    try {
      return File(path).existsSync();
    } catch (_) {
      return false;
    }
  }

  static String _basename(String path) {
    final int slash = path.lastIndexOf('/');
    return slash >= 0 ? path.substring(slash + 1) : path;
  }

  static Map<String, dynamic> _stringMap(Map<dynamic, dynamic> source) {
    final Map<String, dynamic> out = <String, dynamic>{};
    source.forEach((dynamic key, dynamic value) {
      out[key.toString()] = _normalize(value);
    });
    return out;
  }

  static dynamic _normalize(dynamic value) {
    if (value is Map) return _stringMap(value);
    if (value is List) return value.map<dynamic>(_normalize).toList();
    return value;
  }
}
