import 'package:flutter_foreground_task/flutter_foreground_task.dart';

/// Entry point of the flutter_foreground_task isolate (Android).
///
/// flutter_foreground_task starts a second, headless FlutterEngine for the
/// foreground service and runs this callback in it. No translation work runs
/// here: the job runs on a Python thread in the main engine's process, and the
/// service only keeps that process alive. The handler forwards notification
/// buttons, taps and the service's end (including the Android 15 dataSync
/// timeout) to the main isolate, where GlossarionNativeService turns them into
/// `on_foreground` events.
@pragma('vm:entry-point')
void glossarionStartCallback() {
  FlutterForegroundTask.setTaskHandler(GlossarionTaskHandler());
}

class GlossarionTaskHandler extends TaskHandler {
  // Only null/bool/num/String/List/Map values may cross isolate groups.
  void _send(Map<String, Object?> data) {
    FlutterForegroundTask.sendDataToMain(data);
  }

  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {
    _send({'type': 'started', 'starter': starter.name});
  }

  @override
  void onRepeatEvent(DateTime timestamp) {}

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    _send({
      'type': isTimeout ? 'timeout' : 'destroyed',
      'is_timeout': isTimeout,
    });
  }

  @override
  void onReceiveData(Object data) {}

  @override
  void onNotificationButtonPressed(String id) {
    _send({'type': 'button', 'button_id': id});
  }

  @override
  void onNotificationPressed() {
    // The notification's content intent already brings the app to the front.
    _send({'type': 'tap'});
  }

  @override
  void onNotificationDismissed() {
    _send({'type': 'dismissed'});
  }
}
