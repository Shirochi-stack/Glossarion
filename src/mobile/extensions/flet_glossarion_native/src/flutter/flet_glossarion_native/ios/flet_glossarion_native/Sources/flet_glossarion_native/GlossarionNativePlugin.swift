import BackgroundTasks
import Flutter
import UIKit
import UserNotifications

/// iOS side of GlossarionNative (FlutterMethodChannel "glossarion_native/platform").
///
/// Dart -> Swift: attach, get_platform_info, clear_shared, init_notifications,
/// show_notification, cancel_notification, begin_background_task,
/// end_background_task, background_time_remaining, start_continued_processing,
/// update_continued_processing, finish_continued_processing, and the document
/// destination methods handled by DocumentDestinations (pick_folder, ...).
/// Swift -> Dart: "share" {items}, "notification" {...}, "background_task" {...},
/// "document" {type, op_id, ...}.
///
/// "Open in" / "Copy to Glossarion" file URLs are copied (security-scoped,
/// coordinated) into tmp/shared/<batch>/ and returned as shared items. The
/// handlers return true for file URLs so Flutter's deep-link handling never
/// turns a file:// URL into a route. Both lifecycles are supported: the
/// UIApplicationDelegate path and the UIScene path that Flutter 3.44 migrates
/// unmodified app templates to.
public class GlossarionNativePlugin: NSObject, FlutterPlugin, FlutterSceneLifeCycleDelegate {
  private static let channelName = "glossarion_native/platform"
  private static let notificationIdKey = "glossarion_id"
  private static let notificationPayloadKey = "glossarion_payload"
  private static let notificationRequestPrefix = "glossarion."
  /// Local notification posted when a background grant expires (see begin_background_task).
  private static let expirationNotificationId = 41101
  private static let deepLinkScheme = "glossarion"

  private struct ExpirationNotice {
    let title: String
    let body: String
    let payload: String?
  }

  private struct BackgroundTaskInfo {
    let identifier: UIBackgroundTaskIdentifier
    let key: String
    let name: String
    let expiration: ExpirationNotice?
  }

  private var channel: FlutterMethodChannel?
  private var dartAttached = false
  private var pendingShared: [[String: Any]] = []
  private var pendingNotifications: [[String: Any]] = []
  private var launchNotification: [String: Any]?
  private var categories: [String: UNNotificationCategory] = [:]
  private var backgroundTasks: [Int: BackgroundTaskInfo] = [:]

  // BGContinuedProcessingTask (iOS 26) is stored untyped: stored properties
  // cannot carry a newer availability than the deployment target (iOS 13).
  private var continuedTask: AnyObject?
  private var continuedIdentifier: String?
  private var continuedExpiration: ExpirationNotice?
  private var pendingProgress: (completed: Int64, total: Int64, subtitle: String?)?
  private var registeredContinuedIdentifiers = Set<String>()
  private var finishedContinuedIdentifiers = Set<String>()

  private let ioQueue = DispatchQueue(label: "com.glossarion.native.io", qos: .userInitiated)

  /// Document destinations (U10): folder / file pickers with bookmarks and coordinated writes.
  private lazy var documents = DocumentDestinations(
    send: { [weak self] event in
      self?.channel?.invokeMethod("document", arguments: event)
    },
    isDartAttached: { [weak self] in
      self?.dartAttached ?? false
    }
  )

  public static func register(with registrar: FlutterPluginRegistrar) {
    let instance = GlossarionNativePlugin()
    let channel = FlutterMethodChannel(name: channelName, binaryMessenger: registrar.messenger())
    instance.channel = channel
    registrar.addMethodCallDelegate(instance, channel: channel)
    // UIApplicationDelegate events (apps not migrated to UIScene) and
    // UNUserNotificationCenterDelegate callbacks forwarded by FlutterAppDelegate.
    registrar.addApplicationDelegate(instance)
    // UISceneDelegate events (Flutter 3.44 migrates the Flet template to UIScene).
    registrar.addSceneDelegate(instance)
    instance.ensureNotificationDelegate()
  }

  // MARK: - Method channel

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let args = call.arguments as? [String: Any] ?? [:]
    if DocumentDestinations.methods.contains(call.method) {
      documents.handle(call.method, args, result: result)
      return
    }
    switch call.method {
    case "attach":
      dartAttached = true
      var response: [String: Any] = [
        "shared": pendingShared,
        "notifications": pendingNotifications,
        "document_results": documents.onDartAttach(),
      ]
      if let launch = launchNotification {
        response["launch_notification"] = launch
      }
      pendingShared.removeAll()
      pendingNotifications.removeAll()
      result(response)
    case "get_platform_info":
      platformInfo(result: result)
    case "clear_shared":
      pendingShared.removeAll()
      if (args["delete_files"] as? Bool) == true {
        let root = GlossarionNativePlugin.sharedRoot()
        ioQueue.async {
          _ = try? FileManager.default.removeItem(at: root)
        }
      }
      result(nil)
    case "init_notifications":
      initNotifications(requestPermission: (args["request_permission"] as? Bool) ?? false, result: result)
    case "show_notification":
      showNotification(args, result: result)
    case "cancel_notification":
      if let id = GlossarionNativePlugin.intValue(args["id"]) {
        let identifier = GlossarionNativePlugin.notificationRequestPrefix + String(id)
        let center = UNUserNotificationCenter.current()
        center.removePendingNotificationRequests(withIdentifiers: [identifier])
        center.removeDeliveredNotifications(withIdentifiers: [identifier])
      }
      result(nil)
    case "begin_background_task":
      result(beginBackgroundTask(args))
    case "end_background_task":
      if let taskId = GlossarionNativePlugin.intValue(args["task_id"]) {
        endBackgroundTask(taskId)
      }
      result(nil)
    case "background_time_remaining":
      let remaining = UIApplication.shared.backgroundTimeRemaining
      // DBL_MAX while the app is in the foreground.
      if remaining > 1_000_000 {
        result(nil)
      } else {
        result(remaining)
      }
    case "start_continued_processing":
      startContinuedProcessing(args, result: result)
    case "update_continued_processing":
      result(updateContinuedProcessing(args))
    case "finish_continued_processing":
      finishContinuedProcessing(success: (args["success"] as? Bool) ?? false)
      result(nil)
    case "save_to_downloads":
      // Android only: iOS exports go through Share / the Files app.
      result(nil)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func platformInfo(result: @escaping FlutterResult) {
    var info: [String: Any] = [
      "platform": "ios",
      "system_version": UIDevice.current.systemVersion,
      "model": UIDevice.current.model,
      "bundle_id": Bundle.main.bundleIdentifier ?? "",
      "continued_processing": GlossarionNativePlugin.continuedProcessingSupported,
      "background_refresh_available": UIApplication.shared.backgroundRefreshStatus == .available,
      "save_to_downloads": false,
      "documents": true,
      "fgs_types": [String](),
      "shared_dir": GlossarionNativePlugin.sharedRoot().path,
    ]
    if let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String {
      info["version_name"] = version
    }
    if let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String {
      info["version_code"] = build
    }
    let base = info
    UNUserNotificationCenter.current().getNotificationSettings { settings in
      let status = settings.authorizationStatus
      let enabled = status == .authorized || status == .provisional
      let statusName = GlossarionNativePlugin.describe(status)
      DispatchQueue.main.async {
        var out = base
        out["notifications_enabled"] = enabled
        out["notification_status"] = statusName
        result(out)
      }
    }
  }

  // MARK: - Open in / Copy to (file URLs)

  public func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [AnyHashable: Any] = [:]
  ) -> Bool {
    // A launch URL is delivered again through application(_:open:options:).
    ensureNotificationDelegate()
    return true
  }

  public func application(
    _ application: UIApplication,
    open url: URL,
    options: [UIApplication.OpenURLOptionsKey: Any] = [:]
  ) -> Bool {
    guard url.isFileURL else { return false }
    importFiles([url], origin: "open_url")
    return true
  }

  public func scene(
    _ scene: UIScene,
    willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions?
  ) -> Bool {
    ensureNotificationDelegate()
    guard let contexts = connectionOptions?.urlContexts, !contexts.isEmpty else { return false }
    let urls = contexts.map { $0.url }
    // Plugins registered later (receive_sharing_intent 1.9.0) claim every
    // cold-start connection, which makes Flutter skip deep-link handling.
    // Keep the app's own deep links as shared "url" items so Python can route
    // them (de-duplicate against page.route).
    for link in urls where !link.isFileURL {
      recordLaunchLink(link)
    }
    let files = urls.filter { $0.isFileURL }
    guard !files.isEmpty else { return false }
    importFiles(files, origin: "launch")
    return true
  }

  public func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) -> Bool {
    let files = URLContexts.map { $0.url }.filter { $0.isFileURL }
    guard !files.isEmpty else { return false }
    importFiles(files, origin: "open_url")
    return true
  }

  private func recordLaunchLink(_ url: URL) {
    guard url.scheme?.lowercased() == GlossarionNativePlugin.deepLinkScheme else { return }
    let item: [String: Any] = [
      "id": UUID().uuidString,
      "kind": "url",
      "text": url.absoluteString,
      "source": "launch",
    ]
    deliverShared([item])
  }

  private func importFiles(_ urls: [URL], origin: String) {
    let root = GlossarionNativePlugin.sharedRoot()
    let stamp = Int(Date().timeIntervalSince1970 * 1000)
    let batch = root.appendingPathComponent("\(stamp)-\(UUID().uuidString.prefix(8))", isDirectory: true)
    ioQueue.async { [weak self] in
      let items = urls.map { GlossarionNativePlugin.copySharedFile($0, into: batch, origin: origin) }
      DispatchQueue.main.async {
        self?.deliverShared(items)
      }
    }
  }

  private func deliverShared(_ items: [[String: Any]]) {
    guard !items.isEmpty else { return }
    if dartAttached, let channel = channel {
      channel.invokeMethod("share", arguments: ["items": items])
    } else {
      pendingShared.append(contentsOf: items)
    }
  }

  private static func copySharedFile(_ url: URL, into batch: URL, origin: String) -> [String: Any] {
    var item: [String: Any] = [
      "id": UUID().uuidString,
      "kind": "file",
      "uri": url.absoluteString,
      "source": origin,
      "name": url.lastPathComponent,
    ]
    let accessing = url.startAccessingSecurityScopedResource()
    defer {
      if accessing {
        url.stopAccessingSecurityScopedResource()
      }
    }
    do {
      let fileManager = FileManager.default
      try fileManager.createDirectory(at: batch, withIntermediateDirectories: true, attributes: nil)
      let destination = uniqueURL(in: batch, name: safeFileName(url.lastPathComponent))
      var coordinatorError: NSError?
      var copyError: Error?
      NSFileCoordinator(filePresenter: nil).coordinate(
        readingItemAt: url,
        options: .withoutChanges,
        error: &coordinatorError
      ) { readURL in
        do {
          try fileManager.copyItem(at: readURL, to: destination)
        } catch {
          copyError = error
        }
      }
      if let error = coordinatorError {
        throw error
      }
      if let error = copyError {
        throw error
      }
      item["path"] = destination.path
      item["name"] = destination.lastPathComponent
      if let attributes = try? fileManager.attributesOfItem(atPath: destination.path),
         let size = attributes[.size] as? NSNumber {
        item["size"] = size.int64Value
      }
      if let mime = mimeType(forExtension: destination.pathExtension) {
        item["mime_type"] = mime
      }
      // Without in-place opening iOS copies the file into Documents/Inbox;
      // drop that duplicate once we have our own copy.
      if url.path.contains("/Documents/Inbox/") {
        _ = try? fileManager.removeItem(at: url)
      }
    } catch {
      item["error"] = error.localizedDescription
    }
    return item
  }

  // MARK: - Notifications

  private func ensureNotificationDelegate() {
    // FlutterAppDelegate implements UNUserNotificationCenterDelegate and
    // forwards to every plugin registered with addApplicationDelegate, but only
    // once it is the center's delegate. The Flet template's AppDelegate does
    // not set it, so do it here (as flutter_local_notifications' README asks
    // app authors to).
    let center = UNUserNotificationCenter.current()
    if center.delegate == nil,
       let appDelegate = UIApplication.shared.delegate as? UNUserNotificationCenterDelegate {
      center.delegate = appDelegate
    }
  }

  private func initNotifications(requestPermission: Bool, result: @escaping FlutterResult) {
    ensureNotificationDelegate()
    let center = UNUserNotificationCenter.current()
    if requestPermission {
      center.requestAuthorization(options: [.alert, .sound, .badge]) { granted, _ in
        DispatchQueue.main.async {
          result(granted)
        }
      }
    } else {
      center.getNotificationSettings { settings in
        let status = settings.authorizationStatus
        let enabled = status == .authorized || status == .provisional
        DispatchQueue.main.async {
          result(enabled)
        }
      }
    }
  }

  private func showNotification(_ args: [String: Any], result: @escaping FlutterResult) {
    guard let id = GlossarionNativePlugin.intValue(args["id"]) else {
      result(false)
      return
    }
    let content = UNMutableNotificationContent()
    content.title = args["title"] as? String ?? ""
    content.body = args["body"] as? String ?? ""
    if let thread = args["channel_id"] as? String, !thread.isEmpty {
      content.threadIdentifier = thread
    }
    var userInfo: [AnyHashable: Any] = [GlossarionNativePlugin.notificationIdKey: id]
    if let payload = args["payload"] as? String {
      userInfo[GlossarionNativePlugin.notificationPayloadKey] = payload
    }
    content.userInfo = userInfo
    if let progress = args["progress"] as? [Any], progress.count == 2,
       let done = GlossarionNativePlugin.intValue(progress[0]),
       let total = GlossarionNativePlugin.intValue(progress[1]), total > 0 {
      content.subtitle = "\(done)/\(total)"
    }
    if (args["silent"] as? Bool) != true {
      content.sound = .default
    }
    if let actions = args["actions"] as? [[String: Any]], !actions.isEmpty {
      content.categoryIdentifier = registerCategory(actions)
    }
    let request = UNNotificationRequest(
      identifier: GlossarionNativePlugin.notificationRequestPrefix + String(id),
      content: content,
      trigger: nil
    )
    UNUserNotificationCenter.current().add(request) { error in
      let ok = error == nil
      DispatchQueue.main.async {
        result(ok)
      }
    }
  }

  private func registerCategory(_ actions: [[String: Any]]) -> String {
    var notificationActions: [UNNotificationAction] = []
    var ids: [String] = []
    for action in actions {
      guard let actionId = action["id"] as? String, !actionId.isEmpty else { continue }
      let title = action["title"] as? String ?? actionId
      notificationActions.append(
        UNNotificationAction(identifier: actionId, title: title, options: [.foreground])
      )
      ids.append(actionId)
    }
    let identifier = "glossarion.category." + ids.joined(separator: ".")
    if categories[identifier] == nil {
      categories[identifier] = UNNotificationCategory(
        identifier: identifier,
        actions: notificationActions,
        intentIdentifiers: [],
        options: []
      )
      UNUserNotificationCenter.current().setNotificationCategories(Set(categories.values))
    }
    return identifier
  }

  private func postLocalNotice(_ notice: ExpirationNotice) {
    let content = UNMutableNotificationContent()
    content.title = notice.title
    content.body = notice.body
    var userInfo: [AnyHashable: Any] = [
      GlossarionNativePlugin.notificationIdKey: GlossarionNativePlugin.expirationNotificationId
    ]
    if let payload = notice.payload {
      userInfo[GlossarionNativePlugin.notificationPayloadKey] = payload
    }
    content.userInfo = userInfo
    content.sound = .default
    let request = UNNotificationRequest(
      identifier: GlossarionNativePlugin.notificationRequestPrefix
        + String(GlossarionNativePlugin.expirationNotificationId),
      content: content,
      trigger: nil
    )
    UNUserNotificationCenter.current().add(request, withCompletionHandler: nil)
  }

  // Forwarded by FlutterAppDelegate (UNUserNotificationCenterDelegate). Only
  // notifications posted by this plugin are handled; others are left to the
  // plugin that owns them (which then calls the completion handler).
  public func userNotificationCenter(
    _ center: UNUserNotificationCenter,
    willPresent notification: UNNotification,
    withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
  ) {
    guard notification.request.content.userInfo[GlossarionNativePlugin.notificationIdKey] != nil else {
      return
    }
    if #available(iOS 14.0, *) {
      completionHandler([.banner, .list, .sound])
    } else {
      completionHandler([.alert, .sound])
    }
  }

  public func userNotificationCenter(
    _ center: UNUserNotificationCenter,
    didReceive response: UNNotificationResponse,
    withCompletionHandler completionHandler: @escaping () -> Void
  ) {
    let userInfo = response.notification.request.content.userInfo
    guard let id = GlossarionNativePlugin.intValue(userInfo[GlossarionNativePlugin.notificationIdKey]) else {
      return
    }
    var event: [String: Any] = [
      "notification_id": id,
      "launched_app": !dartAttached,
    ]
    if let payload = userInfo[GlossarionNativePlugin.notificationPayloadKey] as? String {
      event["payload"] = payload
    }
    let action = response.actionIdentifier
    if action != UNNotificationDefaultActionIdentifier && action != UNNotificationDismissActionIdentifier {
      event["action_id"] = action
    }
    deliverNotification(event)
    completionHandler()
  }

  private func deliverNotification(_ event: [String: Any]) {
    if dartAttached, let channel = channel {
      channel.invokeMethod("notification", arguments: event)
    } else if launchNotification == nil {
      launchNotification = event
    } else {
      pendingNotifications.append(event)
    }
  }

  // MARK: - beginBackgroundTask

  private func beginBackgroundTask(_ args: [String: Any]) -> Int {
    let name = (args["name"] as? String) ?? "Glossarion job"
    let key = UUID().uuidString
    let identifier = UIApplication.shared.beginBackgroundTask(withName: name) { [weak self] in
      self?.backgroundTaskExpired(key: key)
    }
    if identifier == .invalid {
      return -1
    }
    backgroundTasks[identifier.rawValue] = BackgroundTaskInfo(
      identifier: identifier,
      key: key,
      name: name,
      expiration: GlossarionNativePlugin.expirationNotice(args)
    )
    return identifier.rawValue
  }

  private func endBackgroundTask(_ rawValue: Int) {
    guard let info = backgroundTasks.removeValue(forKey: rawValue) else { return }
    UIApplication.shared.endBackgroundTask(info.identifier)
  }

  /// Called on the main thread shortly before the grant runs out. The task must
  /// be ended here or iOS terminates the app.
  private func backgroundTaskExpired(key: String) {
    guard let entry = backgroundTasks.first(where: { $0.value.key == key }) else { return }
    let info = entry.value
    backgroundTasks.removeValue(forKey: entry.key)
    if let notice = info.expiration {
      postLocalNotice(notice)
    }
    let event: [String: Any] = [
      "type": "expiring",
      "task_id": info.identifier.rawValue,
      "task_name": info.name,
    ]
    channel?.invokeMethod("background_task", arguments: event)
    UIApplication.shared.endBackgroundTask(info.identifier)
  }

  // MARK: - BGContinuedProcessingTask (iOS 26)

  private static var continuedProcessingSupported: Bool {
#if compiler(>=6.2)
    if #available(iOS 26.0, *) {
      return true
    }
#endif
    return false
  }

  private func startContinuedProcessing(_ args: [String: Any], result: @escaping FlutterResult) {
#if compiler(>=6.2)
    if #available(iOS 26.0, *) {
      submitContinuedProcessing(args, result: result)
      return
    }
#endif
    result(false)
  }

#if compiler(>=6.2)
  @available(iOS 26.0, *)
  private func submitContinuedProcessing(_ args: [String: Any], result: @escaping FlutterResult) {
    let bundleId = Bundle.main.bundleIdentifier ?? "com.glossarion.app"
    let prefix = bundleId + ".job."
    var requested = (args["identifier"] as? String) ?? ""
    if requested.isEmpty || requested.hasSuffix("*") || requested == prefix {
      let suffix = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
      requested = prefix + String(suffix.prefix(12))
    }
    let taskIdentifier = requested
    guard taskIdentifier.hasPrefix(prefix) else {
      emitContinuedFailure(taskIdentifier, reason: "identifier must start with \(prefix)")
      result(false)
      return
    }
    // iOS terminates an app that registers the same identifier twice.
    guard !registeredContinuedIdentifiers.contains(taskIdentifier) else {
      emitContinuedFailure(taskIdentifier, reason: "identifier already used in this process")
      result(false)
      return
    }

    let scheduler = BGTaskScheduler.shared
    let registered = scheduler.register(forTaskWithIdentifier: taskIdentifier, using: nil) { [weak self] task in
      task.expirationHandler = { [weak self, weak task] in
        task?.setTaskCompleted(success: false)
        DispatchQueue.main.async {
          self?.continuedTaskExpired(identifier: taskIdentifier)
        }
      }
      DispatchQueue.main.async {
        self?.continuedTaskLaunched(task, identifier: taskIdentifier)
      }
    }
    guard registered else {
      emitContinuedFailure(
        taskIdentifier,
        reason: "register failed: \(prefix)* missing from BGTaskSchedulerPermittedIdentifiers?"
      )
      result(false)
      return
    }
    registeredContinuedIdentifiers.insert(taskIdentifier)

    let title = (args["title"] as? String) ?? "Glossarion"
    let subtitle = (args["subtitle"] as? String) ?? ""
    let request = BGContinuedProcessingTaskRequest(identifier: taskIdentifier, title: title, subtitle: subtitle)
    request.strategy = ((args["strategy"] as? String) == "queue") ? .queue : .fail
    do {
      try scheduler.submit(request)
    } catch {
      emitContinuedFailure(taskIdentifier, reason: error.localizedDescription)
      result(false)
      return
    }

    // A previous job that never called finish: close its task first.
    if let previous = continuedIdentifier {
      finishedContinuedIdentifiers.insert(previous)
      (continuedTask as? BGContinuedProcessingTask)?.setTaskCompleted(success: false)
    }
    continuedTask = nil
    continuedIdentifier = taskIdentifier
    continuedExpiration = GlossarionNativePlugin.expirationNotice(args)
    pendingProgress = nil
    result(true)
  }

  @available(iOS 26.0, *)
  private func continuedTaskLaunched(_ task: BGTask, identifier: String) {
    guard let continued = task as? BGContinuedProcessingTask else {
      task.setTaskCompleted(success: false)
      return
    }
    if finishedContinuedIdentifiers.contains(identifier) || continuedIdentifier != identifier {
      // The job ended (or was replaced) before the system launched the task.
      continued.setTaskCompleted(success: true)
      return
    }
    continuedTask = continued
    if let pending = pendingProgress {
      continued.progress.totalUnitCount = pending.total
      continued.progress.completedUnitCount = pending.completed
      if let subtitle = pending.subtitle {
        continued.updateTitle(continued.title, subtitle: subtitle)
      }
      pendingProgress = nil
    }
    channel?.invokeMethod("background_task", arguments: [
      "type": "continued_started",
      "identifier": identifier,
    ])
  }
#endif

  private func emitContinuedFailure(_ identifier: String, reason: String) {
    channel?.invokeMethod("background_task", arguments: [
      "type": "continued_failed",
      "identifier": identifier,
      "reason": reason,
    ])
  }

  /// The expiration handler already completed the task (unsuccessfully).
  private func continuedTaskExpired(identifier: String) {
    if continuedIdentifier == identifier {
      if let notice = continuedExpiration {
        postLocalNotice(notice)
      }
      continuedTask = nil
      continuedIdentifier = nil
      continuedExpiration = nil
      pendingProgress = nil
    }
    finishedContinuedIdentifiers.insert(identifier)
    channel?.invokeMethod("background_task", arguments: [
      "type": "continued_expired",
      "identifier": identifier,
    ])
  }

  private func updateContinuedProcessing(_ args: [String: Any]) -> Bool {
    let total = Int64(max(GlossarionNativePlugin.intValue(args["total"]) ?? 0, 1))
    let completed = Int64(min(max(GlossarionNativePlugin.intValue(args["completed"]) ?? 0, 0), Int(total)))
    let subtitle = args["subtitle"] as? String
#if compiler(>=6.2)
    if #available(iOS 26.0, *), let task = continuedTask as? BGContinuedProcessingTask {
      task.progress.totalUnitCount = total
      task.progress.completedUnitCount = completed
      if let subtitle = subtitle {
        task.updateTitle(task.title, subtitle: subtitle)
      }
      return true
    }
#endif
    if continuedIdentifier != nil {
      pendingProgress = (completed: completed, total: total, subtitle: subtitle)
    }
    return false
  }

  private func finishContinuedProcessing(success: Bool) {
    if let identifier = continuedIdentifier {
      finishedContinuedIdentifiers.insert(identifier)
    }
#if compiler(>=6.2)
    if #available(iOS 26.0, *), let task = continuedTask as? BGContinuedProcessingTask {
      if success {
        task.progress.completedUnitCount = task.progress.totalUnitCount
      }
      task.setTaskCompleted(success: success)
    }
#endif
    continuedTask = nil
    continuedIdentifier = nil
    continuedExpiration = nil
    pendingProgress = nil
  }

  // MARK: - Helpers

  private static func expirationNotice(_ args: [String: Any]) -> ExpirationNotice? {
    let title = (args["expiration_title"] as? String) ?? ""
    let body = (args["expiration_body"] as? String) ?? ""
    if title.isEmpty && body.isEmpty {
      return nil
    }
    return ExpirationNotice(title: title, body: body, payload: args["expiration_payload"] as? String)
  }

  private static func intValue(_ value: Any?) -> Int? {
    if let number = value as? NSNumber {
      return number.intValue
    }
    if let int = value as? Int {
      return int
    }
    return nil
  }

  private static func describe(_ status: UNAuthorizationStatus) -> String {
    switch status {
    case .notDetermined:
      return "not_determined"
    case .denied:
      return "denied"
    case .authorized:
      return "authorized"
    case .provisional:
      return "provisional"
    default:
      return "other"
    }
  }

  private static func sharedRoot() -> URL {
    return FileManager.default.temporaryDirectory.appendingPathComponent("shared", isDirectory: true)
  }

  static func safeFileName(_ raw: String) -> String {
    let forbidden = CharacterSet(charactersIn: "/\\:*?\"<>|").union(.controlCharacters)
    var name = raw.components(separatedBy: forbidden).joined(separator: "_")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    while name.hasPrefix(".") {
      name.removeFirst()
    }
    if name.isEmpty {
      name = "shared_\(Int(Date().timeIntervalSince1970))"
    }
    if name.count > 120 {
      let ext = (name as NSString).pathExtension
      let keepExtension = !ext.isEmpty && ext.count <= 10
      let stemSource = keepExtension ? (name as NSString).deletingPathExtension : name
      let stem = String(stemSource.prefix(keepExtension ? 120 - ext.count - 1 : 120))
      name = keepExtension ? stem + "." + ext : stem
    }
    return name
  }

  private static func uniqueURL(in directory: URL, name: String) -> URL {
    let fileManager = FileManager.default
    var candidate = directory.appendingPathComponent(name)
    if !fileManager.fileExists(atPath: candidate.path) {
      return candidate
    }
    let ext = (name as NSString).pathExtension
    let stem = (name as NSString).deletingPathExtension
    var counter = 1
    while fileManager.fileExists(atPath: candidate.path) {
      let numbered = ext.isEmpty ? "\(stem) (\(counter))" : "\(stem) (\(counter)).\(ext)"
      candidate = directory.appendingPathComponent(numbered)
      counter += 1
    }
    return candidate
  }

  static func mimeType(forExtension rawExtension: String) -> String? {
    switch rawExtension.lowercased() {
    case "epub": return "application/epub+zip"
    case "pdf": return "application/pdf"
    case "txt": return "text/plain"
    case "md", "markdown": return "text/markdown"
    case "html", "htm": return "text/html"
    case "xhtml": return "application/xhtml+xml"
    case "zip": return "application/zip"
    case "cbz": return "application/vnd.comicbook+zip"
    case "json": return "application/json"
    case "csv": return "text/csv"
    case "srt": return "application/x-subrip"
    case "vtt": return "text/vtt"
    case "ass", "ssa": return "text/x-ssa"
    case "sdlxliff", "xliff", "xlf": return "application/xliff+xml"
    case "png": return "image/png"
    case "jpg", "jpeg": return "image/jpeg"
    case "webp": return "image/webp"
    case "gif": return "image/gif"
    default: return nil
    }
  }
}
