import Flutter
import UIKit
import WebKit

/// In-app browser pages (hotfix 3) on iOS: hidden WKWebViews this plugin instance creates and owns
/// outside the Flutter widget tree, so Flet screen changes never rebuild them. Contract:
/// flet_glossarion_native/web_pages.py.
///
/// Methods (routed by GlossarionNativePlugin): web_open, web_load, web_eval, web_title, web_close,
/// web_reset, web_status. Every answer is a map {ok, error, message, ...}; typed failures never use
/// FlutterError. Event "web" {page_id, type, ...}: started, finished, url, error, gone.
///
/// - The pages live in a 1 x 1 pt clipped UIView at index 0 of the key window (behind the opaque
///   FlutterView), not hidden, alpha 1, no user interaction, hidden from accessibility. Each WKWebView
///   has its viewport size in points.
/// - WKWebsiteDataStore.default() and the default process pool, like flet-webview
///   (webview_flutter_wkwebview): cookies and localStorage carry over from the visible sign-in pages.
/// - No uiDelegate (alert/confirm complete silently, window.open opens nothing), no user scripts (the
///   routes keep their own iframe hook), no decidePolicyFor: WebKit never hands intent: / market: links
///   to other apps, so `prevent_links` is accepted and ignored here.
/// - WebKit still suspends web content soon after the app leaves the screen; requests pause until
///   Glossarion is back. A web content process that terminates reports `gone` (`crashed` null:
///   WebKit does not say whether it crashed or iOS killed it) and the page is lost.
/// - Every FlutterResult is answered exactly once (Reply); a pending web_eval answers closed / gone when
///   its page is destroyed. Everything runs on the main thread (the channel handler, the navigation
///   delegate, the evaluateJavaScript completion, channel.invokeMethod). No @MainActor, no async API,
///   no iOS 14+ API (deployment target iOS 13).
final class HiddenWebPages: NSObject, WKNavigationDelegate {
  static let methods: Set<String> = [
    "web_open", "web_load", "web_eval", "web_title", "web_close", "web_reset", "web_status",
  ]

  // flet_glossarion_native/web_pages.py WEB_PAGES_VERSION / NATIVE_MAX_PAGES.
  static let version = 1
  static let maxPages = 8
  private static let maxSide = 4096
  private static let defaultWidth = 480
  private static let defaultHeight = 800
  private static let maxIdLength = 64

  // Error codes (web_pages.py WebPageError; "timeout" is Python's own).
  static let errUnavailable = "unavailable"
  static let errBadArgs = "bad_args"
  static let errExists = "exists"
  static let errTooMany = "too_many"
  static let errNoPage = "no_page"
  static let errGone = "gone"
  static let errClosed = "closed"
  static let errNative = "native_error"

  // Event types (web_pages.py WEB_EVENT_TYPES).
  static let eventStarted = "started"
  static let eventFinished = "finished"
  static let eventUrl = "url"
  static let eventError = "error"
  static let eventGone = "gone"

  /// A FlutterResult answered at most once.
  private final class Reply {
    private let result: FlutterResult
    private(set) var replied = false

    init(_ result: @escaping FlutterResult) {
      self.result = result
    }

    func send(_ answer: [String: Any]) {
      if replied {
        return
      }
      replied = true
      result(answer)
    }
  }

  private final class Page {
    let id: String
    let webView: WKWebView
    let width: Int
    let height: Int
    var lastUrl: String?
    var mainFrameError: String?
    var gone = false
    var destroyed = false
    var evals: [Reply] = []

    init(id: String, webView: WKWebView, width: Int, height: Int) {
      self.id = id
      self.webView = webView
      self.width = width
      self.height = height
    }
  }

  private let send: ([String: Any]) -> Void
  private var pages: [String: Page] = [:]
  private var containerView: UIView?

  init(send: @escaping ([String: Any]) -> Void) {
    self.send = send
    super.init()
  }

  /// get_platform_info()["web_pages"].
  func capability() -> [String: Any] {
    return ["version": HiddenWebPages.version, "max_pages": HiddenWebPages.maxPages, "provider": "WKWebView"]
  }

  // MARK: - Dispatch (main thread)

  func handle(_ method: String, _ args: [String: Any], result: @escaping FlutterResult) {
    let reply = Reply(result)
    switch method {
    case "web_open":
      openPage(args, reply)
    case "web_load":
      withPage(args, reply) { page in
        self.load(page, args["url"] as? String, reply)
      }
    case "web_eval":
      withPage(args, reply) { page in
        self.eval(page, args["script"] as? String, reply)
      }
    case "web_title":
      withPage(args, reply) { page in
        self.title(page, reply)
      }
    case "web_close":
      close(args["page_id"] as? String, reply)
    case "web_reset":
      var answer = HiddenWebPages.ok()
      answer["closed"] = reset()
      reply.send(answer)
    case "web_status":
      reply.send(status())
    default:
      reply.send(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "unknown in-app browser method"))
    }
  }

  private func withPage(_ args: [String: Any], _ reply: Reply, _ action: (Page) -> Void) {
    guard let id = args["page_id"] as? String, let page = pages[id] else {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errNoPage, "no such in-app page"))
      return
    }
    if page.gone {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errGone, "the in-app page's renderer stopped"))
      return
    }
    if page.destroyed {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errClosed, "the in-app page was closed"))
      return
    }
    action(page)
  }

  // MARK: - Pages

  private func openPage(_ args: [String: Any], _ reply: Reply) {
    guard let id = args["page_id"] as? String, !id.isEmpty, id.count <= HiddenWebPages.maxIdLength else {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "page_id is required"))
      return
    }
    if pages[id] != nil {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errExists, "the page is already open"))
      return
    }
    if pages.count >= HiddenWebPages.maxPages {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errTooMany, "too many in-app pages"))
      return
    }
    let width = HiddenWebPages.side(args["width"], HiddenWebPages.defaultWidth)
    let height = HiddenWebPages.side(args["height"], HiddenWebPages.defaultHeight)
    let configuration = WKWebViewConfiguration()
    // Shared with the flet-webview pages: logins carry over.
    configuration.websiteDataStore = WKWebsiteDataStore.default()
    let webView = WKWebView(
      frame: CGRect(x: 0, y: 0, width: width, height: height), configuration: configuration)
    webView.navigationDelegate = self
    webView.isUserInteractionEnabled = false
    webView.isAccessibilityElement = false
    webView.accessibilityElementsHidden = true
    let page = Page(id: id, webView: webView, width: width, height: height)
    pages[id] = page
    ensureContainer().addSubview(webView)
    attach()
    var answer = HiddenWebPages.ok()
    answer["page_id"] = id
    answer["attached"] = containerView?.window != nil
    reply.send(answer)
  }

  private func load(_ page: Page, _ raw: String?, _ reply: Reply) {
    guard let text = raw, let url = URL(string: text), let scheme = url.scheme?.lowercased(),
          scheme == "http" || scheme == "https" else {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "only http(s) pages can be opened"))
      return
    }
    attach()
    page.lastUrl = text
    _ = page.webView.load(URLRequest(url: url))
    reply.send(HiddenWebPages.ok())
  }

  private func eval(_ page: Page, _ script: String?, _ reply: Reply) {
    guard let source = script else {
      reply.send(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "script is required"))
      return
    }
    attach()
    page.evals.append(reply)
    page.webView.evaluateJavaScript(source) { value, _ in
      if reply.replied {
        // The page was destroyed meanwhile and answered closed / gone already.
        return
      }
      page.evals.removeAll { $0 === reply }
      let url = page.webView.url?.absoluteString ?? page.lastUrl
      page.lastUrl = url
      var answer = HiddenWebPages.ok()
      // A JavaScript string arrives as NSString; anything else (or a script error) is null.
      answer["result"] = DocumentDestinations.orNull(value as? String)
      answer["url"] = DocumentDestinations.orNull(url)
      reply.send(answer)
    }
  }

  private func title(_ page: Page, _ reply: Reply) {
    attach()
    let url = page.webView.url?.absoluteString ?? page.lastUrl
    page.lastUrl = url
    var answer = HiddenWebPages.ok()
    answer["title"] = DocumentDestinations.orNull(page.webView.title)
    answer["url"] = DocumentDestinations.orNull(url)
    reply.send(answer)
  }

  private func close(_ id: String?, _ reply: Reply) {
    var closed = false
    if let id = id, let page = pages.removeValue(forKey: id) {
      destroy(page, HiddenWebPages.errClosed)
      closed = true
    }
    var answer = HiddenWebPages.ok()
    answer["closed"] = closed
    reply.send(answer)
  }

  /// Close every page of this plugin instance (web_reset, engine detach); how many there were.
  @discardableResult
  func reset() -> Int {
    let all = Array(pages.values)
    pages.removeAll()
    for page in all {
      destroy(page, HiddenWebPages.errClosed)
    }
    containerView?.removeFromSuperview()
    return all.count
  }

  private func status() -> [String: Any] {
    var list: [[String: Any]] = []
    for page in pages.values {
      var entry: [String: Any] = [
        "page_id": page.id, "gone": page.gone, "width": page.width, "height": page.height,
        "pending": page.evals.count,
      ]
      entry["url"] = DocumentDestinations.orNull(page.lastUrl)
      list.append(entry)
    }
    var answer = HiddenWebPages.ok()
    answer["pages"] = list
    answer["attached"] = containerView?.window != nil
    answer["window"] = DocumentDestinations.keyWindow() != nil
    return answer
  }

  /// Stops and removes the page's WKWebView; pending evals answer `answer`.
  private func destroy(_ page: Page, _ answer: String) {
    if page.destroyed {
      return
    }
    page.destroyed = true
    let waiting = page.evals
    page.evals.removeAll()
    var message = "the in-app page was closed"
    if answer == HiddenWebPages.errGone {
      message = "the in-app page's renderer stopped"
    }
    for pending in waiting {
      pending.send(HiddenWebPages.fail(answer, message))
    }
    let webView = page.webView
    webView.navigationDelegate = nil
    if !page.gone {
      webView.stopLoading()
    }
    webView.removeFromSuperview()
    if !hasLivePage() {
      containerView?.removeFromSuperview()
    }
  }

  private func emit(_ page: Page, _ type: String, _ extras: [String: Any] = [:]) {
    if page.destroyed {
      return
    }
    var event: [String: Any] = ["page_id": page.id, "type": type]
    for (key, value) in extras {
      event[key] = value
    }
    send(event)
  }

  private func livePage(for webView: WKWebView) -> Page? {
    return pages.values.first { $0.webView === webView && !$0.destroyed }
  }

  private func hasLivePage() -> Bool {
    return pages.values.contains { !$0.destroyed }
  }

  // MARK: - Container

  private func ensureContainer() -> UIView {
    if let view = containerView {
      return view
    }
    let view = UIView(frame: CGRect(x: 0, y: 0, width: 1, height: 1))
    view.clipsToBounds = true
    view.isUserInteractionEnabled = false
    view.isAccessibilityElement = false
    view.accessibilityElementsHidden = true
    view.backgroundColor = UIColor.clear
    containerView = view
    return view
  }

  /// Puts the container behind the root view of the key window (an identity check when it is there).
  /// A WKWebView outside any window is throttled by WebKit, so every page call attaches first.
  private func attach() {
    guard let view = containerView else {
      return
    }
    if !hasLivePage() {
      view.removeFromSuperview()
      return
    }
    guard let window = DocumentDestinations.keyWindow() else {
      return
    }
    if view.superview !== window {
      view.removeFromSuperview()
      window.insertSubview(view, at: 0)
    }
  }

  private static func side(_ value: Any?, _ fallback: Int) -> Int {
    guard let number = value as? NSNumber else {
      return fallback
    }
    let raw = number.doubleValue
    if raw.isNaN {
      return fallback
    }
    return Int(min(max(raw.rounded(), 1), Double(maxSide)))
  }

  static func ok(_ values: [String: Any] = [:]) -> [String: Any] {
    var out: [String: Any] = ["ok": true, "error": NSNull(), "message": NSNull()]
    for (key, value) in values {
      out[key] = value
    }
    return out
  }

  static func fail(_ code: String, _ message: String) -> [String: Any] {
    return ["ok": false, "error": code, "message": message]
  }

  // MARK: - WKNavigationDelegate (main thread)

  func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
    guard let page = livePage(for: webView) else {
      return
    }
    page.mainFrameError = nil
    let url = webView.url?.absoluteString
    if url != nil {
      page.lastUrl = url
    }
    emit(page, HiddenWebPages.eventStarted, ["url": DocumentDestinations.orNull(url)])
  }

  func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
    guard let page = livePage(for: webView) else {
      return
    }
    let url = webView.url?.absoluteString
    if url != nil {
      page.lastUrl = url
    }
    emit(page, HiddenWebPages.eventUrl, ["url": DocumentDestinations.orNull(url)])
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    guard let page = livePage(for: webView) else {
      return
    }
    let url = webView.url?.absoluteString ?? page.lastUrl
    page.lastUrl = url
    let error = page.mainFrameError
    var extras: [String: Any] = ["ok": error == nil]
    extras["url"] = DocumentDestinations.orNull(url)
    extras["error"] = DocumentDestinations.orNull(error)
    emit(page, HiddenWebPages.eventFinished, extras)
  }

  func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
    failed(webView, error)
  }

  func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error
  ) {
    failed(webView, error)
  }

  func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
    guard let page = livePage(for: webView), !page.gone else {
      return
    }
    page.gone = true
    // WebKit gives no reason (a crash, or iOS reclaiming a suspended app's memory): crashed is unknown.
    emit(page, HiddenWebPages.eventGone, ["crashed": NSNull()])
    destroy(page, HiddenWebPages.errGone)
  }

  /// A main-frame failure. WebKit sends no didFinish after it, so `finished` (ok false) follows.
  private func failed(_ webView: WKWebView, _ error: Error) {
    guard let page = livePage(for: webView) else {
      return
    }
    let ns = error as NSError
    // A navigation replaced by the next one (cancelled / frame load interrupted): a new one follows.
    if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled {
      return
    }
    if ns.domain == "WebKitErrorDomain" && ns.code == 102 {
      return
    }
    let text = ns.localizedDescription
    page.mainFrameError = text
    let url = webView.url?.absoluteString ?? page.lastUrl
    var failure: [String: Any] = ["code": ns.code, "error": text, "main_frame": true]
    failure["url"] = DocumentDestinations.orNull(url)
    emit(page, HiddenWebPages.eventError, failure)
    var finished: [String: Any] = ["ok": false, "error": text]
    finished["url"] = DocumentDestinations.orNull(url)
    emit(page, HiddenWebPages.eventFinished, finished)
  }
}
