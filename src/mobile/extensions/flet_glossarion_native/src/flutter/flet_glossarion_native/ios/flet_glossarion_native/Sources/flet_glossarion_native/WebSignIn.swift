import UIKit
import WebKit

/// A site sign-in the user completes in a visible WKWebView, and the cookie store it leaves behind
/// (authgem/: the Gemini app runs on the google.com session cookies). Contract:
/// flet_glossarion_native/web_pages.py (web_sign_in, web_cookies, web_clear_cookies).
///
/// - web_sign_in presents a full-screen view controller over the top one: a bar with Cancel and a
///   WKWebView on `url`, WKWebsiteDataStore.default() (the store the hidden pages use). It sends
///   `user_agent` (default: Mobile Safari's agent) because Google refuses its sign-in page to agents
///   that look like an embedded web view. It ends signed in once a page under `done_url` finished loading
///   and the store holds `done_cookie` for the host of `cookie_url`; Cancel ends it signed out.
/// - web_cookies {url} -> {cookies: "name=value; ..." | null}: every cookie the store would send to
///   `url` (HttpOnly included). web_clear_cookies {url, domain} deletes the cookies of `domain`.
/// - Main thread only; each FlutterResult is answered once. No iOS 14+ API (deployment target iOS 13).
final class WebSignIn: NSObject, WKNavigationDelegate {
  static let methods: Set<String> = ["web_sign_in", "web_cookies", "web_clear_cookies"]

  private static let safariAgent =
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
    + "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"

  private final class Session {
    let controller: UIViewController
    let webView: WKWebView
    let reply: ([String: Any]) -> Void
    let doneUrl: String
    let doneCookie: String
    let cookieHost: String
    var finished = false

    init(controller: UIViewController, webView: WKWebView, reply: @escaping ([String: Any]) -> Void,
         doneUrl: String, doneCookie: String, cookieHost: String) {
      self.controller = controller
      self.webView = webView
      self.reply = reply
      self.doneUrl = doneUrl
      self.doneCookie = doneCookie
      self.cookieHost = cookieHost
    }
  }

  private var active: Session?

  func handle(_ method: String, _ args: [String: Any], reply: @escaping ([String: Any]) -> Void) {
    switch method {
    case "web_sign_in":
      signIn(args, reply)
    case "web_cookies":
      cookies(args["url"] as? String, reply)
    case "web_clear_cookies":
      clearCookies(args["domain"] as? String, reply)
    default:
      reply(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "unknown in-app browser method"))
    }
  }

  /// The engine is going away: end a pending sign-in signed out.
  func dispose() {
    if let session = active {
      finish(session, false)
    }
  }

  private static func httpUrl(_ text: String?) -> URL? {
    guard let raw = text, let url = URL(string: raw), let scheme = url.scheme?.lowercased(),
          scheme == "http" || scheme == "https", url.host != nil else {
      return nil
    }
    return url
  }

  private func signIn(_ args: [String: Any], _ reply: @escaping ([String: Any]) -> Void) {
    guard let url = WebSignIn.httpUrl(args["url"] as? String),
          let doneUrl = args["done_url"] as? String, WebSignIn.httpUrl(doneUrl) != nil,
          let cookieHost = WebSignIn.httpUrl(args["cookie_url"] as? String)?.host,
          let doneCookie = args["done_cookie"] as? String, !doneCookie.isEmpty else {
      reply(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "url, done_url, cookie_url and done_cookie are required"))
      return
    }
    guard var top = DocumentDestinations.keyWindow()?.rootViewController else {
      reply(HiddenWebPages.fail(HiddenWebPages.errUnavailable, "the app is not on screen"))
      return
    }
    while let presented = top.presentedViewController {
      top = presented
    }
    if let session = active {
      finish(session, false)
    }
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = WKWebsiteDataStore.default()
    let webView = WKWebView(frame: .zero, configuration: configuration)
    let agent = (args["user_agent"] as? String) ?? ""
    webView.customUserAgent = agent.isEmpty ? WebSignIn.safariAgent : agent
    webView.navigationDelegate = self

    let content = UIViewController()
    content.title = (args["title"] as? String) ?? "Sign in"
    content.view.backgroundColor = UIColor.white
    webView.translatesAutoresizingMaskIntoConstraints = false
    content.view.addSubview(webView)
    NSLayoutConstraint.activate([
      webView.leadingAnchor.constraint(equalTo: content.view.leadingAnchor),
      webView.trailingAnchor.constraint(equalTo: content.view.trailingAnchor),
      webView.topAnchor.constraint(equalTo: content.view.safeAreaLayoutGuide.topAnchor),
      webView.bottomAnchor.constraint(equalTo: content.view.bottomAnchor),
    ])
    let navigation = UINavigationController(rootViewController: content)
    navigation.modalPresentationStyle = .fullScreen
    let session = Session(controller: navigation, webView: webView, reply: reply, doneUrl: doneUrl,
                          doneCookie: doneCookie, cookieHost: cookieHost.lowercased())
    content.navigationItem.leftBarButtonItem = UIBarButtonItem(
      barButtonSystemItem: .cancel, target: self, action: #selector(cancelTapped))
    active = session
    top.present(navigation, animated: true)
    _ = webView.load(URLRequest(url: url))
  }

  @objc private func cancelTapped() {
    if let session = active {
      finish(session, false)
    }
  }

  private func check(_ session: Session) {
    guard !session.finished, let url = session.webView.url?.absoluteString,
          url.lowercased().hasPrefix(session.doneUrl.lowercased()) else {
      return
    }
    WKWebsiteDataStore.default().httpCookieStore.getAllCookies { cookies in
      let found = cookies.contains { cookie in
        cookie.name == session.doneCookie && !cookie.value.isEmpty
          && WebSignIn.domainMatches(cookie.domain, session.cookieHost)
      }
      if found {
        self.finish(session, true)
      }
    }
  }

  private func finish(_ session: Session, _ signedIn: Bool) {
    if session.finished {
      return
    }
    session.finished = true
    if active === session {
      active = nil
    }
    session.webView.navigationDelegate = nil
    session.webView.stopLoading()
    session.controller.dismiss(animated: true)
    var answer = HiddenWebPages.ok()
    answer["signed_in"] = signedIn
    session.reply(answer)
  }

  /// RFC 6265 domain match: a cookie for ".google.com" / "google.com" goes to www.google.com.
  private static func domainMatches(_ cookieDomain: String, _ host: String) -> Bool {
    var domain = cookieDomain.lowercased()
    if domain.hasPrefix(".") {
      domain.removeFirst()
    }
    return host == domain || host.hasSuffix("." + domain)
  }

  private func cookies(_ raw: String?, _ reply: @escaping ([String: Any]) -> Void) {
    guard let url = WebSignIn.httpUrl(raw), let host = url.host?.lowercased() else {
      reply(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "only http(s) cookies can be read"))
      return
    }
    let path = url.path.isEmpty ? "/" : url.path
    WKWebsiteDataStore.default().httpCookieStore.getAllCookies { all in
      let sent = all.filter { cookie in
        WebSignIn.domainMatches(cookie.domain, host) && path.hasPrefix(cookie.path)
          && (cookie.expiresDate == nil || cookie.expiresDate! > Date())
      }
      var answer = HiddenWebPages.ok()
      if sent.isEmpty {
        answer["cookies"] = NSNull()
      } else {
        answer["cookies"] = sent.map { "\($0.name)=\($0.value)" }.joined(separator: "; ")
      }
      reply(answer)
    }
  }

  private func clearCookies(_ domain: String?, _ reply: @escaping ([String: Any]) -> Void) {
    guard var target = domain?.lowercased(), !target.isEmpty else {
      reply(HiddenWebPages.fail(HiddenWebPages.errBadArgs, "domain is required"))
      return
    }
    if target.hasPrefix(".") {
      target.removeFirst()
    }
    let store = WKWebsiteDataStore.default().httpCookieStore
    store.getAllCookies { all in
      let doomed = all.filter { WebSignIn.domainMatches($0.domain, target) || $0.domain.lowercased().hasSuffix(target) }
      let group = DispatchGroup()
      for cookie in doomed {
        group.enter()
        store.delete(cookie) { group.leave() }
      }
      group.notify(queue: .main) {
        var answer = HiddenWebPages.ok()
        answer["cleared"] = doomed.count
        reply(answer)
      }
    }
  }

  // MARK: - WKNavigationDelegate (main thread)

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    if let session = active, session.webView === webView {
      check(session)
    }
  }

  func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
    if let session = active, session.webView === webView {
      check(session)
    }
  }

  func webView(
    _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) {
    // Only web pages: another app's link never leaves the sign-in.
    let scheme = navigationAction.request.url?.scheme?.lowercased() ?? ""
    decisionHandler(scheme == "http" || scheme == "https" || scheme == "about" ? .allow : .cancel)
  }
}
