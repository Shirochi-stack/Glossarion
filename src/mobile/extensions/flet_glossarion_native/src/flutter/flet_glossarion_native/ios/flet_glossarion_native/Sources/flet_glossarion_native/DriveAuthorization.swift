import AuthenticationServices
import Flutter
import UIKit

/// The browser returns the callback to this session. Python validates state and exchanges the code.
final class DriveAuthorization: NSObject, ASWebAuthenticationPresentationContextProviding {
  private var session: ASWebAuthenticationSession?
  private var anchor: ASPresentationAnchor?

  func open(_ arguments: Any?, result: @escaping FlutterResult) {
    guard session == nil else {
      result(["error": "busy"])
      return
    }
    guard let args = arguments as? [String: Any],
          let raw = args["url"] as? String, let url = URL(string: raw),
          url.scheme == "https", url.host == "accounts.google.com",
          let scheme = args["scheme"] as? String, scheme.hasPrefix("com.googleusercontent.apps."),
          let window = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene })
            .flatMap({ $0.windows }).first(where: { $0.isKeyWindow }) else {
      result(["error": "unavailable"])
      return
    }
    anchor = window
    let auth = ASWebAuthenticationSession(url: url, callbackURLScheme: scheme) { [weak self] callback, error in
      DispatchQueue.main.async {
        self?.session = nil
        self?.anchor = nil
        if let callback {
          result(["callback": callback.absoluteString])
        } else {
          let cancelled = (error as? ASWebAuthenticationSessionError)?.code == .canceledLogin
          result(["error": cancelled ? "cancelled" : "failed"])
        }
      }
    }
    auth.presentationContextProvider = self
    session = auth
    if !auth.start() {
      session = nil
      anchor = nil
      result(["error": "failed"])
    }
  }

  func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
    anchor ?? ASPresentationAnchor()
  }
}
