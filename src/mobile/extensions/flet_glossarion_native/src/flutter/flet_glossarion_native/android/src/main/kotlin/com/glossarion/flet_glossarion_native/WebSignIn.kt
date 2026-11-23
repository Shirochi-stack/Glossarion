package com.glossarion.flet_glossarion_native

import android.annotation.SuppressLint
import android.app.Activity
import android.app.Dialog
import android.graphics.Color
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.CookieManager
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.Locale

/**
 * A site sign-in the user completes in a visible WebView, and the process-wide cookie jar it leaves
 * behind (authgem/: the Gemini app runs on the google.com session cookies). Contract:
 * flet_glossarion_native/web_pages.py (web_sign_in, web_cookies, web_clear_cookies).
 *
 * - web_sign_in opens a full-screen Dialog over the Activity: a title bar with Cancel and a WebView on
 *   [url]. The WebView sends [user_agent] (default: the system WebView agent without its "; wv" and
 *   "Version/4.0" markers, i.e. the phone's Chrome agent) because Google refuses its sign-in page to
 *   agents that name an embedded WebView. It ends signed in once a page under [done_url] finished
 *   loading and the jar holds [done_cookie] for [cookie_url]; Cancel / Back end it signed out. The
 *   cookies are flushed before the answer {signed_in}.
 * - web_cookies {url} -> {cookies: "name=value; ..." | null}: CookieManager.getCookie, which includes
 *   HttpOnly cookies. web_clear_cookies {url, domain} expires every cookie the jar sends to [url]
 *   (domain and host-only variants) -> {cleared}.
 * - Everything runs on the main thread; each channel result is answered once.
 */
internal class WebSignIn(private val pages: HiddenWebPages) {
    private var active: Session? = null

    private class Session(
        val dialog: Dialog,
        val webView: WebView,
        val reply: (Map<String, Any?>) -> Unit,
        val doneUrl: String,
        val doneCookie: String,
        val cookieUrl: String,
    ) {
        var finished = false
    }

    @SuppressLint("SetJavaScriptEnabled")
    fun signIn(
        activity: Activity?,
        url: String?,
        doneUrl: String?,
        doneCookie: String?,
        cookieUrl: String?,
        userAgent: String?,
        title: String?,
        reply: (Map<String, Any?>) -> Unit,
    ) {
        if (activity == null || activity.isFinishing) {
            reply(HiddenWebPages.fail(HiddenWebPages.ERR_UNAVAILABLE, "the app is not on screen"))
            return
        }
        if (!isHttp(url) || !isHttp(doneUrl) || !isHttp(cookieUrl) || doneCookie.isNullOrEmpty()) {
            reply(HiddenWebPages.fail(HiddenWebPages.ERR_BAD_ARGS, "url, done_url, cookie_url and done_cookie are required"))
            return
        }
        active?.let { finish(it, false) }
        val webView = try {
            WebView(activity)
        } catch (e: Throwable) {
            Log.w(TAG, "WebView unavailable: ${e.javaClass.simpleName}")
            reply(HiddenWebPages.fail(HiddenWebPages.ERR_UNAVAILABLE, e.javaClass.simpleName))
            return
        }
        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.allowFileAccess = false
        settings.allowContentAccess = false
        settings.userAgentString = if (userAgent.isNullOrBlank()) browserAgent(settings.userAgentString) else userAgent
        val cookies = CookieManager.getInstance()
        cookies.setAcceptCookie(true)
        cookies.setAcceptThirdPartyCookies(webView, true)

        val dialog = Dialog(activity, android.R.style.Theme_DeviceDefault_NoActionBar)
        val layout = LinearLayout(activity)
        layout.orientation = LinearLayout.VERTICAL
        layout.setBackgroundColor(Color.WHITE)
        val bar = LinearLayout(activity)
        bar.orientation = LinearLayout.HORIZONTAL
        bar.gravity = Gravity.CENTER_VERTICAL
        val pad = dp(activity, 12)
        bar.setPadding(pad, pad / 2, pad / 2, pad / 2)
        bar.setBackgroundColor(Color.rgb(32, 33, 36))
        val label = TextView(activity)
        label.text = title ?: "Sign in"
        label.setTextColor(Color.WHITE)
        label.setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
        bar.addView(label, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        val cancel = Button(activity)
        cancel.text = "Cancel"
        bar.addView(cancel, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        layout.addView(bar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        layout.addView(webView, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        dialog.setContentView(layout)
        dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        @Suppress("DEPRECATION")
        dialog.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)

        val session = Session(dialog, webView, reply, doneUrl!!, doneCookie, cookieUrl!!)
        active = session
        cancel.setOnClickListener { finish(session, false) }
        dialog.setOnCancelListener { finish(session, false) }
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                // Only web pages: an intent: / market: link never leaves the sign-in.
                return !isHttp(request?.url?.toString())
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                check(session, url)
            }

            override fun doUpdateVisitedHistory(view: WebView?, url: String?, isReload: Boolean) {
                check(session, url)
            }
        }
        try {
            dialog.show()
        } catch (e: Exception) {
            Log.w(TAG, "sign-in dialog: ${e.javaClass.simpleName}")
            active = null
            webView.destroy()
            reply(HiddenWebPages.fail(HiddenWebPages.ERR_NATIVE, e.javaClass.simpleName))
            return
        }
        webView.loadUrl(url!!)
    }

    private fun check(session: Session, url: String?) {
        if (session.finished || url == null || !url.startsWith(session.doneUrl, ignoreCase = true)) return
        if (hasCookie(session.cookieUrl, session.doneCookie)) finish(session, true)
    }

    private fun finish(session: Session, signedIn: Boolean) {
        if (session.finished) return
        session.finished = true
        if (active === session) active = null
        flush()
        try {
            session.webView.stopLoading()
            (session.webView.parent as? ViewGroup)?.removeView(session.webView)
            session.webView.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "sign-in WebView cleanup: ${e.javaClass.simpleName}")
        }
        try {
            session.dialog.setOnCancelListener(null)
            session.dialog.dismiss()
        } catch (e: Exception) {
            Log.w(TAG, "sign-in dialog dismiss: ${e.javaClass.simpleName}")
        }
        session.reply(HiddenWebPages.ok("signed_in" to signedIn))
    }

    /** The engine or Activity is going away: end a pending sign-in signed out. */
    fun dispose() {
        active?.let { finish(it, false) }
    }

    fun cookies(url: String?): Map<String, Any?> {
        if (!isHttp(url)) return HiddenWebPages.fail(HiddenWebPages.ERR_BAD_ARGS, "only http(s) cookies can be read")
        return HiddenWebPages.ok("cookies" to CookieManager.getInstance().getCookie(url))
    }

    fun clearCookies(url: String?, domain: String?): Map<String, Any?> {
        if (!isHttp(url)) return HiddenWebPages.fail(HiddenWebPages.ERR_BAD_ARGS, "only http(s) cookies can be cleared")
        val manager = CookieManager.getInstance()
        val names = (manager.getCookie(url) ?: "").split(";").map { it.substringBefore('=').trim() }.filter { it.isNotEmpty() }
        val cleanDomain = domain?.trim()?.takeIf { it.isNotEmpty() }
        for (name in names.distinct()) {
            // __Host- cookies are host-only by definition; the rest may be set for the parent domain.
            if (cleanDomain != null && !name.startsWith("__Host-")) {
                manager.setCookie(url, "$name=; Max-Age=0; Path=/; Domain=$cleanDomain; Secure")
            }
            manager.setCookie(url, "$name=; Max-Age=0; Path=/; Secure")
        }
        flush()
        return HiddenWebPages.ok("cleared" to names.distinct().size)
    }

    companion object {
        private const val TAG = "GlossarionSignIn"
        val METHODS = setOf("web_sign_in", "web_cookies", "web_clear_cookies")

        fun isHttp(url: String?): Boolean {
            val scheme = url?.substringBefore(':', "")?.lowercase(Locale.ROOT)
            return scheme == "http" || scheme == "https"
        }

        /** The phone's Chrome agent: the WebView agent without its embedded-browser markers. */
        fun browserAgent(webViewAgent: String?): String {
            val agent = webViewAgent ?: return ""
            return agent.replace("; wv)", ")").replace(Regex("\\s*Version/\\d+(\\.\\d+)*"), "")
        }

        private fun hasCookie(url: String, name: String): Boolean {
            val all = CookieManager.getInstance().getCookie(url) ?: return false
            return all.split(";").any { part ->
                val pair = part.trim()
                pair.startsWith("$name=") && pair.length > name.length + 1
            }
        }

        private fun flush() {
            try {
                CookieManager.getInstance().flush()
            } catch (e: Exception) {
                Log.w(TAG, "CookieManager.flush: ${e.javaClass.simpleName}")
            }
        }

        private fun dp(activity: Activity, value: Int): Int =
            TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), activity.resources.displayMetrics).toInt()
    }
}
