package com.glossarion.flet_glossarion_native

import android.annotation.SuppressLint
import android.annotation.TargetApi
import android.app.Activity
import android.content.Context
import android.content.MutableContextWrapper
import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.ConsoleMessage
import android.webkit.CookieManager
import android.webkit.GeolocationPermissions
import android.webkit.JsPromptResult
import android.webkit.JsResult
import android.webkit.PermissionRequest
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.util.Locale
import kotlin.math.roundToInt
import org.json.JSONTokener

/**
 * In-app browser pages (hotfix 3): hidden [WebView]s this plugin instance creates and owns outside the
 * Flutter widget tree, so Flet screen changes never rebuild them and they keep running while the app is
 * in the background. Contract: flet_glossarion_native/web_pages.py.
 *
 * Methods (MethodChannel `glossarion_native/platform`, routed by GlossarionNativePlugin): web_open,
 * web_load, web_eval, web_title, web_close, web_reset, web_status. Every answer is a map
 * {ok, error, message, ...}; typed failures never use result.error(). Event `web` {page_id, type, ...}
 * goes to Dart through [sendToDart]: started, finished, url, error, gone.
 *
 * - One [HiddenContainer] (application context, 1 x 1 px, clipChildren) sits at index 0 of the Activity's
 *   android.R.id.content, behind the opaque FlutterView. Each WebView is laid out at its viewport
 *   (dp * density) at (0,0): the 1 px clip keeps it "drawn" (View.draw's quickReject passes), alpha stays
 *   1 and it is never GONE / INVISIBLE, so Chromium treats the page as visible while the app is.
 * - The ActivityAware callbacks only store [activity] (FlutterActivity attaches plugins before
 *   setContentView: touching the content view there would be dropped by PhoneWindow.setContentView);
 *   the attach is posted, finds the content view with peekDecorView (never creates the decor) and is
 *   skipped while no live page exists, so the app makes no view-hierarchy call at start-up.
 * - ViewRootImpl lays nothing out while the Activity is stopped: every WebView is measured and laid out
 *   by hand after addView and after re-parenting, so a page opened in the background has its viewport.
 * - Never called: onPause, pauseTimers, setWebContentsDebuggingEnabled, addJavascriptInterface. Console
 *   messages are swallowed; dialogs cancelled; permission requests denied; window.open opens nothing.
 * - A renderer that is gone (onRenderProcessGone returns true, or the system would kill the app) loses
 *   its page at once: event gone, pending evals answer gone, the WebView is removed and destroyed and
 *   never called again; the id keeps answering gone until web_close.
 * - Every channel result is answered exactly once ([Reply]); the page registry belongs to this
 *   instance (flutter_foreground_task's headless engine has its own instance, which never opens a page).
 * All of it runs on the main thread (the channel handler, WebViewClient / WebChromeClient callbacks and
 * the evaluateJavascript callback).
 */
internal class HiddenWebPages(
    private val appContext: Context,
    private val mainHandler: Handler,
    private val sendToDart: (Map<String, Any?>) -> Unit,
) {
    /** Set by the plugin from the ActivityAware callbacks; the setter only stores it and posts the attach. */
    var activity: Activity? = null
        set(value) {
            field = value
            if (value == null) {
                detachContainer()
            } else {
                mainHandler.post { if (activity === value) attachContainer() }
            }
        }

    private val pages = LinkedHashMap<String, Page>()
    private var container: HiddenContainer? = null
    private val capabilityInfo: Map<String, Any?>? by lazy { computeCapability() }

    /** A MethodChannel.Result answered at most once (a second reply throws on the main thread). */
    private class Reply(private val result: MethodChannel.Result) {
        var replied = false
            private set

        fun send(answer: Map<String, Any?>) {
            if (replied) return
            replied = true
            result.success(answer)
        }
    }

    private class Page(
        val id: String,
        val context: MutableContextWrapper,
        val webView: WebView,
        val preventLinks: List<String>,
        val widthDp: Int,
        val heightDp: Int,
        val widthPx: Int,
        val heightPx: Int,
    ) {
        var lastUrl: String? = null
        var mainFrameError: String? = null
        var subresourceErrors = 0
        var gone = false
        var destroyed = false
        val evals = ArrayList<Reply>()
    }

    @SuppressLint("ViewConstructor")
    private class HiddenContainer(context: Context) : FrameLayout(context) {
        init {
            clipChildren = true
            isClickable = false
            isFocusable = false
            isFocusableInTouchMode = false
            descendantFocusability = ViewGroup.FOCUS_BLOCK_DESCENDANTS
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                importantForAutofill = View.IMPORTANT_FOR_AUTOFILL_NO_EXCLUDE_DESCENDANTS
            }
        }

        // Never an offscreen layer around a WebView's draw functor.
        override fun hasOverlappingRendering(): Boolean = false
    }

    fun handles(method: String): Boolean = method in METHODS

    /** get_platform_info()["web_pages"], or null when this phone has no WebView provider. */
    fun capability(): Map<String, Any?>? = capabilityInfo

    /** The engine is going away: destroy this instance's pages only. */
    fun dispose() {
        reset()
        container = null
    }

    // ------------------------------------------------------------------ dispatch

    fun handle(call: MethodCall, result: MethodChannel.Result) {
        val reply = Reply(result)
        try {
            when (call.method) {
                "web_open" -> openPage(call, reply)
                "web_load" -> withPage(call, reply) { page -> load(page, call.argument<String>("url"), reply) }
                "web_eval" -> withPage(call, reply) { page -> eval(page, call.argument<String>("script"), reply) }
                "web_title" -> withPage(call, reply) { page -> title(page, reply) }
                "web_close" -> close(call.argument<String>("page_id"), reply)
                "web_reset" -> reply.send(ok("closed" to reset()))
                "web_status" -> reply.send(status())
                else -> reply.send(fail(ERR_BAD_ARGS, "unknown in-app browser method"))
            }
        } catch (e: Exception) {
            // Exception text can carry URLs: log the type only.
            Log.w(TAG, "${call.method} failed: ${e.javaClass.simpleName}")
            reply.send(fail(ERR_NATIVE, e.javaClass.simpleName))
        }
    }

    private fun withPage(call: MethodCall, reply: Reply, action: (Page) -> Unit) {
        val id = call.argument<String>("page_id")
        val page = if (id == null) null else pages[id]
        when {
            page == null -> reply.send(fail(ERR_NO_PAGE, "no such in-app page"))
            page.gone -> reply.send(fail(ERR_GONE, "the in-app page's renderer stopped"))
            page.destroyed -> reply.send(fail(ERR_CLOSED, "the in-app page was closed"))
            else -> action(page)
        }
    }

    // --------------------------------------------------------------------- pages

    @SuppressLint("SetJavaScriptEnabled")
    private fun openPage(call: MethodCall, reply: Reply) {
        val id = call.argument<String>("page_id")
        if (id.isNullOrEmpty() || id.length > MAX_ID_LENGTH) {
            reply.send(fail(ERR_BAD_ARGS, "page_id is required"))
            return
        }
        if (pages.containsKey(id)) {
            reply.send(fail(ERR_EXISTS, "the page is already open"))
            return
        }
        if (pages.size >= MAX_PAGES) {
            reply.send(fail(ERR_TOO_MANY, "too many in-app pages"))
            return
        }
        val density = appContext.resources.displayMetrics.density
        val widthDp = dp(call.argument<Number>("width"), DEFAULT_WIDTH_DP)
        val heightDp = dp(call.argument<Number>("height"), DEFAULT_HEIGHT_DP)
        val widthPx = px(widthDp, density)
        val heightPx = px(heightDp, density)
        val links = (call.argument<List<Any?>>("prevent_links") ?: emptyList())
            .mapNotNull { it as? String }
            .filter { it.isNotEmpty() }
        val wrapper = MutableContextWrapper(activity ?: appContext)
        val webView = try {
            WebView(wrapper)
        } catch (e: Throwable) {
            // MissingWebViewPackageException, a provider being updated, ...
            Log.w(TAG, "WebView unavailable: ${e.javaClass.simpleName}")
            reply.send(fail(ERR_UNAVAILABLE, e.javaClass.simpleName))
            return
        }
        val page = Page(id, wrapper, webView, links, widthDp, heightDp, widthPx, heightPx)
        try {
            configure(webView)
            webView.webViewClient = PageClient(page)
            webView.webChromeClient = PageChrome()
            ensureContainer().addView(webView, FrameLayout.LayoutParams(widthPx, heightPx, Gravity.TOP or Gravity.START))
            layoutPage(page)
        } catch (e: Exception) {
            Log.w(TAG, "web_open failed: ${e.javaClass.simpleName}")
            try {
                (webView.parent as? ViewGroup)?.removeView(webView)
                webView.destroy()
            } catch (cleanup: Exception) {
                Log.w(TAG, "WebView cleanup: ${cleanup.javaClass.simpleName}")
            }
            removeContainerIfEmpty()
            reply.send(fail(ERR_NATIVE, e.javaClass.simpleName))
            return
        }
        // Registered only once the WebView is in the container and laid out.
        pages[id] = page
        attachContainer()
        reply.send(ok("page_id" to id, "attached" to isAttached()))
    }

    /** webview_flutter_android's AndroidWebViewController defaults + flet-webview's JavaScriptMode.unrestricted. */
    @SuppressLint("SetJavaScriptEnabled")
    private fun configure(webView: WebView) {
        webView.isFocusable = false
        webView.isFocusableInTouchMode = false
        webView.isVerticalScrollBarEnabled = false
        webView.isHorizontalScrollBarEnabled = false
        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.javaScriptCanOpenWindowsAutomatically = true
        // No onCreateWindow: window.open() opens nothing and never navigates the hidden page away.
        settings.setSupportMultipleWindows(true)
        settings.loadWithOverviewMode = true
        settings.useWideViewPort = true
        settings.builtInZoomControls = true
        settings.displayZoomControls = false
        settings.mediaPlaybackRequiresUserGesture = true
        // Hidden pages only open http(s).
        settings.allowFileAccess = false
        settings.allowContentAccess = false
        // Left at the defaults on purpose: the system user agent (like flet-webview), mixed content
        // (never allowed), cache mode, Safe Browsing, third-party cookies (off).
        CookieManager.getInstance().setAcceptCookie(true)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // The renderer keeps the app's (foreground service) importance while it is in the background.
            webView.setRendererPriorityPolicy(WebView.RENDERER_PRIORITY_IMPORTANT, false)
        }
    }

    private fun load(page: Page, url: String?, reply: Reply) {
        val scheme = url?.substringBefore(':', "")?.lowercase(Locale.ROOT)
        if (url == null || !(scheme == "http" || scheme == "https")) {
            reply.send(fail(ERR_BAD_ARGS, "only http(s) pages can be opened"))
            return
        }
        attachContainer()
        page.lastUrl = url
        page.webView.loadUrl(url)
        reply.send(ok())
    }

    private fun eval(page: Page, script: String?, reply: Reply) {
        if (script == null) {
            reply.send(fail(ERR_BAD_ARGS, "script is required"))
            return
        }
        attachContainer()
        page.evals.add(reply)
        try {
            page.webView.evaluateJavascript(script) { raw ->
                page.evals.remove(reply)
                // A page destroyed meanwhile already answered closed / gone: send() is a no-op then.
                reply.send(ok("result" to decodeJsString(raw), "url" to page.lastUrl))
            }
        } catch (e: Exception) {
            page.evals.remove(reply)
            Log.w(TAG, "web_eval failed: ${e.javaClass.simpleName}")
            reply.send(fail(ERR_NATIVE, e.javaClass.simpleName))
        }
    }

    private fun title(page: Page, reply: Reply) {
        attachContainer()
        val url = page.webView.url ?: page.lastUrl
        page.lastUrl = url
        reply.send(ok("title" to page.webView.title, "url" to url))
    }

    private fun close(id: String?, reply: Reply) {
        val page = if (id == null) null else pages.remove(id)
        if (page != null) destroy(page, ERR_CLOSED)
        reply.send(ok("closed" to (page != null)))
    }

    /**
     * Close every page of this instance; how many there were. The container goes and the cookies are
     * flushed once, after the last page (CookieManager.flush blocks the main thread and may do I/O).
     */
    private fun reset(): Int {
        val all = ArrayList(pages.values)
        val live = all.any { !it.destroyed }
        pages.clear()
        for (page in all) destroy(page, ERR_CLOSED, settle = false)
        removeContainerIfEmpty()
        if (live) flushCookies()
        return all.size
    }

    private fun status(): Map<String, Any?> {
        val list = ArrayList<Map<String, Any?>>()
        for (page in pages.values) {
            list.add(
                hashMapOf<String, Any?>(
                    "page_id" to page.id,
                    "url" to page.lastUrl,
                    "gone" to page.gone,
                    "width" to page.widthDp,
                    "height" to page.heightDp,
                    "pending" to page.evals.size,
                )
            )
        }
        return ok(
            "pages" to list,
            "attached" to isAttached(),
            "activity" to (activity != null),
            "density" to appContext.resources.displayMetrics.density.toDouble(),
        )
    }

    /**
     * Destroys the WebView of [page]; pending evals answer [answer]. Never calls a dead WebView's page API.
     * [settle]: when no live page is left, remove the container and flush the cookies (reset() does that
     * once for all its pages instead).
     */
    private fun destroy(page: Page, answer: String, settle: Boolean = true) {
        if (page.destroyed) return
        page.destroyed = true
        val waiting = ArrayList(page.evals)
        page.evals.clear()
        val message = if (answer == ERR_GONE) "the in-app page's renderer stopped" else "the in-app page was closed"
        for (pending in waiting) pending.send(fail(answer, message))
        val webView = page.webView
        if (!page.gone) {
            try {
                webView.stopLoading()
            } catch (e: Exception) {
                Log.w(TAG, "WebView.stopLoading: ${e.javaClass.simpleName}")
            }
        }
        try {
            (webView.parent as? ViewGroup)?.removeView(webView)
        } catch (e: Exception) {
            Log.w(TAG, "removeView: ${e.javaClass.simpleName}")
        }
        try {
            webView.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "WebView.destroy: ${e.javaClass.simpleName}")
        }
        page.context.setBaseContext(appContext)
        if (settle && !hasLivePage()) {
            removeContainerIfEmpty()
            flushCookies()
        }
    }

    /** The renderer is gone: keep the id (it answers gone), drop the WebView now. */
    private fun lost(page: Page, crashed: Boolean) {
        if (page.gone || page.destroyed) return
        page.gone = true
        emit(page, EVENT_GONE, "crashed" to crashed)
        destroy(page, ERR_GONE)
    }

    private fun emit(page: Page, type: String, vararg extras: Pair<String, Any?>) {
        if (page.destroyed) return
        val event = HashMap<String, Any?>()
        event["page_id"] = page.id
        event["type"] = type
        for ((key, value) in extras) event[key] = value
        try {
            sendToDart(event)
        } catch (e: Exception) {
            Log.w(TAG, "web event: ${e.javaClass.simpleName}")
        }
    }

    // ----------------------------------------------------------------- container

    private fun ensureContainer(): HiddenContainer =
        container ?: HiddenContainer(appContext).also { container = it }

    private fun hasLivePage(): Boolean = pages.values.any { !it.destroyed }

    private fun isAttached(): Boolean = container?.isAttachedToWindow == true

    /**
     * Puts the container behind the FlutterView of the current Activity (an identity check when it is
     * there). Best effort: the pages keep working out of any window, so a failure is only logged.
     */
    private fun attachContainer() {
        val c = container ?: return
        if (!hasLivePage()) {
            removeContainerIfEmpty()
            return
        }
        try {
            val act = activity
            val content = act?.window?.peekDecorView()?.findViewById<View>(android.R.id.content) as? ViewGroup
            if (act == null || content == null) {
                if (c.parent == null) layoutDetached(c)
                return
            }
            if (c.parent === content) return
            (c.parent as? ViewGroup)?.removeView(c)
            content.addView(c, 0, FrameLayout.LayoutParams(1, 1, Gravity.TOP or Gravity.START))
            for (page in pages.values) {
                if (page.destroyed) continue
                page.context.setBaseContext(act)
                layoutPage(page)
            }
        } catch (e: Exception) {
            Log.w(TAG, "attach: ${e.javaClass.simpleName}")
        }
    }

    /** No Activity: the container keeps running out of any window, laid out by hand. */
    private fun detachContainer() {
        val c = container ?: return
        // Runs in an ActivityAware callback: an exception here would end the app.
        try {
            (c.parent as? ViewGroup)?.removeView(c)
            for (page in pages.values) {
                if (!page.destroyed) page.context.setBaseContext(appContext)
            }
            if (hasLivePage()) layoutDetached(c)
        } catch (e: Exception) {
            Log.w(TAG, "detach: ${e.javaClass.simpleName}")
        }
    }

    private fun removeContainerIfEmpty() {
        if (hasLivePage()) return
        val c = container ?: return
        (c.parent as? ViewGroup)?.removeView(c)
    }

    private fun layoutDetached(c: HiddenContainer) {
        c.measure(exactly(1), exactly(1))
        c.layout(0, 0, 1, 1)
        for (page in pages.values) {
            if (!page.destroyed) layoutPage(page)
        }
    }

    /** ViewRootImpl skips layout while the Activity is stopped: give the page its viewport now (idempotent). */
    private fun layoutPage(page: Page) {
        page.webView.measure(exactly(page.widthPx), exactly(page.heightPx))
        page.webView.layout(0, 0, page.widthPx, page.heightPx)
    }

    private fun flushCookies() {
        try {
            CookieManager.getInstance().flush()
        } catch (e: Exception) {
            Log.w(TAG, "CookieManager.flush: ${e.javaClass.simpleName}")
        }
    }

    private fun computeCapability(): Map<String, Any?>? {
        var provider: String? = null
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            var missing = false
            try {
                val pkg = WebView.getCurrentWebViewPackage()
                if (pkg == null) missing = true else provider = "${pkg.packageName} ${pkg.versionName}"
            } catch (e: Throwable) {
                Log.w(TAG, "getCurrentWebViewPackage: ${e.javaClass.simpleName}")
            }
            if (missing) return null
        }
        return hashMapOf<String, Any?>("version" to VERSION, "max_pages" to MAX_PAGES, "provider" to provider)
    }

    // ------------------------------------------------------------------- clients

    private inner class PageClient(private val page: Page) : WebViewClient() {
        override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
            val url = request?.url?.toString() ?: return false
            return page.preventLinks.any { url.startsWith(it, ignoreCase = true) }
        }

        override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
            page.mainFrameError = null
            page.subresourceErrors = 0
            if (url != null) page.lastUrl = url
            emit(page, EVENT_STARTED, "url" to url)
        }

        override fun onPageFinished(view: WebView?, url: String?) {
            if (url != null) page.lastUrl = url
            val error = page.mainFrameError
            emit(page, EVENT_FINISHED, "url" to url, "ok" to (error == null), "error" to error)
        }

        override fun doUpdateVisitedHistory(view: WebView?, url: String?, isReload: Boolean) {
            if (url != null) page.lastUrl = url
            emit(page, EVENT_URL, "url" to url)
        }

        override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
            val main = request?.isForMainFrame == true
            val code = error?.errorCode ?: 0
            val text = error?.description?.toString() ?: "error $code"
            if (main) {
                page.mainFrameError = text
            } else {
                if (page.subresourceErrors >= MAX_SUBRESOURCE_ERRORS) return
                page.subresourceErrors += 1
            }
            emit(page, EVENT_ERROR, "url" to request?.url?.toString(), "code" to code, "error" to text, "main_frame" to main)
        }

        @TargetApi(Build.VERSION_CODES.O)
        override fun onRenderProcessGone(view: WebView?, detail: RenderProcessGoneDetail?): Boolean {
            // Synchronously, like the platform's termination-handling sample: nothing queued after this
            // callback may reach the dead WebView. Returning false would kill the app.
            lost(page, detail?.didCrash() == true)
            return true
        }
    }

    private class PageChrome : WebChromeClient() {
        // Swallowed: page console output (model answers, prompts) never reaches logcat.
        override fun onConsoleMessage(consoleMessage: ConsoleMessage?): Boolean = true

        override fun onJsAlert(view: WebView?, url: String?, message: String?, result: JsResult?): Boolean {
            result?.cancel()
            return true
        }

        override fun onJsConfirm(view: WebView?, url: String?, message: String?, result: JsResult?): Boolean {
            result?.cancel()
            return true
        }

        override fun onJsPrompt(
            view: WebView?,
            url: String?,
            message: String?,
            defaultValue: String?,
            result: JsPromptResult?,
        ): Boolean {
            result?.cancel()
            return true
        }

        // A streaming chat's beforeunload never blocks the next load.
        override fun onJsBeforeUnload(view: WebView?, url: String?, message: String?, result: JsResult?): Boolean {
            result?.confirm()
            return true
        }

        override fun onPermissionRequest(request: PermissionRequest?) {
            request?.deny()
        }

        override fun onGeolocationPermissionsShowPrompt(origin: String?, callback: GeolocationPermissions.Callback?) {
            callback?.invoke(origin, false, false)
        }
    }

    companion object {
        private const val TAG = "GlossarionWeb"

        // flet_glossarion_native/web_pages.py WEB_PAGES_VERSION / NATIVE_MAX_PAGES / MAX_SUBRESOURCE_ERRORS.
        const val VERSION = 1
        const val MAX_PAGES = 8
        private const val MAX_SUBRESOURCE_ERRORS = 8
        private const val MAX_SIDE_DP = 4096
        private const val MAX_SIDE_PX = 16384
        private const val DEFAULT_WIDTH_DP = 480
        private const val DEFAULT_HEIGHT_DP = 800
        private const val MAX_ID_LENGTH = 64

        val METHODS = setOf(
            "web_open", "web_load", "web_eval", "web_title", "web_close", "web_reset", "web_status",
        )

        // Error codes (web_pages.py WebPageError; "timeout" is Python's own).
        const val ERR_UNAVAILABLE = "unavailable"
        const val ERR_BAD_ARGS = "bad_args"
        const val ERR_EXISTS = "exists"
        const val ERR_TOO_MANY = "too_many"
        const val ERR_NO_PAGE = "no_page"
        const val ERR_GONE = "gone"
        const val ERR_CLOSED = "closed"
        const val ERR_NATIVE = "native_error"

        // Event types (web_pages.py WEB_EVENT_TYPES).
        const val EVENT_STARTED = "started"
        const val EVENT_FINISHED = "finished"
        const val EVENT_URL = "url"
        const val EVENT_ERROR = "error"
        const val EVENT_GONE = "gone"

        fun ok(vararg values: Pair<String, Any?>): HashMap<String, Any?> {
            val out = hashMapOf<String, Any?>("ok" to true, "error" to null, "message" to null)
            for ((key, value) in values) out[key] = value
            return out
        }

        fun fail(code: String, message: String): HashMap<String, Any?> =
            hashMapOf<String, Any?>("ok" to false, "error" to code, "message" to message)

        private fun exactly(size: Int): Int = View.MeasureSpec.makeMeasureSpec(size, View.MeasureSpec.EXACTLY)

        private fun dp(value: Number?, fallback: Int): Int {
            val raw = value?.toDouble() ?: return fallback
            if (raw.isNaN()) return fallback
            return raw.coerceIn(1.0, MAX_SIDE_DP.toDouble()).roundToInt()
        }

        private fun px(dp: Int, density: Float): Int = (dp * density).roundToInt().coerceIn(1, MAX_SIDE_PX)

        /** The completion value evaluateJavascript hands back is a JSON literal; only a string is kept. */
        private fun decodeJsString(raw: String?): String? {
            if (raw == null || raw == "null") return null
            return try {
                JSONTokener(raw).nextValue() as? String
            } catch (e: Exception) {
                null
            }
        }
    }
}
