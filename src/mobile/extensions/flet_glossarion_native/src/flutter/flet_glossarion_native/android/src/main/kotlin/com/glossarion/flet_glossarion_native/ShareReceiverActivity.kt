package com.glossarion.flet_glossarion_native

import android.app.Activity
import android.content.ClipData
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log

/**
 * Exported trampoline for "Open with" (ACTION_VIEW) and "Share"
 * (ACTION_SEND / ACTION_SEND_MULTIPLE).
 *
 * Flutter pushes `intent.data.toString()` as a route whenever MainActivity
 * receives an intent with data (FlutterActivityAndFragmentDelegate), which
 * would put a content:// URI into page.route. This activity therefore moves
 * every URI into extras (EXTRA_GLOSSARION_URI / EXTRA_GLOSSARION_URIS), keeps
 * them in ClipData so FLAG_GRANT_READ_URI_PERMISSION carries the read grant,
 * and forwards to the app's launch activity with data == null and
 * FLAG_ACTIVITY_SINGLE_TOP (MainActivity is singleTop: a running app gets
 * onNewIntent). GlossarionNativePlugin copies the files and emits `share`.
 *
 * Only well-known extras are copied: re-parcelling a foreign Bundle could
 * contain classes this process cannot load.
 */
class ShareReceiverActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            forward(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Could not forward shared content", e)
        } finally {
            finish()
        }
    }

    private fun forward(source: Intent?) {
        if (source == null) return
        val launch = packageManager.getLaunchIntentForPackage(packageName)
        val target = launch?.component
        if (target == null) {
            Log.e(TAG, "No launch activity for $packageName")
            return
        }

        val uris = ArrayList<Uri>()
        var text: String? = null
        var subject: String? = null
        val origin: String

        when (source.action) {
            Intent.ACTION_VIEW -> {
                origin = ORIGIN_VIEW
                source.data?.let { uris.add(it) }
            }
            Intent.ACTION_SEND -> {
                origin = ORIGIN_SEND
                safeExtra { readUriExtra(source, Intent.EXTRA_STREAM) }?.let { uris.add(it) }
                text = safeExtra { source.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString() }
                subject = safeExtra { source.getStringExtra(Intent.EXTRA_SUBJECT) }
            }
            Intent.ACTION_SEND_MULTIPLE -> {
                origin = ORIGIN_SEND_MULTIPLE
                safeExtra { readUriListExtra(source, Intent.EXTRA_STREAM) }?.let { uris.addAll(it) }
                subject = safeExtra { source.getStringExtra(Intent.EXTRA_SUBJECT) }
            }
            else -> {
                Log.w(TAG, "Ignoring action ${source.action}")
                return
            }
        }

        // ClipData is filled by the sender (or by the framework for SEND) and
        // carries the URIs the read grant applies to.
        source.clipData?.let { clip ->
            for (i in 0 until clip.itemCount) {
                val uri = clip.getItemAt(i).uri ?: continue
                if (!uris.contains(uri)) uris.add(uri)
            }
        }

        if (uris.isEmpty() && text.isNullOrEmpty()) {
            Log.w(TAG, "Nothing to forward for ${source.action}")
            return
        }

        val forwarded = Intent(ACTION_SHARED).apply {
            setComponent(target)
            addFlags(
                Intent.FLAG_ACTIVITY_NEW_TASK or
                    Intent.FLAG_ACTIVITY_SINGLE_TOP or
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
            putExtra(EXTRA_GLOSSARION_ORIGIN, origin)
            source.type?.let { putExtra(EXTRA_GLOSSARION_MIME, it) }
            if (uris.isNotEmpty()) {
                putExtra(EXTRA_GLOSSARION_URI, uris[0])
                putParcelableArrayListExtra(EXTRA_GLOSSARION_URIS, uris)
                val clip = ClipData.newRawUri("glossarion", uris[0])
                for (i in 1 until uris.size) clip.addItem(ClipData.Item(uris[i]))
                clipData = clip
            }
            text?.let { putExtra(EXTRA_GLOSSARION_TEXT, it) }
            subject?.let { putExtra(EXTRA_GLOSSARION_SUBJECT, it) }
        }
        try {
            startActivity(forwarded)
        } catch (e: SecurityException) {
            // A sender passed a URI without granting us read access, so we
            // cannot re-grant it. Forward anyway; the plugin reports a per-item
            // error if the copy fails.
            Log.w(TAG, "Forwarding without URI grant", e)
            forwarded.removeFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            forwarded.clipData = null
            startActivity(forwarded)
        }
    }

    private inline fun <T> safeExtra(block: () -> T?): T? =
        try {
            block()
        } catch (e: Exception) {
            Log.w(TAG, "Unreadable share extra", e)
            null
        }

    companion object {
        private const val TAG = "GlossarionShare"

        const val ACTION_SHARED = "com.glossarion.flet_glossarion_native.action.SHARED"
        const val EXTRA_GLOSSARION_URI = "com.glossarion.flet_glossarion_native.extra.URI"
        const val EXTRA_GLOSSARION_URIS = "com.glossarion.flet_glossarion_native.extra.URIS"
        const val EXTRA_GLOSSARION_MIME = "com.glossarion.flet_glossarion_native.extra.MIME"
        const val EXTRA_GLOSSARION_TEXT = "com.glossarion.flet_glossarion_native.extra.TEXT"
        const val EXTRA_GLOSSARION_SUBJECT = "com.glossarion.flet_glossarion_native.extra.SUBJECT"
        const val EXTRA_GLOSSARION_ORIGIN = "com.glossarion.flet_glossarion_native.extra.ORIGIN"

        const val ORIGIN_VIEW = "view"
        const val ORIGIN_SEND = "send"
        const val ORIGIN_SEND_MULTIPLE = "send_multiple"

        // getParcelableExtra(String, Class) is unreliable on API 33; use it from 34.
        @Suppress("DEPRECATION")
        fun readUriExtra(intent: Intent, key: String): Uri? =
            if (Build.VERSION.SDK_INT >= 34) {
                intent.getParcelableExtra(key, Uri::class.java)
            } else {
                intent.getParcelableExtra<Uri>(key)
            }

        @Suppress("DEPRECATION")
        fun readUriListExtra(intent: Intent, key: String): ArrayList<Uri>? =
            if (Build.VERSION.SDK_INT >= 34) {
                intent.getParcelableArrayListExtra(key, Uri::class.java)
            } else {
                intent.getParcelableArrayListExtra<Uri>(key)
            }
    }
}
