package com.glossarion.flet_glossarion_native

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.core.content.FileProvider
import java.io.File

/**
 * In-app updates (About > Updates): Python downloads and verifies the release APK into
 * `<cache>/updates`, then [install] hands it to the system installer.
 *
 * A sideloaded app can never update itself silently (that needs the device owner or the Play
 * Store): Android always shows its own "Do you want to update this app?" screen. The first time
 * it also needs the user's "Install unknown apps" switch for Glossarion
 * (REQUEST_INSTALL_PACKAGES); [openInstallSettings] opens exactly that page.
 *
 * The APK is shared as a content:// URI through [GlossarionUpdateFileProvider] (authority
 * `<applicationId>.glossarion_native.updates`, res/xml/glossarion_native_update_paths.xml), which
 * shares the updates folder only. Answers are maps for Python:
 * {ok: true} or {ok: false, error: permission | missing | bad_args | native_error, message}.
 */
internal object AppUpdates {
    private const val TAG = "GlossarionNative"
    const val UPDATES_DIR = "updates"
    const val AUTHORITY_SUFFIX = ".glossarion_native.updates"
    private const val APK_MIME = "application/vnd.android.package-archive"

    fun canInstall(context: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.O || context.packageManager.canRequestPackageInstalls()

    fun openInstallSettings(context: Context, activity: Activity?): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
        val intent = Intent(
            Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
            Uri.parse("package:" + context.packageName),
        )
        return start(context, activity, intent)
    }

    fun install(context: Context, activity: Activity?, path: String?): Map<String, Any?> {
        if (path.isNullOrEmpty()) return error("bad_args", "No APK path")
        val root = File(context.cacheDir, UPDATES_DIR).canonicalFile
        val file = File(path).canonicalFile
        if (file.parentFile != root || !file.name.endsWith(".apk", ignoreCase = true)) {
            return error("bad_args", "Only an APK in the app's update folder can be installed")
        }
        if (!file.isFile) return error("missing", "The downloaded APK is gone")
        if (!canInstall(context)) {
            return error("permission", "Allow Glossarion to install apps first")
        }
        val uri = FileProvider.getUriForFile(context, context.packageName + AUTHORITY_SUFFIX, file)
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, APK_MIME)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return if (start(context, activity, intent)) {
            hashMapOf("ok" to true)
        } else {
            error("native_error", "Android has no installer for APK files")
        }
    }

    private fun start(context: Context, activity: Activity?, intent: Intent): Boolean =
        try {
            if (activity != null) {
                activity.startActivity(intent)
            } else {
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
            }
            true
        } catch (e: ActivityNotFoundException) {
            Log.w(TAG, "No activity for ${intent.action}", e)
            false
        }

    private fun error(code: String, message: String): Map<String, Any?> =
        hashMapOf("ok" to false, "error" to code, "message" to message)
}

/** Its own FileProvider class, so the app's other providers (image_picker's) never clash with it. */
class GlossarionUpdateFileProvider : FileProvider()
