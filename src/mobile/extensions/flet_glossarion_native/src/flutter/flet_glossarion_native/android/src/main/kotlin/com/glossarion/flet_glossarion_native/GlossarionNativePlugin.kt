package com.glossarion.flet_glossarion_native

import android.Manifest
import android.annotation.SuppressLint
import android.app.PendingIntent
import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import android.webkit.MimeTypeMap
import androidx.core.app.NotificationChannelCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.core.content.pm.PackageInfoCompat
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Android side of GlossarionNative (MethodChannel `glossarion_native/platform`).
 *
 * Dart -> Kotlin: attach, get_platform_info, clear_shared, init_notifications,
 * show_notification, cancel_notification, save_to_downloads.
 * Kotlin -> Dart: `share` {items}, `notification` {notification_id, action_id,
 * payload, launched_app}.
 *
 * Items received before Dart calls `attach` (cold start) are queued and
 * returned by `attach`; later ones are pushed. File copies and Downloads
 * exports run on a background executor, never on the UI thread.
 *
 * The foreground service itself is flutter_foreground_task's; this plugin only
 * declares its manifest entry (see AndroidManifest.xml).
 */
class GlossarionNativePlugin :
    FlutterPlugin,
    MethodChannel.MethodCallHandler,
    ActivityAware,
    PluginRegistry.NewIntentListener {

    private var appContext: Context? = null
    private var channel: MethodChannel? = null
    private var activityBinding: ActivityPluginBinding? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var io: ExecutorService? = null

    // Main-thread state.
    private var dartAttached = false
    private val pendingShared = ArrayList<Map<String, Any?>>()
    private val pendingNotifications = ArrayList<Map<String, Any?>>()
    private var launchNotification: Map<String, Any?>? = null

    // ------------------------------------------------------------ FlutterPlugin

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext = binding.applicationContext
        io = Executors.newFixedThreadPool(2)
        channel = MethodChannel(binding.binaryMessenger, CHANNEL).also {
            it.setMethodCallHandler(this)
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel?.setMethodCallHandler(null)
        channel = null
        io?.shutdown()
        io = null
        dartAttached = false
    }

    // ------------------------------------------------------------- ActivityAware

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activityBinding = binding
        binding.addOnNewIntentListener(this)
        handleIntent(binding.activity.intent, initial = true)
    }

    override fun onDetachedFromActivityForConfigChanges() {
        activityBinding?.removeOnNewIntentListener(this)
        activityBinding = null
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        activityBinding = binding
        binding.addOnNewIntentListener(this)
    }

    override fun onDetachedFromActivity() {
        activityBinding?.removeOnNewIntentListener(this)
        activityBinding = null
    }

    override fun onNewIntent(intent: Intent): Boolean = handleIntent(intent, initial = false)

    // ----------------------------------------------------------------- intents

    private fun handleIntent(intent: Intent?, initial: Boolean): Boolean {
        if (intent == null || intent.getBooleanExtra(EXTRA_HANDLED, false)) return false
        return when (intent.action) {
            ShareReceiverActivity.ACTION_SHARED -> {
                intent.putExtra(EXTRA_HANDLED, true)
                importSharedIntent(intent)
                true
            }
            ACTION_NOTIFICATION -> {
                intent.putExtra(EXTRA_HANDLED, true)
                handleNotificationIntent(intent, initial)
                true
            }
            Intent.ACTION_VIEW -> {
                // Fallback: a content:/file: VIEW intent that reached MainActivity
                // directly (not through ShareReceiverActivity). Flutter will also
                // push it as a route; the Python router ignores content:/file:.
                val data = intent.data
                val scheme = data?.scheme
                if (data != null &&
                    (scheme == ContentResolver.SCHEME_CONTENT || scheme == ContentResolver.SCHEME_FILE)
                ) {
                    intent.putExtra(EXTRA_HANDLED, true)
                    importUris(listOf(data), intent.type, ShareReceiverActivity.ORIGIN_VIEW, null, null)
                    true
                } else {
                    false
                }
            }
            else -> false
        }
    }

    private fun importSharedIntent(intent: Intent) {
        val origin = intent.getStringExtra(ShareReceiverActivity.EXTRA_GLOSSARION_ORIGIN)
            ?: ShareReceiverActivity.ORIGIN_SEND
        val mime = intent.getStringExtra(ShareReceiverActivity.EXTRA_GLOSSARION_MIME)
        val uris = ArrayList<Uri>()
        try {
            ShareReceiverActivity.readUriListExtra(intent, ShareReceiverActivity.EXTRA_GLOSSARION_URIS)
                ?.let { uris.addAll(it) }
            if (uris.isEmpty()) {
                ShareReceiverActivity.readUriExtra(intent, ShareReceiverActivity.EXTRA_GLOSSARION_URI)
                    ?.let { uris.add(it) }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Unreadable forwarded URIs", e)
        }
        if (uris.isEmpty()) {
            intent.clipData?.let { clip ->
                for (i in 0 until clip.itemCount) {
                    clip.getItemAt(i).uri?.let { uris.add(it) }
                }
            }
        }
        val text = intent.getStringExtra(ShareReceiverActivity.EXTRA_GLOSSARION_TEXT)
        val subject = intent.getStringExtra(ShareReceiverActivity.EXTRA_GLOSSARION_SUBJECT)
        importUris(uris, mime, origin, text, subject)
    }

    private fun importUris(
        uris: List<Uri>,
        mimeHint: String?,
        origin: String,
        text: String?,
        subject: String?,
    ) {
        val context = appContext ?: return
        val executor = io ?: return
        val batch = "${System.currentTimeMillis()}-${SEQUENCE.incrementAndGet()}"
        val singleMime = if (uris.size == 1) mimeHint else null
        try {
            executor.execute {
                val items = ArrayList<Map<String, Any?>>()
                val batchDir = File(sharedRoot(context), batch)
                for (uri in uris) {
                    items.add(copyUri(context, uri, singleMime, batchDir, origin))
                }
                if (!text.isNullOrEmpty()) {
                    items.add(
                        hashMapOf(
                            "id" to UUID.randomUUID().toString(),
                            "kind" to "text",
                            "text" to text,
                            "subject" to subject,
                            "mime_type" to "text/plain",
                            "source" to origin,
                        )
                    )
                }
                mainHandler.post { deliverShared(items) }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Could not schedule shared-file import", e)
        }
    }

    private fun copyUri(
        context: Context,
        uri: Uri,
        mimeHint: String?,
        batchDir: File,
        origin: String,
    ): Map<String, Any?> {
        val item = HashMap<String, Any?>()
        item["id"] = UUID.randomUUID().toString()
        item["kind"] = "file"
        item["uri"] = uri.toString()
        item["source"] = origin
        try {
            val resolver = context.contentResolver
            var displayName: String? = null
            if (uri.scheme == ContentResolver.SCHEME_CONTENT) {
                resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (index >= 0 && !cursor.isNull(index)) displayName = cursor.getString(index)
                    }
                }
            }
            val providerMime =
                if (uri.scheme == ContentResolver.SCHEME_CONTENT) resolver.getType(uri) else null
            val rawName = displayName ?: uri.lastPathSegment
            val mime = providerMime ?: mimeHint ?: guessMime(rawName)
            val name = safeFileName(rawName, mime)
            item["name"] = name

            if (!batchDir.isDirectory && !batchDir.mkdirs()) {
                throw IOException("Cannot create ${batchDir.path}")
            }
            val dest = uniqueFile(batchDir, name)
            val input = if (uri.scheme == ContentResolver.SCHEME_FILE) {
                FileInputStream(File(uri.path ?: throw IOException("Empty file URI")))
            } else {
                resolver.openInputStream(uri) ?: throw IOException("Cannot open $uri")
            }
            input.use { source ->
                FileOutputStream(dest).use { sink -> source.copyTo(sink, COPY_BUFFER) }
            }
            item["path"] = dest.absolutePath
            item["name"] = dest.name
            item["mime_type"] = mime
            item["size"] = dest.length()
        } catch (e: Exception) {
            Log.w(TAG, "Could not copy $uri", e)
            item["error"] = e.toString()
        }
        return item
    }

    private fun deliverShared(items: List<Map<String, Any?>>) {
        if (items.isEmpty()) return
        val ch = channel
        if (dartAttached && ch != null) {
            ch.invokeMethod("share", hashMapOf("items" to items))
        } else {
            pendingShared.addAll(items)
        }
    }

    // ------------------------------------------------------------- MethodChannel

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        val context = appContext
        if (context == null) {
            result.error("not_attached", "GlossarionNativePlugin is not attached to an engine", null)
            return
        }
        try {
            when (call.method) {
                "attach" -> {
                    dartAttached = true
                    val shared = ArrayList(pendingShared)
                    pendingShared.clear()
                    val notifications = ArrayList(pendingNotifications)
                    pendingNotifications.clear()
                    result.success(
                        hashMapOf(
                            "shared" to shared,
                            "notifications" to notifications,
                            "launch_notification" to launchNotification,
                        )
                    )
                }
                "get_platform_info" -> result.success(platformInfo(context))
                "clear_shared" -> {
                    pendingShared.clear()
                    if (call.argument<Boolean>("delete_files") == true) {
                        io?.execute { sharedRoot(context).deleteRecursively() }
                    }
                    result.success(null)
                }
                "init_notifications" -> {
                    val channels = call.argument<List<Any?>>("channels") ?: emptyList()
                    result.success(initNotifications(context, channels))
                }
                "show_notification" -> {
                    val args = call.arguments as? Map<*, *> ?: emptyMap<String, Any?>()
                    result.success(showNotification(context, args))
                }
                "cancel_notification" -> {
                    call.argument<Number>("id")?.let {
                        NotificationManagerCompat.from(context).cancel(it.toInt())
                    }
                    result.success(null)
                }
                "save_to_downloads" -> saveToDownloadsAsync(context, call, result)
                else -> result.notImplemented()
            }
        } catch (e: Exception) {
            Log.e(TAG, "${call.method} failed", e)
            result.error("native_error", e.toString(), null)
        }
    }

    // ------------------------------------------------------------ platform info

    private fun platformInfo(context: Context): Map<String, Any?> {
        val info = HashMap<String, Any?>()
        info["platform"] = "android"
        info["sdk_int"] = Build.VERSION.SDK_INT
        info["release"] = Build.VERSION.RELEASE
        info["manufacturer"] = Build.MANUFACTURER
        info["model"] = Build.MODEL
        info["package_name"] = context.packageName
        try {
            val pkg = packageInfo(context)
            info["version_name"] = pkg.versionName
            info["version_code"] = PackageInfoCompat.getLongVersionCode(pkg)
        } catch (e: Exception) {
            info["version_error"] = e.toString()
        }
        info["notifications_enabled"] = NotificationManagerCompat.from(context).areNotificationsEnabled()
        info["post_notifications_granted"] = Build.VERSION.SDK_INT < 33 ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        val power = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        info["ignoring_battery_optimizations"] =
            power?.isIgnoringBatteryOptimizations(context.packageName) ?: false
        info["fgs_types"] = listOf("dataSync")
        // Android 15 (API 35) limits dataSync foreground services to 6 h per 24 h.
        info["fgs_data_sync_time_limited"] = Build.VERSION.SDK_INT >= 35
        info["save_to_downloads"] = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ||
            hasLegacyWritePermission(context)
        info["continued_processing"] = false
        info["shared_dir"] = sharedRoot(context).absolutePath
        info["cache_dir"] = context.cacheDir.absolutePath
        info["files_dir"] = context.filesDir.absolutePath
        return info
    }

    @Suppress("DEPRECATION")
    private fun packageInfo(context: Context): PackageInfo =
        if (Build.VERSION.SDK_INT >= 33) {
            context.packageManager.getPackageInfo(
                context.packageName,
                PackageManager.PackageInfoFlags.of(0)
            )
        } else {
            context.packageManager.getPackageInfo(context.packageName, 0)
        }

    // ------------------------------------------------------------ notifications

    private fun initNotifications(context: Context, channels: List<Any?>): Boolean {
        val manager = NotificationManagerCompat.from(context)
        for (raw in channels) {
            val spec = raw as? Map<*, *> ?: continue
            val id = spec["id"] as? String ?: continue
            val name = spec["name"] as? String ?: id
            val importance = when ((spec["importance"] as? String)?.lowercase()) {
                "min" -> NotificationManagerCompat.IMPORTANCE_MIN
                "low" -> NotificationManagerCompat.IMPORTANCE_LOW
                "high" -> NotificationManagerCompat.IMPORTANCE_HIGH
                "max" -> NotificationManagerCompat.IMPORTANCE_MAX
                else -> NotificationManagerCompat.IMPORTANCE_DEFAULT
            }
            val builder = NotificationChannelCompat.Builder(id, importance).setName(name)
            (spec["description"] as? String)?.let { builder.setDescription(it) }
            (spec["show_badge"] as? Boolean)?.let { builder.setShowBadge(it) }
            (spec["vibration"] as? Boolean)?.let { builder.setVibrationEnabled(it) }
            if (spec["sound"] == false) builder.setSound(null, null)
            manager.createNotificationChannel(builder.build())
        }
        return manager.areNotificationsEnabled()
    }

    private fun ensureChannel(context: Context, channelId: String) {
        val manager = NotificationManagerCompat.from(context)
        if (manager.getNotificationChannelCompat(channelId) != null) return
        manager.createNotificationChannel(
            NotificationChannelCompat.Builder(channelId, NotificationManagerCompat.IMPORTANCE_DEFAULT)
                .setName(channelId)
                .build()
        )
    }

    @SuppressLint("MissingPermission")
    private fun showNotification(context: Context, args: Map<*, *>): Boolean {
        val id = (args["id"] as? Number)?.toInt() ?: return false
        val manager = NotificationManagerCompat.from(context)
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            return false
        }
        if (!manager.areNotificationsEnabled()) return false

        val channelId = (args["channel_id"] as? String)?.takeIf { it.isNotEmpty() } ?: DEFAULT_CHANNEL
        ensureChannel(context, channelId)
        val title = args["title"] as? String ?: ""
        val body = args["body"] as? String ?: ""
        val payload = args["payload"] as? String
        val ongoing = args["ongoing"] as? Boolean ?: false
        val silent = args["silent"] as? Boolean ?: false
        val autoCancel = args["auto_cancel"] as? Boolean ?: !ongoing

        val builder = NotificationCompat.Builder(context, channelId)
            .setSmallIcon(R.drawable.glossarion_native_ic_notification)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setOngoing(ongoing)
            .setAutoCancel(autoCancel)
            .setOnlyAlertOnce(true)
            .setSilent(silent)
            .setContentIntent(notificationIntent(context, id, payload, null, 0))

        val progress = args["progress"] as? List<*>
        val indeterminate = args["indeterminate"] as? Boolean ?: false
        if (progress != null && progress.size == 2) {
            val done = (progress[0] as? Number)?.toInt() ?: 0
            val total = (progress[1] as? Number)?.toInt() ?: 0
            val max = maxOf(total, 0)
            builder.setProgress(max, done.coerceIn(0, max), indeterminate || max == 0)
        } else if (indeterminate) {
            builder.setProgress(0, 0, true)
        }

        (args["actions"] as? List<*>)?.forEachIndexed { index, raw ->
            val action = raw as? Map<*, *> ?: return@forEachIndexed
            val actionId = action["id"] as? String ?: return@forEachIndexed
            val label = action["title"] as? String ?: actionId
            builder.addAction(0, label, notificationIntent(context, id, payload, actionId, index + 1))
        }

        manager.notify(id, builder.build())
        return true
    }

    private fun notificationIntent(
        context: Context,
        id: Int,
        payload: String?,
        actionId: String?,
        slot: Int,
    ): PendingIntent? {
        val launch = context.packageManager.getLaunchIntentForPackage(context.packageName)
        val target = launch?.component ?: return null
        // data stays null so Flutter does not push it as a route.
        val intent = Intent(ACTION_NOTIFICATION).apply {
            setComponent(target)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            putExtra(EXTRA_NOTIFICATION_ID, id)
            putExtra(EXTRA_NOTIFICATION_PAYLOAD, payload)
            putExtra(EXTRA_NOTIFICATION_ACTION, actionId)
        }
        val requestCode = (id and 0x00FFFFFF) * 16 + (slot and 0x0F)
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        return PendingIntent.getActivity(context, requestCode, intent, flags)
    }

    private fun handleNotificationIntent(intent: Intent, initial: Boolean) {
        val id = intent.getIntExtra(EXTRA_NOTIFICATION_ID, -1)
        val actionId = intent.getStringExtra(EXTRA_NOTIFICATION_ACTION)
        val event = hashMapOf<String, Any?>(
            "notification_id" to id,
            "action_id" to actionId,
            "payload" to intent.getStringExtra(EXTRA_NOTIFICATION_PAYLOAD),
            "launched_app" to initial,
        )
        if (actionId != null && id >= 0) {
            // Action buttons do not auto-cancel the notification.
            appContext?.let { NotificationManagerCompat.from(it).cancel(id) }
        }
        if (initial && launchNotification == null) {
            launchNotification = event
            return
        }
        val ch = channel
        if (dartAttached && ch != null) {
            ch.invokeMethod("notification", event)
        } else {
            pendingNotifications.add(event)
        }
    }

    // ------------------------------------------------------------- Downloads

    private fun saveToDownloadsAsync(context: Context, call: MethodCall, result: MethodChannel.Result) {
        val path = call.argument<String>("path")
        if (path.isNullOrEmpty()) {
            result.error("bad_args", "path is required", null)
            return
        }
        val displayName = call.argument<String>("display_name")
        val mimeType = call.argument<String>("mime_type")
        val subdir = call.argument<String>("subdir")
        val executor = io
        if (executor == null) {
            result.error("not_attached", "executor unavailable", null)
            return
        }
        executor.execute {
            try {
                val saved = saveToDownloads(context, File(path), displayName, mimeType, subdir)
                mainHandler.post { result.success(saved) }
            } catch (e: Exception) {
                Log.e(TAG, "save_to_downloads failed", e)
                mainHandler.post { result.error("save_failed", e.toString(), null) }
            }
        }
    }

    private fun saveToDownloads(
        context: Context,
        source: File,
        displayName: String?,
        mimeType: String?,
        subdir: String?,
    ): String? {
        if (!source.isFile) throw IOException("Source file not found: ${source.path}")
        val mime = mimeType?.takeIf { it.isNotEmpty() } ?: guessMime(source.name) ?: "application/octet-stream"
        val name = safeFileName(displayName?.takeIf { it.isNotBlank() } ?: source.name, mime)
        val folder = safeSubdir(subdir)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val resolver = context.contentResolver
            val relativePath = if (folder.isEmpty()) {
                Environment.DIRECTORY_DOWNLOADS
            } else {
                Environment.DIRECTORY_DOWNLOADS + "/" + folder
            }
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, relativePath)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val collection = MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val uri = resolver.insert(collection, values)
                ?: throw IOException("MediaStore refused the Downloads entry")
            try {
                val out = resolver.openOutputStream(uri) ?: throw IOException("Cannot write $uri")
                out.use { sink ->
                    FileInputStream(source).use { input -> input.copyTo(sink, COPY_BUFFER) }
                }
                val done = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
                resolver.update(uri, done, null, null)
            } catch (e: Exception) {
                try {
                    resolver.delete(uri, null, null)
                } catch (cleanupError: Exception) {
                    Log.w(TAG, "Could not remove partial Downloads entry", cleanupError)
                }
                throw e
            }
            return uri.toString()
        }

        // API 26-28: direct file write; needs WRITE_EXTERNAL_STORAGE, which the
        // app manifest removes by default (see README).
        if (!hasLegacyWritePermission(context)) return null
        @Suppress("DEPRECATION")
        val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val dir = if (folder.isEmpty()) downloads else File(downloads, folder)
        if (!dir.isDirectory && !dir.mkdirs()) throw IOException("Cannot create ${dir.path}")
        val dest = uniqueFile(dir, name)
        source.copyTo(dest, overwrite = false, bufferSize = COPY_BUFFER)
        MediaScannerConnection.scanFile(context, arrayOf(dest.absolutePath), arrayOf(mime), null)
        return dest.absolutePath
    }

    private fun hasLegacyWritePermission(context: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) ==
            PackageManager.PERMISSION_GRANTED

    companion object {
        private const val TAG = "GlossarionNative"
        private const val CHANNEL = "glossarion_native/platform"
        private const val SHARED_DIR = "shared"
        private const val DEFAULT_CHANNEL = "jobs.done"
        private const val COPY_BUFFER = 256 * 1024
        private const val MAX_NAME_LENGTH = 120

        const val ACTION_NOTIFICATION = "com.glossarion.flet_glossarion_native.action.NOTIFICATION"
        const val EXTRA_NOTIFICATION_ID = "com.glossarion.flet_glossarion_native.extra.NOTIFICATION_ID"
        const val EXTRA_NOTIFICATION_PAYLOAD = "com.glossarion.flet_glossarion_native.extra.NOTIFICATION_PAYLOAD"
        const val EXTRA_NOTIFICATION_ACTION = "com.glossarion.flet_glossarion_native.extra.NOTIFICATION_ACTION"
        private const val EXTRA_HANDLED = "com.glossarion.flet_glossarion_native.extra.HANDLED"

        private val SEQUENCE = AtomicInteger(0)

        private fun sharedRoot(context: Context): File = File(context.cacheDir, SHARED_DIR)

        private fun guessMime(name: String?): String? {
            if (name.isNullOrEmpty()) return null
            val ext = name.substringAfterLast('.', "").lowercase()
            if (ext.isEmpty()) return null
            return when (ext) {
                "epub" -> "application/epub+zip"
                "cbz" -> "application/vnd.comicbook+zip"
                "srt" -> "application/x-subrip"
                "vtt" -> "text/vtt"
                "ass", "ssa" -> "text/x-ssa"
                "md" -> "text/markdown"
                "sdlxliff", "xliff", "xlf" -> "application/xliff+xml"
                else -> MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext)
            }
        }

        private fun safeFileName(raw: String?, mime: String?): String {
            var name = (raw ?: "").substringAfterLast('/')
            name = name.map { ch ->
                if (ch.code < 32 || "\\/:*?\"<>|".indexOf(ch) >= 0) '_' else ch
            }.joinToString("").trim().trimStart('.')
            if (name.isEmpty()) name = "shared_${System.currentTimeMillis()}"
            if (!name.contains('.') && mime != null) {
                val ext = MimeTypeMap.getSingleton().getExtensionFromMimeType(mime)
                    ?: when (mime) {
                        "application/epub+zip" -> "epub"
                        "application/vnd.comicbook+zip", "application/x-cbz" -> "cbz"
                        "application/x-subrip" -> "srt"
                        else -> null
                    }
                if (ext != null) name = "$name.$ext"
            }
            if (name.length > MAX_NAME_LENGTH) {
                val dot = name.lastIndexOf('.')
                val ext = if (dot > 0 && name.length - dot <= 12) name.substring(dot) else ""
                name = name.substring(0, MAX_NAME_LENGTH - ext.length) + ext
            }
            return name
        }

        private fun safeSubdir(raw: String?): String =
            (raw ?: "")
                .split('/')
                .map { part -> part.trim().filter { it.code >= 32 && "\\:*?\"<>|".indexOf(it) < 0 } }
                .filter { it.isNotEmpty() && it != "." && it != ".." }
                .joinToString("/")

        private fun uniqueFile(dir: File, name: String): File {
            var candidate = File(dir, name)
            if (!candidate.exists()) return candidate
            val dot = name.lastIndexOf('.')
            val stem = if (dot > 0) name.substring(0, dot) else name
            val ext = if (dot > 0) name.substring(dot) else ""
            var n = 1
            while (candidate.exists()) {
                candidate = File(dir, "$stem ($n)$ext")
                n += 1
            }
            return candidate
        }
    }
}
