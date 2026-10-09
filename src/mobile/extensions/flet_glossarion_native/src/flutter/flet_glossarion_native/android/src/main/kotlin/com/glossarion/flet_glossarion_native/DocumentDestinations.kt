package com.glossarion.flet_glossarion_native

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.UriPermission
import android.content.pm.PackageManager
import android.content.pm.ProviderInfo
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.ParcelFileDescriptor
import android.os.Process
import android.os.SystemClock
import android.provider.DocumentsContract
import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import android.util.Log
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Document destinations (U10): a folder or file the user picked once in the system picker
 * (Storage Access Framework), written to later without asking again.
 *
 * Methods (MethodChannel `glossarion_native/platform`, routed by GlossarionNativePlugin):
 * pick_folder (ACTION_OPEN_DOCUMENT_TREE), pick_save_location (ACTION_CREATE_DOCUMENT),
 * pick_document (ACTION_OPEN_DOCUMENT), list_children, create_file, create_folder, write_file,
 * stat, delete, query_root, release, list_grants, cancel_document_op.
 * Every answer is a map {ok, error, message, scope, retryable, ...} (see the Python
 * flet_glossarion_native/documents.py); typed failures never use result.error().
 *
 * Rules taken from the U10 research critic:
 * - Persist only the flags the picker actually granted (data.flags & READ|WRITE).
 * - FileNotFoundException is not "deleted": Drive throws it for an unsupported mode, Nextcloud while
 *   offline. A document is `missing` only when its tree root answers and the document does not.
 * - Non-truncating modes (w, rw) are used only when the new file is not shorter than the cloud copy
 *   (or the descriptor is a real file we can ftruncate); the written length is read back with
 *   statSize ('r'), never COLUMN_SIZE, and a stale tail is reported as size_mismatch/needs_replace.
 * - A failed create is retried once, after looking for a file that appeared anyway.
 * - A picker answer whose call is gone (activity or process recreated) is still persisted and
 *   reported as a `document` event / attach result; a picker left open across a process restart is
 *   reported as cancelled.
 * Blocking work runs on [io]; answers are posted on the main thread.
 */
internal class DocumentDestinations(
    private val context: Context,
    private val mainHandler: Handler,
    private val isDartAttached: () -> Boolean,
    private val sendToDart: (Map<String, Any?>) -> Unit,
) {
    /** Set by the plugin from the ActivityAware callbacks. */
    var activity: Activity? = null

    private val io: ExecutorService = Executors.newFixedThreadPool(2)
    private val resolver: ContentResolver get() = context.contentResolver
    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val cancelFlags = ConcurrentHashMap<String, AtomicBoolean>()

    // Main-thread state.
    private var pending: PendingPick? = null
    private val lateResults = ArrayList<Map<String, Any?>>()

    private class PendingPick(
        val opId: String,
        val kind: String,
        val requestCode: Int,
        val result: MethodChannel.Result?,
        val name: String?,
        val mime: String?,
        val sourcePath: String?,
        val modes: List<String>,
    )

    /** A folder or file: [tree] is set when it lives inside a picked folder. */
    private class Ref(val kind: String, val tree: Uri?, val document: Uri)

    private class Row(
        val documentId: String?,
        val name: String?,
        val mime: String?,
        val size: Long?,
        val modified: Long?,
        val flags: Int,
    ) {
        val isDir: Boolean get() = mime == DocumentsContract.Document.MIME_TYPE_DIR
    }

    private class Listing(val rows: List<Row>?, val loading: Boolean)

    private class CancelledOp : IOException("cancelled")

    fun handles(method: String): Boolean = method in METHODS

    fun dispose() {
        io.shutdown()
        pending = null
    }

    // ------------------------------------------------------------------ dispatch

    fun handle(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "pick_folder" -> startPick(KIND_FOLDER, call, result)
            "pick_save_location" -> startPick(KIND_SAVE, call, result)
            "pick_document" -> startPick(KIND_OPEN, call, result)
            "cancel_document_op" -> {
                val opId = call.argument<String>("op_id")
                if (opId.isNullOrEmpty()) {
                    result.success(false)
                } else {
                    cancelFlags.getOrPut(opId) { AtomicBoolean(false) }.set(true)
                    result.success(true)
                }
            }
            else -> runIo(call, result)
        }
    }

    private fun runIo(call: MethodCall, result: MethodChannel.Result) {
        val method = call.method
        val args: Map<*, *> = (call.arguments as? Map<*, *>) ?: emptyMap<String, Any?>()
        try {
            io.execute {
                val payload: Any? = try {
                    when (method) {
                        "list_children" -> listChildren(args)
                        "create_file" -> createFile(args, directory = false)
                        "create_folder" -> createFile(args, directory = true)
                        "write_file" -> writeFile(args)
                        "rename_document" -> renameDocument(args)
                        "stat" -> stat(args)
                        "delete" -> delete(args)
                        "query_root" -> queryRoot(args)
                        "release" -> release(args)
                        "list_grants" -> listGrants()
                        else -> fail(ERR_BAD_ARGS, "Unknown document method $method", null)
                    }
                } catch (e: Exception) {
                    // Exception text can carry content URIs: log the type only.
                    Log.w(TAG, "$method failed: ${e.javaClass.simpleName}")
                    when (method) {
                        "release" -> false
                        "list_grants" -> ArrayList<Map<String, Any?>>()
                        else -> fail(classify(e, notFound = ERR_PROVIDER), e.toString(), null)
                    }
                }
                mainHandler.post { result.success(payload) }
            }
        } catch (e: Exception) {
            result.success(fail(ERR_PROVIDER, "Could not schedule $method: $e", null))
        }
    }

    // ------------------------------------------------------------------- pickers

    private fun startPick(kind: String, call: MethodCall, result: MethodChannel.Result) {
        if (pending != null) {
            result.success(fail(ERR_BUSY, "Another picker is already open", null))
            return
        }
        val act = activity
        if (act == null) {
            result.success(fail(ERR_UNAVAILABLE, "Glossarion is not in the foreground", null))
            return
        }
        val args: Map<*, *> = (call.arguments as? Map<*, *>) ?: emptyMap<String, Any?>()
        val opId = (args["op_id"] as? String)?.takeIf { it.isNotEmpty() } ?: UUID.randomUUID().toString()
        val name = args["name"] as? String
        val mime = (args["mime_type"] as? String)?.takeIf { it.isNotEmpty() }
        val grantFlags = Intent.FLAG_GRANT_READ_URI_PERMISSION or
            Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
            Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
        val intent = when (kind) {
            KIND_FOLDER -> Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
                addFlags(grantFlags or Intent.FLAG_GRANT_PREFIX_URI_PERMISSION)
            }
            KIND_SAVE -> Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = mime ?: "application/octet-stream"
                putExtra(Intent.EXTRA_TITLE, GlossarionNativePlugin.safeFileName(name, mime))
                addFlags(grantFlags)
            }
            else -> Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                val types = (args["mime_types"] as? List<*>)
                    ?.mapNotNull { it as? String }
                    ?.filter { it.isNotEmpty() }
                    .orEmpty()
                type = if (types.size == 1) types[0] else "*/*"
                if (types.size > 1) putExtra(Intent.EXTRA_MIME_TYPES, types.toTypedArray())
                addFlags(grantFlags)
            }
        }
        val initial = parseRef(args["initial"])?.document
        if (initial != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            intent.putExtra(DocumentsContract.EXTRA_INITIAL_URI, initial)
        }
        val record = PendingPick(
            opId = opId,
            kind = kind,
            requestCode = requestCodeFor(kind),
            result = result,
            name = name,
            mime = mime,
            sourcePath = (args["source_path"] as? String)?.takeIf { it.isNotEmpty() },
            modes = parseModes(args["mode_chain"]) ?: DEFAULT_MODES,
        )
        pending = record
        savePendingRecord(record)
        try {
            act.startActivityForResult(intent, record.requestCode)
        } catch (e: ActivityNotFoundException) {
            pending = null
            clearPendingRecord()
            result.success(fail(ERR_UNAVAILABLE, "No app on this phone can show the system file picker", null))
        } catch (e: Exception) {
            pending = null
            clearPendingRecord()
            result.success(fail(ERR_PROVIDER, "Could not open the picker: $e", null))
        }
    }

    /** ActivityResultListener entry (main thread). */
    fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode !in REQUEST_CODES) return false
        val live = pending?.takeIf { it.requestCode == requestCode }
        if (live != null) pending = null
        val record = live ?: loadPendingRecord()?.takeIf { it.requestCode == requestCode }
        clearPendingRecord()
        if (record == null) return true
        val uri = data?.data
        if (resultCode != Activity.RESULT_OK || uri == null) {
            deliver(record, fail(ERR_CANCELLED, "No location was chosen", null))
            return true
        }
        val grantFlags = data?.flags ?: 0
        try {
            io.execute {
                val payload = try {
                    completePick(record, uri, grantFlags)
                } catch (e: Exception) {
                    Log.w(TAG, "pick completion failed: ${e.javaClass.simpleName}")
                    fail(classify(e, notFound = ERR_PROVIDER), e.toString(), null)
                }
                mainHandler.post { deliver(record, payload) }
            }
        } catch (e: Exception) {
            deliver(record, fail(ERR_PROVIDER, "Could not finish the pick: $e", null))
        }
        return true
    }

    /** Dart attached: answers that arrived without a live call, plus a picker lost to a restart. */
    fun onDartAttach(): List<Map<String, Any?>> {
        val out = ArrayList<Map<String, Any?>>(lateResults)
        lateResults.clear()
        val record = loadPendingRecord()
        if (record != null && pending == null && prefs.getInt(KEY_PID, -1) != Process.myPid()) {
            // The picker was open when the previous process died and no answer came back.
            clearPendingRecord()
            out.add(pickEvent(record, fail(ERR_CANCELLED, "Glossarion was restarted while the picker was open", null)))
        }
        return out
    }

    private fun deliver(record: PendingPick, payload: Map<String, Any?>) {
        val result = record.result
        if (result != null) {
            result.success(payload)
            return
        }
        val event = pickEvent(record, payload)
        if (isDartAttached()) sendToDart(event) else lateResults.add(event)
    }

    private fun pickEvent(record: PendingPick, payload: Map<String, Any?>): Map<String, Any?> {
        val status = when {
            payload["ok"] == true -> "ok"
            payload["error"] == ERR_CANCELLED -> "cancelled"
            else -> "error"
        }
        return hashMapOf(
            "type" to EVENT_PICK_RESULT,
            "op_id" to record.opId,
            "kind" to record.kind,
            "status" to status,
            "result" to payload,
        )
    }

    private fun completePick(record: PendingPick, uri: Uri, grantFlags: Int): Map<String, Any?> {
        // Only the flags the picker granted may be persisted (asking for more throws).
        val flags = grantFlags and
            (Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        var persisted = false
        var persistError: String? = null
        if (flags != 0) {
            try {
                resolver.takePersistableUriPermission(uri, flags)
                persisted = true
            } catch (e: SecurityException) {
                persistError = e.toString()
            }
        }
        val canWrite = (flags and Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0
        if (record.kind == KIND_FOLDER) {
            val rootId = DocumentsContract.getTreeDocumentId(uri)
            val root = DocumentsContract.buildDocumentUriUsingTree(uri, rootId)
            val row = try {
                queryRows(root)?.firstOrNull()
            } catch (e: Exception) {
                null
            }
            val target = refMap(uri, root, row, stableId("android:$uri"))
            target["kind"] = KIND_FOLDER
            decorate(target, uri.authority, rootId, persisted, canWrite, row, folder = true)
            return ok("target" to target, "persisted" to persisted, "persist_error" to persistError)
        }
        val row = try {
            queryRows(uri)?.firstOrNull()
        } catch (e: Exception) {
            null
        }
        val document = refMap(null, uri, row, stableId("android:$uri"))
        document["kind"] = KIND_FILE
        decorate(document, uri.authority, documentIdOrNull(uri), persisted, canWrite, row, folder = false)
        val out = ok("document" to document, "persisted" to persisted, "persist_error" to persistError, "write" to null)
        if (record.kind == KIND_SAVE && document["own_folder"] == true) {
            // Glossarion's own folder (Download/Glossarion, the app's storage) is refused as a save location:
            // write nothing, and take back the empty document the Save dialog made (the app releases the grant).
            try {
                DocumentsContract.deleteDocument(resolver, uri)
                out["removed"] = true
            } catch (e: Exception) {
                Log.w(TAG, "removing a refused save location failed: ${e.javaClass.simpleName}")
            }
            out["own_folder"] = true
            return out
        }
        val source = record.sourcePath?.let { File(it) }
        if (record.kind == KIND_SAVE && source != null && source.isFile) {
            cancelFlags.putIfAbsent(record.opId, AtomicBoolean(false))
            try {
                out["write"] = writeChain(Ref(KIND_FILE, null, uri), source, record.modes, record.opId, true)
            } finally {
                cancelFlags.remove(record.opId)
            }
        }
        return out
    }

    // ----------------------------------------------------------- pending record

    private fun savePendingRecord(record: PendingPick) {
        // commit(): the process may die right after the picker opens.
        prefs.edit()
            .putString(KEY_OP, record.opId)
            .putString(KEY_KIND, record.kind)
            .putInt(KEY_REQUEST, record.requestCode)
            .putInt(KEY_PID, Process.myPid())
            .putLong(KEY_STARTED, System.currentTimeMillis())
            .putString(KEY_NAME, record.name)
            .putString(KEY_MIME, record.mime)
            .putString(KEY_SOURCE, record.sourcePath)
            .putString(KEY_MODES, record.modes.joinToString(","))
            .commit()
    }

    private fun loadPendingRecord(): PendingPick? {
        val opId = prefs.getString(KEY_OP, null) ?: return null
        val kind = prefs.getString(KEY_KIND, null) ?: return null
        return PendingPick(
            opId = opId,
            kind = kind,
            requestCode = prefs.getInt(KEY_REQUEST, -1),
            result = null,
            name = prefs.getString(KEY_NAME, null),
            mime = prefs.getString(KEY_MIME, null),
            sourcePath = prefs.getString(KEY_SOURCE, null),
            modes = parseModes(prefs.getString(KEY_MODES, null)?.split(",")) ?: DEFAULT_MODES,
        )
    }

    private fun clearPendingRecord() {
        prefs.edit().clear().commit()
    }

    // ------------------------------------------------------------------ queries

    private fun parseRef(raw: Any?): Ref? {
        val map = raw as? Map<*, *> ?: return null
        val tree = (map["uri"] as? String)?.takeIf { it.isNotEmpty() }?.let { Uri.parse(it) }
        val docString = (map["document"] as? String)?.takeIf { it.isNotEmpty() }
        val document: Uri = when {
            docString != null -> Uri.parse(docString)
            tree != null -> DocumentsContract.buildDocumentUriUsingTree(
                tree, DocumentsContract.getTreeDocumentId(tree)
            )
            else -> return null
        }
        val kind = if ((map["kind"] as? String) == KIND_FOLDER || docString == null) KIND_FOLDER else KIND_FILE
        return Ref(kind, tree, document)
    }

    private fun rootOf(ref: Ref): Uri? = ref.tree?.let {
        DocumentsContract.buildDocumentUriUsingTree(it, DocumentsContract.getTreeDocumentId(it))
    }

    private fun isRoot(ref: Ref): Boolean = ref.tree != null && ref.document == rootOf(ref)

    private fun idFor(ref: Ref): String =
        if (isRoot(ref)) stableId("android:${ref.tree}") else stableId("android:${ref.document}")

    private fun documentIdOrNull(uri: Uri): String? = try {
        DocumentsContract.getDocumentId(uri)
    } catch (e: Exception) {
        null
    }

    /** null: the provider gave no cursor (DocumentsProvider.query returns null on FileNotFoundException). */
    private fun queryRows(uri: Uri): List<Row>? = queryListing(uri).rows

    private fun queryListing(uri: Uri): Listing {
        val cursor = resolver.query(uri, COLUMNS, null, null, null) ?: return Listing(null, false)
        return cursor.use { c ->
            val rows = ArrayList<Row>()
            while (c.moveToNext()) rows.add(readRow(c))
            val loading = c.extras?.getBoolean(DocumentsContract.EXTRA_LOADING, false) ?: false
            Listing(rows, loading)
        }
    }

    private fun readRow(c: Cursor): Row {
        fun text(column: String): String? {
            val index = c.getColumnIndex(column)
            return if (index >= 0 && !c.isNull(index)) c.getString(index) else null
        }
        fun number(column: String): Long? {
            val index = c.getColumnIndex(column)
            return if (index >= 0 && !c.isNull(index)) c.getLong(index) else null
        }
        return Row(
            documentId = text(DocumentsContract.Document.COLUMN_DOCUMENT_ID),
            name = text(DocumentsContract.Document.COLUMN_DISPLAY_NAME),
            mime = text(DocumentsContract.Document.COLUMN_MIME_TYPE),
            size = number(DocumentsContract.Document.COLUMN_SIZE),
            modified = number(DocumentsContract.Document.COLUMN_LAST_MODIFIED),
            flags = (number(DocumentsContract.Document.COLUMN_FLAGS) ?: 0L).toInt(),
        )
    }

    private fun persistedGrant(uri: Uri): UriPermission? =
        resolver.persistedUriPermissions.firstOrNull { it.uri == uri }

    private fun refMap(tree: Uri?, document: Uri, row: Row?, id: String): HashMap<String, Any?> {
        val flags = row?.flags ?: 0
        val isDir = row?.isDir == true
        return hashMapOf(
            "platform" to "android",
            "kind" to if (isDir) KIND_FOLDER else KIND_FILE,
            "id" to id,
            "uri" to tree?.toString(),
            "document" to document.toString(),
            "bookmark" to null,
            "root" to null,
            "path" to null,
            "name" to row?.name,
            "mime" to row?.mime,
            "size" to row?.size,
            "mtime" to row?.modified,
            "flags" to flags,
            "provider" to document.authority,
            "can_write" to ((flags and DocumentsContract.Document.FLAG_SUPPORTS_WRITE) != 0),
            "can_create" to (isDir && (flags and DocumentsContract.Document.FLAG_DIR_SUPPORTS_CREATE) != 0),
            "can_delete" to ((flags and DocumentsContract.Document.FLAG_SUPPORTS_DELETE) != 0),
            "virtual" to ((flags and DocumentsContract.Document.FLAG_VIRTUAL_DOCUMENT) != 0),
        )
    }

    private fun decorate(
        target: HashMap<String, Any?>,
        authority: String?,
        documentId: String?,
        persisted: Boolean,
        grantWrite: Boolean,
        row: Row?,
        folder: Boolean,
    ) {
        val flags = row?.flags ?: 0
        target["provider_label"] = providerLabel(authority)
        target["persisted"] = persisted
        target["own_folder"] = isOwnFolder(authority, documentId)
        if (folder) {
            target["can_write"] = grantWrite
            target["can_create"] = grantWrite && (row == null ||
                (flags and DocumentsContract.Document.FLAG_DIR_SUPPORTS_CREATE) != 0)
        } else {
            target["can_write"] = grantWrite &&
                (row == null || (flags and DocumentsContract.Document.FLAG_SUPPORTS_WRITE) != 0)
        }
    }

    private fun providerInfo(authority: String): ProviderInfo? = try {
        val pm = context.packageManager
        if (Build.VERSION.SDK_INT >= 33) {
            pm.resolveContentProvider(authority, PackageManager.ComponentInfoFlags.of(0))
        } else {
            @Suppress("DEPRECATION")
            pm.resolveContentProvider(authority, 0)
        }
    } catch (e: Exception) {
        null
    }

    private fun providerLabel(authority: String?): String? {
        if (authority.isNullOrEmpty()) return null
        val info = providerInfo(authority) ?: return null
        return try {
            info.loadLabel(context.packageManager)?.toString()
        } catch (e: Exception) {
            null
        }
    }

    /** Glossarion's own storage: never a cloud destination (the app refuses it). */
    private fun isOwnFolder(authority: String?, documentId: String?): Boolean {
        if (authority.isNullOrEmpty() || documentId.isNullOrEmpty()) return false
        val pkg = context.packageName.lowercase(Locale.ROOT)
        val id = documentId.lowercase(Locale.ROOT)
        return when (authority) {
            EXTERNAL_STORAGE_AUTHORITY -> {
                val path = id.substringAfter(':', "").trim('/')
                path == OWN_DOWNLOADS || path.startsWith("$OWN_DOWNLOADS/") ||
                    path.startsWith("android/data/$pkg") || path.startsWith("android/media/$pkg")
            }
            DOWNLOADS_AUTHORITY -> id.startsWith("raw:") && id.contains("/$OWN_DOWNLOADS")
            else -> authority.lowercase(Locale.ROOT).startsWith(pkg)
        }
    }

    /**
     * Error for an operation that failed with a not-found-like or permission error, and what it
     * affects. Only says `missing` when the folder root answers (critic: FNFE is not "deleted").
     */
    private fun diagnose(ref: Ref): Pair<String, String> {
        val tree = ref.tree
        if (tree != null) {
            if (persistedGrant(tree) == null) return ERR_PERMISSION to SCOPE_TARGET
            val root = rootOf(ref) ?: return ERR_PROVIDER to SCOPE_TARGET
            val rootRows = try {
                queryRows(root)
            } catch (e: SecurityException) {
                return ERR_PERMISSION to SCOPE_TARGET
            } catch (e: Exception) {
                return ERR_PROVIDER to SCOPE_TARGET
            }
            if (rootRows.isNullOrEmpty()) return ERR_PROVIDER to SCOPE_TARGET // cannot prove anything
            if (ref.document == root) return ERR_PROVIDER to SCOPE_TARGET
            val rows = try {
                queryRows(ref.document)
            } catch (e: SecurityException) {
                // The root answers but this document is no longer inside it (moved out).
                return ERR_MISSING to SCOPE_DOCUMENT
            } catch (e: Exception) {
                return ERR_PROVIDER to SCOPE_DOCUMENT
            }
            return if (rows.isNullOrEmpty()) ERR_MISSING to SCOPE_DOCUMENT else ERR_PROVIDER to SCOPE_DOCUMENT
        }
        // A single picked file: only the document itself can be asked.
        if (persistedGrant(ref.document) == null) return ERR_PERMISSION to SCOPE_DOCUMENT
        val rows = try {
            queryRows(ref.document)
        } catch (e: SecurityException) {
            return ERR_PERMISSION to SCOPE_DOCUMENT
        } catch (e: Exception) {
            return ERR_PROVIDER to SCOPE_DOCUMENT
        }
        return if (rows.isNullOrEmpty()) ERR_MISSING to SCOPE_DOCUMENT else ERR_PROVIDER to SCOPE_DOCUMENT
    }

    private fun diagnosed(ref: Ref, message: String, vararg extra: Pair<String, Any?>): HashMap<String, Any?> {
        val (code, scope) = diagnose(ref)
        val out = fail(code, message, scope, *extra)
        if (code == ERR_MISSING) out["proven"] = ref.tree != null
        return out
    }

    // ---------------------------------------------------------------- listing

    private fun listChildren(args: Map<*, *>): Map<String, Any?> {
        val folder = parseRef(args["folder"]) ?: return fail(ERR_BAD_ARGS, "folder is required", null)
        val tree = folder.tree
            ?: return fail(ERR_BAD_ARGS, "Only folders picked with pick_folder can be listed", null)
        val names = (args["names"] as? List<*>)?.mapNotNull { it as? String }?.toSet()
        val parentId = DocumentsContract.getDocumentId(folder.document)
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId)
        val listing = try {
            queryListing(childrenUri)
        } catch (e: SecurityException) {
            return diagnosed(folder, e.toString())
        }
        val rows = listing.rows ?: return diagnosed(folder, "The cloud app did not list the folder")
        val children = ArrayList<Map<String, Any?>>()
        for (row in rows) {
            val id = row.documentId ?: continue
            if (names != null && (row.name == null || row.name !in names)) continue
            val uri = DocumentsContract.buildDocumentUriUsingTree(tree, id)
            children.add(refMap(tree, uri, row, stableId("android:$uri")))
        }
        return ok("children" to children, "complete" to !listing.loading)
    }

    /** Documents directly in [folder] whose display name is [name] (empty when listing fails). */
    private fun childrenNamed(folder: Ref, name: String): List<HashMap<String, Any?>>? {
        val tree = folder.tree ?: return null
        val parentId = DocumentsContract.getDocumentId(folder.document)
        val rows = queryRows(DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId)) ?: return null
        return rows.filter { it.name == name }.mapNotNull { row ->
            row.documentId?.let { id ->
                val uri = DocumentsContract.buildDocumentUriUsingTree(tree, id)
                refMap(tree, uri, row, stableId("android:$uri"))
            }
        }
    }

    // ---------------------------------------------------------------- creating

    private fun createFile(args: Map<*, *>, directory: Boolean): Map<String, Any?> {
        val folder = parseRef(args["folder"]) ?: return fail(ERR_BAD_ARGS, "folder is required", null)
        val rawName = args["name"] as? String
        if (rawName.isNullOrBlank()) return fail(ERR_BAD_ARGS, "name is required", null)
        val mime = if (directory) {
            DocumentsContract.Document.MIME_TYPE_DIR
        } else {
            (args["mime_type"] as? String)?.takeIf { it.isNotEmpty() }
                ?: GlossarionNativePlugin.guessMime(rawName) ?: "application/octet-stream"
        }
        val name = if (directory) GlossarionNativePlugin.safeFileName(rawName, null)
        else GlossarionNativePlugin.safeFileName(rawName, mime)
        val onExists = (args["on_exists"] as? String) ?: if (directory) ON_EXISTS_ADOPT else ON_EXISTS_RENAME
        val made = createChild(folder, name, mime, onExists)
        if (directory && made["ok"] == true) {
            // create_folder answers {"folder": ref}
            made["folder"] = made.remove("document")
        }
        return made
    }

    private fun createChild(folder: Ref, name: String, mime: String, onExists: String): HashMap<String, Any?> {
        val tree = folder.tree
            ?: return fail(ERR_BAD_ARGS, "Files can only be created inside a folder picked with pick_folder", null)
        // Same-name items before creating: adopt / fail on them, and tell a late-appearing create
        // (Drive is eventually consistent) from a file that was already there.
        val before: List<HashMap<String, Any?>>? = try {
            childrenNamed(folder, name)
        } catch (e: SecurityException) {
            return diagnosed(folder, e.toString())
        } catch (e: Exception) {
            null
        }
        val existing = before?.firstOrNull()
        if (existing != null && onExists != ON_EXISTS_RENAME) {
            if (onExists == ON_EXISTS_FAIL) {
                return fail(ERR_EXISTS, "\"$name\" already exists", SCOPE_DOCUMENT, "document" to existing)
            }
            return ok("document" to existing, "created" to false, "adopted" to true)
        }
        val beforeIds = before?.map { it["document"] }?.toSet()
        var lastError: Exception? = null
        for (attempt in 0..1) {
            try {
                val uri = DocumentsContract.createDocument(resolver, folder.document, mime, name)
                if (uri != null) {
                    val row = try {
                        queryRows(uri)?.firstOrNull()
                    } catch (e: Exception) {
                        null
                    }
                    val made = refMap(tree, uri, row ?: Row(documentIdOrNull(uri), name, mime, 0L, null, 0), stableId("android:$uri"))
                    if (row == null) made["kind"] = if (mime == DocumentsContract.Document.MIME_TYPE_DIR) KIND_FOLDER else KIND_FILE
                    return ok("document" to made, "created" to true, "adopted" to false)
                }
                lastError = IOException("The cloud app did not create \"$name\"")
            } catch (e: SecurityException) {
                return diagnosed(folder, e.toString())
            } catch (e: Exception) {
                if (isNoSpace(e)) return fail(ERR_NO_SPACE, e.toString(), SCOPE_TARGET)
                lastError = e
                if (e is UnsupportedOperationException) break // the provider's default "Create not supported"
            }
            if (attempt == 0) {
                SystemClock.sleep(CREATE_RETRY_DELAY_MS)
                if (beforeIds != null) {
                    val appeared = try {
                        childrenNamed(folder, name)?.filter { it["document"] !in beforeIds }
                    } catch (e: Exception) {
                        null
                    }
                    if (appeared != null && appeared.size == 1) {
                        return ok("document" to appeared[0], "created" to true, "adopted" to false, "late" to true)
                    }
                }
            }
        }
        val error = lastError ?: IOException("create failed")
        if (error is FileNotFoundException) return diagnosed(folder, error.toString())
        val directory = mime == DocumentsContract.Document.MIME_TYPE_DIR
        if (error is UnsupportedOperationException ||
            (directory && error is IllegalArgumentException && !mentionsMode(error))
        ) {
            // DocumentsProvider.createDocument's default UnsupportedOperationException("Create not supported"),
            // or a provider rejecting the directory MIME type: this folder takes no new items of that kind (a
            // cloud app that accepts files but no sub-folders). Not retried: the app lays books out flat.
            return fail(ERR_READ_ONLY, error.toString(), SCOPE_DOCUMENT, "create_unsupported" to true)
        }
        return fail(classify(error, notFound = ERR_PROVIDER), error.toString(), SCOPE_TARGET)
    }

    // ------------------------------------------------------------------ writing

    private fun writeFile(args: Map<*, *>): Map<String, Any?> {
        val ref = parseRef(args["ref"]) ?: return fail(ERR_BAD_ARGS, "ref is required", null)
        val opId = (args["op_id"] as? String)?.takeIf { it.isNotEmpty() } ?: UUID.randomUUID().toString()
        val modes = parseModes(args["mode_chain"])
            ?: return fail(ERR_BAD_ARGS, "mode_chain must list wt, rwt, w or rw", null)
        val source = (args["source_path"] as? String)?.takeIf { it.isNotEmpty() }?.let { File(it) }
        if (source == null || !source.isFile) {
            return fail(ERR_SOURCE_MISSING, "The file to upload is missing", SCOPE_SOURCE, "op_id" to opId)
        }
        val verify = args["verify"] != false
        cancelFlags.putIfAbsent(opId, AtomicBoolean(false))
        try {
            var target = ref
            var created = false
            var adopted = false
            if (ref.kind == KIND_FOLDER) {
                val rawName = (args["name"] as? String)?.takeIf { it.isNotBlank() } ?: source.name
                val mime = (args["mime_type"] as? String)?.takeIf { it.isNotEmpty() }
                    ?: GlossarionNativePlugin.guessMime(rawName) ?: "application/octet-stream"
                val name = GlossarionNativePlugin.safeFileName(rawName, mime)
                val made = createChild(ref, name, mime, (args["on_exists"] as? String) ?: ON_EXISTS_RENAME)
                if (made["ok"] != true) {
                    made["op_id"] = opId
                    return made
                }
                target = parseRef(made["document"]) ?: return fail(ERR_PROVIDER, "Created file has no URI", SCOPE_DOCUMENT)
                created = made["created"] == true
                adopted = made["adopted"] == true
            }
            val outcome = writeChain(target, source, modes, opId, verify)
            outcome["created"] = created
            outcome["adopted"] = adopted
            outcome["op_id"] = opId
            if (outcome["document"] == null) {
                outcome["document"] = refMap(target.tree, target.document, null, idFor(target)).also {
                    it["kind"] = KIND_FILE
                }
            }
            return outcome
        } finally {
            cancelFlags.remove(opId)
        }
    }

    /** Try the write modes in order; see the class comment for the safety rules. */
    private fun writeChain(ref: Ref, source: File, modes: List<String>, opId: String, verify: Boolean): HashMap<String, Any?> {
        val total = source.length()
        val sourceModified = source.lastModified()
        val attempts = ArrayList<Map<String, Any?>>()
        var remoteBefore: Long? = null
        var remoteBeforeKnown = false
        var permissionDenied = false
        for (mode in modes) {
            if (isCancelled(opId)) {
                return fail(ERR_CANCELLED, "Cancelled", SCOPE_DOCUMENT, "attempts" to attempts, "written" to 0L)
            }
            if (mode in NON_TRUNCATING_MODES) {
                if (!remoteBeforeKnown) {
                    remoteBefore = remoteSize(ref.document)
                    remoteBeforeKnown = true
                }
                val old = remoteBefore
                if (old == null || total < old) {
                    attempts.add(attempt(mode, ATTEMPT_SKIPPED,
                        if (old == null) "cloud copy size unknown" else "new file is shorter than the cloud copy"))
                    continue
                }
            }
            val pfd: ParcelFileDescriptor = try {
                resolver.openFileDescriptor(ref.document, mode)
                    ?: throw FileNotFoundException("No descriptor for mode $mode")
            } catch (e: Exception) {
                val code = classify(e, notFound = ATTEMPT_NOT_FOUND)
                attempts.add(attempt(mode, code, e.toString()))
                if (code == ERR_NO_SPACE) {
                    return fail(ERR_NO_SPACE, e.toString(), SCOPE_DOCUMENT, "attempts" to attempts, "written" to 0L)
                }
                if (code == ERR_PERMISSION) {
                    permissionDenied = true
                    break
                }
                continue
            }
            return streamInto(pfd, mode, ref, source, total, sourceModified, opId, verify, attempts, remoteBefore)
        }
        // No mode could be opened.
        val onlyModeProblems = !permissionDenied && attempts.isNotEmpty() &&
            attempts.all { it["error"] == ERR_UNSUPPORTED_MODE || it["error"] == ATTEMPT_SKIPPED }
        if (onlyModeProblems) {
            val row = try {
                queryRows(ref.document)?.firstOrNull()
            } catch (e: Exception) {
                null
            }
            val writable = row == null || (row.flags and DocumentsContract.Document.FLAG_SUPPORTS_WRITE) != 0
            return fail(
                if (writable) ERR_UNSUPPORTED_MODE else ERR_READ_ONLY,
                "The cloud app accepted none of the write modes ${modes.joinToString(", ")}",
                SCOPE_DOCUMENT,
                "attempts" to attempts,
                "needs_replace" to writable,
                "remote_size" to remoteBefore,
                "written" to 0L,
            )
        }
        return diagnosed(ref, describe(attempts), "attempts" to attempts, "written" to 0L)
    }

    private fun streamInto(
        pfd: ParcelFileDescriptor,
        mode: String,
        ref: Ref,
        source: File,
        total: Long,
        sourceModified: Long,
        opId: String,
        verify: Boolean,
        attempts: ArrayList<Map<String, Any?>>,
        remoteBefore: Long?,
    ): HashMap<String, Any?> {
        val truncating = mode in TRUNCATING_MODES
        var written = 0L
        try {
            // Not closed on its own: closing pfd closes the descriptor and tells the provider.
            val out = FileOutputStream(pfd.fileDescriptor)
            FileInputStream(source).use { input ->
                val buffer = ByteArray(CHUNK_BYTES)
                var lastProgress = 0L
                while (true) {
                    if (isCancelled(opId)) throw CancelledOp()
                    val n = input.read(buffer)
                    if (n < 0) break
                    out.write(buffer, 0, n)
                    written += n
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastProgress >= PROGRESS_INTERVAL_MS) {
                        lastProgress = now
                        progress(opId, written, total)
                    }
                }
            }
            out.flush()
            if (!truncating && pfd.statSize >= 0) {
                // A non-truncating mode on a real file: cut what is left of a longer older copy.
                Os.ftruncate(pfd.fileDescriptor, written)
            }
            try {
                pfd.fileDescriptor.sync()
            } catch (e: Exception) {
                // Pipes and sockets cannot sync.
            }
            pfd.close()
        } catch (e: Exception) {
            val code = if (e is CancelledOp) ERR_CANCELLED else classify(e, notFound = ERR_PROVIDER)
            try {
                // Providers that listen for it (OnCloseListener) drop the partial file.
                pfd.closeWithError("Glossarion: upload stopped ($code)")
            } catch (closeError: Exception) {
                // Already closed.
            }
            attempts.add(attempt(mode, code, e.toString()))
            val extra = arrayOf<Pair<String, Any?>>(
                "attempts" to attempts,
                "mode" to mode,
                "written" to written,
                "remote_damaged" to (truncating || written > 0),
            )
            if (code == ERR_PERMISSION) return diagnosed(ref, e.toString(), *extra)
            return fail(code, e.toString(), SCOPE_DOCUMENT, *extra)
        }
        progress(opId, written, total)
        if (written != total || source.length() != total || source.lastModified() != sourceModified) {
            attempts.add(attempt(mode, ERR_SOURCE_CHANGED, null))
            return fail(
                ERR_SOURCE_CHANGED, "The local file changed while it was copied", SCOPE_SOURCE,
                "attempts" to attempts, "mode" to mode, "written" to written, "remote_damaged" to true,
            )
        }
        attempts.add(attempt(mode, ATTEMPT_OK, null))
        val verifiedSize = if (verify) readBackSize(ref.document) else null
        val row = try {
            queryRows(ref.document)?.firstOrNull()
        } catch (e: Exception) {
            null
        }
        val document = refMap(ref.tree, ref.document, row, idFor(ref))
        if (row == null) document["kind"] = KIND_FILE
        if (verifiedSize != null && verifiedSize != written) {
            val staleTail = verifiedSize > written
            return fail(
                ERR_SIZE_MISMATCH,
                if (staleTail) "The cloud copy kept old bytes after the new end" else "The cloud copy is shorter than the file",
                SCOPE_DOCUMENT,
                "attempts" to attempts,
                "mode" to mode,
                "written" to written,
                "verified_size" to verifiedSize,
                "stale_tail" to staleTail,
                "needs_replace" to staleTail,
                "remote_damaged" to true,
                "document" to document,
            )
        }
        return ok(
            "document" to document,
            "mode" to mode,
            "written" to written,
            "total" to total,
            "verified_size" to verifiedSize,
            "verified" to (verifiedSize != null),
            "reported_size" to row?.size,
            "remote_size_before" to remoteBefore,
            "attempts" to attempts,
        )
    }

    /** Length of the cloud copy as a real file ('r' + fstat); null when the provider streams it. */
    private fun readBackSize(uri: Uri): Long? = try {
        resolver.openFileDescriptor(uri, "r")?.use { fd -> fd.statSize.takeIf { it >= 0 } }
    } catch (e: Exception) {
        null
    }

    private fun remoteSize(uri: Uri): Long? = readBackSize(uri) ?: try {
        queryRows(uri)?.firstOrNull()?.size
    } catch (e: Exception) {
        null
    }

    private fun isCancelled(opId: String): Boolean = cancelFlags[opId]?.get() == true

    private fun progress(opId: String, written: Long, total: Long) {
        val event = hashMapOf<String, Any?>(
            "type" to EVENT_PROGRESS,
            "op_id" to opId,
            "written" to written,
            "total" to total,
        )
        mainHandler.post { if (isDartAttached()) sendToDart(event) }
    }

    // ------------------------------------------------------- stat, delete, root

    private fun stat(args: Map<*, *>): Map<String, Any?> {
        val ref = parseRef(args["document"]) ?: return fail(ERR_BAD_ARGS, "document is required", null)
        val rows = try {
            queryRows(ref.document)
        } catch (e: SecurityException) {
            return diagnosed(ref, e.toString())
        }
        val row = rows?.firstOrNull() ?: return diagnosed(ref, "The cloud app did not answer for this document")
        return ok("document" to refMap(ref.tree, ref.document, row, idFor(ref)))
    }

    private fun delete(args: Map<*, *>): Map<String, Any?> {
        val ref = parseRef(args["document"]) ?: return fail(ERR_BAD_ARGS, "document is required", null)
        val deleted = try {
            DocumentsContract.deleteDocument(resolver, ref.document)
        } catch (e: SecurityException) {
            return diagnosed(ref, e.toString())
        } catch (e: FileNotFoundException) {
            return diagnosed(ref, e.toString())
        }
        return if (deleted) ok("deleted" to true) else fail(ERR_PROVIDER, "The cloud app did not delete it", SCOPE_DOCUMENT)
    }

    /**
     * Rename a document inside its folder (the cloud sync gives a replaced copy its first name back). The
     * provider may answer with a new URI; providers without FLAG_SUPPORTS_RENAME answer `unavailable`.
     */
    private fun renameDocument(args: Map<*, *>): Map<String, Any?> {
        val ref = parseRef(args["document"]) ?: return fail(ERR_BAD_ARGS, "document is required", null)
        val name = (args["name"] as? String)?.takeIf { it.isNotBlank() }
            ?: return fail(ERR_BAD_ARGS, "name is required", null)
        if (isRoot(ref)) return fail(ERR_BAD_ARGS, "The picked folder itself is not renamed", null)
        val renamed: Uri? = try {
            DocumentsContract.renameDocument(resolver, ref.document, name)
        } catch (e: SecurityException) {
            return diagnosed(ref, e.toString())
        } catch (e: FileNotFoundException) {
            return diagnosed(ref, e.toString())
        } catch (e: UnsupportedOperationException) {
            return fail(ERR_UNAVAILABLE, "The cloud app cannot rename files", SCOPE_DOCUMENT)
        } catch (e: IllegalStateException) {
            return fail(ERR_EXISTS, e.toString(), SCOPE_DOCUMENT)
        }
        val document = renamed ?: return fail(ERR_PROVIDER, "The cloud app did not rename it", SCOPE_DOCUMENT)
        val row = try {
            queryRows(document)?.firstOrNull()
        } catch (e: Exception) {
            null
        }
        return ok("document" to refMap(ref.tree, document, row, stableId("android:$document")), "renamed" to true)
    }

    private fun queryRoot(args: Map<*, *>): Map<String, Any?> {
        val ref = parseRef(args["target"]) ?: return fail(ERR_BAD_ARGS, "target is required", null)
        val authority = ref.document.authority
        if (!authority.isNullOrEmpty() && providerInfo(authority) == null) {
            return fail(ERR_PERMISSION, "The app that holds this location is no longer installed", SCOPE_TARGET,
                "provider_missing" to true)
        }
        val grantUri = if (ref.tree != null) ref.tree else ref.document
        val grant = persistedGrant(grantUri)
            ?: return fail(ERR_PERMISSION, "Glossarion no longer has access to this location", SCOPE_TARGET)
        val root = rootOf(ref) ?: ref.document
        val rows = try {
            queryRows(root)
        } catch (e: SecurityException) {
            return fail(ERR_PERMISSION, e.toString(), SCOPE_TARGET)
        } catch (e: Exception) {
            return fail(ERR_PROVIDER, e.toString(), SCOPE_TARGET)
        }
        val row = rows?.firstOrNull()
            ?: return fail(
                if (ref.tree == null && rows != null) ERR_MISSING else ERR_PROVIDER,
                "The cloud app did not answer", SCOPE_TARGET,
            )
        val folder = ref.tree != null
        val id = if (folder) stableId("android:${ref.tree}") else stableId("android:${ref.document}")
        val target = refMap(ref.tree, root, row, id)
        target["kind"] = if (folder) KIND_FOLDER else KIND_FILE
        decorate(target, authority, documentIdOrNull(root), true, grant.isWritePermission, row, folder)
        return ok("target" to target)
    }

    private fun release(args: Map<*, *>): Boolean {
        val raw = args["target"]
        val uri: Uri = when (raw) {
            is String -> if (raw.isEmpty()) return false else Uri.parse(raw)
            is Map<*, *> -> {
                val ref = parseRef(raw) ?: return false
                // The folder's own grant for a picked folder, the document's for a single file.
                if (ref.tree != null && (isRoot(ref) || ref.kind == KIND_FOLDER && raw["document"] == null)) ref.tree
                else ref.document
            }
            else -> return false
        }
        val grant = persistedGrant(uri) ?: return false
        var flags = 0
        if (grant.isReadPermission) flags = flags or Intent.FLAG_GRANT_READ_URI_PERMISSION
        if (grant.isWritePermission) flags = flags or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
        return try {
            resolver.releasePersistableUriPermission(uri, flags)
            true
        } catch (e: SecurityException) {
            false
        }
    }

    private fun listGrants(): List<Map<String, Any?>> = resolver.persistedUriPermissions.map {
        hashMapOf(
            "uri" to it.uri.toString(),
            "read" to it.isReadPermission,
            "write" to it.isWritePermission,
            "persisted_time" to it.persistedTime,
            "tree" to DocumentsContract.isTreeUri(it.uri),
        )
    }

    // ------------------------------------------------------------------ helpers

    private fun attempt(mode: String, error: String, message: String?): Map<String, Any?> =
        hashMapOf("mode" to mode, "error" to error, "message" to message)

    private fun describe(attempts: List<Map<String, Any?>>): String =
        attempts.joinToString("; ") { "${it["mode"]}: ${it["error"]}" }.ifEmpty { "write failed" }

    companion object {
        private const val TAG = "GlossarionNativeDocs"
        private const val PREFS_NAME = "glossarion_native_documents"

        val METHODS = setOf(
            "pick_folder", "pick_save_location", "pick_document", "list_children", "create_file",
            "create_folder", "write_file", "rename_document", "stat", "delete", "query_root", "release",
            "list_grants", "cancel_document_op",
        )

        const val KIND_FOLDER = "folder"
        const val KIND_FILE = "file"
        private const val KIND_SAVE = "save_location"
        private const val KIND_OPEN = "document"

        // Request codes (16 bit; distinct from file_picker's).
        private const val REQUEST_FOLDER = 0x4C31
        private const val REQUEST_SAVE = 0x4C32
        private const val REQUEST_OPEN = 0x4C33
        private val REQUEST_CODES = setOf(REQUEST_FOLDER, REQUEST_SAVE, REQUEST_OPEN)

        private fun requestCodeFor(kind: String): Int = when (kind) {
            KIND_FOLDER -> REQUEST_FOLDER
            KIND_SAVE -> REQUEST_SAVE
            else -> REQUEST_OPEN
        }

        // Error codes (flet_glossarion_native/documents.py DocumentError).
        const val ERR_CANCELLED = "cancelled"
        const val ERR_PERMISSION = "permission_lost"
        const val ERR_MISSING = "missing"
        const val ERR_UNSUPPORTED_MODE = "unsupported_mode"
        const val ERR_PROVIDER = "provider_error"
        const val ERR_NO_SPACE = "no_space"
        const val ERR_SOURCE_MISSING = "source_missing"
        const val ERR_SOURCE_CHANGED = "source_changed"
        const val ERR_READ_ONLY = "read_only"
        const val ERR_SIZE_MISMATCH = "size_mismatch"
        const val ERR_EXISTS = "exists"
        const val ERR_BUSY = "busy"
        const val ERR_UNAVAILABLE = "unavailable"
        const val ERR_BAD_ARGS = "bad_args"
        private val RETRYABLE = setOf(ERR_PROVIDER, ERR_SOURCE_CHANGED, ERR_BUSY)

        // Per-mode attempt outcomes that are not error codes.
        private const val ATTEMPT_OK = "ok"
        private const val ATTEMPT_SKIPPED = "skipped"
        private const val ATTEMPT_NOT_FOUND = "not_found"

        const val SCOPE_TARGET = "target"
        const val SCOPE_DOCUMENT = "document"
        const val SCOPE_SOURCE = "source"

        const val EVENT_PROGRESS = "progress"
        const val EVENT_PICK_RESULT = "pick_result"

        private const val ON_EXISTS_RENAME = "rename"
        private const val ON_EXISTS_ADOPT = "adopt"
        private const val ON_EXISTS_FAIL = "fail"

        val VALID_MODES = listOf("wt", "rwt", "w", "rw")
        val DEFAULT_MODES = listOf("wt", "rwt", "w")
        private val TRUNCATING_MODES = setOf("wt", "rwt")
        private val NON_TRUNCATING_MODES = setOf("w", "rw")

        private const val CHUNK_BYTES = 1 shl 20
        private const val PROGRESS_INTERVAL_MS = 250L
        private const val CREATE_RETRY_DELAY_MS = 1500L

        private const val EXTERNAL_STORAGE_AUTHORITY = "com.android.externalstorage.documents"
        private const val DOWNLOADS_AUTHORITY = "com.android.providers.downloads.documents"
        /** Download/Glossarion: the phone-folder destination (save_to_downloads), lower case. */
        private const val OWN_DOWNLOADS = "download/glossarion"

        private const val KEY_OP = "op_id"
        private const val KEY_KIND = "kind"
        private const val KEY_REQUEST = "request_code"
        private const val KEY_PID = "pid"
        private const val KEY_STARTED = "started"
        private const val KEY_NAME = "name"
        private const val KEY_MIME = "mime"
        private const val KEY_SOURCE = "source_path"
        private const val KEY_MODES = "mode_chain"

        private val COLUMNS = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
            DocumentsContract.Document.COLUMN_SIZE,
            DocumentsContract.Document.COLUMN_LAST_MODIFIED,
            DocumentsContract.Document.COLUMN_FLAGS,
        )

        private const val FNV_OFFSET: ULong = 0xcbf29ce484222325uL
        private const val FNV_PRIME: ULong = 0x100000001b3uL

        /** FNV-1a/64 over UTF-8, "d" + 16 hex digits (documents.stable_id in Python). */
        fun stableId(identity: String): String {
            var hash = FNV_OFFSET
            for (byte in identity.toByteArray(Charsets.UTF_8)) {
                hash = hash xor byte.toUByte().toULong()
                hash *= FNV_PRIME
            }
            return "d" + hash.toString(16).padStart(16, '0')
        }

        fun parseModes(raw: Any?): List<String>? {
            if (raw == null) return DEFAULT_MODES
            val list = raw as? List<*> ?: return null
            val modes = list.mapNotNull { (it as? String)?.trim()?.lowercase(Locale.ROOT) }
                .filter { it in VALID_MODES }
                .distinct()
            return modes.ifEmpty { null }
        }

        fun ok(vararg pairs: Pair<String, Any?>): HashMap<String, Any?> {
            val out = hashMapOf<String, Any?>("ok" to true, "error" to null, "message" to null, "retryable" to false)
            for ((key, value) in pairs) out[key] = value
            return out
        }

        fun fail(code: String, message: String?, scope: String?, vararg extra: Pair<String, Any?>): HashMap<String, Any?> {
            val out = hashMapOf<String, Any?>(
                "ok" to false,
                "error" to code,
                "message" to message,
                "scope" to scope,
                "retryable" to (code in RETRYABLE),
            )
            for ((key, value) in extra) out[key] = value
            return out
        }

        fun isNoSpace(error: Throwable?): Boolean {
            var current = error
            var depth = 0
            while (current != null && depth < 8) {
                if (current is ErrnoException &&
                    (current.errno == OsConstants.ENOSPC || current.errno == OsConstants.EDQUOT)
                ) {
                    return true
                }
                val message = current.message ?: ""
                if (message.contains("ENOSPC") || message.contains("No space left", ignoreCase = true)) return true
                current = current.cause
                depth += 1
            }
            return false
        }

        private fun mentionsMode(error: Throwable): Boolean =
            (error.message ?: "").lowercase(Locale.ROOT).contains("mode")

        /**
         * Error code of an exception. [notFound] is what a plain FileNotFoundException maps to: the
         * callers decide (see diagnose) because it does not prove that a document was deleted.
         */
        fun classify(error: Throwable, notFound: String): String = when {
            error is CancelledOp -> ERR_CANCELLED
            isNoSpace(error) -> ERR_NO_SPACE
            error is SecurityException -> ERR_PERMISSION
            (error is FileNotFoundException || error is IllegalArgumentException ||
                error is UnsupportedOperationException) && mentionsMode(error) -> ERR_UNSUPPORTED_MODE
            error is FileNotFoundException -> notFound
            else -> ERR_PROVIDER
        }
    }
}
