import Flutter
import UIKit

/// Document destinations (U10) on iOS: a folder from the Files picker (iCloud Drive, On My
/// iPhone) or one exported file (Google Drive, OneDrive, Dropbox: their File Provider extensions
/// cannot be picked as folders), both kept as minimal bookmarks and written later without asking.
///
/// Methods (routed by GlossarionNativePlugin): pick_folder, pick_save_location, pick_document,
/// list_children, create_file, create_folder, write_file, stat, delete, query_root, release,
/// list_grants, cancel_document_op. Answers are maps {ok, error, message, scope, retryable, ...}
/// (flet_glossarion_native/documents.py); typed failures never use FlutterError.
///
/// - Folder picks use the iOS 13 initializers (`documentTypes:in:` with "public.folder"), so no
///   UniformTypeIdentifiers import (an iOS 14 framework) is linked into an iOS 13 target.
/// - Writes copy the snapshot into a replacement directory on the destination's volume (progress,
///   cancel between 1 MiB chunks), then swap it in under NSFileCoordinator (.forReplacing +
///   replaceItemAt / moveItem): the cloud file never holds a mix of old and new bytes and the old
///   version is not downloaded first. When no replacement directory is available the file is
///   written in place (mode "in_place", not atomic).
/// - Items inside a picked folder carry the folder's bookmark ("root") and their relative path; a
///   child bookmark that now resolves outside the folder or into a trash (.Trash, iCloud
///   "Recently Deleted") is treated as missing.
/// - A picker left open when the process died is reported as cancelled on the next attach.
/// The app wraps background writes in begin_background_task; this class does not manage time.
final class DocumentDestinations: NSObject, UIDocumentPickerDelegate {
  static let methods: Set<String> = [
    "pick_folder", "pick_save_location", "pick_document", "list_children", "create_file",
    "create_folder", "write_file", "rename_document", "stat", "delete", "query_root", "release",
    "list_grants", "cancel_document_op",
  ]

  private static let pendingKey = "com.glossarion.native.documents.pending"
  private static let launchId = UUID().uuidString
  private static let chunkBytes = 1 << 20
  private static let progressInterval: TimeInterval = 0.25

  private static let errCancelled = "cancelled"
  private static let errPermission = "permission_lost"
  private static let errMissing = "missing"
  private static let errProvider = "provider_error"
  private static let errNoSpace = "no_space"
  private static let errSourceMissing = "source_missing"
  private static let errSourceChanged = "source_changed"
  private static let errReadOnly = "read_only"
  private static let errSizeMismatch = "size_mismatch"
  private static let errExists = "exists"
  private static let errBusy = "busy"
  private static let errUnavailable = "unavailable"
  private static let errBadArgs = "bad_args"
  private static let retryable: Set<String> = [errProvider, errSourceChanged, errBusy]

  private static let scopeTarget = "target"
  private static let scopeDocument = "document"
  private static let scopeSource = "source"

  private struct PendingPick {
    let opId: String
    let kind: String
    let result: FlutterResult
    let picker: UIDocumentPickerViewController
    let exportDirectory: URL?
    let wroteSource: Bool
  }

  private struct DocFailure: Error {
    let code: String
    let message: String
    let scope: String?
    let extra: [String: Any]

    init(_ code: String, _ message: String, scope: String? = nil, extra: [String: Any] = [:]) {
      self.code = code
      self.message = message
      self.scope = scope
      self.extra = extra
    }

    var payload: [String: Any] {
      var out = DocumentDestinations.fail(code, message, scope: scope)
      for (key, value) in extra {
        out[key] = value
      }
      return out
    }
  }

  /// A ref resolved to a URL, with the security scope that was started for it.
  private struct Resolved {
    let url: URL
    let accessURL: URL
    let accessing: Bool
    let rootURL: URL?
    let rootBookmark: String?
    let basePath: String
    var refreshed: [String: String]

    func stop() {
      if accessing {
        accessURL.stopAccessingSecurityScopedResource()
      }
    }
  }

  private let send: ([String: Any]) -> Void
  private let isDartAttached: () -> Bool
  private let queue = DispatchQueue(label: "com.glossarion.native.documents", qos: .utility)
  private let cancelLock = NSLock()
  private var cancelledOps = Set<String>()
  private var pending: PendingPick?

  init(send: @escaping ([String: Any]) -> Void, isDartAttached: @escaping () -> Bool) {
    self.send = send
    self.isDartAttached = isDartAttached
    super.init()
  }

  // MARK: - Dispatch (main thread)

  func handle(_ method: String, _ args: [String: Any], result: @escaping FlutterResult) {
    switch method {
    case "pick_folder":
      dropStalePick()
      let picker = UIDocumentPickerViewController(documentTypes: ["public.folder"], in: .open)
      presentPicker(picker, kind: "folder", args: args, exportDirectory: nil, wroteSource: false, result: result)
    case "pick_save_location":
      dropStalePick()
      pickSaveLocation(args, result: result)
    case "pick_document":
      dropStalePick()
      let types = DocumentDestinations.typeIdentifiers(args["mime_types"])
      let picker = UIDocumentPickerViewController(documentTypes: types, in: .open)
      presentPicker(picker, kind: "document", args: args, exportDirectory: nil, wroteSource: false, result: result)
    case "cancel_document_op":
      guard let opId = args["op_id"] as? String, !opId.isEmpty else {
        result(false)
        return
      }
      cancelLock.lock()
      cancelledOps.insert(opId)
      cancelLock.unlock()
      result(true)
    case "release":
      // Bookmarks are only data in the app's own storage: forgetting them releases them.
      result(true)
    case "list_grants":
      result([Any]())
    default:
      queue.async {
        let payload = self.runIO(method, args)
        DispatchQueue.main.async {
          result(payload)
        }
      }
    }
  }

  private func runIO(_ method: String, _ args: [String: Any]) -> [String: Any] {
    do {
      switch method {
      case "list_children":
        return try listChildren(args)
      case "create_file":
        return try createChild(args, directory: false)
      case "create_folder":
        return try createChild(args, directory: true)
      case "write_file":
        return try writeFile(args)
      case "rename_document":
        return try renameItem(args)
      case "stat":
        return try statItem(args)
      case "delete":
        return try deleteItem(args)
      case "query_root":
        return try queryRoot(args)
      default:
        return DocumentDestinations.fail(DocumentDestinations.errBadArgs, "Unknown document method \(method)")
      }
    } catch let failure as DocFailure {
      return failure.payload
    } catch {
      return DocumentDestinations.fail(DocumentDestinations.classify(error), error.localizedDescription)
    }
  }

  // MARK: - Pickers

  private func pickSaveLocation(_ args: [String: Any], result: @escaping FlutterResult) {
    guard pending == nil else {
      result(DocumentDestinations.fail(DocumentDestinations.errBusy, "Another picker is already open"))
      return
    }
    let name = GlossarionNativePlugin.safeFileName((args["name"] as? String) ?? "Glossarion output")
    let sourcePath = args["source_path"] as? String
    queue.async {
      // The export picker moves this staged copy to the chosen place and keeps access to it.
      let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("documents-export", isDirectory: true)
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
      let staged = directory.appendingPathComponent(name)
      var wroteSource = false
      do {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
        if let sourcePath = sourcePath, FileManager.default.fileExists(atPath: sourcePath) {
          try FileManager.default.copyItem(at: URL(fileURLWithPath: sourcePath), to: staged)
          wroteSource = true
        } else if !FileManager.default.createFile(atPath: staged.path, contents: Data(), attributes: nil) {
          throw CocoaError(.fileWriteUnknown)
        }
      } catch {
        try? FileManager.default.removeItem(at: directory)
        let failure = DocumentDestinations.fail(DocumentDestinations.classify(error), error.localizedDescription)
        DispatchQueue.main.async {
          result(failure)
        }
        return
      }
      let wrote = wroteSource
      DispatchQueue.main.async {
        let picker: UIDocumentPickerViewController
        if #available(iOS 14.0, *) {
          picker = UIDocumentPickerViewController(forExporting: [staged], asCopy: false)
        } else {
          picker = UIDocumentPickerViewController(urls: [staged], in: .moveToService)
        }
        self.presentPicker(
          picker, kind: "save_location", args: args, exportDirectory: directory, wroteSource: wrote,
          result: result)
      }
    }
  }

  /// A picker that was dismissed without any delegate call would keep later picks "busy".
  private func dropStalePick() {
    guard let stale = pending, stale.picker.presentingViewController == nil else { return }
    pending = nil
    UserDefaults.standard.removeObject(forKey: DocumentDestinations.pendingKey)
    finishPick(stale, DocumentDestinations.fail(DocumentDestinations.errCancelled, "No location was chosen"))
  }

  private func presentPicker(
    _ picker: UIDocumentPickerViewController,
    kind: String,
    args: [String: Any],
    exportDirectory: URL?,
    wroteSource: Bool,
    result: @escaping FlutterResult
  ) {
    guard pending == nil else {
      removeLater(exportDirectory)
      result(DocumentDestinations.fail(DocumentDestinations.errBusy, "Another picker is already open"))
      return
    }
    guard let presenter = DocumentDestinations.topViewController() else {
      removeLater(exportDirectory)
      result(DocumentDestinations.fail(DocumentDestinations.errUnavailable, "No window to show the picker in"))
      return
    }
    var opId = (args["op_id"] as? String) ?? ""
    if opId.isEmpty {
      opId = UUID().uuidString
    }
    picker.delegate = self
    picker.allowsMultipleSelection = false
    if #available(iOS 13.0, *) {
      picker.shouldShowFileExtensions = true
    }
    pending = PendingPick(
      opId: opId, kind: kind, result: result, picker: picker, exportDirectory: exportDirectory,
      wroteSource: wroteSource)
    UserDefaults.standard.set(
      ["op_id": opId, "kind": kind, "launch": DocumentDestinations.launchId],
      forKey: DocumentDestinations.pendingKey)
    // Swiping the sheet down also ends in documentPickerWasCancelled.
    presenter.present(picker, animated: true, completion: nil)
  }

  func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
    guard let pick = pending, pick.picker === controller else { return }
    pending = nil
    UserDefaults.standard.removeObject(forKey: DocumentDestinations.pendingKey)
    guard let url = urls.first else {
      finishPick(pick, DocumentDestinations.fail(DocumentDestinations.errCancelled, "Nothing was chosen"))
      return
    }
    queue.async {
      let payload = self.completePick(pick, url: url)
      DispatchQueue.main.async {
        self.finishPick(pick, payload)
      }
    }
  }

  func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
    guard let pick = pending, pick.picker === controller else { return }
    pending = nil
    UserDefaults.standard.removeObject(forKey: DocumentDestinations.pendingKey)
    finishPick(pick, DocumentDestinations.fail(DocumentDestinations.errCancelled, "No location was chosen"))
  }

  private func finishPick(_ pick: PendingPick, _ payload: [String: Any]) {
    removeLater(pick.exportDirectory)
    pick.result(payload)
  }

  private func removeLater(_ directory: URL?) {
    guard let directory = directory else { return }
    queue.async {
      try? FileManager.default.removeItem(at: directory)
    }
  }

  private func completePick(_ pick: PendingPick, url: URL) -> [String: Any] {
    let accessing = url.startAccessingSecurityScopedResource()
    defer {
      if accessing {
        url.stopAccessingSecurityScopedResource()
      }
    }
    let bookmark: Data
    do {
      bookmark = try url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
    } catch {
      return DocumentDestinations.fail(
        DocumentDestinations.errPermission, "Glossarion could not keep access: \(error.localizedDescription)")
    }
    let folder = pick.kind == "folder"
    var ref = DocumentDestinations.describe(
      url, kind: folder ? "folder" : "file", bookmark: bookmark.base64EncodedString(), root: nil, path: nil)
    ref["persisted"] = true
    if folder {
      return DocumentDestinations.ok(["target": ref, "persisted": true])
    }
    var out = DocumentDestinations.ok(["document": ref, "persisted": true, "write": NSNull()])
    if pick.kind == "save_location" && pick.wroteSource {
      let size = (ref["size"] as? Int64) ?? 0
      out["write"] = DocumentDestinations.ok([
        "document": ref, "mode": "export", "written": size, "total": size, "verified_size": size,
        "verified": true, "created": true,
      ])
    }
    return out
  }

  /// Dart attached: a picker that was open when the previous process died never answers.
  func onDartAttach() -> [[String: Any]] {
    guard let record = UserDefaults.standard.dictionary(forKey: DocumentDestinations.pendingKey) else {
      return []
    }
    if pending != nil || (record["launch"] as? String) == DocumentDestinations.launchId {
      return []
    }
    UserDefaults.standard.removeObject(forKey: DocumentDestinations.pendingKey)
    let event: [String: Any] = [
      "type": "pick_result",
      "op_id": (record["op_id"] as? String) ?? "",
      "kind": (record["kind"] as? String) ?? "",
      "status": "cancelled",
      "result": DocumentDestinations.fail(
        DocumentDestinations.errCancelled, "Glossarion was restarted while the picker was open"),
    ]
    return [event]
  }

  // MARK: - Resolving refs (IO queue)

  private func resolveBookmark(_ base64: String) throws -> (URL, Bool) {
    guard let data = Data(base64Encoded: base64) else {
      throw DocFailure(DocumentDestinations.errBadArgs, "Invalid bookmark")
    }
    var stale = false
    let url = try URL(resolvingBookmarkData: data, options: [.withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale)
    return (url, stale)
  }

  private func resolve(_ raw: Any?) throws -> Resolved {
    guard let ref = raw as? [String: Any] else {
      throw DocFailure(DocumentDestinations.errBadArgs, "A document reference is required")
    }
    let kind = (ref["kind"] as? String) ?? "file"
    let own = (ref["bookmark"] as? String) ?? ""
    let path = (ref["path"] as? String) ?? ""
    if let rootB64 = ref["root"] as? String, !rootB64.isEmpty {
      // An item inside a picked folder.
      let found: (URL, Bool)
      do {
        found = try resolveBookmark(rootB64)
      } catch let failure as DocFailure {
        throw failure
      } catch {
        throw DocFailure(
          DocumentDestinations.errPermission, "The picked folder cannot be reached: \(error.localizedDescription)",
          scope: DocumentDestinations.scopeTarget)
      }
      let rootURL = found.0
      let accessing = rootURL.startAccessingSecurityScopedResource()
      var refreshed: [String: String] = [:]
      if found.1, let fresh = try? rootURL.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil) {
        refreshed["root"] = fresh.base64EncodedString()
      }
      var itemURL: URL?
      if !own.isEmpty, let child = try? resolveBookmark(own) {
        if DocumentDestinations.isInside(child.0, root: rootURL) && !DocumentDestinations.isTrashed(child.0) {
          itemURL = child.0
          if child.1, let fresh = try? child.0.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil) {
            refreshed["bookmark"] = fresh.base64EncodedString()
          }
        }
      }
      if itemURL == nil && !path.isEmpty {
        itemURL = DocumentDestinations.child(of: rootURL, path: path, directory: kind == "folder")
      }
      guard let url = itemURL else {
        if accessing {
          rootURL.stopAccessingSecurityScopedResource()
        }
        throw DocFailure(
          DocumentDestinations.errMissing, "The item is no longer in the picked folder",
          scope: DocumentDestinations.scopeDocument, extra: ["proven": true])
      }
      return Resolved(
        url: url, accessURL: rootURL, accessing: accessing, rootURL: rootURL,
        rootBookmark: refreshed["root"] ?? rootB64, basePath: path, refreshed: refreshed)
    }
    // A picked folder itself, or a single picked / exported file.
    guard !own.isEmpty else {
      throw DocFailure(DocumentDestinations.errBadArgs, "The reference has no bookmark")
    }
    let found: (URL, Bool)
    do {
      found = try resolveBookmark(own)
    } catch let failure as DocFailure {
      throw failure
    } catch {
      throw DocFailure(
        DocumentDestinations.errPermission, "Glossarion lost access: \(error.localizedDescription)",
        scope: kind == "folder" ? DocumentDestinations.scopeTarget : DocumentDestinations.scopeDocument)
    }
    let url = found.0
    let accessing = url.startAccessingSecurityScopedResource()
    var refreshed: [String: String] = [:]
    if found.1, let fresh = try? url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil) {
      refreshed["bookmark"] = fresh.base64EncodedString()
    }
    if DocumentDestinations.isTrashed(url) {
      if accessing {
        url.stopAccessingSecurityScopedResource()
      }
      throw DocFailure(
        DocumentDestinations.errMissing, "The item was moved to the trash",
        scope: kind == "folder" ? DocumentDestinations.scopeTarget : DocumentDestinations.scopeDocument,
        extra: ["proven": true])
    }
    let isFolder = kind == "folder"
    return Resolved(
      url: url, accessURL: url, accessing: accessing, rootURL: isFolder ? url : nil,
      rootBookmark: isFolder ? (refreshed["bookmark"] ?? own) : nil, basePath: "", refreshed: refreshed)
  }

  /// Ref of an item inside the folder [folder] was resolved from.
  private func childRef(_ url: URL, name: String, in folder: Resolved, directory: Bool) -> [String: Any] {
    let relative = folder.basePath.isEmpty ? name : folder.basePath + "/" + name
    var bookmark: Any = NSNull()
    if let data = try? url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil) {
      bookmark = data.base64EncodedString()
    }
    var ref = DocumentDestinations.describe(
      url, kind: directory ? "folder" : "file", bookmark: bookmark,
      root: DocumentDestinations.orNull(folder.rootBookmark), path: relative)
    if ref["name"] == nil || ref["name"] is NSNull {
      ref["name"] = name
    }
    return ref
  }

  /// Fresh ref for an item that was resolved from [raw] (keeps root / path, adds new bookmarks).
  private func updatedRef(_ raw: Any?, resolved: Resolved) -> [String: Any] {
    let original = raw as? [String: Any] ?? [:]
    let kind = (original["kind"] as? String) ?? "file"
    let oldBookmark: String? = resolved.refreshed["bookmark"] ?? (original["bookmark"] as? String)
    var bookmark: Any = DocumentDestinations.orNull(oldBookmark)
    if let data = try? resolved.url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil) {
      bookmark = data.base64EncodedString()
    }
    let rootBookmark: String? = resolved.refreshed["root"] ?? (original["root"] as? String)
    let root: Any = DocumentDestinations.orNull(rootBookmark)
    let path: Any = DocumentDestinations.orNull(original["path"] as? String)
    var ref = DocumentDestinations.describe(resolved.url, kind: kind, bookmark: bookmark, root: root, path: path)
    for key in ["provider_label", "persisted"] where original[key] != nil {
      ref[key] = original[key]
    }
    return ref
  }

  // MARK: - Listing and creating

  private func listChildren(_ args: [String: Any]) throws -> [String: Any] {
    let folder = try resolve(args["folder"])
    defer { folder.stop() }
    let names = (args["names"] as? [Any])?.compactMap { $0 as? String }
    var coordinatorError: NSError?
    var listError: Error?
    var children: [[String: Any]] = []
    NSFileCoordinator(filePresenter: nil).coordinate(
      readingItemAt: folder.url, options: [.immediatelyAvailableMetadataOnly], error: &coordinatorError
    ) { url in
      do {
        let keys: [URLResourceKey] = [.nameKey, .fileSizeKey, .contentModificationDateKey, .isDirectoryKey]
        let items = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: keys, options: [])
        for item in items {
          guard let name = DocumentDestinations.visibleName(item.lastPathComponent) else { continue }
          if let names = names, !names.contains(name) { continue }
          let real = item.deletingLastPathComponent().appendingPathComponent(name)
          let isDirectory = (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
          children.append(self.childRef(real, name: name, in: folder, directory: isDirectory))
        }
      } catch {
        listError = error
      }
    }
    if let error = coordinatorError ?? listError.map({ $0 as NSError }) {
      throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeTarget)
    }
    var out = DocumentDestinations.ok(["children": children, "complete": true])
    for (key, value) in folder.refreshed {
      out["refreshed_" + key] = value
    }
    return out
  }

  private func createChild(_ args: [String: Any], directory: Bool) throws -> [String: Any] {
    let folder = try resolve(args["folder"])
    defer { folder.stop() }
    guard let rawName = args["name"] as? String, !rawName.isEmpty else {
      throw DocFailure(DocumentDestinations.errBadArgs, "name is required")
    }
    let name = GlossarionNativePlugin.safeFileName(rawName)
    let onExists = (args["on_exists"] as? String) ?? (directory ? "adopt" : "rename")
    let key = directory ? "folder" : "document"
    var target = folder.url.appendingPathComponent(name, isDirectory: directory)
    if DocumentDestinations.itemExists(target) {
      switch onExists {
      case "adopt":
        return DocumentDestinations.ok([key: childRef(target, name: name, in: folder, directory: directory), "created": false, "adopted": true])
      case "fail":
        throw DocFailure(
          DocumentDestinations.errExists, "\"\(name)\" already exists", scope: DocumentDestinations.scopeDocument,
          extra: ["document": childRef(target, name: name, in: folder, directory: directory)])
      default:
        target = DocumentDestinations.uniqueChild(in: folder.url, name: name, directory: directory)
      }
    }
    var coordinatorError: NSError?
    var innerError: Error?
    NSFileCoordinator(filePresenter: nil).coordinate(
      writingItemAt: target, options: [.forReplacing], error: &coordinatorError
    ) { url in
      do {
        if directory {
          try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false, attributes: nil)
        } else if !FileManager.default.createFile(atPath: url.path, contents: Data(), attributes: nil) {
          throw CocoaError(.fileWriteUnknown)
        }
      } catch {
        innerError = error
      }
    }
    if let error = coordinatorError ?? innerError.map({ $0 as NSError }) {
      if error.domain == NSCocoaErrorDomain && error.code == NSFeatureUnsupportedError {
        // A File Provider that takes files but no sub-folders (or no new items): not worth retrying, the app
        // lays books out flat (DocumentDestinations.kt answers the same for "Create not supported").
        throw DocFailure(
          DocumentDestinations.errReadOnly, error.localizedDescription, scope: DocumentDestinations.scopeDocument,
          extra: ["create_unsupported": true])
      }
      throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeTarget)
    }
    return DocumentDestinations.ok([
      key: childRef(target, name: target.lastPathComponent, in: folder, directory: directory),
      "created": true, "adopted": false,
    ])
  }

  // MARK: - Writing

  private func writeFile(_ args: [String: Any]) throws -> [String: Any] {
    var opId = (args["op_id"] as? String) ?? ""
    if opId.isEmpty {
      opId = UUID().uuidString
    }
    defer { clearCancel(opId) }
    guard let sourcePath = args["source_path"] as? String, !sourcePath.isEmpty,
          FileManager.default.fileExists(atPath: sourcePath) else {
      throw DocFailure(
        DocumentDestinations.errSourceMissing, "The file to upload is missing",
        scope: DocumentDestinations.scopeSource, extra: ["op_id": opId])
    }
    let source = URL(fileURLWithPath: sourcePath)
    let before = DocumentDestinations.fileStamp(source)
    let raw = args["ref"]
    let ref = raw as? [String: Any] ?? [:]
    let resolved = try resolve(raw)
    defer { resolved.stop() }

    var destination = resolved.url
    var created = false
    var adopted = false
    if (ref["kind"] as? String) == "folder" {
      let rawName = (args["name"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? source.lastPathComponent
      let name = GlossarionNativePlugin.safeFileName(rawName)
      destination = resolved.url.appendingPathComponent(name)
      if DocumentDestinations.itemExists(destination) {
        switch (args["on_exists"] as? String) ?? "rename" {
        case "fail":
          throw DocFailure(DocumentDestinations.errExists, "\"\(name)\" already exists", scope: DocumentDestinations.scopeDocument)
        case "adopt":
          adopted = true
        default:
          destination = DocumentDestinations.uniqueChild(in: resolved.url, name: name, directory: false)
        }
      }
      created = !DocumentDestinations.itemExists(destination)
    } else if !DocumentDestinations.itemExists(destination) {
      throw DocFailure(
        DocumentDestinations.errMissing, "The file is no longer there", scope: DocumentDestinations.scopeDocument,
        extra: ["proven": resolved.rootURL != nil])
    }

    var mode = created ? "create" : "replace"
    var written: Int64 = 0
    let total = before.size
    let parent = destination.deletingLastPathComponent()
    let staging = try? FileManager.default.url(
      for: .itemReplacementDirectory, in: .userDomainMask, appropriateFor: created ? parent : destination,
      create: true)
    if let staging = staging {
      defer { try? FileManager.default.removeItem(at: staging) }
      let staged = staging.appendingPathComponent(destination.lastPathComponent)
      // 1. Stage the new content (progress, cancel, ENOSPC) without touching the cloud file.
      do {
        written = try copyChunked(from: source, to: staged, opId: opId, total: total)
      } catch let failure as DocFailure {
        throw failure
      } catch {
        throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeDocument)
      }
      try checkSource(source, before: before, written: written, remoteDamaged: false)
      // 2. Swap it in under coordination.
      do {
        try coordinate(writingAt: destination) { url in
          if created || !FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.moveItem(at: staged, to: url)
          } else {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: staged, backupItemName: nil, options: [])
          }
        }
      } catch where resolved.rootURL == nil && !created {
        // A single picked / exported file: its access may not cover renaming inside its folder.
        // Copy the finished staged file over it instead (short, but not atomic).
        mode = "in_place"
        var copied: Int64 = 0
        do {
          try coordinate(writingAt: destination) { url in
            copied = try self.copyChunked(from: staged, to: url, opId: opId, total: total)
          }
        } catch let failure as DocFailure {
          throw DocFailure(failure.code, failure.message, scope: failure.scope, extra: ["remote_damaged": true])
        }
        written = copied
      }
    } else {
      // No replacement directory on this volume: write in place (not atomic).
      mode = "in_place"
      var copied: Int64 = 0
      do {
        try coordinate(writingAt: destination) { url in
          copied = try self.copyChunked(from: source, to: url, opId: opId, total: total)
        }
      } catch let failure as DocFailure {
        throw DocFailure(failure.code, failure.message, scope: failure.scope, extra: ["remote_damaged": true])
      }
      written = copied
      try checkSource(source, before: before, written: written, remoteDamaged: true)
    }
    emitProgress(opId, written: written, total: total)

    let size = (try? destination.resourceValues(forKeys: [.fileSizeKey]))?.fileSize.map { Int64($0) }
    var document: [String: Any]
    if (ref["kind"] as? String) == "folder" {
      document = childRef(destination, name: destination.lastPathComponent, in: resolved, directory: false)
    } else {
      document = updatedRef(raw, resolved: Resolved(
        url: destination, accessURL: resolved.accessURL, accessing: false, rootURL: resolved.rootURL,
        rootBookmark: resolved.rootBookmark, basePath: resolved.basePath, refreshed: resolved.refreshed))
    }
    var out: [String: Any] = [
      "document": document, "created": created, "adopted": adopted, "mode": mode, "written": written,
      "total": total, "op_id": opId, "attempts": [["mode": mode, "error": "ok"]],
    ]
    if let size = size {
      out["verified_size"] = size
      out["verified"] = true
      if size != written {
        return DocFailure(
          DocumentDestinations.errSizeMismatch, "The saved copy has a different length", scope: DocumentDestinations.scopeDocument,
          extra: out.merging(["stale_tail": size > written, "needs_replace": size > written, "remote_damaged": true]) { _, new in new }
        ).payload
      }
    } else {
      out["verified_size"] = NSNull()
      out["verified"] = false
    }
    if mode == "in_place" {
      out["warnings"] = ["non_atomic"]
    }
    return DocumentDestinations.ok(out)
  }

  private func coordinate(writingAt url: URL, _ body: @escaping (URL) throws -> Void) throws {
    var coordinatorError: NSError?
    var innerError: Error?
    NSFileCoordinator(filePresenter: nil).coordinate(
      writingItemAt: url, options: [.forReplacing], error: &coordinatorError
    ) { target in
      do {
        try body(target)
      } catch {
        innerError = error
      }
    }
    if let failure = innerError as? DocFailure {
      throw failure
    }
    if let error = coordinatorError ?? innerError.map({ $0 as NSError }) {
      throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeDocument)
    }
  }

  private func copyChunked(from source: URL, to destination: URL, opId: String, total: Int64) throws -> Int64 {
    guard let input = InputStream(url: source) else {
      throw DocFailure(DocumentDestinations.errSourceMissing, "Cannot read the file to upload", scope: DocumentDestinations.scopeSource)
    }
    guard let output = OutputStream(url: destination, append: false) else {
      throw DocFailure(DocumentDestinations.errProvider, "Cannot write the destination", scope: DocumentDestinations.scopeDocument)
    }
    input.open()
    output.open()
    defer {
      input.close()
      output.close()
    }
    let capacity = DocumentDestinations.chunkBytes
    var buffer = [UInt8](repeating: 0, count: capacity)
    var written: Int64 = 0
    var lastProgress = Date.distantPast
    while true {
      if isCancelled(opId) {
        throw DocFailure(DocumentDestinations.errCancelled, "Cancelled", scope: DocumentDestinations.scopeDocument)
      }
      let count = input.read(&buffer, maxLength: capacity)
      if count < 0 {
        let error: Error = input.streamError ?? CocoaError(.fileReadUnknown)
        throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeSource)
      }
      if count == 0 {
        break
      }
      var offset = 0
      while offset < count {
        let sent = buffer.withUnsafeBufferPointer { pointer -> Int in
          guard let base = pointer.baseAddress else { return -1 }
          return output.write(base + offset, maxLength: count - offset)
        }
        if sent <= 0 {
          let error: Error = output.streamError ?? CocoaError(.fileWriteUnknown)
          throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeDocument)
        }
        offset += sent
      }
      written += Int64(count)
      let now = Date()
      if now.timeIntervalSince(lastProgress) >= DocumentDestinations.progressInterval {
        lastProgress = now
        emitProgress(opId, written: written, total: total)
      }
    }
    return written
  }

  private func checkSource(_ source: URL, before: (size: Int64, modified: Date?), written: Int64, remoteDamaged: Bool) throws {
    let after = DocumentDestinations.fileStamp(source)
    if written != before.size || after.size != before.size || after.modified != before.modified {
      throw DocFailure(
        DocumentDestinations.errSourceChanged, "The local file changed while it was copied",
        scope: DocumentDestinations.scopeSource, extra: ["written": written, "remote_damaged": remoteDamaged])
    }
  }

  private func emitProgress(_ opId: String, written: Int64, total: Int64) {
    let event: [String: Any] = ["type": "progress", "op_id": opId, "written": written, "total": total]
    DispatchQueue.main.async {
      if self.isDartAttached() {
        self.send(event)
      }
    }
  }

  private func isCancelled(_ opId: String) -> Bool {
    cancelLock.lock()
    defer { cancelLock.unlock() }
    return cancelledOps.contains(opId)
  }

  private func clearCancel(_ opId: String) {
    cancelLock.lock()
    cancelledOps.remove(opId)
    cancelLock.unlock()
  }

  // MARK: - stat, delete, query_root

  private func statItem(_ args: [String: Any]) throws -> [String: Any] {
    let resolved = try resolve(args["document"])
    defer { resolved.stop() }
    guard DocumentDestinations.itemExists(resolved.url) else {
      throw DocFailure(
        DocumentDestinations.errMissing, "The item is no longer there", scope: DocumentDestinations.scopeDocument,
        extra: ["proven": resolved.rootURL != nil])
    }
    return DocumentDestinations.ok(["document": updatedRef(args["document"], resolved: resolved)])
  }

  private func deleteItem(_ args: [String: Any]) throws -> [String: Any] {
    let resolved = try resolve(args["document"])
    defer { resolved.stop() }
    guard DocumentDestinations.itemExists(resolved.url) else {
      throw DocFailure(
        DocumentDestinations.errMissing, "The item is no longer there", scope: DocumentDestinations.scopeDocument,
        extra: ["proven": resolved.rootURL != nil])
    }
    var coordinatorError: NSError?
    var innerError: Error?
    NSFileCoordinator(filePresenter: nil).coordinate(
      writingItemAt: resolved.url, options: [.forDeleting], error: &coordinatorError
    ) { url in
      do {
        try FileManager.default.removeItem(at: url)
      } catch {
        innerError = error
      }
    }
    if let error = coordinatorError ?? innerError.map({ $0 as NSError }) {
      throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeDocument)
    }
    return DocumentDestinations.ok(["deleted": true])
  }

  /// Renames an item inside a picked folder with a coordinated move in the same folder (the cloud sync
  /// gives a replaced copy its first name back). A single exported file has no folder access: unavailable.
  private func renameItem(_ args: [String: Any]) throws -> [String: Any] {
    guard let rawName = args["name"] as? String, !rawName.isEmpty else {
      throw DocFailure(DocumentDestinations.errBadArgs, "name is required")
    }
    let name = GlossarionNativePlugin.safeFileName(rawName)
    let raw = args["document"]
    let ref = raw as? [String: Any] ?? [:]
    let directory = (ref["kind"] as? String) == "folder"
    let resolved = try resolve(raw)
    defer { resolved.stop() }
    guard let rootURL = resolved.rootURL, !resolved.basePath.isEmpty else {
      throw DocFailure(
        DocumentDestinations.errUnavailable, "Only items inside a picked folder can be renamed",
        scope: DocumentDestinations.scopeDocument)
    }
    guard DocumentDestinations.itemExists(resolved.url) else {
      throw DocFailure(
        DocumentDestinations.errMissing, "The item is no longer there", scope: DocumentDestinations.scopeDocument,
        extra: ["proven": true])
    }
    let source = resolved.url
    let target = source.deletingLastPathComponent().appendingPathComponent(name, isDirectory: directory)
    if DocumentDestinations.itemExists(target) {
      throw DocFailure(
        DocumentDestinations.errExists, "\"\(name)\" already exists", scope: DocumentDestinations.scopeDocument)
    }
    let coordinator = NSFileCoordinator(filePresenter: nil)
    var coordinatorError: NSError?
    var innerError: Error?
    coordinator.coordinate(
      writingItemAt: source, options: [.forMoving], writingItemAt: target, options: [.forReplacing],
      error: &coordinatorError
    ) { from, to in
      do {
        try FileManager.default.moveItem(at: from, to: to)
        coordinator.item(at: from, didMoveTo: to)
      } catch {
        innerError = error
      }
    }
    if let error = coordinatorError ?? innerError.map({ $0 as NSError }) {
      throw DocFailure(DocumentDestinations.classify(error), error.localizedDescription, scope: DocumentDestinations.scopeDocument)
    }
    let parentPath = (resolved.basePath as NSString).deletingLastPathComponent
    let folder = Resolved(
      url: target.deletingLastPathComponent(), accessURL: rootURL, accessing: false, rootURL: rootURL,
      rootBookmark: resolved.rootBookmark, basePath: parentPath, refreshed: [:])
    return DocumentDestinations.ok([
      "document": childRef(target, name: name, in: folder, directory: directory), "renamed": true,
    ])
  }

  private func queryRoot(_ args: [String: Any]) throws -> [String: Any] {
    let resolved = try resolve(args["target"])
    defer { resolved.stop() }
    do {
      _ = try resolved.url.checkResourceIsReachable()
    } catch {
      let code = DocumentDestinations.classify(error)
      throw DocFailure(
        code == DocumentDestinations.errProvider ? DocumentDestinations.errMissing : code,
        "The destination is not reachable: \(error.localizedDescription)", scope: DocumentDestinations.scopeTarget)
    }
    var target = updatedRef(args["target"], resolved: resolved)
    target["persisted"] = true
    return DocumentDestinations.ok(["target": target])
  }

  // MARK: - Helpers

  static func ok(_ values: [String: Any] = [:]) -> [String: Any] {
    var out: [String: Any] = ["ok": true, "error": NSNull(), "message": NSNull(), "retryable": false]
    for (key, value) in values {
      out[key] = value
    }
    return out
  }

  static func fail(_ code: String, _ message: String, scope: String? = nil) -> [String: Any] {
    return [
      "ok": false,
      "error": code,
      "message": message,
      "scope": orNull(scope),
      "retryable": retryable.contains(code),
    ]
  }

  /// The value, or NSNull for a missing one (the channel codec sends null).
  static func orNull(_ value: Any?) -> Any {
    if let value = value {
      return value
    }
    return NSNull()
  }

  /// Error code of a Foundation / POSIX error (scope is decided by the caller).
  static func classify(_ error: Error) -> String {
    if let failure = error as? DocFailure {
      return failure.code
    }
    let ns = error as NSError
    if ns.domain == NSCocoaErrorDomain {
      switch ns.code {
      case NSUserCancelledError:
        return errCancelled
      case NSFileWriteOutOfSpaceError:
        return errNoSpace
      case NSFileWriteVolumeReadOnlyError:
        return errReadOnly
      case NSFileReadNoPermissionError, NSFileWriteNoPermissionError:
        return errPermission
      case NSFileNoSuchFileError, NSFileReadNoSuchFileError:
        return errMissing
      default:
        break
      }
    }
    if ns.domain == NSPOSIXErrorDomain {
      switch Int32(truncatingIfNeeded: ns.code) {
      case ENOSPC, EDQUOT:
        return errNoSpace
      case EACCES, EPERM:
        return errPermission
      case ENOENT:
        return errMissing
      case EROFS:
        return errReadOnly
      default:
        break
      }
    }
    if let underlying = ns.userInfo[NSUnderlyingErrorKey] as? Error {
      let inner = classify(underlying)
      if inner != errProvider {
        return inner
      }
    }
    return errProvider
  }

  /// FNV-1a/64 over UTF-8, "d" + 16 hex digits (documents.stable_id in Python).
  static func stableId(_ identity: String) -> String {
    var hash: UInt64 = 0xcbf29ce484222325
    for byte in identity.utf8 {
      hash ^= UInt64(byte)
      hash = hash &* 0x100000001b3
    }
    let hex = String(hash, radix: 16)
    return "d" + String(repeating: "0", count: max(0, 16 - hex.count)) + hex
  }

  static func canonicalPath(_ url: URL) -> String {
    return url.standardizedFileURL.resolvingSymlinksInPath().path
  }

  static func isInside(_ url: URL, root: URL) -> Bool {
    let path = canonicalPath(url)
    let base = canonicalPath(root)
    return path == base || path.hasPrefix(base.hasSuffix("/") ? base : base + "/")
  }

  /// iCloud "Recently Deleted" and File Provider trashes live in .Trash folders.
  static func isTrashed(_ url: URL) -> Bool {
    return url.pathComponents.contains(".Trash") || url.path.contains("/.Trash/")
  }

  /// Glossarion's own container (Files › On My iPhone › Glossarion): never a cloud destination.
  static func isOwnFolder(_ url: URL) -> Bool {
    return isInside(url, root: URL(fileURLWithPath: NSHomeDirectory()))
  }

  static func provider(of url: URL) -> (String, Any) {
    let path = canonicalPath(url)
    if path.contains("/Mobile Documents/") {
      return ("icloud", "iCloud Drive")
    }
    if isOwnFolder(url) {
      return ("local", "Glossarion")
    }
    if path.contains("/File Provider Storage/") || path.contains("/CloudStorage/") {
      return ("file_provider", NSNull())
    }
    return ("local", "On My iPhone")
  }

  static func describe(_ url: URL, kind: String, bookmark: Any, root: Any?, path: Any?) -> [String: Any] {
    let keys: Set<URLResourceKey> = [.nameKey, .fileSizeKey, .contentModificationDateKey, .isWritableKey]
    let values = try? url.resourceValues(forKeys: keys)
    let origin = DocumentDestinations.provider(of: url)
    var ref: [String: Any] = [
      "platform": "ios",
      "kind": kind,
      "id": stableId("ios:" + canonicalPath(url)),
      "uri": NSNull(),
      "document": NSNull(),
      "bookmark": bookmark,
      "root": orNull(root),
      "path": orNull(path),
      "name": values?.name ?? url.lastPathComponent,
      "mime": orNull(GlossarionNativePlugin.mimeType(forExtension: url.pathExtension)),
      "provider": origin.0,
      "provider_label": origin.1,
      "flags": 0,
      "can_write": values?.isWritable ?? true,
      "can_create": kind == "folder" && (values?.isWritable ?? true),
      "can_delete": values?.isWritable ?? true,
      "own_folder": isOwnFolder(url),
      "virtual": false,
    ]
    if let size = values?.fileSize {
      ref["size"] = Int64(size)
    } else {
      ref["size"] = NSNull()
    }
    if let modified = values?.contentModificationDate {
      ref["mtime"] = Int64(modified.timeIntervalSince1970 * 1000)
    } else {
      ref["mtime"] = NSNull()
    }
    if kind == "folder" {
      ref["mime"] = NSNull()
    }
    return ref
  }

  static func fileStamp(_ url: URL) -> (size: Int64, modified: Date?) {
    let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
    let size = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
    return (size, attributes?[.modificationDate] as? Date)
  }

  /// "x.epub" also exists while iCloud keeps it as a ".x.epub.icloud" placeholder (pre-iOS 16 layout).
  static func itemExists(_ url: URL) -> Bool {
    if FileManager.default.fileExists(atPath: url.path) {
      return true
    }
    let placeholder = url.deletingLastPathComponent().appendingPathComponent("." + url.lastPathComponent + ".icloud")
    return FileManager.default.fileExists(atPath: placeholder.path)
  }

  /// Display name of a directory entry: ".x.icloud" placeholders map to "x", other dot files are hidden.
  static func visibleName(_ raw: String) -> String? {
    if raw.hasPrefix(".") && raw.hasSuffix(".icloud") && raw.count > ".icloud".count + 1 {
      return String(raw.dropFirst().dropLast(".icloud".count))
    }
    if raw.hasPrefix(".") {
      return nil
    }
    return raw
  }

  static func child(of root: URL, path: String, directory: Bool) -> URL {
    var url = root
    let parts = path.split(separator: "/").map(String.init).filter { !$0.isEmpty && $0 != "." && $0 != ".." }
    for (index, part) in parts.enumerated() {
      url = url.appendingPathComponent(part, isDirectory: directory || index < parts.count - 1)
    }
    return url
  }

  static func uniqueChild(in folder: URL, name: String, directory: Bool) -> URL {
    let ext = (name as NSString).pathExtension
    let stem = (name as NSString).deletingPathExtension
    var counter = 1
    var candidate = folder.appendingPathComponent(name, isDirectory: directory)
    while itemExists(candidate) {
      let numbered = ext.isEmpty || directory ? "\(name) (\(counter))" : "\(stem) (\(counter)).\(ext)"
      candidate = folder.appendingPathComponent(numbered, isDirectory: directory)
      counter += 1
    }
    return candidate
  }

  static func typeIdentifiers(_ raw: Any?) -> [String] {
    let mimes = (raw as? [Any])?.compactMap { $0 as? String } ?? []
    var types: [String] = []
    for mime in mimes {
      switch mime.lowercased() {
      case "application/epub+zip":
        types.append("org.idpf.epub-container")
      case "application/pdf":
        types.append("com.adobe.pdf")
      case "text/plain":
        types.append("public.plain-text")
      case "text/html", "application/xhtml+xml":
        types.append("public.html")
      case "application/zip":
        types.append("public.zip-archive")
      default:
        types.append("public.item")
      }
    }
    return types.isEmpty ? ["public.item"] : Array(Set(types))
  }

  static func topViewController() -> UIViewController? {
    var root: UIViewController?
    if #available(iOS 13.0, *) {
      let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
      let scene = scenes.first { $0.activationState == .foregroundActive } ?? scenes.first
      let window = scene?.windows.first { $0.isKeyWindow } ?? scene?.windows.first
      root = window?.rootViewController
    }
    if root == nil {
      root = UIApplication.shared.delegate?.window??.rootViewController
    }
    var top = root
    while let presented = top?.presentedViewController, !presented.isBeingDismissed {
      top = presented
    }
    return top
  }
}
