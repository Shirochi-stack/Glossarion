(function () {
  const SKIP_TAGS = new Set([
    "SCRIPT",
    "STYLE",
    "NOSCRIPT",
    "TEXTAREA",
    "INPUT",
    "SELECT",
    "OPTION",
    "CODE",
    "PRE",
    "KBD",
    "SAMP",
    "SVG",
    "CANVAS"
  ]);

  const state = {
    nextId: 1,
    ids: new WeakMap(),
    nodes: new Map(),
    originals: new Map(),
    translations: new Map(),
    lastTranslated: 0,
    activeJobId: "",
    activeTotal: 0,
    streamedIds: new Set(),
    streamSaveTimer: null,
    running: false
  };
  const DEFAULT_CHUNK_COMPRESSION_FACTOR = 3.0;

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || !message.type) {
      return false;
    }

    if (message.type === "GLOSSARION_PING") {
      sendResponse({ ok: true });
      return false;
    }

    if (message.type === "GLOSSARION_COLLECT_TEXT_ITEMS") {
      try {
        const items = collectVisibleTextItems();
        sendResponse({ ok: true, items: items.map(({ id, text }) => ({ id, text })) });
      } catch (error) {
        sendResponse({ ok: false, error: error.message || String(error) });
      }
      return false;
    }

    if (message.type === "GLOSSARION_APPLY_TRANSLATIONS") {
      try {
        applyTranslations(message.items || []);
        sendResponse({ ok: true, applied: message.items?.length || 0 });
      } catch (error) {
        sendResponse({ ok: false, error: error.message || String(error) });
      }
      return false;
    }

    if (message.type === "GLOSSARION_STREAM_TRANSLATIONS") {
      try {
        applyStreamedTranslations(String(message.jobId || ""), message.items || []);
        sendResponse({ ok: true, applied: message.items?.length || 0 });
      } catch (error) {
        sendResponse({ ok: false, error: error.message || String(error) });
      }
      return false;
    }

    if (message.type === "GLOSSARION_RESTORE_PAGE") {
      restorePage();
      sendResponse({ ok: true, restored: state.originals.size });
      return false;
    }

    if (message.type === "GLOSSARION_TRANSLATE_PAGE_DETACHED") {
      const jobId = String(message.jobId || "");
      translatePage(jobId)
        .catch((error) => {
          sendProgress(jobId, {
            status: "error",
            error: error.message || String(error)
          });
        });
      sendResponse({ ok: true });
      return false;
    }

    if (message.type === "GLOSSARION_TRANSLATE_PAGE") {
      translatePage()
        .then((result) => sendResponse({ ok: true, ...result }))
        .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
      return true;
    }

    return false;
  });

  queueAutoRestore();

  async function translatePage(jobId = "") {
    if (state.running) {
      throw new Error("Glossarion is already translating this page.");
    }
    state.running = true;
    state.translations.clear();
    state.streamedIds.clear();
    state.activeJobId = jobId;
    state.activeTotal = 0;
    try {
      const items = collectVisibleTextItems();
      state.activeTotal = items.length;
      if (items.length === 0) {
        sendProgress(jobId, { status: "complete", translated: 0, total: 0 });
        return { translated: 0, total: 0 };
      }
      sendProgress(jobId, { status: "running", translated: 0, total: items.length });

      const settings = await chrome.storage.local.get({
        batchSize: 1000,
        maxTokens: 16384,
        chunkCompressionFactor: DEFAULT_CHUNK_COMPRESSION_FACTOR
      });
      const batchSize = Math.max(1, Math.min(Number(settings.batchSize) || 1000, 5000));
      const maxTokens = Math.max(256, Math.min(Number(settings.maxTokens) || 16384, 200000));
      const chunkCompressionFactor = Math.max(
        1,
        Math.min(Number(settings.chunkCompressionFactor) || DEFAULT_CHUNK_COMPRESSION_FACTOR, 20)
      );
      let translated = 0;

      for (const batch of chunkByOutputBudget(items, batchSize, maxTokens, chunkCompressionFactor)) {
        const response = await chrome.runtime.sendMessage({
          type: "GLOSSARION_TRANSLATE_BATCH",
          jobId,
          items: batch.map(({ id, text }) => ({ id, text }))
        });

        if (!response?.ok) {
          throw new Error(response?.error || "Translation batch failed.");
        }

        applyTranslations(response.items || [], batch);
        for (const item of response.items || []) {
          state.streamedIds.add(String(item.id));
        }
        translated = state.streamedIds.size;
        await saveTranslationSnapshot(jobId, translated, items.length, "partial");
        sendProgress(jobId, { status: "running", translated, total: items.length });
      }

      state.lastTranslated = translated;
      const savedRecordId = await saveTranslationSnapshot(jobId, translated, items.length, "complete");
      sendProgress(jobId, { status: "complete", translated, total: items.length, savedRecordId });
      return { translated, total: items.length };
    } catch (error) {
      if (state.translations.size > 0) {
        await saveTranslationSnapshot(jobId, state.streamedIds.size, state.activeTotal, "partial");
      }
      throw error;
    } finally {
      state.running = false;
      state.activeJobId = "";
      state.activeTotal = 0;
      if (state.streamSaveTimer) {
        clearTimeout(state.streamSaveTimer);
        state.streamSaveTimer = null;
      }
    }
  }

  function collectVisibleTextItems() {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node.nodeValue || !node.nodeValue.trim()) {
            return NodeFilter.FILTER_REJECT;
          }
          const parent = node.parentElement;
          if (!parent || shouldSkipElement(parent) || !isVisible(parent)) {
            return NodeFilter.FILTER_REJECT;
          }
          if (!/[^\d\s.,:;!?()[\]{}'"`~@#$%^&*+=|\\/<>-]/.test(node.nodeValue)) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );

    const items = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const fullText = node.nodeValue;
      const text = fullText.trim();
      if (!text || text.length > 4000) {
        continue;
      }

      const id = getNodeId(node);
      const leading = fullText.match(/^\s*/)?.[0] || "";
      const trailing = fullText.match(/\s*$/)?.[0] || "";
      state.nodes.set(id, node);
      if (!state.originals.has(id)) {
        state.originals.set(id, fullText);
      }
      items.push({ id, text, leading, trailing, index: items.length });
    }
    return items;
  }

  function getNodeId(node) {
    let id = state.ids.get(node);
    if (!id) {
      id = `g${Date.now().toString(36)}_${state.nextId++}`;
      state.ids.set(node, id);
    }
    return id;
  }

  function shouldSkipElement(element) {
    for (let current = element; current && current !== document.body; current = current.parentElement) {
      if (SKIP_TAGS.has(current.tagName)) {
        return true;
      }
      if (current.isContentEditable) {
        return true;
      }
      if (current.closest("[data-glossarion-skip]")) {
        return true;
      }
    }
    return false;
  }

  function isVisible(element) {
    const style = window.getComputedStyle(element);
    if (
      style.display === "none"
      || style.visibility === "hidden"
      || style.opacity === "0"
      || style.contentVisibility === "hidden"
    ) {
      return false;
    }
    return element.getClientRects().length > 0;
  }

  function chunkByOutputBudget(items, maxItems, maxTokens, compressionFactor) {
    const chunks = [];
    let current = [];
    let outputTokens = 0;
    let inputChars = 0;
    const tokenBudget = Math.max(128, Math.floor(maxTokens / compressionFactor));
    const charBudget = Math.max(12000, Math.floor(tokenBudget * 8));

    for (const item of items) {
      const itemTokens = estimateTranslatedJsonTokens(item);
      const nextTokens = outputTokens + itemTokens;
      const nextChars = inputChars + item.text.length;

      if (current.length && (current.length >= maxItems || nextTokens > tokenBudget || nextChars > charBudget)) {
        chunks.push(current);
        current = [];
        outputTokens = 0;
        inputChars = 0;
      }

      current.push(item);
      outputTokens += itemTokens;
      inputChars += item.text.length;
    }

    if (current.length) {
      chunks.push(current);
    }
    return chunks;
  }

  function estimateTranslatedJsonTokens(item) {
    const text = String(item?.text || "");
    const id = String(item?.id || "");
    return Math.ceil(text.length / 2.6) + Math.ceil(id.length / 4) + 18;
  }

  function applyTranslations(items, sourceItems = []) {
    const sourceById = new Map(sourceItems.map((item) => [String(item.id), item]));
    for (const item of items) {
      const node = state.nodes.get(String(item.id));
      if (!node) {
        continue;
      }
      const original = state.originals.get(String(item.id)) || node.nodeValue || "";
      const leading = original.match(/^\s*/)?.[0] || "";
      const trailing = original.match(/\s*$/)?.[0] || "";
      const translated = String(item.text || "").trim();
      if (translated) {
        node.nodeValue = `${leading}${translated}${trailing}`;
        const source = sourceById.get(String(item.id));
        state.translations.set(String(item.id), {
          id: String(item.id),
          index: Number.isFinite(Number(source?.index)) ? Number(source.index) : null,
          original: original.trim(),
          text: translated
        });
      }
    }
  }

  function applyStreamedTranslations(jobId, items) {
    if (!jobId || jobId !== state.activeJobId || !Array.isArray(items) || !items.length) {
      return;
    }
    applyTranslations(items);
    for (const item of items) {
      state.streamedIds.add(String(item.id));
    }
    sendProgress(jobId, {
      status: "running",
      translated: state.streamedIds.size,
      total: state.activeTotal
    });
    schedulePartialSave(jobId);
  }

  function schedulePartialSave(jobId) {
    if (!jobId) {
      return;
    }
    if (state.streamSaveTimer) {
      clearTimeout(state.streamSaveTimer);
    }
    state.streamSaveTimer = setTimeout(() => {
      state.streamSaveTimer = null;
      saveTranslationSnapshot(jobId, state.streamedIds.size, state.activeTotal, "partial").catch(() => {});
    }, 700);
  }

  function restorePage() {
    for (const [id, original] of state.originals.entries()) {
      const node = state.nodes.get(id);
      if (node) {
        node.nodeValue = original;
      }
    }
  }

  async function saveTranslationSnapshot(jobId, translated, total, status) {
    const entries = Array.from(state.translations.values())
      .filter((entry) => entry && entry.text)
      .map((entry) => ({
        id: entry.id,
        index: entry.index,
        original: entry.original,
        text: entry.text
      }));

    if (!entries.length) {
      return "";
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: "GLOSSARION_SAVE_TRANSLATION",
        jobId,
        page: {
          url: location.href,
          title: document.title || location.href
        },
        status,
        translated,
        total,
        entries
      });
      return response?.recordId || "";
    } catch {
      // Saving is best-effort; the translated page should remain usable.
      return "";
    }
  }

  function queueAutoRestore() {
    setTimeout(() => autoRestoreSavedTranslation(), 300);
    setTimeout(() => autoRestoreSavedTranslation(), 1800);
  }

  async function autoRestoreSavedTranslation() {
    if (state.running || document.documentElement.dataset.glossarionRestored === "true") {
      return;
    }

    let response;
    try {
      response = await chrome.runtime.sendMessage({
        type: "GLOSSARION_GET_SAVED_TRANSLATION",
        url: location.href
      });
    } catch {
      return;
    }

    const record = response?.record;
    if (!response?.ok || !record?.entries?.length) {
      return;
    }

    const items = collectVisibleTextItems();
    const applied = applySavedTranslationRecord(record, items);
    if (applied > 0) {
      document.documentElement.dataset.glossarionRestored = "true";
      state.lastTranslated = applied;
    }
  }

  function applySavedTranslationRecord(record, items) {
    const entries = Array.isArray(record.entries) ? record.entries : [];
    const used = new Set();
    const byOriginal = new Map();

    for (const entry of entries) {
      const original = String(entry.original || "").trim();
      if (!original || !entry.text) {
        continue;
      }
      if (!byOriginal.has(original)) {
        byOriginal.set(original, []);
      }
      byOriginal.get(original).push(entry);
    }

    let applied = 0;
    for (const item of items) {
      const text = String(item.text || "").trim();
      let entry = entries.find((candidate, entryIndex) => (
        !used.has(entryIndex)
        && Number(candidate.index) === Number(item.index)
        && String(candidate.original || "").trim() === text
      ));
      let entryIndex = entry ? entries.indexOf(entry) : -1;

      if (!entry) {
        const queue = byOriginal.get(text) || [];
        entry = queue.find((candidate) => {
          const idx = entries.indexOf(candidate);
          return idx >= 0 && !used.has(idx);
        });
        entryIndex = entry ? entries.indexOf(entry) : -1;
      }

      if (!entry || entryIndex < 0) {
        continue;
      }

      const node = state.nodes.get(String(item.id));
      if (!node) {
        continue;
      }

      const original = state.originals.get(String(item.id)) || node.nodeValue || "";
      const leading = original.match(/^\s*/)?.[0] || "";
      const trailing = original.match(/\s*$/)?.[0] || "";
      const translated = String(entry.text || "").trim();
      if (!translated || text === translated) {
        continue;
      }

      node.nodeValue = `${leading}${translated}${trailing}`;
      state.translations.set(String(item.id), {
        id: String(item.id),
        index: item.index,
        original: text,
        text: translated
      });
      used.add(entryIndex);
      applied += 1;
    }
    return applied;
  }

  function sendProgress(jobId, payload) {
    if (!jobId) {
      return;
    }
    chrome.runtime.sendMessage({
      type: "GLOSSARION_TRANSLATION_PROGRESS",
      jobId,
      ...payload
    }).catch(() => {
      // Progress is best-effort; translation should not stop if the popup is gone.
    });
  }
})();
