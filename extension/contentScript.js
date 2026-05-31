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
    running: false
  };

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

  async function translatePage(jobId = "") {
    if (state.running) {
      throw new Error("Glossarion is already translating this page.");
    }
    state.running = true;
    state.translations.clear();
    try {
      const items = collectVisibleTextItems();
      if (items.length === 0) {
        sendProgress(jobId, { status: "complete", translated: 0, total: 0 });
        return { translated: 0, total: 0 };
      }
      sendProgress(jobId, { status: "running", translated: 0, total: items.length });

      const settings = await chrome.storage.local.get({ batchSize: 20 });
      const batchSize = Math.max(1, Math.min(Number(settings.batchSize) || 20, 80));
      let translated = 0;

      for (const batch of chunkBySizeAndChars(items, batchSize, 6000)) {
        const response = await chrome.runtime.sendMessage({
          type: "GLOSSARION_TRANSLATE_BATCH",
          items: batch.map(({ id, text }) => ({ id, text }))
        });

        if (!response?.ok) {
          throw new Error(response?.error || "Translation batch failed.");
        }

        applyTranslations(response.items || []);
        translated += response.items?.length || 0;
        sendProgress(jobId, { status: "running", translated, total: items.length });
      }

      state.lastTranslated = translated;
      const savedRecordId = await saveCompletedTranslation(jobId, translated, items.length);
      sendProgress(jobId, { status: "complete", translated, total: items.length, savedRecordId });
      return { translated, total: items.length };
    } finally {
      state.running = false;
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
      items.push({ id, text, leading, trailing });
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

  function chunkBySizeAndChars(items, maxItems, maxChars) {
    const chunks = [];
    let current = [];
    let chars = 0;
    for (const item of items) {
      const nextChars = chars + item.text.length;
      if (current.length && (current.length >= maxItems || nextChars > maxChars)) {
        chunks.push(current);
        current = [];
        chars = 0;
      }
      current.push(item);
      chars += item.text.length;
    }
    if (current.length) {
      chunks.push(current);
    }
    return chunks;
  }

  function applyTranslations(items) {
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
        state.translations.set(String(item.id), {
          id: String(item.id),
          original: original.trim(),
          text: translated
        });
      }
    }
  }

  function restorePage() {
    for (const [id, original] of state.originals.entries()) {
      const node = state.nodes.get(id);
      if (node) {
        node.nodeValue = original;
      }
    }
  }

  async function saveCompletedTranslation(jobId, translated, total) {
    const entries = Array.from(state.translations.values())
      .filter((entry) => entry && entry.text)
      .map((entry) => ({
        id: entry.id,
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
