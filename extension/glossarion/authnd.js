const BUILD_BASE_URL = "https://build.nvidia.com";
const API_BASE_URL = "https://api.ngc.nvidia.com";
const DEFAULT_ORG_ID = "qc69jvmznzxy";
const DEFAULT_PUBLISHER = "z-ai";
const DEFAULT_HCAPTCHA_SITEKEY = "0c6a1e45-75d7-43cc-b836-a0c9d886b8ee";

const metadataCache = new Map();

export async function sendAuthndChatCompletion({ messages, settings }) {
  const modelInfo = normalizeAuthndModel(settings.model);
  const normalizedMessages = normalizeMessages(messages);
  const metadata = await resolveModelMetadata(modelInfo.pageUrl);

  const endpointId = metadata.endpointId || modelInfo.modelId;
  const namespace = metadata.namespace || DEFAULT_ORG_ID;
  const payloadModel = metadata.payloadModel || payloadModelName(modelInfo.modelPath);
  const url = `${API_BASE_URL}/v2/predict/models/${namespace}/${endpointId}`;

  const payload = {
    messages: normalizedMessages,
    model: payloadModel,
    stream: false,
    temperature: Number(settings.temperature ?? 0.2),
    max_tokens: Number(settings.maxTokens ?? 16384)
  };
  applyReasoningPayload(payload, modelInfo.modelPath, settings);

  const response = await postPredictionFromBuildPage({
    pageUrl: modelInfo.pageUrl,
    predictUrl: url,
    payload
  });

  const text = response.text || "";
  const data = parseMaybeJson(text);
  if (!response.ok) {
    const detail = response.headers?.["x-nv-error-msg"]
      || response.headers?.["x-nv-error-code"]
      || data?.error?.message
      || data?.message
      || text
      || response.statusText;
    throw new Error(`AuthND HTTP ${response.status}: ${sanitizeError(detail)}`);
  }

  const content = extractContent(data) || (typeof data === "string" ? data : "");
  if (!content) {
    throw new Error("AuthND returned an empty translation response.");
  }
  return content;
}

export function normalizeAuthndModel(model) {
  let raw = String(model || "").trim();
  raw = raw.replace(/^authnd\d{0,4}\//i, "");
  if (/^authnd/i.test(raw)) {
    raw = raw.replace(/^authnd/i, "").replace(/^\/+/, "");
  }
  raw = raw.replace(/^\/+|\/+$/g, "");
  if (!raw) {
    raw = `${DEFAULT_PUBLISHER}/glm-5.1`;
  }

  let publisher;
  let modelId;
  if (raw.includes("/")) {
    [publisher, modelId] = raw.split(/\/(.+)/, 2);
  } else {
    publisher = DEFAULT_PUBLISHER;
    modelId = raw;
  }

  const pageSlug = buildPageModelSlug(modelId);
  return {
    publisher,
    modelId,
    modelPath: `${publisher}/${modelId}`,
    pageUrl: `${BUILD_BASE_URL}/${publisher}/${pageSlug}`
  };
}

function buildPageModelSlug(modelId) {
  return String(modelId || "").replace(/(?<=\d)\.(?=\d)/g, "_").replace(/^\/+|\/+$/g, "");
}

function payloadModelName(modelPath) {
  return String(modelPath || "").replace(/^\/+|\/+$/g, "");
}

async function resolveModelMetadata(pageUrl) {
  if (metadataCache.has(pageUrl)) {
    return metadataCache.get(pageUrl);
  }

  const response = await fetchWithTimeout(pageUrl, {
    method: "GET",
    headers: {
      accept: "text/html,application/xhtml+xml"
    }
  }, 30000);
  if (!response.ok) {
    throw new Error(`AuthND model page failed to load (${response.status}): ${response.statusText}`);
  }

  const html = await response.text();
  const metadata = {
    endpointId: matchFirst(html, [
      /\\"artifactName\\":\\"([^"\\]+)\\"/,
      /"artifactName"\s*:\s*"([^"]+)"/,
      /\\"nvcfFunctionId\\":\\"([^"\\]+)\\"/,
      /"nvcfFunctionId"\s*:\s*"([^"]+)"/
    ]),
    functionId: matchFirst(html, [
      /\\"nvcfFunctionId\\":\\"([^"\\]+)\\"/,
      /"nvcfFunctionId"\s*:\s*"([^"]+)"/
    ]),
    artifactName: matchFirst(html, [
      /\\"artifactName\\":\\"([^"\\]+)\\"/,
      /"artifactName"\s*:\s*"([^"]+)"/
    ]),
    namespace: matchFirst(html, [
      /\\"namespace\\":\\"([^"\\]+)\\"/,
      /"namespace"\s*:\s*"([^"]+)"/
    ]) || DEFAULT_ORG_ID,
    payloadModel: matchFirst(html, [
      /\\"model\\"\s*:\s*\\"([^"\\]+)\\"/,
      /"model"\s*:\s*"([^"]+)"/
    ])
  };
  metadataCache.set(pageUrl, metadata);
  return metadata;
}

async function postPredictionFromBuildPage({ pageUrl, predictUrl, payload }) {
  const tab = await chrome.tabs.create({ url: pageUrl, active: false });
  try {
    await waitForTabLoaded(tab.id, 60000);
    const [{ result } = {}] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      world: "MAIN",
      args: [DEFAULT_HCAPTCHA_SITEKEY, predictUrl, payload],
      func: runAuthndPredictionInPage
    });
    if (!result) {
      throw new Error("AuthND helper tab returned no result.");
    }
    if (result.error) {
      throw new Error(result.error);
    }
    return result;
  } finally {
    try {
      if (tab?.id) {
        await chrome.tabs.remove(tab.id);
      }
    } catch {
      // Nothing useful to do if the helper tab is already closed.
    }
  }
}

async function runAuthndPredictionInPage(sitekey, predictUrl, payload) {
  const waitFor = (fn, timeoutMs = 20000) => new Promise((resolve, reject) => {
    const start = Date.now();
    const tick = () => {
      try {
        if (fn()) {
          resolve(true);
          return;
        }
      } catch {
        // Keep polling until timeout.
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error("timeout waiting for hcaptcha"));
        return;
      }
      setTimeout(tick, 250);
    };
    tick();
  });

  const loadScript = () => new Promise((resolve, reject) => {
    if (window.hcaptcha) {
      resolve(true);
      return;
    }
    const existing = document.querySelector("script[src*='hcaptcha.com/1/api.js']");
    if (existing) {
      existing.addEventListener("load", () => resolve(true), { once: true });
      existing.addEventListener("error", () => reject(new Error("hcaptcha script failed")), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = "https://js.hcaptcha.com/1/api.js?render=explicit";
    script.async = true;
    script.defer = true;
    script.onload = () => resolve(true);
    script.onerror = () => reject(new Error("hcaptcha script failed"));
    document.head.appendChild(script);
  });

  try {
    await loadScript();
    await waitFor(() => window.hcaptcha && window.hcaptcha.render && window.hcaptcha.execute);
    let container = document.getElementById("__authnd_hcaptcha");
    if (!container) {
      container = document.createElement("div");
      container.id = "__authnd_hcaptcha";
      container.style.position = "fixed";
      container.style.left = "-10000px";
      container.style.top = "0";
      document.body.appendChild(container);
    }
    let widgetId = window.__authndWidgetId;
    if (widgetId === undefined || widgetId === null) {
      widgetId = window.hcaptcha.render(container, { sitekey, size: "invisible" });
      window.__authndWidgetId = widgetId;
    }
    const execResult = await window.hcaptcha.execute(widgetId, { async: true });
    const token = (execResult && execResult.response) || window.hcaptcha.getResponse(widgetId) || "";
    if (!token) {
      throw new Error("AuthND hCaptcha returned no token.");
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 180000);
    try {
      const response = await fetch(predictUrl, {
        method: "POST",
        mode: "cors",
        credentials: "omit",
        headers: {
          accept: "application/json",
          "content-type": "application/json",
          "nv-captcha-token": token
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      const headers = {};
      try {
        response.headers.forEach((value, key) => {
          headers[key] = value;
        });
      } catch {
        // Header exposure depends on the server's CORS policy.
      }
      return {
        ok: response.ok,
        status: response.status,
        statusText: response.statusText,
        headers,
        text: await response.text()
      };
    } finally {
      clearTimeout(timeout);
    }
  } catch (error) {
    return {
      ok: false,
      status: 0,
      statusText: "AuthND browser helper failed",
      headers: {},
      text: "",
      error: String(error && (error.stack || error.message || error))
    };
  }
}

function normalizeMessages(messages) {
  const systemParts = [];
  const normalized = [];

  for (const message of messages || []) {
    let role = String(message.role || "user").toLowerCase();
    const content = message.content == null ? "" : String(message.content);
    if (!content) {
      continue;
    }
    if (role === "system") {
      systemParts.push(content);
      continue;
    }
    if (!["user", "assistant"].includes(role)) {
      role = "user";
    }
    normalized.push({ role, content });
  }

  if (systemParts.length) {
    const systemText = systemParts.join("\n\n");
    const prefix = `System instructions:\n${systemText}`;
    const firstUser = normalized.find((message) => message.role === "user");
    if (firstUser) {
      firstUser.content = firstUser.content ? `${prefix}\n\n${firstUser.content}` : prefix;
    } else {
      normalized.unshift({ role: "user", content: prefix });
    }
  }

  return normalized.length ? normalized : [{ role: "user", content: "" }];
}

function applyReasoningPayload(payload, modelPath, settings) {
  const configured = Object.prototype.hasOwnProperty.call(settings, "thinkingEnabled");
  if (!configured) {
    return;
  }

  const modelLower = String(modelPath || "").toLowerCase();
  const enabled = Boolean(settings.thinkingEnabled);
  const effort = ["low", "medium", "high", "xhigh"].includes(settings.thinkingEffort)
    ? settings.thinkingEffort
    : "medium";

  if (!enabled) {
    payload.chat_template_kwargs = {
      ...(payload.chat_template_kwargs || {}),
      enable_thinking: false
    };
    if (modelLower.includes("nemotron-3-nano")) {
      payload.chat_template_kwargs.parallel_reasoning_mode = "none";
    }
    return;
  }

  if (modelLower.includes("gpt-oss")) {
    payload.reasoning_effort = effort === "xhigh" ? "high" : effort;
    return;
  }

  if (modelLower.includes("nemotron-3-nano")) {
    payload.chat_template_kwargs = {
      ...(payload.chat_template_kwargs || {}),
      enable_thinking: true,
      parallel_reasoning_mode: ["high", "xhigh"].includes(effort) ? "heavy" : effort
    };
    return;
  }

  if (modelLower.includes("deepseek-v4")) {
    payload.reasoning_effort = effort === "xhigh" ? "max" : "high";
    return;
  }

  payload.chat_template_kwargs = {
    ...(payload.chat_template_kwargs || {}),
    enable_thinking: true
  };
}

function extractContent(obj) {
  if (!obj || typeof obj !== "object") {
    return "";
  }
  const choice = Array.isArray(obj.choices) ? obj.choices[0] || {} : {};
  const delta = choice.delta || {};
  const message = choice.message || {};
  for (const candidate of [
    delta.content,
    message.content,
    choice.text,
    choice.content,
    obj.output_text,
    obj.text,
    obj.content,
    obj.response
  ]) {
    if (typeof candidate === "string" && candidate) {
      return candidate;
    }
  }
  return "";
}

function parseMaybeJson(text) {
  if (!text) {
    return "";
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function matchFirst(text, patterns) {
  for (const pattern of patterns) {
    const match = pattern.exec(text);
    if (match) {
      return match[1] || "";
    }
  }
  return "";
}

function waitForTabLoaded(tabId, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(listener);
      reject(new Error("AuthND browser helper tab timed out while loading NVIDIA Build."));
    }, timeoutMs);

    const listener = (updatedTabId, changeInfo) => {
      if (updatedTabId === tabId && changeInfo.status === "complete") {
        clearTimeout(timeout);
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    };

    chrome.tabs.onUpdated.addListener(listener);
    chrome.tabs.get(tabId, (tab) => {
      if (chrome.runtime.lastError) {
        return;
      }
      if (tab?.status === "complete") {
        clearTimeout(timeout);
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    });
  });
}

async function fetchWithTimeout(url, options, timeoutMs = 180000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function sanitizeError(value) {
  return String(value || "").replace(/\s+/g, " ").trim().slice(0, 1200);
}
