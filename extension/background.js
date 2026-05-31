import { buildPageBatchMessages } from "./glossarion/prompts.js";
import { sendAuthndChatCompletion } from "./glossarion/authnd.js";
import {
  getAuthgptStatus,
  loginAuthgptForModel,
  logoutAuthgptForModel,
  sendAuthgptChatCompletion
} from "./glossarion/authgpt.js";
import {
  buildOpenAICompatiblePayload,
  buildResponsesPayload,
  providerNeedsApiKey,
  resolveProviderRoute,
  shouldUseResponsesApi
} from "./glossarion/providerRouter.js";

const DEFAULT_SETTINGS = {
  model: "authgpt/gpt-5.5",
  apiKey: "",
  sourceLanguage: "Auto",
  targetLanguage: "English",
  temperature: 0.2,
  maxTokens: 16384,
  batchSize: 1000,
  chunkCompressionFactor: 3.0,
  thinkingEnabled: false,
  thinkingEffort: "medium",
  useCustomOpenAIEndpoint: false,
  customOpenAIBaseUrl: "",
  customPrefixRoutes: ""
};

const translationJobs = new Map();
let nextJobId = 1;
const SAVED_TRANSLATIONS_KEY = "savedTranslations";
const MAX_SAVED_TRANSLATIONS = 30;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || !message.type) {
    return false;
  }

  if (message.type === "GLOSSARION_TRANSLATE_BATCH") {
    translateBatch(message.items || [], {
      sender,
      jobId: String(message.jobId || "")
    })
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  if (message.type === "GLOSSARION_START_TRANSLATION") {
    startTranslationJob(Number(message.tabId))
      .then((job) => sendResponse({ ok: true, job }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  if (message.type === "GLOSSARION_TRANSLATION_PROGRESS") {
    updateTranslationProgress(message, sender);
    sendResponse({ ok: true });
    return false;
  }

  if (message.type === "GLOSSARION_SAVE_TRANSLATION") {
    saveCompletedTranslation(message, sender)
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  if (message.type === "GLOSSARION_GET_SAVED_TRANSLATION") {
    getSavedTranslation(message.url || sender?.tab?.url || "")
      .then((record) => sendResponse({ ok: true, record }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  if (message.type === "GLOSSARION_GET_TRANSLATION_JOB") {
    sendResponse({ ok: true, job: getJobSnapshot(Number(message.tabId)) });
    return false;
  }

  if (message.type === "GLOSSARION_RESTORE_TAB") {
    restoreTab(Number(message.tabId))
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  if (message.type === "GLOSSARION_AUTHGPT_LOGIN") {
    loginAuthgptForModel(message.model || DEFAULT_SETTINGS.model)
      .then((status) => sendResponse({ ok: true, status }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  if (message.type === "GLOSSARION_AUTHGPT_LOGOUT") {
    logoutAuthgptForModel(message.model || DEFAULT_SETTINGS.model)
      .then((status) => sendResponse({ ok: true, status }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  if (message.type === "GLOSSARION_AUTHGPT_STATUS") {
    getAuthgptStatus(message.model || DEFAULT_SETTINGS.model)
      .then((status) => sendResponse({ ok: true, status }))
      .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
    return true;
  }

  return false;
});

async function startTranslationJob(tabId) {
  if (!Number.isInteger(tabId) || tabId <= 0) {
    throw new Error("No tab selected for translation.");
  }

  const existing = translationJobs.get(tabId);
  if (existing?.status === "running") {
    return getJobSnapshot(tabId);
  }

  const job = {
    id: `job_${Date.now().toString(36)}_${nextJobId++}`,
    tabId,
    status: "running",
    translated: 0,
    total: 0,
    error: "",
    startedAt: Date.now(),
    finishedAt: null
  };
  translationJobs.set(tabId, job);

  await ensureContentScript(tabId);
  const response = await chrome.tabs.sendMessage(tabId, {
    type: "GLOSSARION_TRANSLATE_PAGE_DETACHED",
    jobId: job.id
  });
  if (!response?.ok) {
    job.status = "error";
    job.error = response?.error || "Could not start page translation.";
    job.finishedAt = Date.now();
    throw new Error(job.error);
  }
  return getJobSnapshot(tabId);
}

async function restoreTab(tabId) {
  if (!Number.isInteger(tabId) || tabId <= 0) {
    throw new Error("No tab selected to restore.");
  }
  await ensureContentScript(tabId);
  const response = await chrome.tabs.sendMessage(tabId, { type: "GLOSSARION_RESTORE_PAGE" });
  if (!response?.ok) {
    throw new Error(response?.error || "Restore failed.");
  }
  return response;
}

async function ensureContentScript(tabId) {
  try {
    const response = await chrome.tabs.sendMessage(tabId, { type: "GLOSSARION_PING" });
    if (response?.ok) {
      return;
    }
  } catch {
    // The content script is not present on tabs that were open before install/reload.
  }

  await chrome.scripting.executeScript({
    target: { tabId },
    files: ["contentScript.js"]
  });
}

function getJobSnapshot(tabId) {
  const job = translationJobs.get(tabId);
  if (!job) {
    return null;
  }
  return {
    id: job.id,
    tabId: job.tabId,
    status: job.status,
    translated: job.translated,
    total: job.total,
    error: job.error,
    savedRecordId: job.savedRecordId || "",
    startedAt: job.startedAt,
    finishedAt: job.finishedAt
  };
}

function updateTranslationProgress(message, sender) {
  const tabId = Number(sender?.tab?.id || message.tabId);
  const job = translationJobs.get(tabId);
  if (!job || (message.jobId && message.jobId !== job.id)) {
    return;
  }
  if (typeof message.total === "number") {
    job.total = message.total;
  }
  if (typeof message.translated === "number") {
    job.translated = message.translated;
  }
  if (message.status) {
    job.status = message.status;
  }
  if (message.error) {
    job.error = message.error;
  }
  if (message.savedRecordId) {
    job.savedRecordId = message.savedRecordId;
  }
  if (job.status === "complete" || job.status === "error") {
    job.finishedAt = Date.now();
  }
}

async function saveCompletedTranslation(message, sender) {
  const entries = Array.isArray(message.entries) ? message.entries : [];
  if (!entries.length) {
    return { saved: false };
  }

  const settings = await getSettings();
  const tabId = Number(sender?.tab?.id || message.tabId || 0);
  const record = {
    id: `tr_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    jobId: String(message.jobId || ""),
    tabId,
    url: String(message.page?.url || sender?.tab?.url || ""),
    normalizedUrl: normalizeTranslationUrl(message.page?.url || sender?.tab?.url || ""),
    title: String(message.page?.title || sender?.tab?.title || ""),
    savedAt: new Date().toISOString(),
    status: String(message.status || "complete"),
    model: settings.model,
    sourceLanguage: settings.sourceLanguage,
    targetLanguage: settings.targetLanguage,
    translated: Number(message.translated || entries.length),
    total: Number(message.total || entries.length),
    entries: entries.map((entry) => ({
      id: String(entry.id || ""),
      index: Number.isFinite(Number(entry.index)) ? Number(entry.index) : null,
      original: String(entry.original || ""),
      text: String(entry.text || "")
    })).filter((entry) => entry.id && entry.text)
  };

  const stored = await chrome.storage.local.get({ [SAVED_TRANSLATIONS_KEY]: [] });
  const existing = Array.isArray(stored[SAVED_TRANSLATIONS_KEY]) ? stored[SAVED_TRANSLATIONS_KEY] : [];
  const next = [
    record,
    ...existing.filter((item) => normalizeTranslationUrl(item?.url || item?.normalizedUrl || "") !== record.normalizedUrl)
  ].slice(0, MAX_SAVED_TRANSLATIONS);
  await chrome.storage.local.set({ [SAVED_TRANSLATIONS_KEY]: next });

  const job = translationJobs.get(tabId);
  if (job && (!record.jobId || job.id === record.jobId)) {
    job.savedRecordId = record.id;
  }

  return { saved: true, recordId: record.id };
}

async function getSavedTranslation(url) {
  const target = normalizeTranslationUrl(url);
  if (!target) {
    return null;
  }
  const stored = await chrome.storage.local.get({ [SAVED_TRANSLATIONS_KEY]: [] });
  const records = Array.isArray(stored[SAVED_TRANSLATIONS_KEY]) ? stored[SAVED_TRANSLATIONS_KEY] : [];
  return records.find((record) => normalizeTranslationUrl(record?.url || record?.normalizedUrl || "") === target) || null;
}

function normalizeTranslationUrl(url) {
  try {
    const parsed = new URL(String(url || ""));
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return String(url || "").split("#")[0];
  }
}

chrome.tabs.onRemoved.addListener((tabId) => {
  const job = translationJobs.get(tabId);
  if (job?.status === "running") {
    job.status = "error";
    job.error = "The tab was closed during translation.";
    job.finishedAt = Date.now();
  }
});

async function getSettings() {
  const stored = await chrome.storage.local.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

async function translateBatch(items, context = {}) {
  if (!Array.isArray(items) || items.length === 0) {
    return { items: [] };
  }

  const settings = await getSettings();
  const route = resolveProviderRoute(settings.model, settings);
  if (providerNeedsApiKey(route) && !settings.apiKey) {
    throw new Error("Add an API key in the Glossarion extension popup first.");
  }

  if (isTraditionalTranslationRoute(route)) {
    return translateTraditionalBatch({ route, items, settings });
  }

  const messages = buildPageBatchMessages({
    items,
    sourceLanguage: settings.sourceLanguage,
    targetLanguage: settings.targetLanguage
  });

  const content = await sendChatCompletion({ route, messages, settings, context });
  return { items: parseTranslatedItems(content, items) };
}

async function sendChatCompletion({ route, messages, settings, context = {} }) {
  if (route.provider === "authgpt") {
    return sendAuthgptChatCompletion({
      route,
      messages,
      settings,
      onStreamItems: buildStreamItemsForwarder(context)
    });
  }
  if (route.provider === "authnd") {
    return sendAuthndChatCompletion({ messages, settings });
  }
  if (route.endpointType === "gemini_generate_content") {
    return sendGeminiGenerateContent({ route, messages, settings });
  }
  if (route.endpointType === "/v1/messages") {
    return sendAnthropicMessages({ route, messages, settings });
  }
  if (route.endpointType === "cohere_chat") {
    return sendCohereChat({ route, messages, settings });
  }
  if (route.endpointType === "ai21_complete") {
    return sendAI21Completion({ route, messages, settings });
  }
  if (isDesktopOnlyAuthRoute(route)) {
    throw new Error(`${route.provider} is recognized, but its browser auth flow has not been ported yet.`);
  }
  if (route.endpointType !== "/chat/completions") {
    throw new Error(`Endpoint ${route.endpointType} is recognized but not usable for webpage text translation yet.`);
  }

  return sendOpenAICompatible({ route, messages, settings });
}

function buildStreamItemsForwarder(context = {}) {
  const tabId = Number(context.sender?.tab?.id || 0);
  const jobId = String(context.jobId || "");
  if (!tabId || !jobId) {
    return null;
  }
  return (items) => {
    if (!Array.isArray(items) || !items.length) {
      return;
    }
    chrome.tabs.sendMessage(tabId, {
      type: "GLOSSARION_STREAM_TRANSLATIONS",
      jobId,
      items
    }).catch(() => {
      // Streaming updates are opportunistic; final batch response still applies.
    });
  };
}

async function sendOpenAICompatible({ route, messages, settings }) {
  const useResponses = shouldUseResponsesApi(route);
  const url = `${route.baseUrl.replace(/\/+$/, "")}${useResponses ? "/responses" : "/chat/completions"}`;
  const payload = useResponses
    ? buildResponsesPayload(route, messages, settings)
    : buildOpenAICompatiblePayload(route, messages, settings);

  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: buildHeaders(route, settings),
    body: JSON.stringify(payload)
  });

  const data = await readJsonResponse(response);
  if (!response.ok) {
    const detail = data?.error?.message || data?.message || response.statusText;
    throw new Error(`${route.provider} request failed (${response.status}): ${detail}`);
  }

  const content = useResponses
    ? extractResponsesText(data)
    : data?.choices?.[0]?.message?.content || data?.choices?.[0]?.text;
  if (!content) {
    throw new Error(`${route.provider} returned an empty translation response.`);
  }
  return content;
}

async function sendAnthropicMessages({ route, messages, settings }) {
  const { system, anthropicMessages } = splitSystemMessages(messages);
  const payload = {
    model: route.model,
    messages: anthropicMessages,
    max_tokens: Number(settings.maxTokens ?? 16384),
    temperature: Number(settings.temperature ?? 0.2)
  };
  if (system) {
    payload.system = system;
  }
  if (settings.thinkingEnabled) {
    payload.thinking = {
      type: "enabled",
      budget_tokens: Math.max(1024, Math.min(Number(settings.maxTokens ?? 16384) - 1, effortToBudget(settings.thinkingEffort)))
    };
    payload.temperature = 1;
  }

  const response = await fetchWithTimeout(`${route.baseUrl.replace(/\/+$/, "")}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      "x-api-key": String(settings.apiKey || "").trim(),
      "anthropic-version": "2023-06-01"
    },
    body: JSON.stringify(payload)
  });

  const data = await readJsonResponse(response);
  if (!response.ok) {
    const detail = data?.error?.message || data?.message || response.statusText;
    throw new Error(`anthropic request failed (${response.status}): ${detail}`);
  }

  const content = (data?.content || [])
    .map((part) => typeof part?.text === "string" ? part.text : "")
    .join("");
  if (!content) {
    throw new Error("anthropic returned an empty translation response.");
  }
  return content;
}

async function sendGeminiGenerateContent({ route, messages, settings }) {
  const payload = buildGeminiPayload(messages, settings);
  const modelPath = route.model.startsWith("models/") ? route.model : `models/${route.model}`;
  const response = await fetchWithTimeout(`${route.baseUrl.replace(/\/+$/, "")}/${modelPath}:generateContent`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      "x-goog-api-key": String(settings.apiKey || "").trim()
    },
    body: JSON.stringify(payload)
  });

  const data = await readJsonResponse(response);
  if (!response.ok) {
    const detail = data?.error?.message || data?.message || response.statusText;
    throw new Error(`gemini request failed (${response.status}): ${detail}`);
  }

  const content = (data?.candidates?.[0]?.content?.parts || [])
    .map((part) => typeof part?.text === "string" ? part.text : "")
    .join("");
  if (!content) {
    throw new Error("gemini returned an empty translation response.");
  }
  return content;
}

async function sendCohereChat({ route, messages, settings }) {
  const { chatHistory, message } = buildCohereMessages(messages);
  const payload = {
    model: route.model,
    message,
    chat_history: chatHistory,
    temperature: Number(settings.temperature ?? 0.2),
    max_tokens: Number(settings.maxTokens ?? 16384)
  };

  const response = await fetchWithTimeout(`${route.baseUrl.replace(/\/+$/, "")}/chat`, {
    method: "POST",
    headers: buildHeaders(route, settings),
    body: JSON.stringify(payload)
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    const detail = data?.message || data?.error?.message || response.statusText;
    throw new Error(`cohere request failed (${response.status}): ${detail}`);
  }
  const content = data?.text || data?.message?.content?.[0]?.text;
  if (!content) {
    throw new Error("cohere returned an empty translation response.");
  }
  return content;
}

async function sendAI21Completion({ route, messages, settings }) {
  const payload = {
    prompt: formatMessagesAsPrompt(messages),
    temperature: Number(settings.temperature ?? 0.2),
    maxTokens: Number(settings.maxTokens ?? 16384)
  };
  const response = await fetchWithTimeout(`${route.baseUrl.replace(/\/+$/, "")}/${route.model}/complete`, {
    method: "POST",
    headers: buildHeaders(route, settings),
    body: JSON.stringify(payload)
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    const detail = data?.message || data?.error?.message || response.statusText;
    throw new Error(`ai21 request failed (${response.status}): ${detail}`);
  }
  const content = data?.completions?.[0]?.data?.text || data?.text || data?.output;
  if (!content) {
    throw new Error("ai21 returned an empty translation response.");
  }
  return content;
}

function buildHeaders(route, settings) {
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json"
  };

  if (settings.apiKey) {
    headers.Authorization = `Bearer ${String(settings.apiKey).trim()}`;
  }
  if (route.provider === "openrouter") {
    headers["X-Title"] = "Glossarion Page Translator";
    headers["HTTP-Referer"] = "https://github.com/Shirochi-stack/Glossarion";
  }
  return headers;
}

async function translateTraditionalBatch({ route, items, settings }) {
  if (route.endpointType === "deepl_translate") {
    return translateDeepLBatch({ route, items, settings });
  }
  if (route.endpointType === "google_translate") {
    return translateGoogleBatch({ route, items, settings });
  }
  if (route.endpointType === "google_translate_free") {
    return translateGoogleFreeBatch({ route, items, settings });
  }
  throw new Error(`Traditional translation route ${route.endpointType} is not implemented.`);
}

async function translateDeepLBatch({ route, items, settings }) {
  const form = new URLSearchParams();
  for (const item of items) {
    form.append("text", item.text);
  }
  form.append("target_lang", languageToDeepL(settings.targetLanguage));
  form.append("preserve_formatting", "1");

  const response = await fetchWithTimeout(`${route.baseUrl.replace(/\/+$/, "")}/translate`, {
    method: "POST",
    headers: {
      Authorization: `DeepL-Auth-Key ${String(settings.apiKey || "").trim()}`,
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: form.toString()
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    const detail = data?.message || response.statusText;
    throw new Error(`deepl request failed (${response.status}): ${detail}`);
  }
  return {
    items: (data?.translations || []).map((entry, index) => ({
      id: String(items[index]?.id || ""),
      text: String(entry?.text || "")
    })).filter((item) => item.id)
  };
}

async function translateGoogleBatch({ route, items, settings }) {
  const payload = {
    q: items.map((item) => item.text),
    target: languageToGoogle(settings.targetLanguage),
    format: "text"
  };
  const response = await fetchWithTimeout(`${route.baseUrl}?key=${encodeURIComponent(settings.apiKey || "")}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify(payload)
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    const detail = data?.error?.message || data?.message || response.statusText;
    throw new Error(`google translate request failed (${response.status}): ${detail}`);
  }
  const translations = data?.data?.translations || [];
  return {
    items: translations.map((entry, index) => ({
      id: String(items[index]?.id || ""),
      text: String(entry?.translatedText || "")
    })).filter((item) => item.id)
  };
}

async function translateGoogleFreeBatch({ route, items, settings }) {
  const target = languageToGoogle(settings.targetLanguage);
  const translated = [];
  for (const item of items) {
    const url = `${route.baseUrl}?client=gtx&sl=auto&tl=${encodeURIComponent(target)}&dt=t&q=${encodeURIComponent(item.text)}`;
    const response = await fetchWithTimeout(url, { method: "GET" }, 30000);
    const data = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(`google translate free request failed (${response.status}): ${response.statusText}`);
    }
    const text = Array.isArray(data?.[0])
      ? data[0].map((part) => Array.isArray(part) ? part[0] || "" : "").join("")
      : "";
    translated.push({ id: item.id, text });
  }
  return { items: translated };
}

function isTraditionalTranslationRoute(route) {
  return ["deepl_translate", "google_translate", "google_translate_free"].includes(route.endpointType);
}

function isDesktopOnlyAuthRoute(route) {
  return ["authcd", "authgem", "authgem_key", "authgem_vertex", "authza", "antigravity", "vertex_model_garden", "poe", "replicate", "alephalpha", "huggingface", "azure_openai"].includes(route.endpointType);
}

function extractResponsesText(data) {
  if (typeof data?.output_text === "string") {
    return data.output_text;
  }
  const parts = [];
  for (const item of data?.output || []) {
    for (const content of item?.content || []) {
      if (typeof content?.text === "string") {
        parts.push(content.text);
      }
    }
  }
  return parts.join("");
}

function splitSystemMessages(messages) {
  const systemParts = [];
  const anthropicMessages = [];
  for (const message of messages || []) {
    const role = message.role === "assistant" ? "assistant" : message.role === "system" ? "system" : "user";
    const content = String(message.content || "");
    if (!content) {
      continue;
    }
    if (role === "system") {
      systemParts.push(content);
    } else {
      anthropicMessages.push({ role, content });
    }
  }
  return {
    system: systemParts.join("\n\n"),
    anthropicMessages: anthropicMessages.length ? anthropicMessages : [{ role: "user", content: "" }]
  };
}

function buildGeminiPayload(messages, settings) {
  const systemParts = [];
  const contents = [];

  for (const message of messages || []) {
    const role = message.role === "assistant" ? "model" : message.role === "system" ? "system" : "user";
    const text = String(message.content || "");
    if (!text) {
      continue;
    }
    if (role === "system") {
      systemParts.push(text);
    } else {
      contents.push({ role, parts: [{ text }] });
    }
  }

  if (!contents.length) {
    contents.push({ role: "user", parts: [{ text: "" }] });
  }

  const generationConfig = {
    temperature: Number(settings.temperature ?? 0.2),
    maxOutputTokens: Number(settings.maxTokens ?? 16384)
  };
  if (settings.thinkingEnabled) {
    generationConfig.thinkingConfig = {
      thinkingBudget: effortToBudget(settings.thinkingEffort),
      includeThoughts: false
    };
  } else {
    generationConfig.thinkingConfig = {
      thinkingBudget: 0,
      includeThoughts: false
    };
  }

  const payload = {
    contents,
    generationConfig
  };
  if (systemParts.length) {
    payload.systemInstruction = {
      parts: [{ text: systemParts.join("\n\n") }]
    };
  }
  return payload;
}

function buildCohereMessages(messages) {
  const chatHistory = [];
  let message = "";
  for (const item of messages || []) {
    const content = String(item.content || "");
    if (!content) {
      continue;
    }
    if (item.role === "assistant") {
      chatHistory.push({ role: "CHATBOT", message: content });
    } else if (item.role === "system") {
      message = `${content}\n\n${message}`;
    } else {
      message = content;
    }
  }
  return { chatHistory, message: message || formatMessagesAsPrompt(messages) };
}

function formatMessagesAsPrompt(messages) {
  return (messages || [])
    .map((message) => `${String(message.role || "user").toUpperCase()}:\n${String(message.content || "")}`)
    .join("\n\n");
}

function effortToBudget(effort) {
  switch (String(effort || "").toLowerCase()) {
    case "low":
      return 1024;
    case "high":
      return 8192;
    case "xhigh":
      return 16384;
    case "medium":
    default:
      return 16384;
  }
}

function languageToDeepL(language) {
  const key = String(language || "English").trim().toLowerCase();
  const map = {
    english: "EN-US",
    spanish: "ES",
    french: "FR",
    german: "DE",
    italian: "IT",
    portuguese: "PT-BR",
    russian: "RU",
    arabic: "AR",
    hindi: "HI",
    chinese: "ZH",
    "chinese (simplified)": "ZH",
    "chinese simplified": "ZH",
    japanese: "JA",
    korean: "KO",
    turkish: "TR",
    vietnamese: "VI"
  };
  return map[key] || key.toUpperCase().slice(0, 5) || "EN-US";
}

function languageToGoogle(language) {
  const key = String(language || "English").trim().toLowerCase();
  const map = {
    english: "en",
    spanish: "es",
    french: "fr",
    german: "de",
    italian: "it",
    portuguese: "pt",
    russian: "ru",
    arabic: "ar",
    hindi: "hi",
    chinese: "zh",
    "chinese (simplified)": "zh-CN",
    "chinese simplified": "zh-CN",
    "chinese (traditional)": "zh-TW",
    "chinese traditional": "zh-TW",
    japanese: "ja",
    korean: "ko",
    turkish: "tr",
    vietnamese: "vi"
  };
  return map[key] || key || "en";
}

async function fetchWithTimeout(url, options, timeoutMs = 120000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return { message: text };
  }
}

function parseTranslatedItems(content, originalItems) {
  const parsed = parseLooseJson(content);
  const items = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.items)
      ? parsed.items
      : null;

  if (!items) {
    throw new Error("Translator did not return the expected JSON items array.");
  }

  const originalIds = new Set(originalItems.map((item) => String(item.id)));
  return items
    .filter((item) => item && originalIds.has(String(item.id)))
    .map((item) => ({
      id: String(item.id),
      text: String(item.text ?? "")
    }));
}

function parseLooseJson(content) {
  const raw = String(content || "").trim();
  try {
    return JSON.parse(raw);
  } catch {
    const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced) {
      return JSON.parse(fenced[1].trim());
    }
    const firstBrace = raw.indexOf("{");
    const lastBrace = raw.lastIndexOf("}");
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      return JSON.parse(raw.slice(firstBrace, lastBrace + 1));
    }
    const firstBracket = raw.indexOf("[");
    const lastBracket = raw.lastIndexOf("]");
    if (firstBracket >= 0 && lastBracket > firstBracket) {
      return JSON.parse(raw.slice(firstBracket, lastBracket + 1));
    }
    throw new Error("Response was not valid JSON.");
  }
}
