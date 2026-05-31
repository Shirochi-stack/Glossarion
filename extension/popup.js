import { MODEL_OPTIONS } from "./glossarion/modelOptions.js";

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
const SETTINGS_VERSION = 3;
const LEGACY_MAX_TOKENS_DEFAULT = 4096;
const TYPO_MAX_TOKENS_DEFAULT = 16378;
const LEGACY_BATCH_SIZE_DEFAULT = 20;

const fields = {
  model: document.querySelector("#model"),
  apiKey: document.querySelector("#apiKey"),
  sourceLanguage: document.querySelector("#sourceLanguage"),
  targetLanguage: document.querySelector("#targetLanguage"),
  batchSize: document.querySelector("#batchSize"),
  maxTokens: document.querySelector("#maxTokens"),
  chunkCompressionFactor: document.querySelector("#chunkCompressionFactor"),
  thinkingEnabled: document.querySelector("#thinkingEnabled"),
  thinkingEffort: document.querySelector("#thinkingEffort"),
  useCustomOpenAIEndpoint: document.querySelector("#useCustomOpenAIEndpoint"),
  customOpenAIBaseUrl: document.querySelector("#customOpenAIBaseUrl"),
  customPrefixRoutes: document.querySelector("#customPrefixRoutes")
};

const statusEl = document.querySelector("#status");
const apiKeyLabel = document.querySelector("#apiKeyLabel");
const thinkingControls = document.querySelector("#thinkingControls");
const authgptControls = document.querySelector("#authgptControls");
const authgptStatus = document.querySelector("#authgptStatus");
const saveButton = document.querySelector("#save");
const translateButton = document.querySelector("#translate");
const restoreButton = document.querySelector("#restore");
const authgptLoginButton = document.querySelector("#authgptLogin");
const authgptLogoutButton = document.querySelector("#authgptLogout");

let activeStatusTimer = null;

init();

async function init() {
  fillModels();
  const settings = await loadSettings();
  applySettings(settings);
  bindEvents();
}

async function loadSettings() {
  const stored = await chrome.storage.local.get({
    ...DEFAULT_SETTINGS,
    settingsVersion: 0
  });
  if (Number(stored.settingsVersion || 0) < SETTINGS_VERSION) {
    const updates = { settingsVersion: SETTINGS_VERSION };
    if ([LEGACY_MAX_TOKENS_DEFAULT, TYPO_MAX_TOKENS_DEFAULT].includes(Number(stored.maxTokens))) {
      updates.maxTokens = DEFAULT_SETTINGS.maxTokens;
      stored.maxTokens = DEFAULT_SETTINGS.maxTokens;
    }
    if (Number(stored.batchSize) === LEGACY_BATCH_SIZE_DEFAULT) {
      updates.batchSize = DEFAULT_SETTINGS.batchSize;
      stored.batchSize = DEFAULT_SETTINGS.batchSize;
    }
    await chrome.storage.local.set(updates);
  }
  return stored;
}

function fillModels() {
  const datalist = document.querySelector("#models");
  const fragment = document.createDocumentFragment();
  for (const model of MODEL_OPTIONS) {
    const option = document.createElement("option");
    option.value = model;
    fragment.appendChild(option);
  }
  datalist.appendChild(fragment);
}

function bindEvents() {
  saveButton.addEventListener("click", async () => {
    await saveSettings();
    setStatus("Saved.");
  });
  translateButton.addEventListener("click", translateActiveTab);
  restoreButton.addEventListener("click", restoreActiveTab);
  fields.thinkingEnabled.addEventListener("change", updateThinkingVisibility);
  fields.model.addEventListener("input", updateModelDependentVisibility);
  authgptLoginButton.addEventListener("click", loginAuthgpt);
  authgptLogoutButton.addEventListener("click", logoutAuthgpt);
}

function applySettings(settings) {
  fields.model.value = settings.model;
  fields.apiKey.value = settings.apiKey;
  fields.sourceLanguage.value = settings.sourceLanguage;
  fields.targetLanguage.value = settings.targetLanguage;
  fields.batchSize.value = settings.batchSize;
  fields.maxTokens.value = settings.maxTokens;
  fields.chunkCompressionFactor.value = settings.chunkCompressionFactor;
  fields.thinkingEnabled.checked = Boolean(settings.thinkingEnabled);
  fields.thinkingEffort.value = settings.thinkingEffort;
  fields.useCustomOpenAIEndpoint.checked = Boolean(settings.useCustomOpenAIEndpoint);
  fields.customOpenAIBaseUrl.value = settings.customOpenAIBaseUrl;
  fields.customPrefixRoutes.value = settings.customPrefixRoutes;
  updateThinkingVisibility();
  updateModelDependentVisibility();
}

function readSettings() {
  return {
    model: fields.model.value.trim(),
    apiKey: fields.apiKey.value.trim(),
    sourceLanguage: fields.sourceLanguage.value.trim() || "Auto",
    targetLanguage: fields.targetLanguage.value.trim() || "English",
    batchSize: clampNumber(fields.batchSize.value, 1, 5000, 1000),
    maxTokens: clampNumber(fields.maxTokens.value, 256, 200000, 16384),
    chunkCompressionFactor: clampNumber(fields.chunkCompressionFactor.value, 1, 20, 3.0, false),
    thinkingEnabled: fields.thinkingEnabled.checked,
    thinkingEffort: fields.thinkingEffort.value,
    useCustomOpenAIEndpoint: fields.useCustomOpenAIEndpoint.checked,
    customOpenAIBaseUrl: fields.customOpenAIBaseUrl.value.trim(),
    customPrefixRoutes: fields.customPrefixRoutes.value.trim()
  };
}

async function saveSettings() {
  const settings = readSettings();
  await chrome.storage.local.set(settings);
  return settings;
}

async function translateActiveTab() {
  await saveSettings();
  const tab = await getActiveTab();
  if (!tab?.id) {
    setStatus("No active tab.");
    return;
  }

  setBusy(true, "Starting translation...");
  try {
    const response = await chrome.runtime.sendMessage({
      type: "GLOSSARION_START_TRANSLATION",
      tabId: tab.id
    });
    if (!response?.ok) {
      throw new Error(response?.error || "Page translation failed.");
    }
    setStatus("Translation started. You can switch tabs; Glossarion will keep working.");
    startStatusPolling(tab.id);
  } catch (error) {
    setStatus(error.message || String(error), true);
    setBusy(false);
  }
}

function startStatusPolling(tabId) {
  stopStatusPolling();

  const poll = async () => {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "GLOSSARION_GET_TRANSLATION_JOB",
        tabId
      });
      const job = response?.job;
      if (!response?.ok || !job) {
        return;
      }

      if (job.status === "running") {
        setStatus(`Translating ${job.translated}/${job.total || "..."} text nodes.`);
        return;
      }

      if (job.status === "complete") {
        const saved = job.savedRecordId ? " Saved." : " Save skipped.";
        setStatus(`Translated ${job.translated}/${job.total} text nodes.${saved}`);
        stopStatusPolling();
        setBusy(false);
        return;
      }

      if (job.status === "error") {
        setStatus(job.error || "Translation failed.", true);
        stopStatusPolling();
        setBusy(false);
      }
    } catch {
      stopStatusPolling();
      setBusy(false);
    }
  };

  poll();
  activeStatusTimer = setInterval(poll, 1200);
}

function stopStatusPolling() {
  if (activeStatusTimer) {
    clearInterval(activeStatusTimer);
    activeStatusTimer = null;
  }
}

async function loginAuthgpt() {
  await saveSettings();
  setStatus("Opening ChatGPT login...");
  authgptLoginButton.disabled = true;
  try {
    const response = await chrome.runtime.sendMessage({
      type: "GLOSSARION_AUTHGPT_LOGIN",
      model: fields.model.value.trim() || DEFAULT_SETTINGS.model
    });
    if (!response?.ok) {
      throw new Error(response?.error || "AuthGPT login failed.");
    }
    renderAuthgptStatus(response.status);
    setStatus("ChatGPT login saved.");
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    authgptLoginButton.disabled = false;
  }
}

async function logoutAuthgpt() {
  authgptLogoutButton.disabled = true;
  try {
    const response = await chrome.runtime.sendMessage({
      type: "GLOSSARION_AUTHGPT_LOGOUT",
      model: fields.model.value.trim() || DEFAULT_SETTINGS.model
    });
    if (!response?.ok) {
      throw new Error(response?.error || "AuthGPT logout failed.");
    }
    renderAuthgptStatus(response.status);
    setStatus("ChatGPT login removed.");
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    authgptLogoutButton.disabled = false;
  }
}

async function restoreActiveTab() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    setStatus("No active tab.");
    return;
  }

  try {
    const response = await chrome.runtime.sendMessage({
      type: "GLOSSARION_RESTORE_TAB",
      tabId: tab.id
    });
    if (!response?.ok) {
      throw new Error(response?.error || "Restore failed.");
    }
    setStatus("Original text restored.");
  } catch (error) {
    setStatus(error.message || String(error), true);
  }
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function updateThinkingVisibility() {
  thinkingControls.classList.toggle("hidden", !fields.thinkingEnabled.checked);
}

function updateModelDependentVisibility() {
  updateApiKeyVisibility();
  updateAuthgptVisibility();
}

function updateApiKeyVisibility() {
  apiKeyLabel.classList.toggle("hidden", /auth/i.test(fields.model.value.trim()));
}

async function updateAuthgptVisibility() {
  const isAuthgpt = /^authgpt\d{0,4}\//i.test(fields.model.value.trim());
  authgptControls.classList.toggle("hidden", !isAuthgpt);
  if (!isAuthgpt) {
    return;
  }
  try {
    const response = await chrome.runtime.sendMessage({
      type: "GLOSSARION_AUTHGPT_STATUS",
      model: fields.model.value.trim() || DEFAULT_SETTINGS.model
    });
    if (response?.ok) {
      renderAuthgptStatus(response.status);
    }
  } catch {
    authgptStatus.textContent = "Status unavailable";
  }
}

function renderAuthgptStatus(status) {
  if (!status?.loggedIn) {
    authgptStatus.textContent = "Not logged in";
    return;
  }
  const account = status.email || `account ${status.accountId || 0}`;
  const suffix = status.expired ? " - refresh needed" : status.planType ? ` - ${status.planType}` : "";
  authgptStatus.textContent = `${account}${suffix}`;
}

function setBusy(isBusy, message = "") {
  translateButton.disabled = isBusy;
  saveButton.disabled = isBusy;
  restoreButton.disabled = isBusy;
  if (message) {
    setStatus(message);
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#a43d2f" : "";
}

function clampNumber(value, min, max, fallback, round = true) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  const clamped = Math.max(min, Math.min(max, parsed));
  return round ? Math.round(clamped) : Number(clamped.toFixed(2));
}

window.addEventListener("unload", () => {
  stopStatusPolling();
});
