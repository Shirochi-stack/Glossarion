const OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize";
const OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token";
const CHATGPT_BASE_URL = "https://chatgpt.com/backend-api";
const RESPONSES_ENDPOINT = "/codex/responses";
const CALLBACK_URL = "http://localhost:1455/auth/callback";
const SCOPES = "openid profile email offline_access";
const TOKEN_REFRESH_MARGIN_SECONDS = 300;

const loginPromises = new Map();

export async function sendAuthgptChatCompletion({ route, messages, settings, onStreamItems = null }) {
  const modelForAccount = route.rawModel || settings.model || "authgpt/gpt-5.5";
  const actualModel = normalizeAuthgptModel(stripAuthgptPrefix(route.model || modelForAccount) || "gpt-5.5");
  const tokenParamCandidates = ["max_tokens", ""];

  authLoop:
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const accessToken = await getValidAccessToken(modelForAccount, {
      autoLogin: true,
      forceRefresh: attempt > 0
    });

    for (const tokenParam of tokenParamCandidates) {
      const body = buildAuthgptResponsesBody(messages, actualModel, settings, tokenParam);
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 600000);
      try {
        const response = await fetch(`${CHATGPT_BASE_URL}${RESPONSES_ENDPOINT}`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${accessToken}`,
            "Content-Type": "application/json",
            Accept: "text/event-stream, application/json"
          },
          body: JSON.stringify(body),
          signal: controller.signal
        });

        if (response.status === 401 && attempt === 0) {
          await refreshAuthgptForModel(modelForAccount);
          continue authLoop;
        }

        if (!response.ok) {
          const detail = await readAuthgptError(response);
          if (response.status === 400 && tokenParam && isUnsupportedTokenParameterError(detail, tokenParam)) {
            continue;
          }
          throw new Error(`AuthGPT HTTP ${response.status}: ${detail}`);
        }

        const result = await readAuthgptStream(response, { onStreamItems });
        if (result.error_details) {
          throw new Error(`AuthGPT failed: ${formatErrorDetail(result.error_details)}`);
        }
        if (!result.content) {
          throw new Error("AuthGPT returned an empty translation response.");
        }
        return result.content;
      } finally {
        clearTimeout(timeout);
      }
    }
  }

  throw new Error("AuthGPT authentication failed.");
}

export async function loginAuthgptForModel(model) {
  const accountId = extractAuthgptAccountId(model);
  if (loginPromises.has(accountId)) {
    return loginPromises.get(accountId);
  }

  const promise = runOAuthFlow(accountId)
    .then(async (tokens) => {
      await saveTokens(accountId, tokens);
      return getAuthgptStatus(model);
    })
    .finally(() => {
      loginPromises.delete(accountId);
    });

  loginPromises.set(accountId, promise);
  return promise;
}

export async function logoutAuthgptForModel(model) {
  const accountId = extractAuthgptAccountId(model);
  await chrome.storage.local.remove(tokenStorageKey(accountId));
  return getAuthgptStatus(model);
}

export async function getAuthgptStatus(model) {
  const accountId = extractAuthgptAccountId(model);
  const tokens = await loadTokens(accountId);
  const info = extractAccountInfo(tokens?.id_token || "");
  const expiresAt = Number(tokens?.expires_at || 0);
  const now = Math.floor(Date.now() / 1000);
  return {
    accountId,
    loggedIn: Boolean(tokens?.access_token),
    expired: Boolean(tokens?.access_token && expiresAt && now >= expiresAt - TOKEN_REFRESH_MARGIN_SECONDS),
    email: info.email || "",
    planType: info.plan_type || "",
    expiresAt
  };
}

async function getValidAccessToken(model, { autoLogin = true, forceRefresh = false } = {}) {
  const accountId = extractAuthgptAccountId(model);
  const tokens = await loadTokens(accountId);

  if (!forceRefresh && tokens?.access_token && !isTokenExpired(tokens)) {
    return tokens.access_token;
  }

  if (tokens?.refresh_token) {
    try {
      const refreshed = await refreshAccessToken(tokens.refresh_token);
      const merged = { ...tokens, ...refreshed };
      await saveTokens(accountId, merged);
      return merged.access_token;
    } catch {
      // Fall through to interactive login.
    }
  }

  if (!autoLogin) {
    throw new Error("AuthGPT needs a ChatGPT login first.");
  }

  const status = await loginAuthgptForModel(model);
  const fresh = await loadTokens(accountId);
  if (!fresh?.access_token) {
    throw new Error(status?.email ? "AuthGPT login completed but no access token was saved." : "AuthGPT login did not complete.");
  }
  return fresh.access_token;
}

async function refreshAuthgptForModel(model) {
  const accountId = extractAuthgptAccountId(model);
  const tokens = await loadTokens(accountId);
  if (!tokens?.refresh_token) {
    await chrome.storage.local.remove(tokenStorageKey(accountId));
    return null;
  }
  const refreshed = await refreshAccessToken(tokens.refresh_token);
  const merged = { ...tokens, ...refreshed };
  await saveTokens(accountId, merged);
  return merged;
}

async function runOAuthFlow(accountId) {
  const codeVerifier = base64UrlRandom(32);
  const codeChallenge = await pkceChallenge(codeVerifier);
  const state = base64UrlRandom(32);

  const authUrl = buildAuthUrl(codeChallenge, state);
  const tab = await chrome.tabs.create({ url: authUrl, active: true });
  if (!tab?.id) {
    throw new Error("Could not open the AuthGPT login tab.");
  }

  const callback = await waitForOAuthCallback(tab.id, state);
  try {
    await chrome.tabs.remove(tab.id);
  } catch {
    // The user may already have closed it after the redirect.
  }

  if (callback.error) {
    throw new Error(`AuthGPT OAuth error: ${callback.error}`);
  }
  if (!callback.code) {
    throw new Error("AuthGPT OAuth did not return an authorization code.");
  }

  const tokens = await exchangeCodeForTokens(callback.code, codeVerifier);
  return {
    ...tokens,
    glossarion_authgpt_account_id: accountId
  };
}

function buildAuthUrl(codeChallenge, state) {
  const params = new URLSearchParams({
    client_id: OPENAI_CLIENT_ID,
    response_type: "code",
    redirect_uri: CALLBACK_URL,
    scope: SCOPES,
    state,
    code_challenge: codeChallenge,
    code_challenge_method: "S256",
    audience: "https://api.openai.com/v1"
  });
  return `${OPENAI_AUTH_URL}?${params.toString()}`;
}

function waitForOAuthCallback(tabId, expectedState) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("AuthGPT login timed out."));
    }, 300000);

    const cleanup = () => {
      clearTimeout(timeout);
      chrome.tabs.onUpdated.removeListener(onUpdated);
      chrome.tabs.onRemoved.removeListener(onRemoved);
    };

    const inspectUrl = (url) => {
      if (!url || !url.startsWith(CALLBACK_URL)) {
        return;
      }
      let parsed;
      try {
        parsed = new URL(url);
      } catch {
        return;
      }
      const returnedState = parsed.searchParams.get("state") || "";
      if (returnedState !== expectedState) {
        cleanup();
        reject(new Error("AuthGPT OAuth state mismatch."));
        return;
      }
      cleanup();
      resolve({
        code: parsed.searchParams.get("code") || "",
        error: parsed.searchParams.get("error") || ""
      });
    };

    const onUpdated = (updatedTabId, changeInfo, tab) => {
      if (updatedTabId !== tabId) {
        return;
      }
      inspectUrl(changeInfo.url || tab?.url || "");
    };

    const onRemoved = (removedTabId) => {
      if (removedTabId !== tabId) {
        return;
      }
      cleanup();
      reject(new Error("AuthGPT login tab was closed before login completed."));
    };

    chrome.tabs.onUpdated.addListener(onUpdated);
    chrome.tabs.onRemoved.addListener(onRemoved);
    chrome.tabs.get(tabId)
      .then((tab) => inspectUrl(tab?.url || ""))
      .catch(() => {});
  });
}

async function exchangeCodeForTokens(code, codeVerifier) {
  const payload = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: OPENAI_CLIENT_ID,
    code,
    redirect_uri: CALLBACK_URL,
    code_verifier: codeVerifier
  });
  return postTokenRequest(payload);
}

async function refreshAccessToken(refreshToken) {
  const payload = new URLSearchParams({
    grant_type: "refresh_token",
    client_id: OPENAI_CLIENT_ID,
    refresh_token: refreshToken
  });
  return postTokenRequest(payload);
}

async function postTokenRequest(payload) {
  const response = await fetch(OPENAI_TOKEN_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "application/json"
    },
    body: payload.toString()
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { error_description: text };
  }
  if (!response.ok) {
    throw new Error(data.error_description || data.error || response.statusText);
  }
  data.expires_at = Math.floor(Date.now() / 1000) + Number(data.expires_in || 3600);
  return data;
}

function buildAuthgptResponsesBody(messages, model, settings = {}, tokenParam = "max_tokens") {
  const instructions = [];
  const input = [];

  for (const message of messages || []) {
    const role = message.role || "user";
    const content = message.content ?? "";
    if (role === "system" || role === "developer") {
      instructions.push(contentToText(content));
      continue;
    }

    input.push({
      type: "message",
      role,
      content: contentToResponsesParts(content, role)
    });
  }

  const body = {
    model,
    instructions: instructions.join("\n\n"),
    input: input.length ? input : [{
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "" }]
    }],
    store: false,
    stream: true,
    reasoning: buildAuthgptReasoning(settings)
  };

  const tokenLimit = Number(settings.maxTokens ?? 16384);
  if (tokenParam) {
    body[tokenParam] = tokenLimit;
  }

  return body;
}

function buildAuthgptReasoning(settings = {}) {
  if (!settings.thinkingEnabled) {
    return { effort: "none" };
  }
  let effort = String(settings.thinkingEffort || "medium").toLowerCase();
  if (!["none", "low", "medium", "high", "xhigh"].includes(effort)) {
    effort = "medium";
  }
  const reasoning = { effort };
  if (effort !== "none") {
    reasoning.summary = "detailed";
  }
  return reasoning;
}

function contentToText(content) {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return String(content ?? "");
  }
  return content.map((part) => {
    if (typeof part === "string") {
      return part;
    }
    if (part && typeof part === "object") {
      return String(part.text || "");
    }
    return "";
  }).filter(Boolean).join("\n");
}

function contentToResponsesParts(content, role) {
  const textType = role === "assistant" ? "output_text" : "input_text";
  if (typeof content === "string") {
    return [{ type: textType, text: content }];
  }
  if (!Array.isArray(content)) {
    return [{ type: textType, text: String(content ?? "") }];
  }

  const parts = [];
  for (const part of content) {
    if (typeof part === "string") {
      parts.push({ type: textType, text: part });
      continue;
    }
    if (!part || typeof part !== "object") {
      continue;
    }
    if (part.type === "text") {
      parts.push({ type: textType, text: String(part.text || "") });
    } else if (part.type === "input_text" || part.type === "output_text" || part.type === "input_image") {
      parts.push(part);
    } else if (part.type === "image_url") {
      const imageUrl = typeof part.image_url === "string" ? part.image_url : part.image_url?.url;
      if (imageUrl) {
        parts.push({ type: "input_image", image_url: imageUrl });
      }
    }
  }
  return parts.length ? parts : [{ type: textType, text: "" }];
}

async function readAuthgptStream(response, { onStreamItems = null } = {}) {
  if (!response.body) {
    return parseSseText(await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const state = newStreamState();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (processSseLine(line, state, onStreamItems)) {
        try {
          await reader.cancel();
        } catch {
          // The stream may already be closed.
        }
        return finalizeSseState(state);
      }
    }
  }

  buffer += decoder.decode();
  if (buffer) {
    for (const line of buffer.split(/\r?\n/)) {
      processSseLine(line, state, onStreamItems);
    }
  }
  return finalizeSseState(state);
}

function newStreamState() {
  return {
    pendingEventType: "",
    lastData: null,
    completedResult: null,
    failedResult: null,
    contentParts: [],
    emittedItemIds: new Set(),
    done: false
  };
}

function processSseLine(rawLine, state, onStreamItems = null) {
  const line = String(rawLine || "").trim();
  if (!line) {
    return false;
  }
  if (line.startsWith("event:")) {
    state.pendingEventType = line.slice(6).trim();
    return false;
  }
  if (!line.startsWith("data:")) {
    return false;
  }

  const payload = line.slice(5).trimStart();
  if (payload === "[DONE]") {
    state.done = true;
    return true;
  }

  let data;
  try {
    data = JSON.parse(payload);
  } catch {
    return false;
  }

  const eventType = String(data.type || state.pendingEventType || "");
  state.lastData = data;

  if (eventType === "response.output_text.delta") {
    state.contentParts.push(String(data.delta || ""));
    emitCompletedStreamItems(state, onStreamItems);
    return false;
  }

  if (eventType === "response.output_text.done" && !state.contentParts.length && typeof data.text === "string") {
    state.contentParts.push(data.text);
    emitCompletedStreamItems(state, onStreamItems);
    return false;
  }

  if (eventType === "response.completed" || eventType === "response.incomplete") {
    const result = parseResponsesResult(data.response || data);
    if (!result.content && state.contentParts.length) {
      result.content = state.contentParts.join("");
    }
    if (eventType === "response.incomplete" && result.finish_reason === "stop") {
      result.finish_reason = "length";
    }
    state.completedResult = result;
    state.done = true;
    return true;
  }

  if (eventType === "response.failed") {
    const result = parseResponsesResult(data.response || data);
    result.content = "";
    result.finish_reason = "error";
    result.error_details = data.response?.error || data.error || { type: "response.failed" };
    state.failedResult = result;
    state.done = true;
    return true;
  }

  return false;
}

function emitCompletedStreamItems(state, onStreamItems) {
  if (typeof onStreamItems !== "function") {
    return;
  }
  const content = state.contentParts.join("");
  const items = collectCompleteJsonItems(content, state.emittedItemIds);
  if (items.length) {
    onStreamItems(items);
  }
}

function collectCompleteJsonItems(content, emittedItemIds) {
  const text = String(content || "");
  const arrayStart = findItemsArrayStart(text);
  if (arrayStart < 0) {
    return [];
  }

  const items = [];
  let inString = false;
  let escape = false;
  let depth = 0;
  let objectStart = -1;

  for (let i = arrayStart + 1; i < text.length; i += 1) {
    const char = text[i];
    if (inString) {
      if (escape) {
        escape = false;
      } else if (char === "\\") {
        escape = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }

    if (char === "\"") {
      inString = true;
      continue;
    }

    if (char === "{") {
      if (depth === 0) {
        objectStart = i;
      }
      depth += 1;
      continue;
    }

    if (char === "}") {
      if (depth > 0) {
        depth -= 1;
      }
      if (depth === 0 && objectStart >= 0) {
        const rawObject = text.slice(objectStart, i + 1);
        objectStart = -1;
        try {
          const parsed = JSON.parse(rawObject);
          const id = String(parsed?.id || "");
          const translated = String(parsed?.text ?? "");
          if (id && translated && !emittedItemIds.has(id)) {
            emittedItemIds.add(id);
            items.push({ id, text: translated });
          }
        } catch {
          // The object may not be complete JSON yet; wait for more stream data.
        }
      }
    }
  }

  return items;
}

function findItemsArrayStart(text) {
  const source = String(text || "");
  const itemsMatch = source.match(/"items"\s*:/);
  if (itemsMatch?.index >= 0) {
    return source.indexOf("[", itemsMatch.index + itemsMatch[0].length);
  }

  const trimmedStart = source.search(/\S/);
  if (trimmedStart >= 0 && source[trimmedStart] === "[") {
    return trimmedStart;
  }

  return -1;
}

function finalizeSseState(state) {
  if (state.failedResult) {
    return state.failedResult;
  }
  if (state.completedResult) {
    return state.completedResult;
  }
  if (state.contentParts.length) {
    return {
      content: state.contentParts.join(""),
      finish_reason: "stop",
      conversation_id: null,
      message_id: null,
      usage: null
    };
  }
  if (state.lastData) {
    return parseResponsesResult(state.lastData.response || state.lastData);
  }
  return {
    content: "",
    finish_reason: "error",
    conversation_id: null,
    message_id: null,
    usage: null
  };
}

function parseSseText(rawText) {
  const stripped = String(rawText || "").trim();
  if (stripped.startsWith("{")) {
    try {
      return parseResponsesResult(JSON.parse(stripped));
    } catch {
      // Fall through to SSE parsing.
    }
  }
  const state = newStreamState();
  for (const line of stripped.split(/\r?\n/)) {
    if (processSseLine(line, state)) {
      break;
    }
  }
  return finalizeSseState(state);
}

function parseResponsesResult(data) {
  let content = "";
  for (const item of data?.output || []) {
    if (item?.type !== "message") {
      continue;
    }
    for (const part of item.content || []) {
      if (part?.type === "output_text" && typeof part.text === "string") {
        content += part.text;
      }
    }
  }

  const status = String(data?.status || "");
  let finishReason = "stop";
  if (status === "incomplete") {
    const reason = String(data?.incomplete_details?.reason || "");
    finishReason = reason.includes("tokens") ? "length" : reason || "incomplete";
  } else if (status === "failed") {
    finishReason = "error";
  }

  const rawUsage = data?.usage;
  const usage = rawUsage ? {
    prompt_tokens: rawUsage.input_tokens || 0,
    completion_tokens: rawUsage.output_tokens || 0,
    total_tokens: rawUsage.total_tokens || 0
  } : null;

  const result = {
    content,
    finish_reason: finishReason,
    conversation_id: data?.id || null,
    message_id: data?.id || null,
    usage
  };
  if (data?.error) {
    result.error_details = data.error;
  }
  return result;
}

async function readAuthgptError(response) {
  const text = await response.text();
  if (!text) {
    return response.statusText || "empty response";
  }
  try {
    const data = JSON.parse(text);
    return data.detail || data.error?.message || data.error_description || data.message || text;
  } catch {
    return text;
  }
}

function formatErrorDetail(value) {
  if (typeof value === "string") {
    return value;
  }
  if (value?.message) {
    return value.message;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function isUnsupportedTokenParameterError(detail, tokenParam) {
  const text = String(detail || "").toLowerCase();
  return (
    text.includes("unsupported parameter")
    && text.includes(String(tokenParam || "").toLowerCase())
  );
}

function extractAuthgptAccountId(model) {
  const match = String(model || "").toLowerCase().match(/^authgpt(\d{1,4})(?:\/|$)/);
  return match ? Number(match[1]) : 0;
}

function stripAuthgptPrefix(model) {
  let value = String(model || "").trim();
  value = value.replace(/^authgpt\d{0,4}\//i, "");
  if (/^authgpt/i.test(value)) {
    value = value.replace(/^authgpt/i, "").replace(/^\/+/, "");
  }
  return value;
}

function normalizeAuthgptModel(model) {
  const value = String(model || "").trim();
  return /^\d+(?:[.-]\d+)*(?:$|[-_])/.test(value) ? `gpt-${value}` : value;
}

function tokenStorageKey(accountId) {
  return accountId ? `authgptTokens_${accountId}` : "authgptTokens";
}

async function loadTokens(accountId) {
  const key = tokenStorageKey(accountId);
  const stored = await chrome.storage.local.get({ [key]: null });
  return stored[key] || null;
}

async function saveTokens(accountId, tokens) {
  await chrome.storage.local.set({ [tokenStorageKey(accountId)]: tokens });
}

function isTokenExpired(tokens) {
  const expiresAt = Number(tokens?.expires_at || 0);
  return !expiresAt || Math.floor(Date.now() / 1000) >= expiresAt - TOKEN_REFRESH_MARGIN_SECONDS;
}

function extractAccountInfo(idToken) {
  const claims = decodeJwtPayload(idToken) || {};
  const authClaims = claims["https://api.openai.com/auth"] || {};
  return {
    chatgpt_account_id: authClaims.chatgpt_account_id || "",
    plan_type: authClaims.chatgpt_plan_type || "",
    email: claims.email || ""
  };
}

function decodeJwtPayload(token) {
  try {
    const payload = String(token || "").split(".")[1];
    if (!payload) {
      return null;
    }
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(payload.length / 4) * 4, "=");
    return JSON.parse(atob(base64));
  } catch {
    return null;
  }
}

function base64UrlRandom(byteCount) {
  const bytes = new Uint8Array(byteCount);
  crypto.getRandomValues(bytes);
  return base64UrlEncodeBytes(bytes);
}

async function pkceChallenge(codeVerifier) {
  const bytes = new TextEncoder().encode(codeVerifier);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return base64UrlEncodeBytes(new Uint8Array(digest));
}

function base64UrlEncodeBytes(bytes) {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}
