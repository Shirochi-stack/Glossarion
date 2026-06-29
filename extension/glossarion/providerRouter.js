const CHAT_COMPLETIONS_ENDPOINT = "/chat/completions";

const MODEL_PROVIDER_PREFIXES = [
  ["authgem-vertex/", "authgem_vertex"],
  ["authgem-vertex", "authgem_vertex"],
  ["authgem-key/", "authgem_key"],
  ["authgem-key", "authgem_key"],
  ["opencode-go/", "opencode"],
  ["electronhub/", "electronhub"],
  ["google-translate-free", "google_translate_free"],
  ["google-translate", "google_translate"],
  ["labs-leanstral", "mistral"],
  ["open-mistral", "mistral"],
  ["antigravity/", "antigravity"],
  ["antigravity", "antigravity"],
  ["openrouter", "openrouter"],
  ["fireworks", "fireworks"],
  ["authgpt/", "authgpt"],
  ["authgem/", "authgem"],
  ["authza/", "authza"],
  ["authnd/", "authnd"],
  ["authcd/", "authcd"],
  ["electron/", "electronhub"],
  ["opencode/", "opencode"],
  ["perplexity", "perplexity"],
  ["replicate", "replicate"],
  ["databricks", "databricks"],
  ["salesforce", "salesforce"],
  ["sambanova", "sambanova"],
  ["deepseek", "deepseek"],
  ["codestral", "mistral"],
  ["devstral", "mistral"],
  ["ministral", "mistral"],
  ["magistral", "mistral"],
  ["pixtral", "mistral"],
  ["voxtral", "mistral"],
  ["mistral", "mistral"],
  ["mixtral-groq", "groq"],
  ["llama-groq", "groq"],
  ["together", "together"],
  ["baichuan", "baichuan"],
  ["chatglm", "zhipu"],
  ["moonshot", "moonshot"],
  ["hunyuan", "tencent"],
  ["minimax", "minimax"],
  ["sensenova", "sensenova"],
  ["internlm", "internlm"],
  ["luminous", "alephalpha"],
  ["starcoder", "huggingface"],
  ["codellama", "meta"],
  ["authgpt", "authgpt"],
  ["authgem", "authgem"],
  ["authza", "authza"],
  ["authnd", "authnd"],
  ["authcd", "authcd"],
  ["vertex/", "vertex_model_garden"],
  ["gemini", "gemini"],
  ["gemma", "gemini"],
  ["claude", "anthropic"],
  ["sonnet", "anthropic"],
  ["opus", "anthropic"],
  ["haiku", "anthropic"],
  ["chutes/", "chutes"],
  ["chutes", "chutes"],
  ["groq/", "groq"],
  ["groq", "groq"],
  ["command", "cohere"],
  ["cohere", "cohere"],
  ["jurassic", "ai21"],
  ["perplexity", "perplexity"],
  ["pplx", "perplexity"],
  ["sonar", "perplexity"],
  ["qwen", "qwen"],
  ["glm", "zhipu"],
  ["kimi", "moonshot"],
  ["ernie", "baidu"],
  ["spark", "iflytek"],
  ["doubao", "bytedance"],
  ["abab", "minimax"],
  ["intern", "internlm"],
  ["llama4", "meta"],
  ["llama3", "meta"],
  ["llama2", "meta"],
  ["llama", "together"],
  ["grok", "xai"],
  ["openchat", "together"],
  ["wizardlm", "together"],
  ["vicuna", "together"],
  ["alpaca", "together"],
  ["falcon", "tii"],
  ["jamba", "ai21"],
  ["azure", "azure"],
  ["palm", "google"],
  ["bard", "google"],
  ["chatgpt", "openai"],
  ["codex", "openai"],
  ["deepl", "deepl"],
  ["poe", "poe"],
  ["or/", "openrouter"],
  ["lr/", "literouter"],
  ["oc/", "opencode"],
  ["nd/", "nvidia"],
  ["eh/", "electronhub"],
  ["za/", "za"],
  ["nan/", "nanogpt"],
  ["sam/", "sambanova"],
  ["gpt", "openai"],
  ["o1", "openai"],
  ["o3", "openai"],
  ["o4", "openai"],
  ["j2", "ai21"],
  ["yi", "yi"],
  ["phi", "microsoft"],
  ["orca", "microsoft"],
  ["bloom", "bigscience"],
  ["opt", "meta"],
  ["aya", "cohere"],
  ["lr", "literouter"],
  ["nan", "nanogpt"],
  ["sam", "sambanova"],
  ["or", "openrouter"]
].sort((a, b) => b[0].length - a[0].length);

const PROVIDER_DEFINITIONS = {
  openai: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.openai.com/v1", apiKey: true },
  custom_openai: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.openai.com/v1", apiKey: true },
  gemini: { endpointType: "gemini_generate_content", baseUrl: "https://generativelanguage.googleapis.com/v1beta", apiKey: true },
  google: { endpointType: "gemini_generate_content", baseUrl: "https://generativelanguage.googleapis.com/v1beta", apiKey: true },
  anthropic: { endpointType: "/v1/messages", baseUrl: "https://api.anthropic.com/v1", apiKey: true },
  mistral: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.mistral.ai/v1", apiKey: true },
  cohere: { endpointType: "cohere_chat", baseUrl: "https://api.cohere.ai/v1", apiKey: true },
  ai21: { endpointType: "ai21_complete", baseUrl: "https://api.ai21.com/studio/v1", apiKey: true },
  together: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.together.xyz/v1", apiKey: true },
  perplexity: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.perplexity.ai", apiKey: true },
  yi: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.01.ai/v1", apiKey: true },
  qwen: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", apiKey: true },
  baichuan: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.baichuan-ai.com/v1", apiKey: true },
  zhipu: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://open.bigmodel.cn/api/paas/v4", apiKey: true },
  moonshot: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.moonshot.cn/v1", apiKey: true },
  groq: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.groq.com/openai/v1", apiKey: true },
  baidu: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop", apiKey: true },
  tencent: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://hunyuan.cloud.tencent.com/v1", apiKey: true },
  iflytek: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://spark-api.xf-yun.com/v1", apiKey: true },
  bytedance: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://maas-api.vercel.app/v1", apiKey: true },
  minimax: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.minimax.chat/v1", apiKey: true },
  sensenova: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.sensenova.cn/v1", apiKey: true },
  internlm: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.internlm.org/v1", apiKey: true },
  tii: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.tii.ae/v1", apiKey: true },
  microsoft: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.microsoft.com/v1", apiKey: true },
  databricks: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://YOUR-WORKSPACE.databricks.com/serving/endpoints", apiKey: true },
  openrouter: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://openrouter.ai/api/v1", apiKey: true },
  literouter: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.literouter.com/v1", apiKey: true },
  opencode: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://opencode.ai/zen/go/v1", apiKey: true },
  fireworks: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.fireworks.ai/inference/v1", apiKey: true },
  xai: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.x.ai/v1", apiKey: true },
  deepseek: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.deepseek.com/v1", apiKey: true },
  chutes: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://llm.chutes.ai/v1", apiKey: true },
  nvidia: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://integrate.api.nvidia.com/v1", apiKey: true },
  salesforce: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.salesforce.com/v1", apiKey: true },
  bigscience: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.together.xyz/v1", apiKey: true },
  meta: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.together.xyz/v1", apiKey: true },
  za: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.z.ai/api/paas/v4", apiKey: true },
  nanogpt: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://nano-gpt.com/api/v1", apiKey: true },
  sambanova: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.sambanova.ai/v1", apiKey: true },
  electronhub: { endpointType: CHAT_COMPLETIONS_ENDPOINT, baseUrl: "https://api.electronhub.ai/v1", apiKey: true },
  deepl: { endpointType: "deepl_translate", baseUrl: "https://api-free.deepl.com/v2", apiKey: true },
  google_translate: { endpointType: "google_translate", baseUrl: "https://translation.googleapis.com/language/translate/v2", apiKey: true },
  google_translate_free: { endpointType: "google_translate_free", baseUrl: "https://translate.googleapis.com/translate_a/single", apiKey: false },
  authnd: { endpointType: "authnd", baseUrl: "https://api.ngc.nvidia.com", apiKey: false },
  authgpt: { endpointType: "authgpt", baseUrl: "https://chatgpt.com/backend-api", apiKey: false },
  authcd: { endpointType: "authcd", baseUrl: "https://claude.ai", apiKey: false },
  authgem: { endpointType: "authgem", baseUrl: "https://generativelanguage.googleapis.com", apiKey: false },
  authgem_key: { endpointType: "authgem_key", baseUrl: "https://generativelanguage.googleapis.com", apiKey: true },
  authgem_vertex: { endpointType: "authgem_vertex", baseUrl: "https://cloud.google.com/vertex-ai", apiKey: false },
  authza: { endpointType: "authza", baseUrl: "https://chat.z.ai", apiKey: false },
  antigravity: { endpointType: "antigravity", baseUrl: "http://localhost:3000", apiKey: false },
  azure: { endpointType: "azure_openai", baseUrl: "", apiKey: true },
  vertex_model_garden: { endpointType: "vertex_model_garden", baseUrl: "https://aiplatform.googleapis.com", apiKey: false },
  poe: { endpointType: "poe", baseUrl: "https://api.poe.com", apiKey: true },
  replicate: { endpointType: "replicate", baseUrl: "https://api.replicate.com/v1", apiKey: true },
  alephalpha: { endpointType: "alephalpha", baseUrl: "https://api.aleph-alpha.com", apiKey: true },
  huggingface: { endpointType: "huggingface", baseUrl: "https://api-inference.huggingface.co", apiKey: true }
};

const STRIP_PREFIXES = {
  openrouter: ["openrouter/", "or/"],
  literouter: ["lr/"],
  opencode: ["opencode-go/", "opencode/", "oc/"],
  electronhub: ["electronhub/", "electron/", "eh/"],
  groq: ["groq/"],
  chutes: ["chutes/"],
  nvidia: ["nd/"],
  za: ["za/"],
  sambanova: ["sam/"],
  nanogpt: ["nan/"],
  fireworks: ["fireworks/"]
};

const LOCAL_URL_RE = /^(https?:\/\/)?(127\.0\.0\.1|localhost|\[::1\]|0\.0\.0\.0|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|host\.docker\.internal)/i;

export function normalizeCustomPrefixEndpointType(endpointType) {
  const value = String(endpointType || "")
    .trim()
    .toLowerCase()
    .replaceAll("-", "_")
    .replaceAll(" ", "_");

  const legacy = {
    "": CHAT_COMPLETIONS_ENDPOINT,
    openai_chat: CHAT_COMPLETIONS_ENDPOINT,
    "{base_url}/chat/completions": CHAT_COMPLETIONS_ENDPOINT,
    "{base_url}/v1/chat/completions": CHAT_COMPLETIONS_ENDPOINT,
    openai_images: "/images/generations",
    anthropic_messages: "/v1/messages",
    mistral_ocr: "/v1/ocr",
    "{base_url}/images/generations": "/images/generations",
    "{base_url}/v1/messages": "/v1/messages",
    "{base_url}/v1/ocr": "/v1/ocr",
    "{base_url}/{model_id}": "/{model_id}"
  };

  if (legacy[value]) {
    return legacy[value];
  }

  let raw = String(endpointType || "").trim();
  if (raw.startsWith("{base_url}/")) {
    raw = raw.slice("{base_url}".length);
  }
  if (raw.startsWith("/") && !/\s/.test(raw)) {
    return raw;
  }
  return CHAT_COMPLETIONS_ENDPOINT;
}

export function parseCustomPrefixRoutes(rawRoutes) {
  if (!String(rawRoutes || "").trim()) {
    return [];
  }

  let parsed;
  try {
    parsed = JSON.parse(rawRoutes);
  } catch (error) {
    throw new Error(`Custom prefix routes must be valid JSON: ${error.message}`);
  }

  const iterable = Array.isArray(parsed)
    ? parsed
    : Object.entries(parsed || {}).map(([prefix, routing]) => ({ prefix, routing }));

  const seen = new Set();
  const routes = [];
  for (const entry of iterable) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    let prefix = String(entry.prefix || "").trim().replaceAll("\\", "/").replace(/^\/+/, "");
    const routing = normalizeBaseUrl(String(entry.routing || entry.base_url || "").trim());
    const endpointType = normalizeCustomPrefixEndpointType(entry.endpoint_type || entry.type || CHAT_COMPLETIONS_ENDPOINT);
    if (!prefix || !routing || !/^https?:\/\//i.test(routing)) {
      continue;
    }
    if (!prefix.endsWith("/")) {
      prefix += "/";
    }
    const key = prefix.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    routes.push({ prefix, routing, endpointType });
  }

  return routes.sort((a, b) => b.prefix.length - a.prefix.length);
}

export function resolveProviderRoute(model, settings = {}) {
  const rawModel = String(model || "").trim();
  if (!rawModel) {
    throw new Error("Model is required.");
  }

  const customRoute = resolveCustomRoute(rawModel, settings);
  if (customRoute) {
    return customRoute;
  }

  const provider = resolveProviderName(rawModel);
  const definition = PROVIDER_DEFINITIONS[provider];
  if (!definition) {
    throw new Error(`Unsupported standalone extension provider for model: ${rawModel}`);
  }

  const customEndpoint = resolveCustomEndpointOverride(provider, settings);
  if (customEndpoint) {
    return {
      provider: "openai",
      baseUrl: customEndpoint,
      endpointType: CHAT_COMPLETIONS_ENDPOINT,
      model: rawModel,
      rawModel,
      requiresApiKey: !isLocalEndpoint(customEndpoint)
    };
  }

  let effectiveModel = normalizeProviderModel(rawModel, provider);
  if (provider === "fireworks" && !effectiveModel.startsWith("accounts/")) {
    effectiveModel = `accounts/fireworks/models/${effectiveModel}`;
  }

  return {
    provider,
    baseUrl: definition.baseUrl,
    endpointType: definition.endpointType,
    model: effectiveModel,
    rawModel,
    requiresApiKey: definition.apiKey && !isLocalEndpoint(definition.baseUrl)
  };
}

export function buildThinkingRequestFields(route, settings = {}) {
  const thinkingEnabled = Boolean(settings.thinkingEnabled);
  const effort = ["none", "low", "medium", "high", "xhigh"].includes(settings.thinkingEffort)
    ? settings.thinkingEffort
    : "medium";

  if (!thinkingEnabled || effort === "none") {
    if (
      route.provider === "opencode"
      && ["deepseek-v4-flash", "deepseek-v4-pro"].includes(String(route.model || "").toLowerCase())
    ) {
      return { thinking: { type: "disabled" } };
    }
    return {};
  }

  if (route.provider === "openrouter") {
    return {
      reasoning: {
        enabled: true,
        exclude: false,
        effort: effort === "xhigh" ? "high" : effort
      }
    };
  }

  if (["deepseek", "chutes"].includes(route.provider)) {
    return {
      thinking: { type: "enabled" },
      reasoning_effort: effort === "xhigh" ? "max" : effort
    };
  }

  if (route.provider === "nanogpt") {
    return {
      thinking: { enabled: true, effort }
    };
  }

  return {
    reasoning_effort: effort
  };
}

export function buildOpenAICompatiblePayload(route, messages, settings = {}) {
  const tokenLimit = clampProviderMaxTokens(route, Number(settings.maxTokens ?? 16384));
  const payload = {
    model: route.model,
    messages
  };

  if (!usesFixedTemperature(route)) {
    payload.temperature = Number(settings.temperature ?? 0.2);
  }

  if (usesMaxCompletionTokens(route)) {
    payload.max_completion_tokens = tokenLimit;
  } else {
    payload.max_tokens = tokenLimit;
  }

  Object.assign(payload, buildThinkingRequestFields(route, settings));
  return payload;
}

function clampProviderMaxTokens(route, tokenLimit) {
  const requested = Number.isFinite(tokenLimit) ? tokenLimit : 16384;
  if (route?.provider !== "antigravity") {
    return requested;
  }

  const model = String(route.model || "").toLowerCase();
  if (model.includes("claude")) {
    return Math.min(requested, 64000);
  }
  if (model.includes("gemini")) {
    return Math.min(requested, 65536);
  }
  return requested;
}

export function buildResponsesPayload(route, messages, settings = {}) {
  const instructions = [];
  const input = [];
  for (const message of messages || []) {
    const role = message.role || "user";
    const content = String(message.content || "");
    if (role === "system" || role === "developer") {
      instructions.push(content);
      continue;
    }
    input.push({
      role,
      content: [{
        type: role === "assistant" ? "output_text" : "input_text",
        text: content
      }]
    });
  }

  const payload = {
    model: route.model,
    input: input.length ? input : [{ role: "user", content: [{ type: "input_text", text: "" }] }],
    max_output_tokens: Number(settings.maxTokens ?? 16384)
  };
  if (instructions.length) {
    payload.instructions = instructions.join("\n\n");
  }
  if (!usesFixedTemperature(route)) {
    payload.temperature = Number(settings.temperature ?? 0.2);
  }
  return payload;
}

export function shouldUseResponsesApi(route) {
  if (route.provider !== "openai") {
    return false;
  }
  const leaf = modelLeaf(route.model);
  return /^gpt-\d+(?:\.\d+)*-pro(?:$|[-_])/.test(leaf);
}

export function usesMaxCompletionTokens(route) {
  const leaf = modelLeaf(route.model);
  return (
    leaf.startsWith("o1")
    || leaf.startsWith("o3")
    || leaf.startsWith("o4")
    || leaf.startsWith("o5")
    || leaf.startsWith("gpt-5")
  );
}

export function usesFixedTemperature(route) {
  const leaf = modelLeaf(route.model);
  return (
    leaf.startsWith("o1")
    || leaf.startsWith("o3")
    || leaf.startsWith("o4")
    || leaf.startsWith("o5")
    || leaf.startsWith("gpt-5")
  );
}

export function isLocalEndpoint(url) {
  return LOCAL_URL_RE.test(String(url || ""));
}

export function providerNeedsApiKey(route) {
  return Boolean(route?.requiresApiKey);
}

function resolveCustomRoute(model, settings) {
  const routes = parseCustomPrefixRoutes(settings.customPrefixRoutes);
  const modelLower = model.toLowerCase();
  const route = routes.find((candidate) => modelLower.startsWith(candidate.prefix.toLowerCase()));
  if (!route) {
    return null;
  }
  return {
    provider: "custom_openai",
    baseUrl: route.routing,
    endpointType: route.endpointType,
    model: stripPrefix(model, route.prefix),
    rawModel: model,
    requiresApiKey: !isLocalEndpoint(route.routing)
  };
}

function resolveProviderName(model) {
  const modelLower = model.toLowerCase();
  if (/^authgem-vertex\d{1,4}(?:\/|$)/.test(modelLower)) {
    return "authgem_vertex";
  }
  if (/^authgem\d{1,4}(?:\/|$)/.test(modelLower)) {
    return "authgem";
  }
  if (/^authgpt\d{1,4}(?:\/|$)/.test(modelLower)) {
    return "authgpt";
  }
  if (/^authza\d{1,4}(?:\/|$)/.test(modelLower)) {
    return "authza";
  }
  if (/^authnd\d{1,4}(?:\/|$)/.test(modelLower)) {
    return "authnd";
  }
  if (/^authcd\d{1,4}(?:\/|$)/.test(modelLower)) {
    return "authcd";
  }

  for (const [prefix, provider] of MODEL_PROVIDER_PREFIXES) {
    if (modelLower.startsWith(prefix)) {
      return provider;
    }
  }
  return "";
}

function normalizeProviderModel(model, provider) {
  let effective = model.trim();
  if (["authgpt", "authnd", "authcd", "authza"].includes(provider)) {
    effective = stripNumberedAuthPrefix(effective, provider);
  } else if (provider === "authgem") {
    effective = stripNumberedAuthPrefix(effective, "authgem");
  } else if (provider === "authgem_vertex") {
    effective = effective.replace(/^authgem-vertex\d{0,4}\//i, "");
  } else if (provider === "authgem_key") {
    effective = effective.replace(/^authgem-key\//i, "");
  }
  for (const prefix of STRIP_PREFIXES[provider] || []) {
    effective = stripPrefix(effective, prefix);
  }
  return effective.trim();
}

function stripNumberedAuthPrefix(model, provider) {
  return String(model || "")
    .replace(new RegExp(`^${provider}\\d{0,4}/`, "i"), "")
    .replace(new RegExp(`^${provider}`, "i"), "")
    .replace(/^\/+/, "");
}

function stripPrefix(model, prefix) {
  return model.toLowerCase().startsWith(prefix.toLowerCase())
    ? model.slice(prefix.length).trim()
    : model;
}

function resolveCustomEndpointOverride(provider, settings = {}) {
  if (!settings.useCustomOpenAIEndpoint || !settings.customOpenAIBaseUrl) {
    return "";
  }
  if ([
    "gemini",
    "google",
    "openrouter",
    "opencode",
    "custom_openai",
    "authgpt",
    "authnd",
    "authcd",
    "authgem",
    "authgem_key",
    "authgem_vertex",
    "authza"
  ].includes(provider)) {
    return "";
  }
  return normalizeBaseUrl(settings.customOpenAIBaseUrl);
}

function normalizeBaseUrl(url) {
  return String(url || "").trim().replace(/\/+$/, "");
}

function modelLeaf(model) {
  return String(model || "").toLowerCase().trim().split("/").pop() || "";
}
