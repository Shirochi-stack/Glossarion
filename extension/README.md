# Glossarion Page Translator Extension

Standalone Chrome Manifest V3 extension for translating the visible text of the current page with Glossarion-style prompts.

## MVP scope

- Runs fully in the browser; no local Glossarion desktop bridge is required.
- Uses the Halgakos icon from `src/Halgakos.ico` for the extension and popup.
- Defaults to `authgpt/gpt-5.5` in the model field.
- Defaults to a 16,384 output-token limit; the popup exposes this as "Output tokens".
- Packs intact webpage text nodes into each request using the configurable chunk-compression factor, with "Node cap" as a safety ceiling. The default factor is 3.0.
- Auto-saves page translation snapshots incrementally in Chrome extension storage, keeps the latest 30 pages, and reapplies a saved translation when the same URL reloads.
- Models with `auth` in the name do not require an API key in the popup.
- Uses the exported Glossarion model catalog from `src/model_options.py`.
- Uses the Universal prompt as the default page translation system prompt.
- Mirrors the `UnifiedClient` model-prefix routing table so every model from `src/model_options.py` resolves to a browser route.
- Supports OpenAI-compatible chat-completions routes:
  - OpenAI models such as `gpt-*`, `o3-*`, `o4-*`
  - OpenRouter: `or/*`, `openrouter/*`
  - LiteRouter: `lr/*`
  - ElectronHub: `eh/*`, `electronhub/*`, `electron/*`
  - OpenCode: `oc/*`, `opencode/*`
  - Chutes: `chutes/*`
  - Groq: `groq/*`
  - Fireworks: `fireworks/*`
  - NVIDIA: `nd/*`
  - SambaNova: `sam/*`
  - DeepSeek, xAI/Grok, NanoGPT, and Z.AI prefixes
- Supports Glossarion-style custom prefix routes.
- Supports the browser-backed AuthGPT route (`authgpt/*`) with a Chrome-tab PKCE login, token refresh, and the ChatGPT Codex Responses endpoint ported from `src/authgpt_auth.py`.
- AuthGPT streams through browser `fetch()`/`ReadableStream`; complete JSON items are applied to the page as they arrive.
- Supports the browser-backed AuthND route (`authnd/*`) by opening a temporary NVIDIA Build helper tab to mint the hCaptcha token, matching the flow from `src/authnd_auth.py`.
- Supports native Gemini `generateContent`, native Anthropic `/v1/messages`, Cohere chat, AI21 completions, DeepL, Google Translate, and Google Translate free routes.
- Uses `max_completion_tokens` instead of `max_tokens` for o-series and GPT-5-style OpenAI-compatible models.

```json
[
  {
    "prefix": "my/",
    "routing": "https://example.com/v1",
    "endpoint_type": "/chat/completions"
  }
]
```

Some desktop-only OAuth/browser routes such as `authcd/*`, `authgem/*`, and `authza/*` are recognized but still rejected until their browser auth flows are ported.

The `authgpt/*` and `authnd/*` routes do not need an API key. AuthGPT opens a normal ChatGPT login tab the first time it needs tokens. AuthND may briefly open a background NVIDIA Build helper tab while obtaining the hCaptcha token.

## Thinking toggle

Thinking is off by default. In that state, the extension sends no reasoning fields, and it explicitly disables OpenCode DeepSeek V4 thinking where Glossarion does the same. When enabled, it adds the relevant OpenAI-compatible reasoning fields for supported routes.

## Load in Chrome

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click "Load unpacked".
4. Select this `extension` folder.

## Refresh model options

The generated `glossarion/modelOptions.js` file mirrors `src/model_options.py`.

```powershell
python extension/tools/export_model_options.py
```
