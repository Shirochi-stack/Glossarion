# 📚 Glossarion — The Complete, Monkey-Proof User Guide

**Version 9.3.3 · Updated June 22, 2026**

This guide explains **every button, box, and toggle** in Glossarion in plain English. You do **not** need to know anything about coding, AI, or computers beyond clicking, typing, and dragging files. If you can use a web browser, you can use this guide.

> **How to read this guide:** Follow it top to bottom the first time. After that, jump to the section you need using the table of contents. Anything marked **⚠️ DON'T** is a common mistake — read those carefully. Anything marked **✅ DO** is the safe, recommended choice.

---

## 📑 Table of Contents

1. [Words you need to know (read this first)](#1-words-you-need-to-know-read-this-first)
2. [Installing and starting Glossarion](#2-installing-and-starting-glossarion)
3. [Your very first translation (the 7-step golden path)](#3-your-very-first-translation-the-7-step-golden-path)
4. [The main window, explained piece by piece](#4-the-main-window-explained-piece-by-piece)
5. [Getting an API key and choosing a model](#5-getting-an-api-key-and-choosing-a-model)
6. [Main settings (the middle of the window)](#6-main-settings-the-middle-of-the-window)
7. [Other Settings (the ⚙️ Other Setting button)](#7-other-settings-the-️-other-setting-button)
8. [The Glossary system (read this — it matters a lot)](#8-the-glossary-system-read-this--it-matters-a-lot)
9. [Local AI / custom endpoints — the 3 methods](#9-local-ai--custom-endpoints--the-3-methods)
10. [Manga & comic translation](#10-manga--comic-translation)
11. [Retranslation (fixing single chapters)](#11-retranslation-fixing-single-chapters)
12. [EPUB Converter (building your finished book)](#12-epub-converter-building-your-finished-book)
13. [QA Scanner (checking quality)](#13-qa-scanner-checking-quality)
14. [Multiple API keys (rotation)](#14-multiple-api-keys-rotation)
15. [Refinement & output modes (new stuff in 9.3.3)](#15-refinement--output-modes)
16. [Translation editing & review (.sdlxliff and formats)](#16-translation-editing--review-sdlxliff-and-supported-formats)
17. [How to translate for free](#17-how-to-translate-for-free)
18. [When things go wrong (troubleshooting)](#18-when-things-go-wrong-troubleshooting)
19. [One-page cheat sheet](#19-one-page-cheat-sheet)

---

## 1. Words you need to know (read this first)

Glossarion uses a few words over and over. Learn these five and the rest of the guide will make sense.

| Word | What it actually means | Real example |
|------|------------------------|--------------|
| **API key** | A secret password that lets Glossarion talk to an AI company's computers on your behalf. You usually pay the AI company, not Glossarion. Treat it like a credit card number. | `sk-abc123...` from OpenAI |
| **Model** | The specific "AI brain" doing the translating. Smarter models cost more and are slower. | `gpt-5.5`, `gemini-3.5-flash`, `claude-opus-4-8` |
| **Provider** | The company that makes the model. | OpenAI, Google, Anthropic, DeepSeek |
| **Profile** | A saved set of instructions telling the AI *how* to translate (which language, what tone, what rules). Glossarion comes with ready-made profiles for Korean, Japanese, and Chinese. | `Korean`, `Japanese`, `korean_OCR` |
| **Glossary** | A name list that keeps the AI consistent — so the hero is "Kim Dokja" in chapter 1 *and* chapter 90, not "Reader Kim" halfway through. | A `.csv` or `.json` file |

A few more you'll bump into:

- **EPUB** — the most common e-book file (`.epub`). Glossarion's favorite input.
- **Token** — a chunk of text (roughly ¾ of a word). AI companies bill per token and limit how many you can send at once. You don't have to count them; just know "token limit" = "how much text at a time."
- **Chunk** — when a chapter is too big, Glossarion splits it into smaller pieces called chunks and translates them one by one.
- **OCR** — "reading" text out of an image (used for manga and scanned pages).
- **Inpainting** — erasing the original text from a manga bubble so the translation can be drawn in its place.

> **✅ DO:** If a setting confuses you, hover your mouse over it. Almost every box in Glossarion shows a yellow help bubble (a *tooltip*) explaining what it does. This guide is built from those same explanations.

---

## 2. Installing and starting Glossarion

There are two kinds of users. Pick the one that matches you.

### A) You downloaded a ready-made `.exe` (most people)

1. Find the Glossarion `.exe` file you downloaded (for example a file whose name contains `Glossarion` and a version number).
2. **Double-click it.** That's it. A splash screen with a spinning flower logo appears while it loads (this can take 10–60 seconds the first time — be patient).
3. The main window opens. Skip to [Section 3](#3-your-very-first-translation-the-7-step-golden-path).

> **Which build should I download?** There are several `.exe` packages. Here's what each one is:
>
> | Package | What it is |
> |---------|------------|
> | **`L_Glossarion.v9.3.3.exe`** ⭐ | **The standard build — recommended for most people.** Full-featured translation (everything *except* manga). Most optimal build for **novel** translation. Includes the **EPUB Library/Reader**. |
> | `L_Glossarion_Lite.v9.3.3.exe` | Same, but **excludes the EPUB Reader and `authnd/` prefix routing**. |
> | `L_Glossarion_TurboLite.v9.3.3.exe` | Excludes the EPUB Reader, `authnd/` routing, Vertex AI SDK, and PDF generation. Smallest/fastest to start. |
> | `N_Glossarion_NoCuda.v9.3.3.exe` | **Full build *with* manga translation** plus some experimental novel features (silent-truncation detection, Argos-translate fallback to Google Translate). Most optimal build for **manga**. Doesn't need an NVIDIA GPU. |
> | `N_Glossarion_Heavy.v9.3.3.exe` | Adds CUDA (NVIDIA GPU) support for the manga translator. ⚠️ May be unstable. |
>
> **✅ Recommended: the standard `L_Glossarion.v9.3.3.exe`.** It keeps the built-in **EPUB Library**, which is what powers **browser-based requests using the `authnd/` provider** (e.g. `authnd/...` models). The `Lite` and `TurboLite` builds drop the EPUB Library *and* `authnd/` routing, so choose the standard build if you want those. Only pick a `N_` build if you specifically need **manga** translation.

### B) You have the source code and Python (advanced)

1. Install Python 3.10 or newer.
2. Open a terminal in the `Glossarion` folder and install the requirements once:
   ```bash
   pip install -r requirements.txt        # Windows
   pip install -r requirements-macos.txt  # macOS
   ```
3. Start the program. **The entry point is `translator_gui.py`.**
   ```bash
   cd src
   python translator_gui.py
   ```
   On Windows you can instead **double-click `src/launch_Glossarion.bat`**, which does the same thing and pauses to show any errors.

> **⚠️ DON'T** double-click `START_Glosarion.bat` expecting the translator — that file starts the optional **Discord bot**, not the GUI. Use `launch_Glossarion.bat` or run `translator_gui.py`.

> **⚠️ DON'T** edit or run `app.py`. It is not the program you want; the real app is `translator_gui.py`.

---

## 3. Your very first translation (the 7-step golden path)

Do exactly this the first time. Don't touch anything not mentioned. You'll have a translated book in a few minutes (plus however long the AI takes).

1. **Pick your file.** Next to **Input File(s)**, click **🔍 Browse ▼** and choose your `.epub` (or `.txt` / `.pdf`).
2. **Choose a Profile.** In the **Profile** dropdown pick the *source* language of the book — `Korean`, `Japanese`, or `Chinese`. (The profile is just instructions for the AI; it works the same whether your input is an EPUB, PDF, or plain `.txt`.)
3. **Pick a Model.** In the **Model** box choose or type a model, e.g. `gemini-3.5-flash` (cheap and fast to start with).
4. **Paste your API key.** In the **API Key** box, paste the key from your AI provider. Click **Show** to make sure you pasted it correctly, then **Hide** again.
5. **Set the Target Language.** Type the language you want to *read*, e.g. `English`.
6. **Press `Run Translation`.** The big start button. Watch the log at the bottom — it tells you exactly what's happening.
7. **Wait, then open the output.** When it finishes, click **Open Output Folder 📁**. Your translated chapters (and a finished EPUB, if you let it build one) are there.

> **✅ DO start small and cheap.** For your first run, set **Chapter range** to `1-2` (see [Section 6](#6-main-settings-the-middle-of-the-window)) so you only translate two chapters. Check they look good *before* spending money on the whole book.

> **🛑 The Stop button is safe.** `Stop Translation` finishes the current piece and stops cleanly. Your already-translated chapters are saved. You can resume later and it skips what's done.

---

## 4. The main window, explained piece by piece

The top of the window is the control strip you'll use every time.

- **Input File(s):** The book(s) or folder you want to translate. Click **🔍 Browse ▼** to pick a file or a whole folder. It shows **No file selected** until you choose something.
- **Model:** The AI brain. **The Model box is NOT a fixed dropdown — it's a free text field.** You can pick a suggestion *or just type any model name yourself*. Any model a routing prefix supports will work if you type it in (e.g. `or/deepseek/deepseek-v4-flash:free`, `nd/moonshotai/kimi-k2-thinking`, `authnd/z-ai/glm-5.1`). Next to it:
  - **Manage Models** (gear/list button) — opens the **Model Manager**, where you add your own models and **custom prefixes** (see [Section 9](#9-local-ai--custom-endpoints--the-3-methods)).
  - **ℹ️** — "Show API provider information and shortcuts," a quick reference of which models belong to which provider.
- **Login buttons (🔐):** If you'd rather use a subscription than an API key, you can log in with your browser:
  - **🔐 ChatGPT Login** — "Log in with your ChatGPT Plus/Pro subscription via browser. No API key needed."
  - **🔐 Claude Login** — "Log in with your Claude Pro/Max subscription via browser. No API key needed."
  - **🔐 Gemini Login** — "Log in with your Google account via browser. No API key needed – uses your Google Cloud project."
  - Each has an account-slot dropdown so you can keep several accounts.
- **Profile:** The instruction set (language + rules). Use the dropdown to switch.
  - **+ New Profile** — make a blank one.
  - **Save Profile** — save your edits to the system prompt.
  - **Delete Profile** — remove a custom profile.
  - **Manage Profiles** — reorder, rename, undo/redo, move up/down/top/bottom.
  - **Name tags:** putting **`_beautifulsoup`** or **`_html2text`** in a profile's name forces that extraction mode (see [Section 7](#7-other-settings-the-️-other-setting-button)).
- **API Key:** Your secret password for the chosen provider. **Show** reveals it; **Hide** masks it again.
- **Target Language:** The language you want to read (e.g., `English`). This fills in the `{target_lang}` blank inside the AI's instructions.
- **Run Translation:** Start. While running it changes to **Stop Translation**.
- **Open Output Folder 📁:** Jump straight to your translated files.
- **📚 Library:** Opens the EPUB Library browser to read/manage your translated files.
- **⚙️ Other Setting:** Opens the big advanced-settings window ([Section 7](#7-other-settings-the-️-other-setting-button)).
- **Save Config:** "Save all settings to config.json." Click this after you set things up the way you like, so they're remembered next time.

There are also buttons to open the other tools — **Manga**, **Glossary**, **EPUB Converter**, **QA Scan**, **Retranslation**, **Multi Key Manager**, **Progress Manager** — each covered in its own section below.

> **✅ DO press `Save Config` once you're happy with your setup.** Otherwise you may have to re-enter settings next launch.

---

## 5. Getting an API key and choosing a model

Glossarion doesn't translate by itself — it sends your text to an AI company and reads back the translation. So you need **one** of these:

### Option 1 — A normal API key (simplest to understand)

1. Make an account with a provider (OpenAI, Google AI Studio, Anthropic, DeepSeek, etc.).
2. Find their "API keys" page and create a key.
3. Copy it and paste it into Glossarion's **API Key** box.
4. Type or pick that provider's model in the **Model** box.

**How Glossarion knows which provider you mean:** it looks at the **start of the model name**. Examples:

| If the model starts with… | It goes to… |
|---|---|
| `gpt-`, `o3` | OpenAI |
| `gemini-` | Google Gemini |
| `claude-` | Anthropic Claude |
| `deepseek-` | DeepSeek |
| `grok-` | xAI |
| `eh/...` | ElectronHub (one key, many providers) |
| `or/...` | OpenRouter |
| `authnd/...` | AuthND (browser/token routing — needs the EPUB Library build) |

> Glossarion supports **40+ providers**. If yours isn't obvious, open **Manage Models → ℹ️ Model Provider Information** for the full list and the exact prefixes.

### Option 2 — Log in with a subscription (no key)

If you already pay for **ChatGPT Plus/Pro**, **Claude Pro/Max**, or have a **Google** account, use the **🔐 login buttons** on the main window instead of an API key. A browser window opens, you log in once, and Glossarion uses that. Pick the matching `authgpt/...`, `authcd/...`, or `authgem/...` model.

### Which model should a beginner pick?

- **Cheapest / fastest to try:** `gemini-3.5-flash` or `deepseek-v4-flash`.
- **Best quality for tricky novels:** `gpt-5.5`, `claude-opus-4-8`, `gemini-3-pro`, or `deepseek-v4-pro`.
- Start cheap, translate 1–2 chapters, and only move to a pricier model if the quality isn't good enough.

> **💸 Don't want to pay anything at all?** You don't have to. See **[Section 17 — How to translate for free](#17-how-to-translate-for-free)** for every no-cost option (free NVIDIA routing, free Google AI Studio keys, free OpenRouter models, your own local AI, and more).

> **⚠️ DON'T paste a key from one company and a model from another.** An OpenAI key with a `claude-` model will fail. Key and model must match the same provider (the multi-provider hubs `eh/`, `or/` are the exception — one key, many models).

---

## 6. Main settings (the middle of the window)

These are the everyday knobs. **The defaults are sensible — you can ignore most of these at first.** Here's what each does, with the safe value.

- **Threading delay (s)** — tiny pause between internal tasks. **Leave at `0.5`.**
- **API call delay (s)** — pause between requests to the AI. **Increase this (e.g., to `2` or more) if you get "rate limit" / "429" errors.** This is the #1 fix for "too many requests."
- **Chapter range** — which chapters to do. `1-10` = chapters 1 through 10. `5,7,9` = just those three. **Empty = the whole book.** Use `1-2` for a cheap test run.
- **Input Token limit** — the most text sent to the AI at once. There's a toggle button:
  - **Enable Input Token Limit** (green) sets a safe cap (e.g., `200000`). **Recommended.**
  - **Disable Input Token Limit** (red) removes the cap — risky for giant chapters.
- **Output Token limit** — "Maximum tokens the model may generate in responses." If the AI keeps cutting off mid-sentence, raise this.
- **Context Mode** — how much the AI "remembers" from earlier chapters (huge for consistency):
  - **Off** — translate each request with no memory of previous chapters (cheapest, least consistent).
  - **Contextual History** — include recent translated chapters as conversation history (good consistency).
  - **Rolling Summary (Replace)** — keep one running summary that gets updated.
  - **Rolling Summary (Append)** — keep adding summaries and retain the most recent ones.
- **Temperature** — AI creativity. "Lower (0.1–0.3) = literal/stable. Higher (0.7+) = creative/random." **For faithful translation, keep it low (around `0.3`).**
- **Batch Translation** — send several chapters in parallel to go faster.
  - **Batch Size** — how many at once (`0` = simultaneous handling). Bigger = faster but more likely to hit rate limits and costs add up quicker.

> **✅ The beginner-safe recipe:** Threading delay `0.5`, API call delay `2`, Input Token Limit **enabled** at `200000`, Context Mode **Contextual History**, Temperature `0.3`, Batch off until you're comfortable.

> **⚠️ DON'T crank Batch Size high on your first book.** If something's wrong with your settings, a big batch just means a bigger, more expensive mistake.

---

## 7. Other Settings (the ⚙️ Other Setting button)

Click **⚙️ Other Setting** to open the advanced window. **You can translate a whole book without ever touching this.** Open it only when you want to fix a specific problem. Settings are grouped; here are the ones that matter, by group.

### Context, memory & summaries

- **Include previous source text in history** — also send the *original* foreign text back as context. **⚠️ Not recommended** — "the translated text is already included; enabling this hurts token efficiency" (costs more for little gain).
- **Rolling Summary settings** — when you use a Rolling Summary context mode, these control how often it summarizes, how it's sent (`system` is best), and how many summaries to keep. **Configure Memory Prompts** lets you edit the summary instructions.

### Response handling & retries (these save failed runs)

- **Auto-retry Truncated Responses** — "Auto-retry logic for clear cases of truncation the API doesn't report" (when the AI silently cuts off). **✅ Keep on.**
- **Preserve Original Text on Failure** — if a translation fails, save the original foreign text wrapped in a failure marker instead of an empty/blocked result. Lets you find and fix it later.
- **Save partial/stopped chapters as QA-failed** — chapters you interrupt with Stop are saved and marked failed (instead of vanishing), so you can retry just them.
- **Streaming** — "Streams tokens as they're generated to reduce wait time. Some providers (e.g. Google Gemini) may truncate streams silently — turn streaming off if you see incomplete output."
- **Show thinking logs** — show the AI's 🧠 reasoning in the log for models that support it.

### Reasoning / "thinking" models

- **GPT-5 / OR Thinking, Gemini Thinking Mode, DeepSeek Thinking Mode** — turn on a model's deep-reasoning mode. **Effort/Budget** controls how hard it thinks (higher = better but slower & pricier; Gemini budget `-1` = automatic).
- **Force adaptive thinking (Claude)** — "⚠️ Not all models support this; enabling it on unsupported models causes API errors." Leave off unless you know your model supports it.

### Extraction mode (how the book's text is pulled out)

- **BS4 (Traditional)** — keeps all the HTML formatting. Best for layout, but uses more tokens.
- **Text (html2text)** — strips formatting to clean text (cheaper, cleaner), then re-adds tags at the end. Good for token savings when exact layout doesn't matter.
- **Render PDF as HTML** — for PDF inputs, convert pages to HTML text before translating.

> **⚠️ The profile name can override this setting.** Profile names carry case-insensitive tags that **hardcode** the extraction mode, ignoring the choice above:
> - a profile whose name contains **`_beautifulsoup`** (e.g. `Korean_BeautifulSoup`) forces **BS4** extraction;
> - a profile whose name contains **`_html2text`** (e.g. `Japanese_html2text`) forces **html2text** extraction.
>
> When you press Run, Glossarion auto-switches and logs `🔄 Auto-switched to … extraction (profile: …)`. So if extraction won't change no matter what you pick here, **check your profile's name** — the tag is winning. (To control extraction with this dropdown instead, use a profile name *without* either tag.)

### Image & PDF output

- **Process Web Novel Images** — also translate text found *inside* the book's images.
- **Create PDF after EPUB** — additionally build a PDF (with images, optional table of contents, page numbers).
- **Image compression / WebP / quality** — shrink images in the output to make smaller files.

### Anti-duplicate (AI Hunter) — a prose-quality & consistency tool

This is **not** mainly about catching an "AI bug." It's a quality tool: it compares each new chapter's output against others and, when two come back too similar, re-runs the chapter so you get **better prose and more consistent results** across the book.

- **Auto-retry Duplicate Content** — when output is too similar to other chapters, retry to improve it. **Detection Method:** `Basic` (fast) or `AI Hunter` (smarter ML-based similarity, slower).
- **Lookback chapters, Similarity Threshold** — how far back to compare and how similar counts as a match (`0.85` = 85% similar).

> **✅ DO use "Save Config" inside Other Settings too**, or your changes here may not stick.

> **⚠️ DON'T turn on every toggle "just in case."** Many options trade money or speed for a niche benefit. Change one thing at a time and re-test.

---

## 8. The Glossary system (read this — it matters a lot)

**What a glossary does:** it's a name list that forces the AI to translate names and terms the *same way every time*. Without it, the same character can be "Seo-yeon," "Seoyeon," and "Lady Seo" across three chapters. With it, she's always "Seo-yeon." For any book longer than a few chapters, **a glossary is the single biggest quality upgrade you can make.**

A glossary is just a table (saved as `.csv` or `.json`) with columns like *raw name* (the original), *translated name* (what it should become), *gender*, and optional *description*.

### 8.1 The Glossary Mode dropdown — the master switch

On the main window there's a **Glossary Mode** dropdown (also mirrored inside the glossary settings). It has **eight** choices. This one dropdown decides *whether* and *how* Glossarion builds and uses a glossary while translating. Here is exactly what each does:

| Mode | What happens | When to use it |
|------|--------------|----------------|
| **Off** | **No** automatic glossary extraction during translation, **but Auto-Mapping is turned ON.** | You built your glossary yourself ahead of time and want Glossarion to auto-attach it by filename. **(See 8.2 — this is the workflow you'll likely want.)** |
| **Off (Fuzzy Mapping)** | Same as Off, but Auto-Mapping also matches *similar* filenames, not just exact ones. | Same as above, but your glossary file's name doesn't exactly match the book's name. |
| **Manual Glossary Only** | No extraction **and Auto-Mapping is OFF.** You must load a glossary by hand with the editor's **Load Glossary** button. | You want full manual control over which glossary file is used. |
| **No Glossary** | Translate with no glossary at all. (Doesn't change your toggle.) | Quick throwaway translations where consistency doesn't matter. |
| **Minimal** | Lightweight extraction *during* translation, run **in-process by `GlossaryManager.py`**. | You want some consistency without much extra cost or setup. |
| **Balanced** ✅ | Smarter extraction using the **same "Extract Glossary" engine**, with request-merging and chapter-splitting, plus **hardcoded settings that push the output-token limit as high as possible** to fit more entries per request. **Recommended of the auto modes.** | A hands-off all-rounder. |
| **Full** | Same **"Extract Glossary" engine** as Balanced, but chapter-by-chapter for maximum context. **Most expensive.** | Big, name-heavy series where you want maximum context and don't mind the cost. |
| **Single Pass** | Extract the glossary inline during each translation request. | Advanced; combines extraction and translation in one step. |

> **Two different engines (worth understanding):**
> - **Minimal** uses the **in-process `GlossaryManager.py`** pipeline — light, fast, runs quietly during translation.
> - **Balanced** and **Full** use the **same engine as the "Extract Glossary" button** (the full extraction pipeline) with request-merging and chapter-splitting. **Balanced** additionally ships with **hardcoded settings that maximize the output token limit**, so each request asks the model to return as many entries as it can.

> **✅ If you just want it to work:** choose **Balanced**. Glossarion will build and apply a glossary as it goes.

### 8.2 ⭐ The recommended power-user workflow: "Off" + Auto-Mapping + manual *Extract Glossary*

This is the workflow many experienced users prefer, and the one you specifically want to understand. The idea: **build the glossary once, by hand, before translating — then let Glossarion attach it automatically.** This gives you a clean, reviewed name list and avoids paying to re-extract names on every run.

**Step by step:**

1. **Set Glossary Mode to `Off`.** Counter-intuitive, but remember: `Off` means *"don't auto-extract during translation, but DO keep Auto-Mapping turned on."* That second half is the important part.
2. **Build the glossary yourself first.** Click **Extract Glossary** (or open the **Glossary Manager**). This scans your book *now* and produces a glossary file. Review and clean it (fix any wrong romanizations, merge duplicates, delete junk). See 8.4 for the manager.
3. **Name the glossary so it matches your book.** Auto-Mapping pairs files by name. If your book is `MyNovel.epub`, name the glossary something like `MyNovel_glossary.csv`. Glossarion creates these names for you automatically when you extract, so usually you don't have to rename anything.
4. **Make sure "Append Glossary to System Prompt" is on.** This is what actually *sends* the glossary to the AI: *"Send the current glossary to the model with every request. Improves consistency across chapters."* When this is on **and** Auto-Mapping is on, Glossarion **auto-fills the glossary for you if none is loaded.**
5. **Press Run Translation.** Because Mode is `Off`, Glossarion does **not** waste money re-extracting names mid-translation. Because Auto-Mapping is on, it **finds and attaches your prepared glossary by filename automatically.** Best of both worlds.

> **Why not just use Balanced?** The real catch is **AI output bias**: language models tend to return *short* outputs no matter how much input you give them. Even though Balanced cranks the output-token limit to the maximum, the model will still under-produce — so a single auto-extraction pass often **misses names** and leaves your glossary incomplete. The "Off + pre-built glossary" workflow sidesteps this: you extract and **review the names once** until the list is actually complete, then reuse that clean, finished glossary on every future run (re-translations, refinements, sequels) for free.

> **If the filenames don't line up**, switch to **`Off (Fuzzy Mapping)`** so similar names match (e.g. `MyNovel_v2.epub` → `MyNovel_glossary.csv`). There's a similarity slider — drag it toward strict if it's grabbing the wrong file, toward loose if it's missing the right one.

> **To force one exact file instead of auto-matching:** set Mode to **`Manual Glossary Only`**, open the editor, browse to your glossary, and click **Load Glossary**. *(The Load Glossary button only takes effect in Manual Glossary Only mode.)* When active, that glossary is also copied into the book's output folder as `glossary.csv`.

### 8.3 Auto-Mapping vs. Auto-Fill (don't mix these up)

- **Auto-Mapping** = automatically choosing *which glossary file* goes with *which book*, by filename. Controlled by the modes above.
- **Auto-Fill / Map Glossaries to EPUBs** = the button you click to do that matching manually if you didn't let it happen automatically.
- **Fuzzy** = allow *close* name matches, not just exact ones.

### 8.4 The Glossary Manager (three tabs)

Open it with **Glossary Manager** / **Extract Glossary**.

**Tab 1 — Manual Extraction (build a master glossary before translating):**
- **Entry Type Configuration** — what to look for: Characters, Terms, and your own **Custom Types** (e.g. "Skill," "Location").
- **Custom Fields** — extra columns (e.g. "Description," "Age").
- **Duplicate Detection** — merges names that are really the same person:
  - **Algorithm:** `Auto` (recommended, tries everything), `Strict` (only near-identical, 95%+), `Balanced` (handles word order like "Park Ji-sung" = "Ji-sung Park"), `Aggressive` (catches "Catherine" vs "Katherine" but may over-merge), `Basic Only` (fast, simple).
  - **Fuzzy Matching Threshold:** "0.90 = very similar (recommended); 1.0 = exact match." Slide right for stricter.
  - **Disable honorifics filtering** — if checked, "Kim-nim" and "Kim" are treated as *different* people.
- **Temperature / Context Limit / Merge Count** — keep temperature low for accurate name-pulling.

**Tab 2 — Automatic Generation (the glossary built *during* translation):**
- **Append Glossary to System Prompt** — sends the terms to the AI every request (the consistency switch).
- **Compress Glossary Prompt** ✅ — "Only send glossary entries that appear in the current text. Saves tokens and cost; recommended ON."
- **Add Additional Glossary** — always include an extra external glossary file (CSV/JSON/TXT/PDF/MD).
- **Include Gender Context** — expands snippets with surrounding sentences so the AI can infer each character's gender (costs more; it's the master switch that unlocks the gender-nuance and description options).
- **Enable Gender Nuance Analysis** — an extra pronoun/honorific-aware scoring pass that prioritizes sentences which reveal gender (slightly higher CPU/time).
- **Include Description Column** — adds a description/context field to every entry (only available while Gender Context is on).
- **Disable Smart Filtering** — "Bypass all filtering and send the entire novel to the extractor. Extremely expensive; debugging only." **⚠️ Leave OFF.**

#### The gender tracker (`*_gender_tracker.json`)

When gender features are on, Glossarion keeps a small sidecar file next to your glossary named **`<glossary>_gender_tracker.json`**. It remembers what gender was decided for each name across chapters so the choice stays consistent, and it powers the **Male/Female dedupe protection** — that protection stops two same-spelled names of *different* genders (e.g. a male "Yuki" and a female "Yuki") from being merged into one entry by mistake.

- There's a setting to **not create/use the gender tracker** ("Do not create or use the `*_gender_tracker.json` sidecar"). Turning that on also **disables the Male/Female dedupe protection**, so same-name entries will dedupe normally.
- A **gender-bias preference** (No Bias / Prefer Female / Prefer Male) controls whether rare gender readings get suppressed: *No Bias* suppresses a rare variant below the flip threshold; *Prefer Female/Male* never suppresses that side's rare readings.

> **✅ Leave the gender tracker ON for character-heavy novels** — it's the thing that keeps a character's gender (and pronouns) consistent from chapter 1 to the end.

**Tab 3 — Glossary Editor (view & clean a file):**
- **Load / Browse** a `.csv` or `.json`.
- **Clean Empty Fields** — drop empty columns.
- **Remove Duplicates** — run the merge algorithm on the loaded file.
- **Load Glossary** — set the browsed file as the active manual glossary (only works in *Manual Glossary Only* mode).

> **✅ DO review your glossary before a big run.** Five minutes fixing names now saves hours of inconsistent output later.

> **⚠️ DON'T enable "Disable Smart Filtering"** unless you're debugging — it sends the *entire book* to the extractor and can be extremely expensive.

> **There's a backup safety net:** use **Restore Glossary** on the main window to bring back the most recent glossary backup if you mess one up.

---

## 9. Local AI / custom endpoints — the 3 methods

You can run Glossarion against a **local AI** on your own computer (free, private, no internet) using tools like **Ollama** or **LM Studio**, or against **any other OpenAI-compatible server**. There are **three different ways** to point Glossarion at such an endpoint. They exist for different situations and they have a clear pecking order. Here they are, simplest to most flexible.

> **First, what's an "endpoint"?** It's just the web address where the AI lives. A cloud provider's endpoint is on the internet. A local AI's endpoint is on your own machine, usually:
> - **Ollama:** `http://localhost:11434/v1`
> - **LM Studio:** `http://localhost:1234/v1`
> - **vLLM:** `http://localhost:8000/v1`

### Method 1 — Custom OpenAI Endpoint (GLOBAL — affects everything)

**What it is:** one address that *all* translation requests use. Set it once and the whole app talks to your local/custom server.

**How to set it:**
1. Open **⚙️ Other Setting → Custom API Endpoints**.
2. Check **Enable Custom OpenAI Endpoint**.
3. In **Override API Endpoint**, type your URL (or **double-click the Ollama / LM Studio shortcut** to fill it in).
4. In the main window's **Model** box, type your local model's exact name (e.g. `llama3`, `mistral:instruct`).
5. For **API Key**, a local server usually accepts any dummy text (e.g. `sk-local`); a private server uses its real key.

**Use it when:** you do *all* your translating on one local model. Simple and global.

### Method 2 — Individual (per-key) Endpoint (applies to ONE key only)

**What it is:** an endpoint attached to a *single API key* inside the **Multi API Key Manager**. Only requests using *that key* go to *that endpoint*. This requires **Multi-Key Mode** (Section 14).

**How to set it:**
1. Open the **Multi API Key Manager** (the **Multi Key Manager** button, or Other Settings → Configure API Keys).
2. Add or select a key, then open **Configure Individual Endpoint** for that key.
3. Tick **Enable**, and type the **Endpoint Base URL** (again, Ollama/LM Studio shortcuts are right there; Azure OpenAI is also supported, with an **Azure API Version** picker).
4. **Save.**

**Use it when:** you want to mix sources in one run — e.g. one key pointing at a local Ollama model *and* another key pointing at a cloud provider, rotating between them.

> **⭐ Precedence rule (important):** A key's **Individual Endpoint OVERRIDES the global Custom OpenAI Endpoint.** So if Method 1 is set globally but a particular key has its own Method-2 endpoint enabled, that key uses *its own* endpoint and ignores the global one. (For any key *without* an individual endpoint, the global one still applies.)

### Method 3 — Custom Prefix routing (route ANY OpenAI-compatible model)

**What it is:** you invent a short **prefix** and tell Glossarion which server it points to. Then any model name you type starting with that prefix is routed there. This is the most flexible — you can have several different OpenAI-compatible servers available at once, each with its own prefix.

**How to set it:**
1. Open **Manage Models** (the Model Manager).
2. Find the **Custom prefixes** table (columns: **Prefix**, **Base URL**, **Endpoint Type**) and click **Add Prefix**.
   - **Prefix** — your made-up label, e.g. `mylocal`.
   - **Base URL** — the provider root (the server address).
   - **Endpoint Type** — the "path shape" (which must be an absolute path like `/v1/chat/completions`).
3. Now type a model in the form **`prefix/model-name`** — e.g. `mylocal/llama3` — in the main **Model** box. Glossarion routes it to that server.

**Use it when:** you regularly switch between several custom/OpenAI-compatible endpoints and want to pick one just by typing a prefix, without changing any global setting.

### Which method should I use? (and who wins)

| You want… | Use | Notes |
|-----------|-----|-------|
| One local model for everything | **Method 1** (global) | Simplest. |
| Different endpoints per key, rotating | **Method 2** (per-key) | Needs Multi-Key Mode. |
| Switch endpoints by typing a prefix | **Method 3** (prefix) | Most flexible. |

**Order of precedence when more than one applies:**

> **Individual per-key endpoint (Method 2)  ▶  beats  ▶  Global Custom OpenAI Endpoint (Method 1).**
>
> **Custom prefixes (Method 3)** are chosen by the *model name* you type (`prefix/model`), so they route independently — if your model name carries a known prefix, that prefix's endpoint is used.

> **⚠️ DON'T forget Method 2 needs Multi-Key Mode turned on.** A per-key endpoint does nothing if you're in single-key mode.

> **✅ DO test a local model with a 1-chapter run first.** Local models vary a lot in quality; confirm output is readable before translating a whole book.

---

## 10. Manga & comic translation

*(Available in NoCuda/Heavy builds — not in Lite.)* This panel translates comics, manga, and manhwa images.

1. **Open the Manga Translator** — click **Open the manga panel translator** on the main window. (First open shows "Loading Manga Translator…" — wait for it.)
2. **Load images** — pick a folder of images or a `.cbz`/`.zip`/`.epub`.
3. **Settings (left panel):**
   - **Translator** — the AI model that does the translating.
   - **OCR Provider** — the tool that *reads* text off the image:
     - `manga-ocr` — best for Japanese.
     - `rapidocr` / `PaddleOCR` / `EasyOCR` — fast, general, run locally.
     - `google-vision` / `azure` — cloud OCR (need their own credentials — set the JSON key path; use **GCloud Creds** button).
   - **Source Language** — the language of the manga.
4. **The translation pipeline (buttons, in order):**
   - **Detect Text** — find the speech bubbles.
   - **OCR** — read the text inside them.
   - **Translate** — translate that text.
   - **Inpaint** — erase the original text from the bubbles.
   - **Render Text** — draw the translation into the bubbles.
   - **Batch Translation** — do the whole folder automatically, start to finish.
5. **Inpainting (erasing original text):**
   - **Skip inpainting** — "render translated text over the original image" (no erasing).
   - **Local inpainting** — runs on your machine (recommended: pick `anime_onnx`).
   - **Cloud (Replicate)** — "Not recommended; performed poorly in tests."
6. **Mask expansion / safe area** — how far past the detected text to erase. `0%` = exact text only; higher = more breathing room around text.

> **Manual editing & preview:** There's a dual-viewer preview (📄 Source / ✨ Translated Output) with drawing tools. **Manual editing is OFF by default for safety** — turn on **Enable Manual Editing** to draw/adjust bubbles yourself. You can cycle a view between translated / cleaned / original by clicking it.

> **⚠️ Cloud OCR (Google/Azure) costs money and needs a credentials file.** Local OCR (`manga-ocr`, `rapidocr`) is free and fine for most jobs — start there.

---

## 11. Retranslation (fixing single chapters)

Don't redo a whole 200-chapter book because three chapters came out badly. Use **Retranslation**.

1. **Open the Retranslation tab** and **Select Folder** — pick your translated output folder.
2. **Read the status colors:**
   - ✅ Green = success
   - ❌ Red = failed (or failed a QA check)
   - ⬜ White = not translated yet
   - 🔄 Spinning = in progress
3. **Tick the chapters** you want to redo.
4. **Retranslate Selected** — deletes the old version and translates those again. **Refresh List** updates the icons.

> Handy extras here: **Show skipped files** reveals files the pipeline normally skips; you can also switch the second column to show **which model** was used per chapter, or view **glossary extraction progress**.

---

## 12. EPUB Converter (building your finished book)

After translating, this stitches your translated HTML chapters into a real `.epub` you can read on any e-reader.

1. **Open EPUB Converter** and select the folder of translated HTML files.
2. **Options worth knowing:**
   - **Attach CSS** — keep the book's styling (fonts, margins). Usually leave on.
   - **Force NCX only** — make an old-style, maximally-compatible table of contents. Turn on only if your e-reader chokes on modern EPUBs.
   - **EPUB structure (Auto / EPUB2 / EPUB3)** — leave on **Auto** unless you have a reason; it copies the source layout.
   - **Retain source extension** — keep `.xhtml` if the original used it (helps compatibility).
   - **Bundle fonts** — embed `.ttf`/`.otf`/`.woff` font files into the EPUB.
3. **Convert** — produces `[YourBook]_translated.epub` in the folder.

> **If conversion fails:** run **Validate EPUB Structure** (in Other Settings) — it checks for missing pieces and tells you what's wrong.

---

## 13. QA Scanner (checking quality)

Scans finished translations for problems and produces a clickable report.

1. **Pick a mode:** `Quick` (fast), `Strict` (thorough), or `Custom` (choose rules).
2. **Point it at your files:** the **Source** (original) and the **Translated Folder** (your output). **Auto-search** tries to find the folder for you.
3. **It checks for:**
   - **Word-count mismatch** — did the AI skip a big chunk?
   - **Broken HTML** — broken tags.
   - **Untranslated text** — leftover Korean/Japanese/Chinese.
   - **Duplicates** — accidentally repeated chapters.
4. **Run Scan** — makes `qa_report.html`. Anything that fails is marked ❌ in the Retranslation tab, so your fix-it list is ready.

> **The natural workflow:** Translate → **QA Scan** → open the Retranslation tab → redo the ❌ chapters → QA Scan again.

---

## 14. Multiple API keys (rotation)

If you hit rate limits a lot, or have several keys, let Glossarion juggle them.

1. Open **Other Settings → Configure API Keys** (or the **Multi Key Manager** button).
2. **Add Key** — enter each key and pick its provider/model.
3. **Enable Multi-Key Mode** — tick the box.
4. **What it does:** rotates through your keys; if one hits a limit (error 429), it instantly switches to the next. You can set fallback keys for specific models, and (as in Section 9) give individual keys their own endpoints.

> **Remember:** the per-key **Individual Endpoint** feature (Section 9, Method 2) lives here and **only works when Multi-Key Mode is on.**

---

## 15. Refinement & output modes

Two newer features worth knowing about in 9.3.3.

### Output Mode (what kind of output you want)

A small selector (also synced into Other Settings → Image Translation) chooses what Glossarion produces:

- **📝 Text** — normal text translation (the default). Most users stay here.
- **👁️ Vision** — **OCR mode.** Reads text *out of images*. Point it at loose images and it OCRs them to text. **Point it at an EPUB and it OCRs every image inside the EPUB and then translates that OCR'd text**, giving you a translated EPUB built from the image text. You can optionally set it to return **only the raw OCR'd text as an EPUB** (no translation). Fine-tune all of this in **Other Settings → the Output Mode section**.
- **🖼️ Image** — **direct image translation / editing.** Use it to translate standalone images, or feed it an **EPUB** and it returns **the same raw EPUB but with every image re-drawn by an image-edit model** (e.g. Nano Banana 2) — handy for translating text baked into illustrations.
- **🎬 Video** — video-generation output.
- **🔊 Audio** — generate text-to-speech audio from your translated text.
- **✨ Refinement** — improve *existing* translated output without using the raw source again.

> **✅ Quick rule:** **Vision** = "turn pictures into translated words." **Image** = "give me the book back with the pictures themselves translated." Detailed knobs for both live in **Other Settings → Output Mode**.

### Refinement (a polish pass after translating)

Turn on a second pass that re-reads and improves your finished translation:

- **Full** — refine *all* translated chapters.
- **Failed** — run a quick QA scan first, then only refine the chapters still marked failed.
- **Partial / Partial.b / Partial.b2** — increasingly targeted/batched passes that fix only the specific problem spots (e.g. chapters with leftover foreign characters), bundling them efficiently to save requests.

> **✅ DO try `Failed` refinement** as a cheap cleanup: it only spends money on the chapters that actually need help.

### Review generator

**Generate Review** produces an AI-written summary/review of a selected EPUB. Options include reviewing the **whole book in chunks** (works around token limits) and **Full Review mode** (includes both the first *and* last chapters, 50/50, for a complete picture instead of just the opening). A ✓ shows when a review already exists for that book.

> **⚠️ You must enable an Input Token Limit to use the Review Generator.** Turn on **Enable Input Token Limit** (Section 6) first — with the limit disabled, the review won't run. Set a sensible cap (e.g. `200000`) and try again.

---

## 16. Translation editing & review (.sdlxliff and supported formats)

Besides translating from scratch, Glossarion can **edit and review existing translations**, including projects from professional CAT tools.

### What files can Glossarion translate?

| Format | Notes |
|--------|-------|
| **EPUB** (`.epub`) | The main, best-supported format. Structure, cover, and images preserved. |
| **TXT** (`.txt`, `.md`) | Plain text with chapter detection. (Any profile works — there's no special "text profile.") |
| **PDF** (`.pdf`) | Text extracted automatically; can be re-rendered to a PDF afterward. |
| **HTML** | Translate/scan folders of HTML chapters. |
| **CBZ / images** | Manga & comics (in the `N_` builds). |
| **SDLXLIFF** (`.sdlxliff`) | Professional bilingual translation files (see below). |

### The SDLXLIFF reviewer / editor

`.sdlxliff` is the bilingual file format used by professional CAT tools like SDL Trados Studio. Glossarion includes a **source-to-output reviewer workflow** for these files (contributed by community member OMORIO). It lets you translate the file's text units and then **check the result side by side** before delivering.

What it gives you:

- **Side-by-side source vs. output review.** Open the **Review source/output text analysis** view to compare each segment. If needed, Glossarion **generates SDLXLIFF sidecar files from your completed entries** automatically.
- **Machine-translation previews.** Generate a quick machine-translation of each source row to sanity-check meaning. **Right-click the button to choose which provider** does the preview.
- **Accuracy flagging.** **Mark rows that fall below a machine-translation accuracy threshold**, so you can jump straight to the segments most likely to be wrong.
- **Structure checks.** Glossarion compares the source and output `<p>` and `<h1>`–`<h6>` tag counts from the matching SDLXLIFF sidecars and **flags files where those text wrappers were dropped or added** (a common source of broken formatting).
- **Auto-refresh.** Press **F5** (or the refresh button) to re-run the SDLXLIFF check after you've made changes.

> The SDLXLIFF tools live alongside the **Retranslation** view, since both are about reviewing and fixing finished translations rather than producing them from scratch.

> **✅ Use this if you receive `.sdlxliff` files** from a client or CAT tool and want Glossarion to translate and review them while keeping the original segment structure intact.

---

## 17. How to translate for free

Yes — you can run Glossarion **without spending a cent.** There are several free routes, from "totally free, no setup" to "free but you host the AI yourself." Pick whichever fits.

> **🔑 The one thing to remember:** the **Model box is a plain text field, not a locked dropdown.** For every option below you just **type the model name in** (including its routing prefix like `authnd/…` or `or/…`). If the routing supports a model, typing it works — you are not limited to the suggestions in the list.

### The free options at a glance

| Method | What you type in **Model** | What you need | The catch |
|--------|----------------------------|---------------|-----------|
| **NVIDIA browser routing** | `authnd/...` (e.g. `authnd/z-ai/glm-5.1`) | **Nothing** — no key, no login | Routes through the built-in NVIDIA browser; needs the **standard `L_Glossarion` build** (has the EPUB Library). |
| **NVIDIA API (free tier)** | `nd/...` (e.g. `nd/deepseek-ai/deepseek-v4-flash`) | A **free NVIDIA API key** | Free, but **only available in certain regions**. |
| **ChatGPT login** | `authgpt/...` | A **ChatGPT login** (🔐 button) | Only **a few free requests**. |
| **Google AI Studio key (Gemini)** | `gemini-...` (e.g. `gemini-3.1-flash-lite`) | A **free Google AI Studio key** | A few free requests on most models — but **~500 free requests/day when used with Gemini 3.1 Flash Lite**. |
| **Gemini coding endpoint** | `authgem/...` | A **Google login** (🔐 button) | Free, but throttled to **1 request per minute (RPM)**. |
| **OpenRouter free models** | `or/...:free` (e.g. `or/deepseek/deepseek-v4-flash:free`) | A **free OpenRouter key** | Limited to OpenRouter's **free-tier models** (the ones ending in `:free`). |
| **Google Translate (free)** | `google-translate-free` | **Nothing** | It's plain **machine translation**, not an AI — fast and free, but lower quality / no context. |
| **Your own local AI** | your model name (e.g. `llama3`) | **LM Studio or Ollama** on your PC | Free and private, but quality and speed depend on your computer. |

### The options explained

**1. `authnd/` — NVIDIA browser routing (easiest free option).**
Type a model like `authnd/z-ai/glm-5.1` and Glossarion routes the request **through NVIDIA's browser** for you. **No API key and no login required.** Because this uses the embedded browser, you need the **standard `Glossarion` build** (the one with the EPUB Library — the `Lite`/`TurboLite` builds remove it). This is the closest thing to "free translation with zero setup."

**2. `nd/` — NVIDIA's free API tier.**
If you create a **free NVIDIA API key**, you can use `nd/...` models (e.g. `nd/deepseek-ai/deepseek-v4-flash`, `nd/moonshotai/kimi-k2-thinking`). It's free, but **availability is limited to specific regions** — if it won't connect, your region may not be supported.

**3. `authgpt/` — free requests via ChatGPT login.**
Click the **🔐 ChatGPT Login** button (Section 4), log in with your browser, and use an `authgpt/...` model. You get **a small number of free requests** this way — handy for testing a chapter or two without an API key.

**4. Free Google AI Studio key (for use with Gemini).**
Go to Google AI Studio, create a **free API key** (this is a *Google AI Studio key*, not a model-specific one), paste it into the API Key box, and use a `gemini-...` model. The free tier gives **a few requests on most models**, but the lightweight **Gemini 3.1 Flash Lite** model allows **around 500 free requests per day** — enough to translate a lot of chapters for free. Type `gemini-3.1-flash-lite` in the Model box.

**5. `authgem/` — Gemini's coding endpoint (free, but slow).**
Log in with the **🔐 Gemini Login** and use an `authgem/...` model. This routes through **Gemini's coding endpoint**, which is free but **capped at 1 request per minute**. Fine for slow background translation; frustrating if you're in a hurry. (Tip: raise your **API call delay** in Section 6 so you don't trip the limit.)

**6. `or/` — free models on OpenRouter.**
Make a **free OpenRouter key** and type an OpenRouter model with the `:free` suffix, e.g. `or/deepseek/deepseek-v4-flash:free`. OpenRouter rotates which models are free, so check their site for the current `:free` list.

**7. `google-translate-free` — free machine translation.**
Type exactly `google-translate-free` in the Model box. This is **classic machine translation** (like the Google Translate website), not an AI model — so it needs no key and costs nothing, but it won't follow your profile/glossary or keep long-range context. Good for a rough, instant draft.

**8. Host your own local AI (totally free, fully private).**
Install **LM Studio** or **Ollama**, download a model, and point Glossarion at it. Nothing leaves your computer and there's no usage cost — the only "price" is your own hardware doing the work. Full setup (all three ways to connect a local model) is in **[Section 9](#9-local-ai--custom-endpoints--the-3-methods)**.

> **✅ Best free starting point:** try **`authnd/`** (zero setup) or a **free Google AI Studio key** used with **Gemini 3.1 Flash Lite** (~500/day). If you have a decent PC and care about privacy, set up a **local model** instead.

> **⚠️ Free tiers are slow or capped on purpose.** Expect rate limits (especially `authgem/` at 1 RPM and `authgpt/`'s few requests). Increase your **API call delay** (Section 6) and translate a small **Chapter range** at a time so you don't burn through a daily quota in one shot.

---

## 18. When things go wrong (troubleshooting)

| Symptom | Most likely cause | Fix |
|---------|------------------|-----|
| **"Rate limit" / error 429** | Sending requests too fast | Raise **API call delay** to `2`+ (Section 6). If you have several keys, enable **Multi-Key Mode** (Section 14). |
| **Output is cut off mid-sentence** | Output token limit too low, or silent truncation | Raise **Output Token limit**; keep **Auto-retry Truncated Responses** on. For Gemini, try turning **Streaming OFF**. |
| **Translation fails / "invalid model" / auth error** | Key and model are from different providers, or wrong key | Make the **Model** prefix match the **API Key**'s provider (Section 5). Re-paste the key and click **Show** to check it. |
| **Names keep changing between chapters** | No glossary, or it isn't being sent | Build a glossary and turn on **Append Glossary to System Prompt** (Section 8). |
| **A character's gender/pronouns flip mid-book** | Gender tracker is off | Turn **Include Gender Context** on and keep the `*_gender_tracker.json` sidecar (Section 8.4). |
| **EPUB won't build** | Missing/broken files in the folder | Run **Validate EPUB Structure** in Other Settings (Section 12). |
| **`authnd/` model won't work** | Using a `Lite`/`TurboLite` build | Those builds drop the EPUB Library and `authnd/` routing — use the standard `L_Glossarion` build (Section 2). |
| **Local model isn't used** | Endpoint not enabled, or wrong precedence | Check the right method in Section 9; remember per-key endpoints need **Multi-Key Mode**. |
| **The window seems frozen during a big job** | It's just working hard | Watch the bottom log — if lines are still appearing, it's fine. **GUI Yield** (Other Settings) reduces freezing. |
| **Settings reset after restart** | Didn't save | Click **Save Config** (Section 4). |

> **The log at the bottom is your best friend.** It says, in plain text, exactly what Glossarion is doing and what failed. Read the last few red lines before asking for help.

> **Still stuck?** Glossarion has a community **Discord** and a **GitHub Issues** page (links are in the project's README). When reporting a problem, **click inside the log, press `Ctrl + A` to select everything, then `Ctrl + C` to copy the *entire* log** — not just the last few lines. The full log usually contains the real cause higher up, so paste the whole thing.

---

## 19. One-page cheat sheet

**To translate a book, every time:**

1. **🔍 Browse ▼** → pick file
2. **Profile** → source language (same profile works for EPUB, PDF, or txt)
3. **Model** → e.g. `gemini-3.5-flash`
4. **API Key** → paste it (or use a 🔐 login)
5. **Target Language** → `English`
6. **Chapter range** → `1-2` for a test, empty for all
7. **Run Translation** → wait → **Open Output Folder 📁**

**Safe starting settings:** API call delay `2` · Input Token Limit **enabled** · Context Mode **Contextual History** · Temperature `0.3` · Batch **off** at first.

**Glossary, the easy way:** Mode **Balanced** = hands-off. Mode **Off** + **Append Glossary** on = use a glossary you built and reviewed yourself (cheaper, cleaner — see 8.2).

**Local AI, pick one:**
- All requests → **Other Settings → Enable Custom OpenAI Endpoint** (global)
- Just one key → **Multi Key Manager → Configure Individual Endpoint** (beats the global one)
- Route by typing `prefix/model` → **Manage Models → Add Prefix**

**Golden rules:**
- ✅ Test 1–2 chapters before committing money to a whole book.
- ✅ Press **Save Config** when you like your setup.
- ✅ Use a **glossary** for anything longer than a few chapters.
- ✅ Want `authnd/` + the EPUB Library? Use the **standard `L_Glossarion` build**, not Lite/TurboLite.
- ⚠️ Match your **model** to your **API key**'s provider.
- ⚠️ Don't run `app.py` — the program is **`translator_gui.py`**.

---

*Made with 🌸 for the translation community. This guide reflects Glossarion v9.3.3 as of June 22, 2026 and is built directly from the in-app tooltips and the program's own code. If a button looks different from this guide, hover it — the live tooltip is always the final word.*

