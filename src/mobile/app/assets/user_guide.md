# Glossarion Mobile — User guide

Glossarion translates novels, documents, subtitles, images and manga with the AI model you choose. The phone app uses the same translation engine, settings and file formats as Glossarion on the desktop, so a workspace, glossary or config made on one opens on the other.

## 1. Getting started

1. **Sign in or add a key.** The default model is ChatGPT (`authgpt/…`): the Welcome guide offers **Sign in with ChatGPT**. Any other provider works with its API key (**Settings › API keys**) or its own sign-in (**Settings › Accounts**: Gemini, Claude, Grok, …).
2. **Pick a model.** Tap the model name in the chat header. Search the list, star favourites, or type any model id and choose **Use “…”** (the desktop model box accepts any id the same way).
3. **Pick the target language and profile** in the same sheet (Profile and Language tabs).

Run **About › Guides › Run the Welcome guide again** at any time.

## 2. Translating in the chat

The home screen is a chat, like the desktop Direct Text window.

- **Type or paste text** and tap **Send**: the reply streams in.
- **Attach a file** with **＋ › Files / Photos / From Library / Clipboard**. EPUB, TXT, PDF, DOCX, HTML, subtitles (SRT, ASS, VTT, LRC), SDLXLIFF, ZIP and images are accepted. An attachment becomes a **job card**: Plan → Queued → Running → Result.
- **Output mode** (the chip in the composer row): Text, Vision, Image, Video, Audio or Refinement.
- The **Result** of a job card shows its output files (EPUB / PDF, `*_translated.txt`, glossary, subtitles, SDLXLIFF): tap one to open, share or save it. **Compile ▾** builds an EPUB or PDF, **Open output** shows the turn's folder in Files, **Read** opens the Reader.
- **Long-press a reply** for Edit translation, Copy as Markdown / HTML / plain text, **Glossary terms used**, Add term to glossary and Delete.
- **Stop** finishes the current request first (graceful stop); **Force stop** cancels at once. A stopped or interrupted job offers **Resume**: it continues from the saved progress.

Long jobs keep running with the screen off (Android shows a notification with Stop). iOS pauses a job shortly after the app leaves the screen; Resume continues it.

## 3. Library and Reader

- **Library** (drawer) lists your books and translation workspaces. Add books with **Import EPUB** (In progress shelf) or **Add translation** (Completed shelf); **Scan for raw** links a workspace to its original file.
- A **Book page** has four tabs: **Overview** (Read, Translate…, Compile, Translate Metadata, Share, Files), **Chapters** (the Progress manager: status per chapter, retranslate, QA results, 🔊 Play audio), **Glossary** (glossary extraction progress and the glossary file) and **Output** (compiled files, workspace files, QA reports, review).
- The **Reader** shows the original, the translation or both side by side. **Aa** sets the font (including fonts loaded with Tools › Compile › Load Font…), size, spacing, theme and layout; page text follows **Aa**, not the system text size. ☰ lists the chapters (a side panel on tablets).

## 4. Glossaries

A glossary keeps names and terms consistent across chapters.

- **Glossary mode** (Settings › Translation defaults, or the chat's glossary chip): Off, Minimal, Balanced, Full, Single Pass, No Glossary and the manual-only modes, exactly as on the desktop.
- **Glossaries** (drawer) opens the Glossary Manager: edit entries, merge, translate, refine, and the Balanced / Full / Minimal / Refinement settings tabs with their prompts and profiles.
- When a translation generates a glossary, the chat asks you to **Edit**, accept (**Yes**) or stop (**No**) before it continues.

## 5. Tools

**Tools** (drawer) holds the Progress manager, Glossary progress, QA scanner, Compile EPUB / PDF, Headers & metadata, Async batch, Review generator, SDLXLIFF reviewer, Manga translator, RPG Maker and the Files browser. The same tools are in the chat's **＋ › Tools**.

## 6. Jobs and background work

**Jobs** (drawer) lists running, queued and finished jobs with their log, progress and files. One job runs at a time; the next one starts when it ends.

## 7. Settings

Settings are grouped like the desktop dialogs (Translation, Models & keys, Glossary, QA, Manga, Reader & Library, Data, About, Advanced). Use **Search** and the chips **Modified · Locked · Advanced · Unavailable on mobile**. Changes apply to the next run. A setting that cannot work on a phone stays visible, disabled, with a reason chip; its desktop value is kept.

## 8. Your files

Outputs are in the app's **Output** folder (**Settings › Storage**; on iOS also in Files › Glossarion, on Android optionally mirrored to Downloads/Glossarion). **Settings › Backup & restore** exports your settings (with or without API keys); **Import from desktop** reads a desktop `config.json`.

## 9. When something goes wrong

- **A request fails:** open the job's **Log** (Jobs › job) — rate limits, sign-in problems and blocked responses are explained there. **Retry failed** retranslates only the failed chapters.
- **Nothing happens on Send:** the Send button names what is missing (sign in, add a key, pick a model).
- **Logs & diagnostics** (Settings, or Help) shows the app log, the environment the next run would get and a self-test you can share.

The desktop user guide (docs/Glossarion_User_Guide in the Glossarion repository) explains every setting in detail; the phone app uses the same names.
