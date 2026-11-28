# 🖼 Interactive Images — User Guide

**Added in Glossarion 9.15.26 · Desktop and mobile**

Interactive Images adds **AI illustrations** and **interactive styles** (status-window tables, glowing system messages, animated level-up lines and so on) to a book you have **already translated**. It is a separate step that runs **after** translation, so it can never break a translation, and you can rerun it for one chapter at a time.

> **How to read this guide:** follow sections 1–4 the first time. Everything after that is reference. **⚠️ DON'T** marks a common mistake; **✅ DO** marks the safe choice.

---

## 📑 Table of Contents

1. [What it does (in one minute)](#1-what-it-does-in-one-minute)
2. [Where to find it](#2-where-to-find-it)
3. [Before you start](#3-before-you-start)
4. [Your first run (the 6-step golden path)](#4-your-first-run-the-6-step-golden-path)
5. [The settings window, explained](#5-the-settings-window-explained)
6. [Reference images (keeping characters consistent)](#6-reference-images-keeping-characters-consistent)
7. [Gender (and how false positives are avoided)](#7-gender-and-how-false-positives-are-avoided)
8. [Prompts, profiles and placeholders](#8-prompts-profiles-and-placeholders)
9. [Interactive styles and the preview panel](#9-interactive-styles-and-the-preview-panel)
10. [The Progress Manager tab](#10-the-progress-manager-tab)
11. [The Gallery (browse and delete images)](#11-the-gallery-browse-and-delete-images)
12. [Batch mode (parallel requests)](#12-batch-mode-parallel-requests)
13. [Keys: the Image Gen/Edit pool](#13-keys-the-image-genedit-pool)
14. [Where the files go](#14-where-the-files-go)
15. [How it works with "Attach CSS"](#15-how-it-works-with-attach-css)
16. [Mobile](#16-mobile)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What it does (in one minute)

For every translated chapter:

1. A **text model** (the "planner") reads the chapter, with its paragraphs numbered, and picks the moments worth illustrating. For each one it says **which paragraph** the picture goes after, **which characters** are in it, and **what the picture shows**. It can also mark paragraphs with style classes (for example a status window).
2. An **image model** draws each picture, using your **reference images** of those characters so they look the same in every chapter.
3. Glossarion puts each picture **right after its paragraph** and adds the style classes.
4. Glossarion then **checks that the chapter's text is exactly the same as before**. If anything in the text would change, that chapter is rejected and left untouched. Only pictures and classes are ever added.

The image model never sees your HTML, and the planner never rewrites your chapter.

---

## 2. Where to find it

**Desktop:**

- **⚙️ Other Setting → EPUB section**, right under *Attach CSS to Chapters*:
  - **Interactive styles** checkbox, the **style dropdown**, **🎨 Styles & Preview…** and its own **Clear**
  - **🖼 Interactive Images…** — the main settings window
- **Progress Manager → 🖼 Interactive Images tab** — build, rebuild, delete and browse (section 10)
- **Progress Manager → 📖 Translation tab** — a **🖼 Interactive Images** button

**Mobile:** **Tools → Interactive Images**, a book's **Images** tab, and **Settings → EPUB output → Interactive images**.

---

## 3. Before you start

✅ **DO** translate the book first. Interactive Images works on finished translated chapters.

✅ **DO** have a glossary for the book. Character names come from it, so the planner uses the same spelling every time. Interactive Images uses **the same glossary translation uses**: the one you loaded with **Load Glossary** in the main window (or the one Glossarion auto-loaded for the book), otherwise the glossary the **Glossary Mode** (Balanced / Full / and so on) wrote. There is nothing to load or pick in Interactive Images itself.

✅ **DO** pick an image model that can take **reference images** (for example Nano Banana / Gemini image models or `authgpt/` models). The settings window warns you when the chosen image model takes none.

⚠️ **DON'T** expect small local text models to plan well on long chapters. The planner must answer in JSON; a weak model may fail some chapters (they are marked failed and can be retried).

💰 **Cost:** each chapter costs **1 planner request + 1 request per image**. A 300-chapter book at 1 image per chapter is about 600 requests. Images are 2–4 MB each, so they also make the EPUB bigger.

---

## 4. Your first run (the 6-step golden path)

1. **Open ⚙️ Other Setting → 🖼 Interactive Images…**
2. **Profile:** leave it on **Default** (1 image per chapter).
3. **Models:** choose the **Text model** (leave empty to use your main translation model) and, if you want, an **Image model** (leave it empty to use the Image Gen/Edit pool model, else the text model, see section 5).
4. **Reference images:** the book's characters are listed for you from the glossary translation uses; add one or two pictures for each main character (section 6). Optional, but strongly recommended.
5. Click **💾 Save**.
6. **Open the Progress Manager → 🖼 Interactive Images tab** and click **▶ Build pending**.

Watch the log. Each finished chapter shows ✅, and the book is recompiled at the end (if *Recompile after* is on), so the new pictures are in your EPUB/PDF.

> **Try it on a few chapters first.** Select a few rows in the tab and use **Build / Rebuild** from the right-click menu.

---

## 5. The settings window, explained

| Setting | What it does | Default |
|---|---|---|
| **Profile** | A saved set of prompts and options (section 8) | Default |
| **Images per chapter** | How many pictures the planner may pick per chapter. **0** = styles only, no image model needed | 1 |
| **Include gender with character names** | Sends a character's gender with their name when it is certain (section 7) | On |
| **Image subjects** | **characters** (people), **backgrounds** (scenery only, no people) or **both** | characters |
| **Art style** | The look of every picture, sent through `{art_style}` in the image prompt (the default image prompt starts with an `Art style:` line). Stored per profile. Empty removes that line and the image model picks its own style | `Anime illustration, clean line art, cel shading` |
| **Text model** | Model for the planner (and for writing styles from a prompt). Empty = main translation model | empty |
| **Image model** | Model that draws the pictures. Empty = the first Image Gen/Edit pool model (when the pool is on), else the **text model**, if it is not known to answer in text only. A model that cannot be told apart (a custom prefix) is tried with one real image request first; if no picture comes back the run stops with a message instead of planning every chapter. A text-only model (for example DeepSeek) is never used: set an image model, or set Images per chapter to 0 | empty |
| **🔑 Image Keys** | Opens the Image Gen/Edit key pool (section 13) | — |
| **Max reference images** | How many reference pictures go with each image request | 5 |
| **Resolution** | Requested image size class | 1K |
| **Aspect ratio** | A fixed ratio, or **auto**. With **auto**, click **Ratios…** and tick the ratios you like; each picture picks one of them (the same one again on a rerun) | 3:4 |
| **Run after each translation** | Runs this step automatically when a translation finishes | Off |
| **Recompile after** | Rebuilds the EPUB/PDF after a run so it contains the pictures | On |
| **✏ Placeholder texts…** | Edit the small texts that fill the placeholders (section 8) | — |
| **🖼 Gallery…** | Opens the gallery for the current book (section 11) | — |

> Some image models ignore the requested size or ratio and pick their own.

---

## 6. Reference images (keeping characters consistent)

Reference images are pictures you give the image model so a character keeps the same face, hair and outfit in every chapter.

- References are kept **per book**. Pick the book, then add pictures under each character name. The characters of the glossary translation uses are **listed automatically** (the path is shown above the list, and the list follows the file when it changes). A name that has pictures stays listed, marked *not in glossary*, if it leaves the glossary. **+ Character** adds a name by hand.
- You can also add **art-style references** (no character). They set the overall look.
- Pictures you add are copied into the book's `Interactive_Images/references/` folder.
- For each picture, the characters **in that scene** get the slots first, then style references fill what is left, up to **Max reference images**.

✅ **DO** use clear, front-facing pictures, one character per picture.

⚠️ **DON'T** add more than you need. More references are not always better, and some models only use the first few.

---

## 7. Gender (and how false positives are avoided)

When **Include gender with character names** is on, the planner and the image request see names like `Lee Seo-yun (female)` instead of just `Lee Seo-yun`. Without it, models sometimes draw a woman as a man (or the reverse).

A gender is sent **only when it is certain**. Glossarion looks at two sources: the **glossary's gender column** and the **Relations & Gender Tracker**.

| Situation | What is sent |
|---|---|
| Glossary and tracker agree | the gender |
| Only one source has a gender, the other has nothing | the gender |
| The two sources disagree | **nothing** (just the name) |
| The tracker shows the character flipping between genders | **nothing** |
| Two glossary rows for the same name disagree | **nothing** |
| Neutral, non-binary, unknown or blank | **nothing** |
| The tracker records **one clear change** at a known chapter, with enough sightings on both sides | the gender for that chapter |
| You made a decision in the tracker | your decision (unless the glossary contradicts it) |

Names are matched exactly. A gender written for a *different* character with a similar name is never used.

The way gender is written (`{name} ({gender})`, and the words `male` / `female`) can be changed under **✏ Placeholder texts…**.

---

## 8. Prompts, profiles and placeholders

The default **Image** prompt is `Art style: {art_style}`, then the scene, the characters and the reference list, so the pictures come out in the art style instead of whatever the image model defaults to. A profile that still holds the earlier default (without the art style line) gets the new one; a prompt you edited is left as it is.

The settings window has four prompt tabs:

| Tab | Used for |
|---|---|
| **Planner** | The text request that picks the moments, characters and classes, and answers in JSON |
| **Image** | The text that goes with each picture request |
| **Image system** | An optional system prompt for the image request (empty by default) |
| **CSS** | The request that writes a new style from your description (section 9) |

**Profiles** keep a full set of prompts and options. The built-ins are **Default** (1 image), **Illustrated** (2 images) and **Styles only** (0 images). Edit one and save it, or make your own.

**Placeholders** are the `{words_in_braces}` in a prompt. Glossarion fills them in before sending. Like the placeholders everywhere else in Glossarion, they show as **links**, and **hovering one shows what it will be filled with** for the current book.

| Placeholder | Filled with |
|---|---|
| `{paragraphs}` | The chapter, one numbered paragraph per line: `[n] text` |
| `{chapter_title}` | The chapter's title |
| `{images_per_chapter}` | The *Images per chapter* number |
| `{character_names}` | The book's characters from the glossary, with gender when certain |
| `{image_subjects}` | The text for the chosen *Image subjects* (empty for **characters**) |
| `{css_classes}` | The style classes the current style defines |
| `{scene_prompt}` | What the planner said the picture shows |
| `{characters}` | The characters in this picture, with gender when certain |
| `{art_style}` | The profile's *Art style* text (an empty text removes its line) |
| `{glossary_entries}` | One line per glossary entry this picture needs: the picture's characters, plus any glossary entry (a place, a term, ...) whose translated name appears as a whole word in the scene description. Glossary order, each once. No relevant entry removes its line. Not in the default image prompt: add it where you want it |
| `{reference_list}` | One line per reference picture, e.g. `Image 1: Lee Seo-yun` |
| `{user_request}` | Your description when writing a new style |

**Placeholder texts** (✏ Placeholder texts…) are the small pieces some placeholders are made of: the gender format and words, the *Image subjects* texts, the reference-line format, the `art style` label for style references and the **glossary entry line** (`{name}: {description}`, the format of each `{glossary_entries}` line; an entry without a description is just its name). Right-click any prompt box → **Available placeholders** / **Edit placeholder text…** to change them.

> Glossarion sends **only** your prompt text and what the placeholders expand to. It never adds hidden sentences of its own.

⚠️ **DON'T** remove the JSON shape from the Planner prompt. Glossarion needs the reply in that shape to place the pictures.

---

## 9. Interactive styles and the preview panel

Interactive styles are **CSS-only** looks for the parts Interactive Images marks: status-window tables, system messages, image frames, glowing or animated lines. E-readers strip JavaScript, so there is none. Animations only play for readers who have not turned off motion in their system settings.

**There are 22 built-in styles**, including Classic Book, Dark Fantasy, Cyberpunk Neon, LitRPG Status Window, Romance Pastel, Wuxia Ink, Gothic Horror, Sci-Fi HUD, Minimal Modern, Royal Court, Cozy Slice-of-Life, Retro Terminal, Arcane Runes, Dragon's Hoard, Elven Grove, Necromancer's Grimoire, Celestial Starfall, Level Up Burst, Quest Log, Dungeon System, Skill Tree Glow and Guild Card.

**🎨 Styles & Preview…** opens the style window:

- Pick a style from the dropdown and the **preview panel** shows a sample chapter in it: headings, dialogue, a status-window table, lists, a quote, animated text, a collapsible section and an image frame.
- **Duplicate** — copy a style to edit it
- **✨ New from prompt…** — describe a look ("dark purple arcane runes, glowing headings") and the text model writes a new style, which then shows in the preview
- **Edit CSS** / **💾 Save** / **Delete** — for your own styles
- **📌 Apply to book…** — write the style into the current book

To use a style: tick **Interactive styles** in Other Settings and pick it in the dropdown. **Clear** next to it removes the book's interactive stylesheet only.

---

## 10. The Progress Manager tab

**Progress Manager → 🖼 Interactive Images** lists **every chapter whose translation is completed**, in the book's **reading order (the OPF spine)** (chapters that are image-only, empty, skipped, failed or still translating are left out; each row shows its place in the book like the Translation tab, `[003]`), including chapters you have not built yet. The tab label counts built chapters (Completed and Styles only) out of that list, for example `🖼 Interactive Images (12/39)`, plus `, 2 stale` when chapters changed since they were built. It shows the count before you open the tab.

| Status | Meaning |
|---|---|
| **Not built** | No pictures or classes yet |
| **Completed** | Built; the text check passed |
| **Partial** | Some pictures could not be made or placed |
| **Failed** | The planner or the image model failed; see the reason |
| **Rejected** | The changes would have altered the chapter's text, so nothing was written |
| **Stale** | The chapter was retranslated after it was built; rebuild it |

**What is skipped:** Interactive Images skips exactly what glossary extraction skips. A chapter is not listed, not counted and never sent to the planner (the log says `⏭️ ... skipped: image-only` and so on) when its translated page is **image-only** or **empty**, or **title/header-only** (only while the Glossary setting *Skip title/header-only chapters* is on), or when its source file is a **special file** (cover, title page, copyright, notice, ...) and the translation setting *Translate special files* is off, with the same keyword lists and the same "never treat in-between files as special" rule the glossary uses.

**Buttons:** **▶ Build pending** (Not built, Failed, Partial, Rejected and Stale chapters), **Stop**, **Settings**, **🎨 Styles**, **🖼 Gallery**, **Refresh**. The stats chips at the top filter the list.

**Right-click a row:** Build / Rebuild, Retry failed, Mark for rebuild, Delete images, Open images, Open chapter, Show plan, Open folder.

✅ **DO** use **Show plan** to see exactly what the planner chose for a chapter.

---

## 11. The Gallery (browse and delete images)

**🖼 Gallery** shows every AI picture of the book as a scrollable grid. Each card shows the picture, its **chapter** and the **characters** in it.

- **Filter** by character (names from the glossary) or by chapter.
- **Select several:** desktop — Ctrl/Shift-click or the card checkboxes; mobile — long-press a card, then tap more.
- **Delete selected (n)** asks first, then removes the pictures **and** their place in the chapters. The text check still runs, and the progress list updates.
- Click a card for the full-size view, with **Delete** and **Go to chapter**.

A chapter left with no pictures and no classes goes back to **Not built**.

---

## 12. Batch mode (parallel requests)

Interactive Images follows the same **Batch Translation** settings as translation: on/off, batch size, threading delay and API call delay.

- **Batch off:** one request at a time.
- **Batch on:** up to *batch size* requests at once (at most 16). Chapters run side by side, and the pictures in one chapter can be drawn at the same time.

Each chapter is still written once, at the end, after its text check. **Stop** cancels everything queued; requests already sent finish first. A failing chapter does not affect the others.

> Web-chat routes that only allow one request at a time stay at one.

---

## 13. Keys: the Image Gen/Edit pool

If you turned on the **Image Gen/Edit key pool** (🔑 Image Keys), **picture requests use keys from that pool**, rotating between them in batch mode. The planner and style requests always use your normal translation keys. With the pool off, everything uses your normal keys.

---

## 14. Where the files go

Everything lives in the book's output folder, inside **`Interactive_Images/`**:

| File | What it is |
|---|---|
| `Glossarion_ch012_1.png` | Picture 1 of the chapter at **position 12 in the book** (its place in the OPF reading order, so two chapters never share a name, even ones without a chapter number such as notices; rebuilding a chapter replaces its pictures). Books without an OPF use the chapter number, or the list order when numbers repeat. Pictures made by an older version were named by chapter number and are not renamed: their chapters show as stale until rebuilt |
| `glossarion_interactive.css` | The book's interactive stylesheet |
| `interactive_images_progress.json` | The progress list |
| `references/` | Copies of your reference images |

This folder is kept apart from the book's own `images/` folder and from the image rename map, so the generated pictures are never confused with the original illustrations. The EPUB builder adds them to the book automatically.

---

## 15. How it works with "Attach CSS"

There is **no conflict**. They are two separate stylesheets:

| | Attach CSS | Interactive styles |
|---|---|---|
| File | The CSS you load with **Load CSS…** | `Interactive_Images/glossarion_interactive.css` |
| Affects | The whole book | Only parts Interactive Images marked (every rule starts with `.glx-`) |
| Order | After the book's own CSS | **Last** |
| Clear | Removes only your loaded CSS | Removes only the interactive stylesheet |

The two toggles are independent. When both are on, your CSS styles the book and the interactive style decorates only its own parts.

Two things to know:

- Processed chapters wrap their content in `<div class="glx-book">`. If your own CSS uses direct-child selectors such as `body > p`, those stop matching in processed chapters. Plain selectors such as `p` are fine.
- Glossarion's own **Reader** shows interactive styles only while **Attach CSS** is on. The compiled EPUB/PDF is not affected.

---

## 16. Mobile

Mobile has the same feature, using the same engine as desktop:

- **Tools → Interactive Images** — four tabs: **Profile** (profile, images per chapter, gender, image subjects, prompts, placeholder texts), **Models** (text/image model, Image Gen/Edit keys, max references, resolution, aspect ratio with Auto chips), **References** (per book; the characters come from the glossary translation uses, whose path is shown) and **Styles** (all styles with a live preview panel).
- **A book's Images tab** — the same list, statuses and actions as desktop, plus **🖼 Gallery** in the ⋯ menu.
- **Settings → EPUB output → Interactive images** — the style toggle and a link to the Tools screen.

---

## 17. Troubleshooting

| Problem | What to do |
|---|---|
| A chapter says **plan was not JSON** | The text model did not answer in JSON. Retry, or pick a stronger text model |
| **Rejected** | The planner's changes would have changed the text. Rebuild; if it keeps happening, check that the Planner prompt still says not to rewrite the chapter |
| **No image could be placed** | The planner pointed at a paragraph that does not exist. Rebuild the chapter |
| A character looks like the wrong gender | Fill in the glossary's gender column, check the tracker for conflicts (section 7) and add reference images |
| Pictures look different every chapter | Add reference images and an art-style reference |
| **no image-capable model found** / the text model returned no image | No image model is set and the text model cannot draw. Set an **Image model** (or add an Image Gen/Edit pool key), or set **Images per chapter** to 0 |
| The image model ignores the size or ratio | Some models choose their own size; try another model |
| The **Interactive styles** row is missing in Other Settings | Restart Glossarion; if you run from source, run `pip install -r requirements.txt`, then check the log for `Interactive styles rows unavailable` |
| The EPUB got much bigger | Each picture is 2–4 MB; use fewer images per chapter, or delete some in the Gallery |
