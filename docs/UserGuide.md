<p align="center">
  <img src="../assets/Halgakos.png" width="220" alt="Glossarion Logo" />
</p>

# 📘 Glossarion User Guide

Welcome to **Glossarion** — a GUI-powered tool for translating Korean, Japanese, or Chinese EPUB novels into clean, context-aware English. This guide walks you through every feature of the interface and how to use it from start to finish.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [User Interface Explained](#user-interface-explained)
5. [Translation Workflow](#translation-workflow)
6. [Glossary Extraction & Trimming](#glossary-extraction--trimming)
7. [Fallback EPUB Compiler](#fallback-epub-compiler)
8. [Profiles and Config](#profiles-and-config)
9. [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## 🧭 Overview

Glossarion is designed for translators who want:

- Automatic EPUB translation with OpenAI / Gemini
- Automatic glossary extraction and trimming
- Flexible profile system for KR/JP/CN novels
- EPUB output with cover image and embedded HTML

---

## 🖥️ System Requirements

- Python 3.10 or newer
- tkinter and ttkbootstrap installed
- Internet connection (for OpenAI or Gemini)
- API key (OpenAI or Gemini)

---

## ⚙️ Installation

1. Install Python 3.10+
2. Run `install_requirements.bat` to install dependencies.
3. Launch the GUI using `launch_gui_silent.bat`

---

## 🎛️ User Interface Explained

Each part of the GUI controls your translation environment:

- **EPUB File**: Path to your source EPUB.
- **Model**: Choose OpenAI (gpt-4.1) or Gemini.
- **Language Profile**: Choose or define your translation tone per language.
- **API Key**: Your GPT or Gemini key.
- **System Prompt**: Custom translation behavior.
- **Translation Settings**:
  - Delay between API calls
  - Temperature (creativity)
  - History window for contextual accuracy

- **Glossary Settings**:
  - Temperature + context limit for glossary extraction
  - Trimming controls (traits, title, group, refer-to, locations)

- **Run Translation**: Starts translation pipeline.
- **Extract Glossary**: Creates glossary from EPUB.
- **Trim Glossary**: Opens editor for refining glossary entries.
- **EPUB Converter**: Rebuilds EPUB from translated HTML.

---

## 🔁 Translation Workflow

1. Select your `.epub` file.
2. Configure your model, delay, API key, and prompt.
3. Click **Run Translation**.
4. Wait for all chapters to be processed.
5. Translated chapters are saved into `output/`.
6. Recompile with **EPUB Converter** to generate `translated_fallback.epub`.

---

## 🧾 Glossary Extraction & Trimming

Click **Extract Glossary** to:
- Sends an API call for every chapter in the EPUB file to Extract character data, and location names. (entries are merged and by appearance time)
- Save to `Glossary/glossary.json` and `glossary.md`.

Click **Trim Glossary** to:
- Remove entries from bottom to top
- Merge and summarize all locations at the end.

---

## 📚 Fallback EPUB Compiler

If automatic EPUB generation fails:
- Click **EPUB Converter**
- It will rebuild the EPUB using all saved HTML and embedded images
- Includes:
  - Cover page (if image present)
  - Table of contents
  - Optional image gallery

---

## 🎨 Profiles and Config

- Save and load prompt profiles for different languages.
- Import/export your entire prompt configuration.
- Use `Save Config` to persist API keys and settings.

---

## 💡 Tips and Troubleshooting

- **API errors**: Make sure your key is valid and quota isn't exceeded.
- **GUI not launching?** Use `pythonw translator_gui.py` to skip the console.
- **Translation cutoff**: Increase `translation_history_limit` or adjust delay.
- **Glossary broken?** Delete `glossary_progress.json` and re-extract.

---

Happy translating with Glossarion!
