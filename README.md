<p align="center">
  <img src="assets/Halgakos.png" width="200" alt="Glossarion Logo" />
</p>

# 📚 Glossarion

Glossarion is an AI translator and AI glossary generator for Korean, Japanese, and Chinese light novels, built to transform EPUB files and raw .txt files into high-quality, context-aware English translations. It supports OpenAI, Gemini, DeepSeek, and Sonnet APIs, with a GUI that gives you total control over every step of the process—from prompt customization to final EPUB export.
---

## 🏷️ Badges

![Build](https://img.shields.io/github/actions/workflow/status/Shirochi-stack/Glossarion/python-app.yml?branch=main)
![License](https://img.shields.io/github/license/Shirochi-stack/Glossarion)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Python application](https://github.com/Shirochi-stack/Glossarion/workflows/Python%20application/badge.svg)
[![GitHub release](https://img.shields.io/github/release/Shirochi-stack/Glossarion.svg)](https://GitHub.com/Shirochi-stack/Glossarion/releases/)

## ✨ Features

### 📖 Core Translation Engine

* Translate EPUB chapters using OpenAI or Gemini chat models.
* Customizable system prompts that allow you to adjust your preferences.
* Fully retains HTML structure and embedded images.
* Manual glossary override supported.
* Configurable context-memory window per chapter.
* Rate-limit handling with delay configuration and retry logic for timeouts & duplicates
  
    ### 🖼️ OCR Translation
  
  * Extract & translate embedded images’ text  
  * Auto-split tall images (>2000 px) for reliable OCR
  * Configurable chunk sizing
  * OpenAI & Gemini API support



### 📓 Glossary Extraction

* AI-powered extraction of:

  * Character names (original + romanized)
  * Titles and group affiliations
  * Traits and reference styles (how characters refer to others)
  * Locations (with original script in brackets)
* Output in `.json` and Markdown `.md` formats.
* Intelligent merging of duplicate entries.


### ✂️ Glossary Management

* GUI glossary trimmer with field-specific controls:

  * Drop traits, affiliations, name-mappings, etc.
  * Aggregate all locations into a summary entry.
* Import and override glossary files manually.


### 🖥️ Full GUI Support

* Built with `ttkbootstrap` and `tkinter` for a modern interface.
* Language prompt profiles with import/export options.
* Configurable API model, temperature, history depth, and more.
* Real-time logging, subprocess output streaming, and fallback behavior.


### 🛡️ QA Scanning

* Scan translated HTML files for duplicates, non-English fragments, spacing issues, and repetitive sentences.
* Generate JSON, CSV, and HTML reports for QA review.
* Trigger via GUI **QA Scan** button or CLI `scan_html_folder.py`.


### 📚 EPUB Export

* Rebuild an EPUB from translated HTML and images, preserving cover art and metadata.
* Image gallery support.
* Accessible via GUI **EPUB Converter** or CLI `epub_converter.py`.
* Preserves cover art, images & metadata  
* Smart chapter extraction & robust XHTML parsing

  
### 🔍 Quality Assurance Tools
* Automated HTML scans for duplicates, non-English fragments & spacing issues  
* Reports in JSON, CSV & HTML

   
### 🔧 Helper Tools

* Unified API client (`unified_api_client.py`): supports OpenAI, Gemini, DeepSeek Chat, and Anthropic/Sonetta models with automatic retries and detailed payload logging.

---

## 📦 Installation

1. Clone or download this repo:

   ```bash
   git clone https://github.com/Shirochi-stack/Glossarion.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

**requirements.txt**:

```text
# Third-party dependencies

tiktoken
ebooklib
beautifulsoup4
ttkbootstrap
Pillow
requests
openai
google-generative-ai
langdetect
tqdm
```

3. Launch the GUI:

   ```bash
   python translator_gui.py
   ```

---

## 🔑 API Keys

To use translation features, provide an API key in the GUI:

* **OpenAI**: Models like `gpt-4.1-*`, `o4-*`
* **Gemini**: Any `gemini-flash-*`, `gemini-pro-*` model supported by Google
* **DeepSeek Chat**: Your DeepSeek Chat API key
* **Sonetta (Anthropic)**: Your Sonetta/Anthropic API key

---


## 🧠 System Prompt Customization

Customize your translation style:

* Per-language prompt profiles (e.g., Korean, Japanese, Chinese)
* Onomatopoeia in Romaji
* Speech tone retention
* Slang/dialect preservation

All settings are saved in `config.json`.

---


## 🧪 Example Workflow

1. Select your `.epub` file.
2. Enter your API key and prompt settings.
3. Click **Run Translation**.
4. After translation, click **EPUB Converter** to recompile.
5. Optionally, use **Extract Glossary** to generate character info.

---

## 🧱 Project Structure

```
.
├── src/
│   ├── translator_gui.py
│   ├── TransateKRtoEN.py
│   ├── extract_glossary_from_epub.py
│   ├── epub_converter.py
│   ├── unified_api_client.py
│   ├── launch_Glossarion.bat
│   └── launch_Glossarion.vbs
│   └── scan_html_folder.py
│   └── image_translator.py
├── docs/
│   └── UserGuide.md
├── Glossary/
├── output/
├── Payloads/
├── assets/
│   └── Halgakos.png
├── README.md
├── LICENSE
├── requirements.txt
└── install_requirements.bat
```

---

## 💬 Acknowledgments

Built with ❤️ using OpenAI, Gemini & Claude APIs. Designed with ChatGPT & Claude.

---

## 📜 License

MIT License
