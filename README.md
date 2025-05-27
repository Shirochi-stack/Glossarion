<p align="center">
  <img src="assets/Halgakos.png" width="200" alt="Glossarion Logo" />
</p>

# 📚 Glossarion

**Glossarion** is a powerful translator and glossary generator for Korean, Japanese, and Chinese light novels, built to transform EPUB files into high-quality, context-aware English translations. It integrates OpenAI and Gemini APIs with a GUI that gives you total control over every step of the process—from prompt customization to final EPUB export.

---

## 🏷️ Badges

![Build](https://img.shields.io/github/actions/workflow/status/Shirochi-stack/Glossarion/python-app.yml?branch=main)
![License](https://img.shields.io/github/license/Shirochi-stack/Glossarion)
![Python](https://img.shields.io/badge/Python-3.10+-blue)


## ✨ Features

### 🔁 Contextual Translation Pipeline
- Translate EPUB chapters using OpenAI or Gemini chat models.
- Customizable system prompts that allow you to adjust your preferences.
- Fully retains HTML structure and embedded images.
- Manual glossary override supported.
- Contextual memory (configurable per chapter window).
- Rate limit handling with delay configuration.

### 📓 Glossary Extraction
- AI-powered extraction of:
  - Character names (original + romanized)
  - Titles and group affiliations
  - Traits and how they refer to others
  - Locations (with original script in brackets)
- Output in `.json` and Markdown `.md` format.
- Merge duplicate entries intelligently.

### ✂️ Glossary Management
- GUI glossary trimmer with field-specific limits:
  - Drop traits, affiliations, name-mappings, etc.
  - Aggregate all locations into a summary entry.
- Load and override glossary files manually.

### 🖥️ Full GUI Support
- Built with `ttkbootstrap` and `tkinter` for modern interface.
- Language prompt profiles with import/export options.
- Configurable API model, temperature, history depth, etc.
- Real-time logging, subprocess streaming, and fallback tools.

---

## 📦 Installation

1. Clone or download this repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Requirements include:
   - `tiktoken`
   - `ebooklib`
   - `beautifulsoup4`
   - `ttkbootstrap`
   - `openai`
   - `google-generativeai`

4. Launch the GUI:
   ```bash
   python translator_gui.py
   ```

---

## 🔑 API Keys

To use translation features, provide an API key in the GUI:

- **OpenAI**: Models like `gpt-4.1-nano`, `gpt-4.1-mini`
- **Gemini**: Any `gemini-pro-*` model supported by Google

---

## 🧠 System Prompt Customization

Customize your translation style:
- Per-language prompt profiles (e.g., Korean, Japanese, Chinese)
- Onomatopoeia in Romaji
- Speech tone retention
- Slang/dialect preservation

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
│   ├── epub_fallback_compiler_with_cover_portable.py
│   ├── unified_api_client.py
│   ├── launch_Glossarion.bat             
│   └── launch_Glossarion.vbs         # Launch GUI with no CMD window
├── docs/
│   └── UserGuide.md
├── Glossary/                         # Extracted glossary JSON and Markdown files
├── output/                           # Generated translations and final EPUB
├── Payloads/                         # API payloads, logs, history
├── assets/Halgakos.png                  # GUI/README logo
├── README.md                         # Instructions and usage guide
├── LICENSE
├── .gitignore
├── requirements.txt                  # Python dependencies
├── install_requirements.bat          # Installer for all Python dependencies
```


---

## 💬 Acknowledgments

Built with ❤️ using OpenAI & Gemini APIs. Designed with ChatGPT.  
GUI and logic heavily assisted by [ChatGPT].

---

## 📜 License

MIT License
