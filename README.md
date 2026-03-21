<p align="center">
  <img src="assets/Halgakos.png" width="200" alt="Glossarion Logo" />
</p>

# 📚 Glossarion

**Glossarion** is a comprehensive AI-powered translation suite for Korean, Japanese, and Chinese light novels, web novels, manga, and documents. Built to transform EPUB files, raw .txt files, PDFs, and manga images into high-quality, context-aware English translations. It supports **40+ AI providers** — including OpenAI, Google Gemini, Anthropic Claude, DeepSeek, xAI Grok, Mistral, and local LLMs via Ollama — with a modern PySide6 GUI that gives you total control over every step of the translation process.

---

## 🏷️ Badges

![Build](https://img.shields.io/github/actions/workflow/status/Shirochi-stack/Glossarion/python-app.yml?branch=main)
![License](https://img.shields.io/github/license/Shirochi-stack/Glossarion)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Python application](https://github.com/Shirochi-stack/Glossarion/workflows/Python%20application/badge.svg)
[![GitHub release](https://img.shields.io/github/release/Shirochi-stack/Glossarion.svg)](https://GitHub.com/Shirochi-stack/Glossarion/releases/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/n7WXRXn533)

---

## ✨ Key Features

### 📖 Novel Translation Engine
* **Multi-Provider AI Support** — 40+ providers out of the box (see [Supported Providers](#-supported-ai-providers) below)
* **Advanced Context Management**
  * Rolling history window for consistent translations
  * Chunk-based processing for long chapters
  * Contextual memory with configurable depth
  * Parallel chapter translation support
  * Full history export/import
* **Smart Retry System**
  * AI Hunter duplicate detection (ML-based similarity analysis with TF-IDF fingerprinting)
  * Automatic retry for timeouts, truncation, and server errors
  * Multi-key rotation with automatic failover on 429/5xx errors
  * Rate limit handling with exponential backoff

### 🎌 Manga Translation Engine
* **Dual OCR System**
  * Google Cloud Vision API for text detection
  * Azure AI Vision / Document Intelligence as alternative OCR backends
* **YOLO Bubble Detection**
  * ONNX-based speech bubble detection for precise text region isolation
  * Automatic bubble segmentation for complex page layouts
* **Visual Context-Aware Translation**
  * AI sees the full manga page image for accurate context
  * Full page context mode for multi-bubble consistency
  * Character expressions inform translation tone
* **Advanced Text Rendering**
  * Customizable fonts, colors, backgrounds, shadows, and outlines
  * Smart inpainting to remove original text (local ONNX or Replicate cloud)
  * Preserve original art option
* **Batch Processing**
  * Process entire manga chapters automatically
  * Stop/resume functionality with progress tracking

### 🧠 AI Hunter — Duplicate Detection
* **ML-based content similarity analysis** with TF-IDF semantic fingerprinting
* **Structural pattern recognition** and statistical outlier detection
* **Configurable sensitivity thresholds** with length ratio checks
* **Key phrase verification** and character name consistency checks

### 📓 Glossary System
* **AI-Powered Extraction** from EPUB and TXT files
* Custom field support with configurable prompts
* Multi-language support (KR/JP/CN)
* Duplicate merging strategies
* Import/export with validation (JSON and Markdown formats)
* Automatic backup system

### 🛡️ Quality Assurance Suite
* **Comprehensive Scanning** — duplicate content, non-English fragments, spacing/formatting issues, repetitive patterns
* **Multiple Report Formats** — interactive HTML, JSON analysis, CSV exports, summary statistics

### 📚 File Format Support
* **EPUB** — structure-preserving translation, metadata/cover retention, image gallery, clean HTML output, EPUB → translated EPUB conversion
* **TXT** — chapter detection, custom delimiters, encoding auto-detection, format preservation
* **PDF** — extraction via PyMuPDF, generation via WeasyPrint/xhtml2pdf
* **HTML** — header translation, scan and batch processing

### 🖼️ Image Translation
* Auto-detection of text in images
* Tall image splitting for reliable OCR
* Batch processing with progress tracking
* Context preservation across chunks

### 🖥️ Modern GUI (PySide6)
* **Cross-Platform** — Windows 10/11, macOS (Apple Silicon & Intel)
* DPI-aware scaling with high-DPI display support
* Animated splash screen and spinner indicators
* Real-time translation progress with API watchdog monitoring
* Comprehensive logging system with rotating log files and crash tracing
* Per-language prompt profiles, temperature/token controls, API endpoint customization

### 🔐 Security & Configuration
* **API Key Encryption** — keys encrypted at rest using the `cryptography` library
* **Multi-Key Management** — key pool with rotation, rate limit caching, and per-key health tracking
* **Config Backup System** — automatic JSON config backups with atomic writes
* **AuthGPT OAuth** — use your ChatGPT subscription directly via OAuth token flow

---

## 🔑 Supported AI Providers

| Provider | Model Prefix | Example Models (2026) |
|----------|-------------|----------------------|
| **OpenAI** | `gpt-*`, `o3-*` | gpt-5.4, gpt-5.4-pro, gpt-5.3-codex, gpt-5.2, gpt-5, gpt-5-mini, gpt-5-nano, o3 |
| **Google Gemini** | `gemini-*` | gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-2.5-flash, gemini-2.5-pro |
| **Anthropic Claude** | `claude-*` | claude-opus-4-6, claude-sonnet-4-6, claude-sonnet-4-5, claude-haiku-4-5 |
| **xAI Grok** | `grok-*`, `xai/*` | grok-4.20-beta, grok-4-fast, grok-4-0709, grok-3, grok-3-mini |
| **DeepSeek** | `deepseek-*` | deepseek-chat, deepseek-reasoner, deepseek-coder |
| **Mistral** | `mistral-*`, `mixtral-*`, `codestral-*` | mistral-large, mixtral-8x22b, codestral-latest |
| **Cohere** | `command-*` | command-r, command-r-plus |
| **ElectronHub** | `eh/*` | eh/gpt-5-chat-latest, eh/claude-sonnet-4-6, eh/gemini-3.1-pro-preview, eh/grok-4-fast |
| **OpenRouter** | `or/*` | or/openai/gpt-5.4, or/google/gemini-3.1-pro-preview, or/deepseek/deepseek-v3.2 |
| **Poe** | `poe/*` | poe/gpt-4.5, poe/claude-4-opus, poe/gemini-2.5-pro |
| **VertexAI** | `vertex/*` | vertex/gemini-3.1-pro-preview, vertex/claude-4-opus |
| **Groq** | `groq/*` | groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-maverick-17b |
| **AuthGPT** | `authgpt/*` | authgpt/gpt-5.4, authgpt/gpt-5.3-codex, authgpt/gpt-5.2 |
| **Antigravity** | `antigravity/*` | antigravity/claude-opus-4-6-thinking, antigravity/gemini-3.1-pro |
| **NVIDIA** | `nd/*` | nd/deepseek-ai/deepseek-v3.2, nd/moonshotai/kimi-k2-thinking |
| **Chutes** | `chutes/*` | chutes/deepseek-ai/DeepSeek-V3.2, chutes/openai/gpt-oss-120b |
| **Fireworks** | `fireworks/*` | fireworks/llama-v3-70b |
| **Together AI** | `together/*` | together/llama-3-70b |
| **Perplexity** | `perplexity/*`, `pplx-*` | perplexity-70b-online, pplx-70b-online |
| **AI21** | `j2-*`, `jamba-*` | j2-ultra, jamba-instruct |
| **Qwen** | `qwen-*` | qwen-72b-chat, qwen-plus, qwen-turbo |
| **Yi** | `yi-*` | yi-34b-chat-200k |
| **DeepL** | `deepl` | deepl (traditional translation API) |
| **Google Translate** | `google-translate*` | google-translate, google-translate-free |

> **Note:** Many more providers are supported — including Baichuan, Zhipu AI (GLM), Moonshot/Kimi, Baidu ERNIE, Tencent Hunyuan, ByteDance Doubao, MiniMax, Meta Llama, Microsoft Phi, Falcon, and others. See `model_options.py` and `unified_api_client.py` for the full catalog.

### API Key Setup
1. **Direct Providers** — use API keys from OpenAI, Google, Anthropic, etc.
2. **ElectronHub** — single API key for access to models from multiple providers
3. **AuthGPT** — use your ChatGPT subscription via OAuth (no API key needed)
4. **Antigravity** — local Cloud Code proxy on `localhost:8080` (no API key needed)
5. **Custom Endpoints** — configure base URL for self-hosted or alternative endpoints

### Manga Translation Setup
1. Create a Google Cloud Project (or Azure AI resource)
2. Enable Cloud Vision API (or Azure AI Vision)
3. Create service account credentials
4. Download JSON key file
5. Set path in Manga Translator interface

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Windows 10/11 or macOS (for full feature support)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Shirochi-stack/Glossarion.git
   cd Glossarion
   ```

2. Install dependencies:
   ```bash
   # Windows
   pip install -r requirements.txt

   # macOS
   pip install -r requirements-macos.txt
   ```

3. Launch the GUI:
   ```bash
   cd src
   python translator_gui.py
   ```

### Building Executable (Optional)

```bash
pip install pyinstaller

# Standard build
pyinstaller src/translator.spec

# Lite build
pyinstaller src/translator_lite.spec

# macOS
pyinstaller src/translator_lite_mac.spec
```

---

## 📋 Key Dependencies

```text
# GUI Framework
PySide6==6.9.3

# AI/API Clients
openai, anthropic, mistralai, cohere, tiktoken
google-genai, google-cloud-aiplatform, vertexai
poe-api-wrapper, deepl, httpx, aiohttp

# File Processing
ebooklib, beautifulsoup4, lxml, html5lib, html2text
pymupdf, weasyprint, xhtml2pdf

# Image Processing & OCR
Pillow, opencv-python-headless, numpy, scipy
google-cloud-vision, azure-ai-vision-imageanalysis
azure-ai-documentintelligence

# Manga / Bubble Detection
onnxruntime, ultralytics (YOLO)
torch, torchvision, transformers

# Text Analysis
langdetect, chardet, datasketch, rapidfuzz, jellyfish, regex

# Security
cryptography
```

---

## 🚀 Usage

### Basic Translation
1. Launch `translator_gui.py`
2. Select your EPUB/TXT/PDF file
3. Choose source language (Korean, Japanese, or Chinese)
4. Enter your API key and select a model
5. Configure translation settings (chunk size, context depth, etc.)
6. Click **"Run Translation"**

### Manga Translation
1. Open **Manga Translator** from the Tools menu
2. Set Google Cloud Vision (or Azure) credentials
3. Select manga images or folder
4. Configure text rendering options (font, color, inpainting)
5. Start batch translation

### Glossary Extraction
1. After translation completes, click **"Extract Glossary"**
2. Review and edit entries
3. Export to JSON or Markdown

### Quality Assurance
1. Complete translation
2. Click **"QA Scan"**
3. Review the interactive HTML report
4. Fix identified issues

---

## 🧱 Project Structure

```
Glossarion/
├── src/
│   ├── translator_gui.py           # Main GUI entry point (PySide6)
│   ├── TransateKRtoEN.py           # Core translation engine
│   ├── unified_api_client.py       # Multi-provider AI client (40+ providers)
│   ├── async_api_processor.py      # Async concurrent chapter processing
│   ├── model_options.py            # Centralized model catalog
│   ├── multi_api_key_manager.py    # API key pool & rotation
│   ├── manga_translator.py         # Manga OCR and translation
│   ├── manga_integration.py        # Manga GUI interface
│   ├── bubble_detector.py          # YOLO-based speech bubble detection
│   ├── local_inpainter.py          # ONNX local inpainting engine
│   ├── ocr_manager.py              # OCR provider manager
│   ├── ai_hunter_enhanced.py       # ML-based duplicate detection
│   ├── epub_converter.py           # EPUB processing & conversion
│   ├── pdf_extractor.py            # PDF text extraction (PyMuPDF)
│   ├── scan_html_folder.py         # QA scanner
│   ├── GlossaryManager.py          # Glossary management engine
│   ├── extract_glossary_from_epub.py  # EPUB glossary extractor
│   ├── review_dialog.py            # Translation review UI
│   ├── other_settings.py           # Advanced settings dialogs
│   ├── authgpt_auth.py             # ChatGPT OAuth integration
│   ├── api_key_encryption.py       # API key encryption at rest
│   ├── config_backup.py            # Config backup management
│   ├── dpi_setup.py                # DPI awareness configuration
│   ├── splash_utils.py             # Animated splash screen
│   ├── update_manager.py           # Auto-update system

│   └── ...
├── assets/                         # App icons and images
├── docs/                           # User guides and documentation
├── .github/workflows/              # CI/CD (Windows, macOS)
├── requirements.txt                # Windows dependencies
├── requirements-macos.txt          # macOS dependencies
├── translator.spec                 # PyInstaller build config
└── LICENSE                         # MIT License
```

---

## 🎯 Advanced Features

### Translation Profiles
- **Japanese** (Manga_JP / Novel) — optimized for manga and novel translation
- **Korean** (Manga_KR / Novel) — manhwa and web novel translation
- **Chinese** (Manga_CN / Novel) — manhua and web novel translation

### Context Window Management
- **Rolling Window** — maintains recent context for consistency
- **Reset on Limit** — clears history at threshold
- **Dynamic Adjustment** — adapts based on model token limits
- **Export/Import** — save and resume translation sessions

### Batch Processing
- **Concurrent Chunks** — process multiple sections simultaneously
- **Auto-retry** — automatic error recovery with key rotation
- **Progress Persistence** — resume interrupted translations via `translation_progress.json`
- **Resource Management** — memory usage tracking and optimization

### API Watchdog
- Real-time monitoring of in-flight API requests
- Per-request tracking with chapter/chunk labels
- Retry attempt logging and duration tracking

---

## 🙏 Acknowledgments

Built using:
- OpenAI, Google, Anthropic, xAI, and many more AI provider APIs
- Designed with assistance from ChatGPT & Claude
- Community feedback and contributions
- Open source libraries and tools
- comic-translate by ogkalu2 — https://github.com/ogkalu2/comic-translate

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Shirochi-stack/Glossarion/issues)
- **Discord**: [Join our Community](https://discord.gg/n7WXRXn533)

---

<p align="center">Made with 🌸 for the translation community</p>
