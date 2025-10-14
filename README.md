<p align="center">
  <img src="assets/Halgakos.png" width="200" alt="Glossarion Logo" />
</p>

# 📚 Glossarion

**Glossarion** is a comprehensive AI-powered translation suite for Korean, Japanese, and Chinese light novels, web novels, and manga. Built to transform EPUB files, raw .txt files, and manga images into high-quality, context-aware English translations. It supports multiple AI providers including OpenAI, Google Gemini, Anthropic Claude, DeepSeek, Mistral, and more (including local LLM's using ollama), with a modern GUI that gives you total control over every step of the translation process.

---

## 🏷️ Badges

![Build](https://img.shields.io/github/actions/workflow/status/Shirochi-stack/Glossarion/python-app.yml?branch=main)
![License](https://img.shields.io/github/license/Shirochi-stack/Glossarion)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Python application](https://github.com/Shirochi-stack/Glossarion/workflows/Python%20application/badge.svg)
[![GitHub release](https://img.shields.io/github/release/Shirochi-stack/Glossarion.svg)](https://GitHub.com/Shirochi-stack/Glossarion/releases/)

## ✨ Key Features

### 📖 Novel Translation Engine
* **Multi-Provider AI Support**
  * OpenAI (GPT-4, o1-preview, o1-mini)
  * Google Gemini (Flash, Pro, experimental models)
  * Anthropic Claude (Opus, Sonnet, Haiku)
  * DeepSeek, Mistral, Cohere, and more
* **Advanced Context Management**
  * Rolling history window for consistent translations
  * Chunk-based processing for long chapters
  * Contextual memory with configurable depth
  * Full history export/import
* **Smart Retry System**
  * AI Hunter duplicate detection
  * Automatic retry for timeouts and errors
  * Intelligent truncation detection and recovery
  * Rate limit handling with exponential backoff
 
### 🎌 Manga Translation Engine
* **Dual API System: OCR + AI Translation**
  * Google Cloud Vision API for text detection (OCR)
  * Your chosen AI provider (OpenAI/Gemini/Claude/etc.) for actual translation
  * Requires BOTH: Google Cloud credentials AND your AI API key
* **Visual Context-Aware Translation**
  * AI sees the full manga page image for accurate context
  * Full page context mode for multi-bubble consistency
  * Character expressions inform translation tone
  * **Best results with advanced models like o3**

* **Advanced Text Rendering**
  * Customizable fonts, colors, and backgrounds
  * Text shadows and outlines for readability
  * Smart inpainting to remove original text
  * Preserve original art option
* **Batch Processing**
  * Process entire manga chapters automatically
  * Stop/resume functionality
  * Progress tracking and error recovery

### 🧠 AI Hunter
* **Advanced Duplicate Detection**
  * Machine learning-based content similarity analysis
  * Semantic fingerprinting using TF-IDF
  * Structural pattern recognition
  * Configurable sensitivity thresholds
* **Smart Filtering**
  * Length ratio checks
  * Key phrase verification
  * Character name consistency
  * Statistical outlier detection

### 📓 Glossary System v2.0
* **Flexible Extraction**
  * Custom field support
  * Configurable prompts
  * Multi-language support (KR/JP/CN)
  * Duplicate merging strategies
* **Advanced Management**
  * Field-specific trimming controls
  * Import/export with validation
  * Automatic backup system
  * JSON and Markdown formats

### 🛡️ Quality Assurance Suite
* **Comprehensive Scanning**
  * Duplicate content detection
  * Non-English fragment identification
  * Spacing and formatting issues
  * Repetitive sentence patterns
* **Multiple Report Formats**
  * Interactive HTML reports
  * Detailed JSON analysis
  * CSV exports for spreadsheets
  * Summary statistics

### 🖼️ Image Translation
* **Smart Processing**
  * Auto-detection of text in images
  * Tall image splitting for reliable OCR
  * Batch processing with progress tracking
  * Context preservation across chunks

### 📚 File Format Support
* **EPUB Processing**
  * Structure-preserving translation
  * Metadata and cover retention
  * Image gallery support
  * Clean HTML generation
* **Text File Support**
  * Chapter detection algorithms
  * Custom delimiters
  * Encoding auto-detection
  * Format preservation

### 🖥️ Modern GUI Interface
* **User-Friendly Design**
  * Dark/light theme support via ttkbootstrap
  * Real-time translation progress
  * Scrollable dialogs for all screens
  * Comprehensive logging system
* **Advanced Configuration**
  * Per-language prompt profiles
  * Temperature and token controls
  * API endpoint customization
  * Batch size optimization

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Windows 10/11 (for full feature support)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Shirochi-stack/Glossarion.git
   cd Glossarion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the GUI:
   ```bash
   python translator_gui.py
   ```

### Building Executable (Optional)
```bash
pip install pyinstaller
pyinstaller translator.spec
```

---

## 📋 Requirements

```text
# Core Translation
tiktoken>=0.5.0
openai>=1.0.0
google-generativeai>=0.3.0
anthropic>=0.7.0
mistralai>=0.0.7
cohere>=4.0.0

# File Processing
ebooklib>=0.18
beautifulsoup4>=4.12.0
lxml>=4.9.0
html5lib>=1.1

# GUI Framework
ttkbootstrap>=1.10.0
tkinter (included with Python)

# Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0

# Manga Translation
google-cloud-vision>=3.4.0

# Text Analysis
langdetect>=1.0.9
chardet>=5.2.0
datasketch>=1.6.0
scipy>=1.11.0

# Utilities
requests>=2.31.0
tqdm>=4.66.0
regex>=2023.0.0
```

---

## 🔑 API Configuration

### Supported AI Providers

Here's the updated API provider table based on the unified API client support:

| Provider | Model Format | Example Models |
|----------|--------------|----------------|
| **OpenAI** | `gpt-*`, `o1-*`, `o3-*`, `o4-*` | gpt-4, gpt-4-turbo, gpt-4o, o1-preview, o1-mini, o3, o4-mini |
| **Google Gemini** | `gemini-*` | gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro |
| **Anthropic** | `claude-*` | claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet |
| **DeepSeek** | `deepseek-*` | deepseek-chat, deepseek-coder |
| **Mistral** | `mistral-*`, `open-mistral-*` | mistral-large, mistral-medium, open-mistral-7b |
| **Cohere** | `command-*` | command-r, command-r-plus |
| **ElectronHub** | `eh/*`, `electronhub/*` | eh/gpt-4, eh/claude-3-opus, eh/gemini-2.5-flash, eh/yi-large |
| **VertexAI** | `vertex/*`, `vertex_ai/*` | vertex/gemini-2.5-flash, vertex/gemini-2.5-pro
| **Groq** | `groq/*`, `llama-*`, `mixtral-*` | groq/llama-3.1-70b, groq/mixtral-8x7b |
| **POE** | [poe/*](https://github.com/Shirochi-stack/Glossarion/blob/main/docs/poeguide.md)     | poe/gemini-2.5-flash, poe/claude-3.5-instant, poe/anthropic-instant-v1 |
| **Together AI** | `together/*`, `meta-llama/*` | together/llama-3-70b, meta-llama/Llama-3-70b |
| **Fireworks** | `fireworks/*`, `accounts/fireworks/*` | fireworks/llama-v3-70b, accounts/fireworks/models/mixtral-8x7b |
| OpenRouter | `or/*`, `openrouter/*` | or/google/gemini-2.5-flash, or/openai/chatgpt-4o-latest
| Chutes | `chutes/*` | chutes/deepseek-ai/DeepSeek-V3.1
| **Perplexity** | `perplexity/*`, `llama-*`, `pplx-*` | perplexity/llama-3.1-70b, pplx-7b-online |
| **Anyscale** | `anyscale/*`, `meta-llama/*` | anyscale/llama-3-70b, meta-llama/Llama-3-70b |
| **Hugging Face** | `huggingface/*` | huggingface/meta-llama/Llama-3-70b |
| **Replicate** | `replicate/*`, `meta/*` | replicate/meta/llama-3-70b |
| **AI21** | `ai21/*`, `jamba-*`, `j2-*` | ai21/jamba-1.5-large, j2-ultra |
| **Voyage AI** | `voyage/*` | voyage/voyage-3 |
| **Reka** | `reka-*` | reka-flash, reka-core, reka-edge |
| **xAI** | `xai/*`, `grok-*` | xai/grok-beta, grok-2 |
| **LeptonAI** | `lepton/*`, `llama-*`, `mixtral-*` | lepton/llama-3.1-70b, lepton/mixtral-8x7b |
| **DeepInfra** | `deepinfra/*` | deepinfra/meta-llama/Llama-3-70b |
| **Qwen** | `qwen-*`, `qwen/*` | qwen-max, qwen-plus, qwen-turbo |
| **Yi** | `yi-*`, `yi/*`, `zero-one-ai/*` | yi-large, yi-medium, zero-one-ai/yi-34b |
| **Moonshot** | `moonshot-*` | moonshot-v1-128k, moonshot-v1-32k |
| **GLM** | `glm-*` | glm-4-plus, glm-4, glm-3-turbo |

### Additional Supported Providers
Yi, Qwen, Baichuan, Zhipu AI, Moonshot, Groq, Baidu, Tencent, iFLYTEK, ByteDance, MiniMax, Together AI, Perplexity, and many more. See the full list in `unified_api_client.py`.

### API Key Setup
1. **Direct Providers**: Use API keys from OpenAI, Google, Anthropic, etc.
2. **ElectronHub**: Single API key for access to models from multiple providers
3. **Custom Endpoints**: Configure base URL for self-hosted or alternative endpoints

### Model Selection
- Enter the model name exactly as shown in the provider's documentation
- The tool automatically detects the provider based on the model prefix
- For ElectronHub, prefix any supported model with `eh/`, `electronhub/`, or `electron/`
  - Example: `eh/gpt-4`, `electronhub/claude-3-opus`, `electron/yi-34b-chat`

### Manga Translation Setup
1. Create a Google Cloud Project
2. Enable Cloud Vision API
3. Create service account credentials
4. Download JSON key file
5. Set path in Manga Translator interface

---

## 🚀 Usage Examples

### Basic Novel Translation
1. Select your EPUB/TXT file
2. Choose source language profile
3. Enter your API key
4. Configure translation settings
5. Click "Run Translation"

### Manga Translation
1. Open Manga Translator from Tools menu
2. Set Google Cloud Vision credentials
3. Select manga images
4. Configure text rendering options
5. Start batch translation

### Glossary Extraction
1. After translation completes
2. Click "Extract Glossary"
3. Review and edit entries
4. Export to JSON/Markdown

### Quality Assurance
1. Complete translation
2. Click "QA Scan"
3. Review HTML report
4. Fix identified issues

---

## 🧱 Project Structure

```
Glossarion/
├── src/
│   ├── translator_gui.py          # Main GUI application
│   ├── TransateKRtoEN.py         # Core translation engine
│   ├── unified_api_client.py     # Multi-provider AI client
│   ├── manga_translator.py       # Manga OCR and translation
│   ├── manga_integration.py      # Manga GUI interface
│   ├── ai_hunter_enhanced.py     # Advanced duplicate detection
│   ├── history_manager.py        # Context management
│   ├── extract_glossary_from_*.py # Glossary extractors
│   ├── epub_converter.py         # EPUB processing
│   ├── scan_html_folder.py       # QA scanner
│   └── [other modules]
├── assets/
│   └── Halgakos.png             # Application icon
├── docs/
│   └── [documentation]
├── translator.spec              # PyInstaller config
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🎯 Advanced Features

### Translation Profiles
- **Japanese (Manga_JP)**: Optimized for manga with visual context
- **Korean (Manga_KR)**: Manhwa-specific translations
- **Chinese (Manga_CN)**: Manhua translations
- **Novel profiles**: Separate profiles for text-only content

### Context Window Management
- **Rolling Window**: Maintains recent context only
- **Reset on Limit**: Clears history at threshold
- **Dynamic Adjustment**: Based on token limits
- **Export/Import**: Save translation sessions

### Batch Processing Options
- **Concurrent Chunks**: Process multiple sections simultaneously
- **Auto-retry Failed**: Automatic error recovery
- **Progress Persistence**: Resume interrupted translations
- **Resource Management**: CPU/memory optimization

---

## 🙏 Acknowledgments

Built using:
- OpenAI, Google, and Anthropic APIs
- Designed with assistance from ChatGPT & Claude
- Community feedback and contributions
- Open source libraries and tools
- comic-translate by ogkalu2 — https://github.com/ogkalu2/comic-translate

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Shirochi-stack/Glossarion/issues)
---

<p align="center">Made with 🌸 for the translation community</p>
