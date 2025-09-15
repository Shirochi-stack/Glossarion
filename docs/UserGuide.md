<p align="center">
  <img src="../assets/Halgakos.png" width="220" alt="Glossarion Logo" />
</p>

# 📖 Glossarion User Guide

A comprehensive guide for using Glossarion - your AI-powered EPUB translator for Korean, Japanese, and Chinese light novels.

---

## 📑 Table of Contents

1. [Getting Started](#-getting-started)
2. [Main Interface Overview](#-main-interface-overview)
3. [Translation Workflow](#-translation-workflow)
4. [Advanced Features](#-advanced-features)
5. [Glossary Management](#-glossary-management)
6. [Image Translation](#-image-translation)
7. [Quality Assurance](#-quality-assurance)
8. [EPUB Export](#-epub-export)
9. [Configuration & Settings](#-configuration--settings)
10. [Troubleshooting](#-troubleshooting)
11. [Best Practices](#-best-practices)
12. [FAQ](#-frequently-asked-questions)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- API key from one of the supported providers:
  - OpenAI (GPT-4, o4)
  - Google (Gemini)
  - DeepSeek
  - Anthropic (Claude/Sonnet)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shirochi-stack/Glossarion.git
   cd Glossarion
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or use the provided batch file:
   ```bash
   install_requirements.bat
   ```

3. **Launch the application:**
   ```bash
   python translator_gui.py
   ```
   
   Or use the launcher scripts:
   - Windows: `launch_Glossarion.bat` or `launch_Glossarion.vbs`

---

## 🖥️ Main Interface Overview

### Key Components

1. **EPUB File Selection**
   - Browse and select your source EPUB file
   - Supported formats: `.epub` files only

2. **Model Selection**
   - Choose your AI model (GPT-4, Gemini, DeepSeek, etc.)
   - Different models have different capabilities and costs

3. **Language Profile**
   - Select source language (Korean, Japanese, Chinese)
   - Create custom profiles with specific translation rules

4. **API Configuration**
   - Enter your API key
   - Set temperature and token limits
   - Configure API call delays

5. **System Prompt**
   - Customize translation style
   - Add specific instructions for the AI

6. **Control Buttons**
   - Run Translation
   - Extract Glossary
   - EPUB Converter
   - QA Scan

---

## 📚 Translation Workflow

### Step 1: Prepare Your EPUB

1. Click **Browse** next to "EPUB File"
2. Select your source EPUB file
3. The output folder will be created automatically based on the EPUB filename

### Step 2: Configure Translation Settings

1. **Select Model**: Choose your preferred AI model
   - `gpt-4.1`: Great quality, highe cost
   - `gpt-4.1-mini`: Good balance of quality and cost
   - `gemini-2.0-flash`: Fast and efficient
   - `deepseek-chat`: Cost-effective alternative

2. **Choose Language Profile**: Select the source language
   - Korean, Japanese, or Chinese
   - Each profile has optimized prompts

3. **Enter API Key**: 
   - Paste your API key
   - Click "Show" to verify it's correct

4. **Set Translation Parameters**:
   - **Temperature** (0.0-1.0): Lower = more consistent, Higher = more creative
   - **Translation History Limit**: Number of previous exchanges to remember
   - **API Call Delay**: Seconds between API calls (prevents rate limiting)

### Step 3: Customize System Prompt (Optional)

The system prompt controls translation style. Default prompts include:
- Honorific retention (-nim, -san, -sama)
- Context-rich translation
- Onomatopoeia in Romaji

You can modify these or create your own.

### Step 4: Start Translation

1. Click **Run Translation**
2. Monitor progress in the log window
3. Translation creates:
   - Individual HTML files for each chapter
   - Metadata file
   - Progress tracking file

### Step 5: Build Final EPUB

1. After translation completes, click **EPUB Converter**
2. Select the output folder containing translated files
3. The final EPUB will be created with "_translated" suffix

---

## 🔧 Advanced Features

### Chapter Range Selection

Translate specific chapters only:
```
5-10    # Chapters 5 through 10
15-20   # Chapters 15 through 20
```

### Token Limit Management

- **Input Token Limit**: Maximum tokens per API call
  - Default: 1,000,000 tokens
  - Can be disabled for unlimited (risky!)
  
- **Output Token Limit**: Maximum response length
  - Default: 8,192 tokens
  - Click button to customize

### Contextual Translation

- **Enabled**: Maintains conversation history between chapters
- **Disabled**: Each chapter translated independently
- Recommended: Keep enabled for better consistency

### Other Settings Menu

Access advanced options via **⚙️ Other Settings**:

#### Context Management
- **Rolling Summary**: Maintains story context across long translations
- **Summary Role**: How summaries are presented to the AI

#### Response Handling
- **Auto-retry Truncated**: Automatically retry if response is cut off
- **Auto-retry Duplicates**: Detect and retry when AI returns same content
- **Auto-retry Slow Chunks**: Retry chunks that take too long

#### Processing Options
- **Emergency Paragraph Restoration**: Fix wall-of-text responses
- **Reset Failed Chapters**: Automatically retry failed chapters
- **Comprehensive Chapter Extraction**: Extract ALL files (not just main content)

---

## 📓 Glossary Management

### Automatic Glossary Generation

1. Click **Extract Glossary** after selecting an EPUB
2. The system extracts:
   - Character names with honorifics
   - Titles and ranks
   - Locations
   - Relationships

3. Output saved to `Glossary/` folder:
   - `.json` format for use in translation
   - `.md` format for human reading

### Glossary Settings

In **Other Settings**:
- **Min Frequency**: Minimum appearances required (default: 2)
- **Max Names**: Maximum character names to extract (default: 50)
- **Max Titles**: Maximum titles/ranks to extract (default: 30)
- **Translation Batch**: Terms per API call (default: 50)

### Manual Glossary Override

1. Click **Load Glossary** 
2. Select a pre-made glossary JSON file
3. This will be used instead of automatic extraction

### Glossary Trimming

1. Click **Trim Glossary**
2. Adjust retention settings:
   - Number of entries to keep
   - Which fields to preserve/remove
3. Options:
   - Aggregate locations into summary
   - Delete empty fields

---

## 🖼️ Image Translation

### Overview

Glossarion can extract and translate text from images within EPUBs, perfect for:
- Web novel screenshots
- Chapter title images
- Embedded text graphics
- Long scrolling images

### Configuration

Enable in **Other Settings** → **Image Translation**:

1. **Enable Image Translation**: Main toggle
2. **Include Long Images**: Process web novel-style tall images
3. **Settings**:
   - **Min Height**: Minimum pixel height to consider (default: 1000px)
   - **Image Output Token Limit**: Max tokens for image responses
   - **Max/Chapter**: Maximum images to process per chapter
   - **Chunk Height**: Pixels per chunk for tall images

### Supported Models

Vision-capable models only:
- Gemini 1.5 Pro/Flash
- GPT-4V/GPT-4o
- Gemini 2.0 models

### How It Works

1. Detects images with potential text content
2. For tall images (>2000px), splits into chunks
3. Sends to vision API for OCR and translation
4. Embeds translations below original images
5. Option to hide labels and remove OCR'd images

---

## 🔍 Quality Assurance

### QA Scan Features

The QA scanner checks for:
- Duplicate translations
- Non-English text fragments
- Spacing issues
- Repetitive sentences
- HTML structure problems

### Running QA Scan

1. Click **QA Scan**
2. Select folder with translated HTML files
3. Review generated reports:
   - `qa_report.json`: Detailed findings
   - `qa_report.csv`: Spreadsheet format
   - `qa_report.html`: Visual report

### Common Issues Found

- **Duplicates**: Same translation used for different chapters
- **Language Fragments**: Untranslated Korean/Japanese/Chinese text
- **Formatting**: Missing spaces, broken HTML tags
- **Repetition**: AI loops or repeated phrases

---

## 📚 EPUB Export

### Building the Final EPUB

1. Ensure translation is complete
2. Click **EPUB Converter**
3. Select the output folder
4. The converter will:
   - Validate all files are present
   - Compile chapters in order
   - Embed images and CSS
   - Generate navigation files
   - Create final EPUB

### EPUB Structure Validation

Use **🔍 Validate EPUB Structure** in Other Settings to check:
- Required files present (container.xml, OPF, NCX)
- Chapter files exist
- Images properly linked
- Metadata complete

### Output Location

Final EPUB saved as:
```
[original_name]_translated.epub
```

---

## ⚙️ Configuration & Settings

### config.json Structure

All settings saved automatically:
```json
{
  "model": "gpt-4o",
  "language": "korean",
  "temperature": 0.3,
  "api_key": "your-key-here",
  "contextual": true,
  "delay": 2,
  ...
}
```

### Profile Management

1. **Save Language Profile**:
   - Modify system prompt
   - Click "Save Language"
   - Creates reusable profile

2. **Import/Export Profiles**:
   - Share profiles with others
   - Backup your custom prompts
   - Import community profiles

### Performance Tuning

- **API Delays**: Increase if hitting rate limits
- **Token Limits**: Decrease if getting errors
- **Chunk Size**: Smaller chunks = more API calls but better reliability
- **Temperature**: 0.3-0.5 recommended for consistency

---

## 🛠️ Troubleshooting

### Common Issues

#### "API Key Invalid"
- Verify key is correct
- Check API account has credits
- Ensure correct model selected for your API

#### "Chapter Translation Failed"
- Check token limits aren't exceeded
- Increase API delay
- Enable retry options in Other Settings

#### "EPUB Converter Failed"
- Run EPUB Structure Validation
- Check all chapters translated
- Ensure no files were manually deleted

#### "Out of Memory"
- Process smaller chapter ranges
- Reduce token limits
- Close other applications

### Debug Mode

Check `Payloads/` folder for:
- `translation_payload.json`: What was sent
- `translation_response.txt`: What was received
- Progress tracking files

### Log Files

All operations logged in main window:
- 🚀 Starting operations
- ✅ Successful completions
- ⚠️ Warnings
- ❌ Errors

---

## 💡 Best Practices

### For Best Translation Quality

1. **Use Latest Models**: GPT-4.1 or Gemini 1.5 Pro
2. **Enable Contextual**: Maintains story consistency
3. **Generate Glossary First**: Ensures name consistency
4. **Custom Prompts**: Add series-specific instructions
5. **QA Check**: Always run QA scan before final export

### For Best Performance

1. **Reasonable Token Limits**: 100k-200k input tokens
2. **API Delays**: 2-3 seconds prevents rate limiting
3. **Chapter Ranges**: Process 10-20 chapters at a time
4. **Enable Retries**: Handles transient failures

### For Cost Management

1. **Use Efficient Models**: Gemini Flash or GPT Nano / Mini models
2. **Disable Image Translation**: If not needed
3. **Lower Token Limits**: Reduces per-call cost
4. **Process Critical Chapters**: Translate important parts first

---

## ❓ Some Questions

### General

**Q: How long does translation take?**
A: Roughly 30-60 seconds per chapter, depending on:
- Chapter length
- API response time
- Retry settings

**Q: Can I resume interrupted translations?**
A: Yes! Progress is saved automatically. Just run translation again.

### Troubleshooting

**Q: Why are some chapters skipped?**
A: Chapters are skipped if:
- Already translated (file exists)
- Duplicate content detected
- Empty chapters

**Q: How do I retranslate specific chapters?**
A: Use the **Retranslate** button to:
- Select specific chapters
- Force retranslation
- Delete old files

**Q: Why is my translation truncated?**
A: Increase Output Token Limit or enable Auto-retry Truncated

### Advanced

**Q: Can I use custom API endpoints?**
A: Yes, just type it in the dropdown.

**Q: How do I translate multiple EPUBs?**
A: Process one at a time currently. Batch processing planned for a later release.

**Q: Can I customize the output format?**
A: EPUB output follows standard format. HTML files can be manually edited before final conversion.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Shirochi-stack/Glossarion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Shirochi-stack/Glossarion/discussions)
- **Updates**: Watch the repository for new releases

---

## 🎯 Quick Start Checklist

- [ ] Install Python 3.10+
- [ ] Run `pip install -r requirements.txt`
- [ ] Get API key from provider
- [ ] Launch with `python translator_gui.py`
- [ ] Select EPUB file
- [ ] Enter API key
- [ ] Click "Run Translation"
- [ ] Run QA Scan
- [ ] Click "EPUB Converter"
- [ ] Enjoy your translated novel! 📖

---

*Happy translating with Glossarion! 🚀*
