# 📖 Glossarion User Guide (GUI Only)

This guide explains every button, toggle, and setting in the Glossarion GUI. It is designed for non-technical users.

**Note:**  
- **API Key** = The password that lets the app talk to the AI (e.g., OpenAI, Google, Anthropic).  
- **Model** = The AI brain doing the work (e.g., `gpt-5.2`, `claude-4.5-sonnet`, `gemini-3.0-flash-preview`).

---

## 🚀 Getting Started

### 1. Launch the Program
Double-click one of the executables:
- **N_Glossarion_NoCuda_v6.6.9.exe**: Full Manga translation features.
- **Glossarion_v6.6.9.exe**: Limited manga translation (Requires bubble detector and inpainter to be manually toggled off)
- **L_Glossarion_Lite_v6.6.9.exe**: (Recommended) Lightweight version. No Manga Translation. Best for most users.

### 2. The Main Window
The top bar controls your translation:
- **Input File(s)**: Click **Browse ▼** to select your file (`.epub`, `.txt`, `.pdf`, etc.) or folder.
- **Model**: Choose the AI model (e.g., `gpt-5.2`, `deepseek-chat`). You can also type in this box.
- **Profile**: Selects the instruction set for the AI.
  - **Universal**: Good for most files.
  - **Korean/Japanese/Chinese**: Optimized prompts for these languages.
  - **_OCR profiles** (e.g., `korean_OCR`): Use when translating text from images.
  - **_TXT profiles**: REQUIRED if your input is a text file (`.txt`, `.md`) or you use the `html2text` extraction mode.
- **Save/Delete Profile**: Saves your custom system prompt or deletes the selected profile.

---

## 1️⃣ Novel Translation Pipeline

This is the main tab for translating books and text files.

### 🔑 Essential Settings
- **API Key**: Paste your AI provider's key here. Click **Show** to verify.
- **Target Language**: Choose the language you want to read (e.g., "English").
- **Start Translation**: Begins the process.
- **Stop**: Pauses/Cancels the operation safely.

### ⚙️ Main Settings (Middle Section)
- **Threading delay (s)**: How long to wait between internal tasks. Default `0.5` is fine.
- **API call delay (s)**: How long to wait between AI requests. Increase this if you get "Rate Limit" errors.
- **Chapter range**: Translate specific chapters (e.g., `1-10` or `5,7,9`). Leave empty for all.
- **Input Token Limit**: The maximum amount of text sent to the AI at once.
  - **Disable Input Token Limit**: (Red button) Click to remove the limit (risky for huge chapters).
  - **Enable Input Token Limit**: (Green button) Click to set a safe limit (e.g., `200000`).
- **Contextual Translation**: Check this to let the AI "remember" previous text for better flow.
  - **Translation History Limit**: How many previous chunks/sentences the AI remembers.
  - **Rolling History Window**: Keeps the *most recent* history instead of clearing it after every chapter.
- **Temperature**: Controls AI creativity (0.0 to 1.0). Lower (e.g., `0.3`) is more accurate; higher is more creative.
- **Batch Translation**: Groups multiple small chapters into one request to save time.
  - **Batch Size**: Number of chapters to send at once.
  - **Spinning Icon**: Indicates batch mode is active.

### 🧩 Other Settings (The "⚙️ Other Setting" Button)
Click this to open advanced options.

#### **Context Management & Memory**
- **Include previous source text in history**: (Not Recommended) Sends the original foreign text back to the AI as context.
- **Use Rolling Summary (Memory)**: Creates a running summary of the story so far. **⚠ Do not use with Contextual Translation.**
  - **Role**: How the summary is sent (`system`, `user`, or `both`). Default `system` is best.
  - **Summarize last N exchanges**: How often to update the summary.
  - **Retain N entries**: How many past summaries to keep (if Mode is `append`).
  - **Configure Memory Prompts**: Edit the instructions the AI uses to create summaries.
  
  #### Rolling Summary Modes
	- **Append Mode**
	  - Generates an isolated summary for each chapter.
	  - Appends each summary to `rolling_summary.txt`.
	  - The **Retain N entries** setting controls how many isolated summaries are included in the request.
	  - **Example:**  
		If **Retain N entries** is set to **5**, the last **5 isolated chapter summaries** will be included as context in the translation request.

	- **Replace Mode**
	  - Generates a summary of the **last N chapters**.
	  - The summary context is **persisted and carried forward** to subsequent requests.
	  - **Example:**  
		If set to summarize the last **5 chapters**, the assistant prompt will always include the summary of those **previous 5 chapters** when generating the summary for the latest chapter.

#### **Response Handling & Retry Logic**
- **GPT-5 / OR Thinking**: Enables "reasoning" features for newer models (like OpenAI's o1/o3 or DeepSeek R1).
  - **Effort**: How hard the AI should think (`low`, `medium`, `high`).
- **Gemini Thinking Mode**: Enables reasoning for Google Gemini models.
  - **Budget**: Token limit for thinking (set to `-1` for auto).
- **DeepSeek Thinking Mode**: Enables DeepSeek's specific reasoning logic.
- **Parallel Extraction**: Uses multiple threads to read the EPUB faster.
  - **Workers**: Number of threads (e.g., `4`).
- **GUI Yield**: Adds tiny pauses so the window doesn't freeze during heavy work. Keep checked.
- **Multi-Key Mode**: (See Section 7).
- **Auto-retry Truncated Responses**: If the AI stops in the middle of a sentence, it tries again automatically.
- **Preserve Original Text on Failure**: If translation fails, it saves the original foreign text instead of an error.
- **Compression Factor**: Controls how much text fits into a chunk based on the token limit. `Auto` is recommended.
- **Auto-retry Duplicate Content**: Detects if the AI repeats itself (a common bug) and retries.
  - **Detection Method**: `Basic` (fast) or `AI Hunter` (smarter but slower).
- **Auto-retry Slow Chunks**: Retries if the AI takes too long (timeout).

#### **Processing Options**
- **Process Web Novel Images**: Extracts and translates images found inside the book.
- **Hide Image Translation Label**: Hides the "Translated by Glossarion" text on images.
- **Disable Automatic Cover Creation**: Prevents generating a default cover if one is missing.
- **Emergency Paragraph Restoration**: Attempts to fix "wall of text" issues where paragraph breaks are lost.
- **Reset Failed Chapters**: Automatically deletes failed output files so they can be retried.
- **Render PDF as HTML**: Converts PDF pages to HTML text before translating.
- **Extraction Mode**:
  - `BS4 (Traditional)`: Standard HTML parsing.
  - `Text (html2text)`: Converts everything to plain text first. **Use `_txt` profiles with this.**

#### **Anti-Duplicate & Anti-Loop (AI Hunter)**
Advanced controls to stop the AI from repeating itself.
- **Lookback Chapters**: How far back to check for duplicates.
- **Similarity Threshold**: How similar text must be to count as a duplicate (0.85 = 85%).
- **Strict Line Matching**: Checks line-by-line.
- **Input Filtering**: Removes repetitive phrases *before* sending to AI.

---

## 2️⃣ Manga Translation Pipeline

A dedicated tool for translating comics, manga, and manhwa.

1. **Open Manga Translator**: Click the button in the main window.
2. **Load Images**: Select a folder of images or a `.cbz`/`.zip` file.
3. **Settings (Left Panel)**:
   - **Translator**: Choose your AI model (e.g., `gpt-5.2`, `claude-3.5-sonnet`, `gemini-3.0-flash`).
   - **OCR Provider**: The tool that reads the text from the image.
     - `manga-ocr`: Best for Japanese.
     - `rapidocr`: Fast, good for general use.
     - `google-vision` / `azure`: Cloud-based (requires their specific keys).
   - **Source Language**: Select the language of the manga.
4. **Operation**:
   - **Render**: Shows the current page.
   - **Detect Text**: Finds speech bubbles.
   - **OCR**: Reads the text in the bubbles.
   - **Translate**: Translates the text.
   - **Inpaint**: Erases the original text.
   - **Render Text**: Puts the English text into the bubbles.
   - **Batch Translation**: Processes the entire folder/file automatically.
5. **Advanced Manga Settings**: (Click the **Settings** button in the Manga window)
   - **Inpainter**: Choose how to erase text (`telea` is fast, `navier-stokes` is standard, `deep-fill` is high quality).
   - **Font Settings**: Choose font, auto-sizing, and color.
   - **Bubble Detection**: Adjust confidence sensitivity if bubbles are missed.

---

## 3️⃣ Retranslation Pipeline

Fix specific chapters without redoing the whole book.

1. **Select Folder**: Click **Retranslation** tab -> **Select Folder**. Pick your output folder (e.g., `MyBook_translated`).
2. **View Status**:
   - ✅ **Green**: Success.
   - ❌ **Red**: Failed or failed QA check.
   - ⬜ **White**: Not translated yet.
   - 🔄 **Spinning**: In progress.
3. **Select Files**: Check the boxes next to chapters you want to fix.
4. **Action**:
   - **Retranslate Selected**: Deletes the old file and translates it again.
   - **Refresh List**: Updates the status icons.

---

## 4️⃣ Glossary Manager

Glossaries help the AI keep names consistent (e.g., ensuring "Kim Dokja" doesn't become "Reader Kim"). Click **Glossary Manager** (or **Extract Glossary**) to open the comprehensive configuration window.

### 📋 Manual Glossary Extraction (Tab 1)
Use this tab to scan your book *before* translating to create a master glossary.

*   **Entry Type Configuration**: Select which types of terms to find (Characters, Terms). You can add **Custom Types** (e.g., "Skill", "Location") here.
*   **Custom Fields**: Add extra columns to your glossary (e.g., "Description", "Age").
*   **Duplicate Detection**: Settings to prevent duplicate entries (see below for algorithm details).
    *   **Detection Algorithm**: Choose how aggressively to merge similar names.
    *   **Fuzzy Matching Threshold**: Slide right (0.95+) for strict matching, left (0.70) for loose matching.
    *   **Disable honorifics filtering**: If checked, "Kim-nim" and "Kim" are treated as different names.
*   **Target Language**: The language for the `translated_name` column.
*   **Extraction Settings**:
    *   **Temperature**: AI creativity (lower is better for factual extraction).
    *   **Context Limit**: How much surrounding text to analyze.
    *   **Merge Count**: How many small chapters to process together.

### 🤖 Automatic Glossary Generation (Tab 2)
This configures the glossary that runs *automatically during* the translation process.

*   **Enable Automatic Glossary Generation**: Master switch. Uses the AI to find names in each chunk as it translates.
*   **Append Glossary to System Prompt**: Sends the found terms to the AI to ensure consistency.
*   **Add Additional Glossary**: Load an external CSV/JSON file to always include in the translation context.
*   **Compress Glossary Prompt**: **(Recommended)** Only sends terms *actually present* in the current text chunk to save money/tokens.
*   **Include Gender / Description**: Adds extra context columns to the extracted glossary.
*   **Disable Smart Filtering**: (Advanced) If checked, sends FULL text to the extractor (Expensive!). Default is unchecked (Smart Filter enabled).
*   **Targeted Extraction Settings** (Sub-tab):
    *   **Min Frequency**: A name must appear this many times to be saved.
    *   **Max Names/Titles**: Limits list size per chapter.
    *   **Context Window**: Sentences to analyze around a name for gender detection.

### 📝 Glossary Editor (Tab 3)
A built-in tool to view and clean your glossary files.
*   **Load/Browse**: Open a `.csv` or `.json` glossary.
*   **Clean Empty Fields**: Removes columns with no data.
*   **Remove Duplicates**: Runs the duplicate detection algorithm on the loaded file.

### 🔍 Duplicate Detection Algorithms
The **Duplicate Detection** settings (in Manual Tab) control how the system merges similar names (e.g., "Jon" vs "John").

*   **Algorithm Modes**:
    *   **Auto**: (Recommended) Uses all available methods (Token sort, Partial match, Jaro-Winkler) and picks the best match.
    *   **Strict**: High precision (95%+ similarity required). Only merges very obvious duplicates.
    *   **Balanced**: Good middle ground. Handles word order ("Park Ji-sung" = "Ji-sung Park") and substrings.
    *   **Aggressive**: Lowers threshold to 80%. Catches variants like "Catherine" vs "Katherine", but might over-merge different names.
    *   **Basic Only**: Simple character difference count (Levenshtein). Fast but less smart.

*   **How it works (The Algorithm)**:
    1.  **Pass 1 (Raw Name)**: Compares the original foreign names. If they are similar (based on your Threshold and Mode), the entries are merged. Honorifics (e.g., -san, -nim) are stripped before comparison unless disabled.
    2.  **Pass 2 (Translated Name)**: Checks the English translations. If two entries have the *exact same* translation (e.g., Raw "A" -> "Sword", Raw "B" -> "Sword"), one is removed to prevent conflicts. The entry with more details (Gender, Description) is kept.

---

## 5️⃣ EPUB Converter

Builds your final book after translation.

1. **Select Input**: Click **EPUB Converter** -> Select the folder containing your translated HTML files.
2. **Options**:
   - **Attach CSS**: Adds styling (fonts, margins) to the book.
   - **Force NCX only**: Creates a simple, compatible table of contents.
   - **Retain source extension**: Keeps `.xhtml` if the original used it (helps with compatibility).
3. **Convert**: Creates `[OriginalName]_translated.epub` in the same folder.

**Tip**: Use **Validate EPUB Structure** (in Other Settings) if conversion fails. It checks for missing files.

---

## 6️⃣ QA Scanner

Checks your translation for quality issues.

1. **Select Mode**:
   - `Quick`: Fast check.
   - `Strict`: Detailed check.
   - `Custom`: Choose specific rules.
2. **Inputs**:
   - **Source**: The original untranslated EPUB/Text.
   - **Translated Folder**: The folder with your finished HTML files.
   - **Auto-search**: Tries to find the folder automatically.
3. **Checks**:
   - **Word Count Mismatch**: Did the AI skip a huge chunk of text?
   - **Broken HTML**: Are tags like `<p>` or `<b>` broken?
   - **Untranslated Text**: Is there still Korean/Japanese left?
   - **Duplicates**: Did the AI accidentally repeat a chapter?
4. **Run Scan**: Produces a report (`qa_report.html`). Failed chapters are marked ❌ in the Retranslation tab.

---

## 7️⃣ Single / Multi API Key Mode

### Single Key Mode
- Just paste one key in the main window. Simple and easy.

### Multi-Key Mode (Rotation)
Use this if you have multiple keys or hit rate limits often.
1. Go to **Other Settings** -> **Configure API Keys**.
2. **Add Key**: Enter a key and select its provider/model.
3. **Enable Multi-Key Mode**: Check the box.
4. **Behavior**:
   - The app will rotate through your keys.
   - If one key hits a limit (Error 429), it switches to the next one instantly.
   - You can set "Fallback Keys" for specific models.

---

## 8️⃣ Custom API Endpoints & Local LLMs

You can connect Glossarion to local AI models (via Ollama, vLLM, LM Studio) or use custom OpenAI-compatible endpoints.

### 🔌 Configuring a Custom Endpoint
1. Go to **Other Settings** -> Scroll down to **Custom API Endpoints**.
2. **Enable Custom OpenAI Endpoint**: Check this box.
3. **Override API Endpoint**: Enter your full API base URL.
   - **Ollama**: `http://localhost:11434/v1`
   - **LM Studio**: `http://localhost:1234/v1`
   - **vLLM**: `http://localhost:8000/v1`
4. **API Key**: In the main window, you can usually enter any dummy key (e.g., `sk-dummy`) if running locally, or your actual key if using a private server.
5. **Model Name**: In the main window's **Model** box, type the name of your local model exactly as it appears in your runner (e.g., `llama3`, `mistral:instruct`).

### ⚙️ Advanced Endpoint Settings
Click **▼ Show More Fields** in the Custom Endpoints section to see additional options:
*   **Groq/Local Base URL**: Dedicated field for Groq or specific local setups if separate from the main override.
*   **Azure API Version**: Select the API version if connecting to Azure OpenAI.

---
