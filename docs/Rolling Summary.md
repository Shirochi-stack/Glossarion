# Enhanced Rolling Summary (Memory) System

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Configuration Options](#configuration-options)
4. [How It Works](#how-it-works)
5. [Customizing Memory Prompts](#customizing-memory-prompts)
6. [Understanding Summary Modes](#understanding-summary-modes)
7. [Practical Examples](#practical-examples)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Introduction

The Enhanced Rolling Summary system, also called the "Memory" system, is an advanced feature in Glossarion v2.0 that maintains translation context across long documents. It works similarly to how AI assistants maintain conversation memory, creating intelligent summaries of previous translations before the conversation history is cleared.

### Key Benefits
- **Consistency**: Maintains character names, terminology, and relationships throughout the translation
- **Context Preservation**: Important plot points and developments are never forgotten
- **Style Continuity**: Preserves the translation tone and style patterns
- **Reduced Errors**: Prevents the AI from contradicting earlier translations
- **Better Flow**: Creates smoother narrative transitions between chapters

---

## Getting Started

### Enabling the Memory System

1. Open **Other Settings** from the main window
2. In the **Context Management & Memory** section, enable **"Use Rolling Summary (Memory)"**
3. Click **Save Settings**

The system will now automatically create summaries when your translation history reaches its limit.

---

## Configuration Options

### Basic Settings

| Setting | Description | Default | Recommended |
|---------|-------------|---------|-------------|
| **Use Rolling Summary** | Enable/disable the memory system | Off | On for novels |
| **Summary Role** | How the AI receives the summary (`user` or `system`) | user | user |
| **Mode** | `append` (build history) or `replace` (keep latest only) | append | append for complex stories |
| **Summarize last X exchanges** | Number of conversation pairs to include in each summary | 5 | 3-8 depending on chapter length |

### Accessing Settings

1. Click **⚙️ Other Settings** in the main window
2. Find the **Context Management & Memory** section (top-left)
3. Adjust settings as needed
4. Click **⚙️ Configure Memory Prompts** to customize the AI prompts

---

## How It Works

### The Memory Cycle

Translation begins → History builds up
↓
History limit reached → Memory system activates
↓
AI creates summary → Preserves key information
↓
History clears → Summary remains as context
↓
Translation continues → With preserved memory


### When Summaries Are Created

The memory system activates when:
- Your translation history reaches the configured limit (set in "Transl. Hist. Limit")
- Before the conversation history is cleared
- Automatically, without interrupting your workflow

### What Gets Summarized

The AI extracts and preserves:
- **Character Information**: Names (with original forms), relationships, roles, and developments
- **Plot Points**: Key events, conflicts, and story progression
- **Locations**: Important places and settings with original names
- **Terminology**: Special terms, abilities, items, or concepts
- **Tone & Style**: Writing style, mood, and notable patterns
- **Unresolved Elements**: Ongoing situations, mysteries, or questions

---

## Customizing Memory Prompts

### Accessing Prompt Configuration

1. In **Other Settings**, click **⚙️ Configure Memory Prompts**
2. Two text areas will appear:
   - **System Prompt**: Defines the AI's role
   - **User Prompt Template**: Specifies what to extract

### System Prompt

The system prompt tells the AI how to behave when creating summaries. The default is:
You are a context summarization assistant. Create concise, informative summaries
that preserve key story elements for translation continuity.

You can customize this for specific genres:

**For Fantasy/Cultivation Novels:**
You are a context summarization assistant for fantasy/cultivation novels.
Focus on power systems, cultivation levels, sect affiliations, and magical
terminology while preserving character relationships and plot progression.

**For Romance Novels:**
You are a context summarization assistant for romance novels. Emphasize
character relationships, emotional developments, romantic tensions, and
relationship milestones while maintaining plot continuity.

### User Prompt Template

The user prompt template uses `{translations}` as a placeholder for the actual content. You can customize what information to prioritize:

**Default Template:**
Analyze the recent translation exchanges and create a structured summary for context continuity.
Focus on extracting and preserving:

Character Information: Names (with original forms), relationships, roles, and important character developments
Plot Points: Key events, conflicts, and story progression
Locations: Important places and settings
Terminology: Special terms, abilities, items, or concepts (with original forms)
Tone & Style: Writing style, mood, and any notable patterns
Unresolved Elements: Questions, mysteries, or ongoing situations

Format the summary clearly with sections. Be concise but comprehensive.
Recent translations to summarize:
{translations}

---

## Understanding Summary Modes

### Append Mode (Recommended)

Builds a complete history file with all summaries:

**Advantages:**
- Complete story progression record
- Can review earlier context
- Best for complex, interconnected stories
- Helpful for maintaining long-term consistency

**File Output Example:**
=== Summary before chapter 5, chunk 1 ===
[2024-01-15 10:23:45]
[First summary content...]
=== Summary before chapter 10, chunk 1 ===
[2024-01-15 10:45:12]
[Second summary content...]
=== Summary before chapter 15, chunk 1 ===
[2024-01-15 11:02:33]
[Third summary content...]

### Replace Mode

Keeps only the most recent summary:

**Advantages:**
- Smaller context size
- Focuses on recent developments
- Good for episodic content
- Reduces token usage

**File Output Example:**
=== Latest Summary (Chapter 15, chunk 1) ===
[2024-01-15 11:02:33]
[Only the most recent summary content...]

---

## Practical Examples

### Example Summary Output

Here's what a typical summary looks like in your translation context:
[MEMORY] Previous context summary:
Main Characters:

Jin Woo (진우): E-rank hunter who acquired the Shadow Monarch system, currently Level 45
Cha Hae-in (차해인-ssi): S-rank hunter from the Hunters Guild, developing feelings for Jin Woo
Yoo Jin-ho (유진호): Jin Woo's friend and guild vice-master, addresses him as "형님" (hyung-nim)

Recent Developments:

Jin Woo successfully cleared the Demon Castle's 75th floor
Received the skill "Shadow Exchange" allowing instant teleportation
Hunters Association Chairman Go Gun-hee (고건희-회장님) offered Jin Woo a special position
Cha Hae-in noticed Jin Woo's unique scent has changed after his recent power-up

Key Terminology:

그림자 군단 (Shadow Army): Jin Woo's undead soldiers
던전 브레이크 (Dungeon Break): When monsters escape dungeons
각성자 (Awakened): People with supernatural abilities

Ongoing Situations:

The Architect's warning about the "Real Enemy" approaching
Jin Woo hiding his true power level from the public
Preparation for the upcoming A-rank dungeon raid

[END MEMORY]

### Visual Indicators in Log

When the memory system is active, you'll see these indicators:

- `📝 Generated rolling summary (append mode, 5 exchanges)` - Summary created
- `📚 Total summaries in memory: 3` - Running count in append mode
- `🔄 Reset glossary context after 3 chapters` - When history clears
- Memory-related messages appear in *green italic text* for easy identification

---

## Troubleshooting

### Common Issues and Solutions

**Issue: Summaries are too long/short**
- **Solution**: Adjust "Summarize last X exchanges" - fewer for concise, more for detailed

**Issue: Important details are missed**
- **Solution**: Customize the User Prompt to emphasize what's important for your genre

**Issue: Summaries aren't being created**
- **Solution**: 
  1. Ensure "Use Rolling Summary" is enabled
  2. Check that your Translation History Limit is not set too high
  3. Verify the feature is working by checking for `rolling_summary.txt` in your output folder

**Issue: Memory seems inconsistent**
- **Solution**: 
  1. Try increasing the number of exchanges to summarize
  2. Switch from "replace" to "append" mode
  3. Review and customize your prompts for better extraction

### Checking Summary Files

Summaries are saved in your translation output folder:
your_epub_name/
├── response_001_chapter_1.html
├── response_002_chapter_2.html
├── rolling_summary.txt          ← Your memory file
└── translation_progress.json

---

## Best Practices

### For Different Content Types

**Long Fantasy/Cultivation Novels:**
- Use **Append Mode** to track power progression
- Set **5-8 exchanges** per summary
- Customize prompts to track cultivation levels, techniques, and sect relationships

**Romance Novels:**
- Use **Append Mode** for relationship development
- Set **3-5 exchanges** per summary
- Focus prompts on emotional developments and relationship milestones

**Mystery/Thriller:**
- Use **Append Mode** to track clues
- Set **8-10 exchanges** for detailed preservation
- Emphasize unresolved elements and plot threads in prompts

**Slice of Life/Episodic:**
- Consider **Replace Mode** as episodes are self-contained
- Set **3-5 exchanges** per summary
- Focus on character consistency rather than plot progression

### Optimization Tips

1. **Start with defaults** - The default settings work well for most novels
2. **Monitor the output** - Check `rolling_summary.txt` after a few chapters
3. **Adjust gradually** - Make small changes to see their effect
4. **Genre-specific prompts** - Customize based on what's important in your content
5. **Balance detail** - Too much detail can confuse, too little loses context

### Token Usage Considerations

- Each summary adds to your context tokens
- Append mode uses more tokens over time
- Replace mode maintains consistent token usage
- Consider your API token limits when choosing modes

---

## Advanced Usage

### Combining with Other Features

**With Glossary System:**
- Memory preserves character name consistency
- Glossary provides the translation rules
- Both work together for maximum consistency

**With Contextual Translation:**
- Contextual = short-term memory (recent chapters)
- Rolling Summary = long-term memory (entire story)
- Enable both for best results

**With Batch Translation:**
- Memory still works but updates less frequently
- Consider increasing exchanges to summarize
- Review summaries more carefully

---

## Conclusion

The Enhanced Rolling Summary system is a powerful tool for maintaining consistency and context in long translations. By acting as the AI's "memory," it ensures that important details, character relationships, and plot developments are never lost, even when translating novels with hundreds of chapters.

Start with the default settings, monitor the results, and adjust based on your specific needs. The system is designed to work quietly in the background, improving your translation quality without requiring constant attention.

For support or questions about this feature, please refer to the Glossarion GitHub repository or community forums.

---

*Last updated for Glossarion v2.0.0*
This document is formatted to be easily added to your user guide. You can save it as a Markdown file (.md) or convert it to PDF/HTML for your documentation. The structure includes everything users need to understand and effectively use the rolling summary feature.