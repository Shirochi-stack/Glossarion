export const UNIVERSAL_PROMPT_TEMPLATE = `You are a professional novel translator. You MUST translate the following text to {target_lang}.
- You MUST output ONLY in {target_lang}. No other languages are permitted.
- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, <img>, etc.
- Preserve any Markdown formatting if present (e.g., headings '#', '##', '###', bold '**text**', italic '*text*', lists '- item'/'1. item', blockquotes '> quote', links '[text](url)', images '![alt](url)', inline code '`code`').
- If the text does not contain HTML tags, use line breaks for proper formatting as expected of a novel.
- Maintain the original meaning, tone, and style.
- Output ONLY the translated text in {target_lang}. Do not add any explanations, notes, or conversational filler.`;

export function buildUniversalPrompt(targetLanguage) {
  const lang = String(targetLanguage || "English").trim() || "English";
  return UNIVERSAL_PROMPT_TEMPLATE.replaceAll("{target_lang}", lang);
}

export function buildPageBatchMessages({ items, targetLanguage, sourceLanguage }) {
  const source = String(sourceLanguage || "Auto").trim() || "Auto";
  const target = String(targetLanguage || "English").trim() || "English";
  const systemPrompt = `${buildUniversalPrompt(target)}

You are translating visible text nodes from a live webpage. Return only valid JSON in this exact shape:
{"items":[{"id":"same-id","text":"translated text"}]}

Rules:
- Preserve each item's id exactly.
- Translate only the item's text field.
- Do not merge, split, reorder, omit, or invent items.
- Preserve inline spacing and punctuation meaning.
- Do not include markdown fences or commentary.`;

  const userPrompt = `Source language: ${source}
Target language: ${target}

Translate this JSON array:
${JSON.stringify(items, null, 2)}`;

  return [
    { role: "system", content: systemPrompt },
    { role: "user", content: userPrompt }
  ];
}
