"""Adaptive script ban of the on-device engine ("Block other scripts in translations", Model manager ›
On-device; Prefs key ``local_llm_block_cjk`` kept from when it banned CJK only).

Small on-device models leak other writing systems into translations: Qwen3.5 0.8B slips Chinese
characters in ("kanji bleed", Korean -> English), Qwen3.5 2B drifts into whole Hindi (Devanagari) lines
(owner 2026-10-28). For a request that writes target-language text, every vocabulary token that produces
a character of a script the target language does not use gets the logit -inf; a request that must copy
source text (glossary extraction, OCR, a single-pass auto glossary whose answer carries the raw names) is
left alone.

Which requests ban which scripts, the script table and which token bytes count as a banned script live in
the shared desktop module ``script_ban_core`` (src/; the universal "Block other scripts in translations
(adaptive, all models)" setting of the API models uses the same table, owner 2026-10-28); they are
re-exported here. This module keeps what is llama.cpp's: the -inf logits processor and the per-model
vocabulary cache.

Cost: ``VocabScripts`` decodes the vocabulary once per loaded model (every token's script bits; the
ASCII tokens are skipped) and keeps one ``BanLogitsProcessor`` per set of banned scripts, so a second
request into the same target language costs nothing (``LlamaEngine.script_ban``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from script_ban_core import (BAN_TASKS, BANNED_RANGES, CJK, SCRIPTS, _banned, banned_scripts,  # noqa: F401
                             banned_token_ids, is_banned_char, is_cjk_target, script_mask, script_of,
                             target_scripts, token_is_banned, token_masks, wants_ban)

__all__ = ["BAN_TASKS", "BANNED_RANGES", "CJK", "SCRIPTS", "BanLogitsProcessor", "VocabScripts",
           "banned_scripts", "banned_token_ids", "is_banned_char", "is_cjk_target", "script_of",
           "target_scripts", "token_is_banned", "wants_ban"]


class BanLogitsProcessor:
    """llama-cpp-python logits processor: sets the banned ids' logits to -inf (numpy, per sampled token)."""

    def __init__(self, ids: Sequence[int], n_vocab: Optional[int] = None) -> None:
        import numpy as np

        array = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.intp)
        if n_vocab is not None:
            array = array[(array >= 0) & (array < int(n_vocab))]
        self.ids = array
        self._inf = -np.inf

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.ids.size:
            scores[..., self.ids] = self._inf
        return scores

    def __len__(self) -> int:
        return int(self.ids.size)


class VocabScripts:
    """A loaded model's vocabulary by script: decoded once (``pieces``: every token's raw bytes, in id
    order), then one cached ``BanLogitsProcessor`` per set of banned scripts."""

    def __init__(self, pieces: Iterable[bytes]) -> None:
        tokens, chars, tails = [], [], []
        n_vocab = 0
        for token, piece in enumerate(pieces):
            n_vocab = token + 1
            if not piece:
                continue
            char_bits, tail_bits = token_masks(piece)
            if char_bits or tail_bits:  # only tokens that touch some script are kept
                tokens.append(token)
                chars.append(char_bits)
                tails.append(tail_bits)
        self.n_vocab = n_vocab
        self._tokens, self._chars, self._tails = tokens, chars, tails
        self._processors: Dict[int, BanLogitsProcessor] = {}

    def __len__(self) -> int:
        """The number of tokens that touch some script of ``SCRIPTS``."""
        return len(self._tokens)

    def ids(self, scripts: Iterable[str]) -> list:
        mask = script_mask(scripts)
        return [token for token, chars, tail in zip(self._tokens, self._chars, self._tails)
                if _banned(chars, tail, mask)]

    def processor(self, scripts: Iterable[str]) -> BanLogitsProcessor:
        mask = script_mask(scripts)
        cached = self._processors.get(mask)
        if cached is None:
            cached = self._processors[mask] = BanLogitsProcessor(self.ids(scripts), self.n_vocab)
        return cached
