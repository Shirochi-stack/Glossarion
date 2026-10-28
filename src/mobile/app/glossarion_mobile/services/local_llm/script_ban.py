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

Emoticons (opt-in, config ``script_ban_allow_kaomoji``, off by default; the request body field
``glossarion_allow_kaomoji`` -> ``params["ban_allow_kaomoji"]``): the ``KAOMOJI_RANGES`` punctuation is
never banned, and the tokens made only of the characters the request's user messages use in emoticons
(``script_ban_core.kaomoji_source_chars``: ツ, ノ, ಠ, ٩ ...) are let through for that request. The
per-request part reads the source once (well under a millisecond for a chapter) and, only when it found
such characters, looks them up in an index of the vocabulary built on first use; the resulting processors
are kept per set of characters (``SOURCE_CACHE``).
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, Optional, Sequence

from script_ban_core import (BAN_TASKS, BANNED_RANGES, CJK, KAOMOJI_RANGES, SCRIPTS, _banned, _truthy,  # noqa: F401
                             banned_scripts, banned_token_ids, is_banned_char, is_cjk_target,
                             kaomoji_source_chars, script_mask, script_of, source_keys, target_scripts,
                             token_is_banned, token_masks, token_requirements, wants_ban)

#: processors kept per (banned scripts, emoticon characters of a source)
SOURCE_CACHE = 16
#: the lead bytes of every ``KAOMOJI_RANGES`` character (U+3000-U+30FF, U+FE10-U+FFEF): only tokens with
#: one of them can change when emoticons are allowed
_KAOMOJI_LEADS = (b"\xe3", b"\xef")

__all__ = ["BAN_TASKS", "BANNED_RANGES", "CJK", "KAOMOJI_RANGES", "SCRIPTS", "BanLogitsProcessor", "VocabScripts",
           "banned_scripts", "banned_token_ids", "is_banned_char", "is_cjk_target", "kaomoji_source_chars",
           "script_of", "target_scripts", "token_is_banned", "wants_ban"]


class BanLogitsProcessor:
    """llama-cpp-python logits processor: sets the banned ids' logits to -inf (numpy, per sampled token)."""

    def __init__(self, ids: Sequence[int], n_vocab: Optional[int] = None) -> None:
        import numpy as np

        array = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.intp)
        if n_vocab is not None:
            array = array[(array >= 0) & (array < int(n_vocab))]
        self.ids = array
        self._inf = -np.inf

    def without(self, ids: Sequence[int]) -> "BanLogitsProcessor":
        """A processor banning these ids except ``ids`` (numpy set difference, no Python loop)."""
        import numpy as np

        other = BanLogitsProcessor.__new__(BanLogitsProcessor)
        other.ids = np.setdiff1d(self.ids, np.asarray(list(ids), dtype=np.intp), assume_unique=True)
        other._inf = self._inf
        return other

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
        tokens, chars, tails, kept = [], [], [], []
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
                kept.append(bytes(piece))
        self.n_vocab = n_vocab
        self._tokens, self._chars, self._tails, self._pieces = tokens, chars, tails, kept
        self._processors: Dict[tuple, BanLogitsProcessor] = {}
        # emoticons allowed (built on first use): the changed masks, the per-character index
        self._kaomoji_masks: Optional[Dict[int, tuple]] = None
        self._requirements: Optional[list] = None
        self._index: Optional[Dict[Any, list]] = None
        self._source_processors: "collections.OrderedDict[tuple, BanLogitsProcessor]" = collections.OrderedDict()

    def __len__(self) -> int:
        """The number of tokens that touch some script of ``SCRIPTS``."""
        return len(self._tokens)

    def _masks(self, allow_kaomoji: bool):
        if not allow_kaomoji:
            return zip(self._tokens, self._chars, self._tails)
        if self._kaomoji_masks is None:  # only tokens with an E3 / EF byte can hold an exempt character
            self._kaomoji_masks = {position: token_masks(piece, True) for position, piece in enumerate(self._pieces)
                                   if any(lead in piece for lead in _KAOMOJI_LEADS)}
        changed = self._kaomoji_masks
        return ((token, *changed.get(position, (chars, tail)))
                for position, (token, chars, tail) in enumerate(zip(self._tokens, self._chars, self._tails)))

    def ids(self, scripts: Iterable[str], *, allow_kaomoji: bool = False) -> list:
        mask = script_mask(scripts)
        return [token for token, chars, tail in self._masks(bool(allow_kaomoji)) if _banned(chars, tail, mask)]

    def source_ids(self, scripts: Iterable[str], chars: Iterable[str]) -> list:
        """The tokens (with emoticons allowed) that only need ``chars`` (``kaomoji_source_chars``) of the
        banned ``scripts``: every character of a banned script they have, and their incomplete tail, comes
        from ``chars`` (``token_requirements``, ``source_keys``)."""
        mask = script_mask(scripts)
        keys = source_keys(chars)
        if not mask or not keys:
            return []
        if self._index is None:
            requirements, index = [], {}
            for position, piece in enumerate(self._pieces):
                needs = token_requirements(piece, True)
                requirements.append(needs)
                for key, _bits, _tail in needs:
                    index.setdefault(key, []).append(position)
            self._requirements, self._index = requirements, index
        found = []
        seen = set()
        for key in keys:
            for position in self._index.get(key, ()):
                if position in seen:
                    continue
                seen.add(position)
                if all(key in keys or not (_banned(0, bits, mask) if tail else bits & mask)
                       for key, bits, tail in self._requirements[position]):
                    found.append(self._tokens[position])
        return found

    def processor(self, scripts: Iterable[str], *, allow_kaomoji: bool = False, source: Any = None) -> BanLogitsProcessor:
        """The processor banning ``scripts``; with ``allow_kaomoji``, without the ``KAOMOJI_RANGES`` tokens and
        (``source``: the request's user text) the tokens of its emoticon characters."""
        mask = script_mask(scripts)
        allow = bool(allow_kaomoji)
        key = (mask, allow)
        cached = self._processors.get(key)
        if cached is None:
            cached = self._processors[key] = BanLogitsProcessor(self.ids(scripts, allow_kaomoji=allow), self.n_vocab)
        if not allow or not source:
            return cached
        chars = kaomoji_source_chars(source, scripts)
        if not chars:
            return cached
        source_key = (mask, chars)
        found = self._source_processors.get(source_key)
        if found is not None:
            self._source_processors.move_to_end(source_key)
            return found
        allowed = self.source_ids(scripts, chars)
        found = cached.without(allowed) if allowed else cached
        self._source_processors[source_key] = found
        while len(self._source_processors) > SOURCE_CACHE:
            self._source_processors.popitem(last=False)
        return found
