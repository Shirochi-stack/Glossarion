"""English named-entity recognition (NER) for the glossary name check.

Runs ``dslim/bert-base-NER`` (CoNLL-2003: person / location / organization / misc; cased) as the
int8 ONNX export ``Xenova/bert-base-NER`` through ONNX Runtime, with a small pure-Python BERT
WordPiece tokenizer, so it needs neither PyTorch nor ``tokenizers``. The model (109 MB) is a
``manga_models`` registry entry (kind ``ner``) and downloads on first use into the ONNX cache
(``ONNX_CACHE_DIR``), verified and resumable like the manga models.

Available where ONNX Runtime, NumPy and ``manga_models`` ship (Heavy / NoCuda desktop builds and
Glossarion Mobile); the standard builds have none of them, so the name check keeps its
capitalization heuristic there (``runtime_available()`` is False).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = ["Entity", "NerTagger", "UNSUPPORTED_REASON", "get_tagger", "model_installed", "runtime_available"]

UNSUPPORTED_REASON = "Not supported in this build"
MAX_TOKENS = 510  # 512 positions minus [CLS] / [SEP]
_LABEL_TYPES = ("PER", "LOC", "ORG", "MISC")
# the letters joined to a name by a hyphen / apostrophe ("So-jeong", "Zhi'an"), but not an English
# contraction ("Walter's", "Yoo'll")
_JOINED_PART = re.compile(r"[-'’]([^\W\d_]+)")
_CONTRACTIONS = frozenset(("s", "t", "d", "m", "ll", "re", "ve"))


@dataclass(frozen=True)
class Entity:
    text: str
    label: str  # PER / LOC / ORG / MISC
    start: int
    end: int
    score: float


def runtime_available() -> bool:
    """ONNX Runtime, NumPy and the model registry are in this build."""
    try:
        return all(importlib.util.find_spec(name) is not None
                   for name in ("onnxruntime", "numpy", "manga_models"))
    except (ImportError, ValueError):
        return False


def _registry():
    import manga_models
    return manga_models


def model_installed() -> bool:
    if not runtime_available():
        return False
    models = _registry()
    return bool(models.is_installed(models.NER_MODEL_KEY))


def ensure_model(log: Optional[Callable[[str], None]] = None, cancel=None, progress=None) -> Optional[str]:
    """The model file, downloading it (and its vocabulary / labels) first when missing; ``progress``
    receives ``manga_models.DownloadProgress`` reports."""
    models = _registry()
    spec = models.get_spec(models.NER_MODEL_KEY)
    if not models.is_installed(spec):
        if log:
            log(f"📥 Downloading {spec.title} ({models.format_size(spec.size)}) for the glossary name check...")
        models.ensure([spec], cancel=cancel, progress=progress)
    return models.model_path(spec)


# ----------------------------------------------------------------------------------------- tokenizer
def _is_punctuation(ch: str) -> bool:
    code = ord(ch)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF
            or 0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F)


def basic_words(text: str) -> list:
    """BERT BasicTokenizer (cased): whitespace / punctuation / CJK split, with character offsets."""
    words = []
    start = None
    for i, ch in enumerate(text):
        code = ord(ch)
        if code == 0 or code == 0xFFFD or (unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\t\n\r"):
            if start is not None:
                words.append((start, i))
                start = None
            continue
        if ch.isspace():
            if start is not None:
                words.append((start, i))
                start = None
        elif _is_punctuation(ch) or _is_cjk(ch):
            if start is not None:
                words.append((start, i))
                start = None
            words.append((i, i + 1))
        elif start is None:
            start = i
    if start is not None:
        words.append((start, len(text)))
    return words


class WordPiece:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): index for index, line in enumerate(f)}
        self.unk = self.vocab["[UNK]"]
        self.cls = self.vocab["[CLS]"]
        self.sep = self.vocab["[SEP]"]

    def pieces(self, word: str) -> list:
        if len(word) > 100:
            return [self.unk]
        out = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            out.append(found)
            start = end
        return out


# ----------------------------------------------------------------------------------------- tagger
def ner_threads(workers: int = 1) -> int:
    """Cores one NER call uses (ONNX Runtime intra-op threads): at most 4, never more than the CPU has;
    with ``workers`` calls at once, the cores shared out between them (at least 1 each)."""
    return max(1, min(4, (os.cpu_count() or 1) // max(1, int(workers))))


class NerTagger:
    def __init__(self, model_path: str, threads: Optional[int] = None):
        import numpy as np
        import onnxruntime as ort

        folder = os.path.dirname(model_path)
        self._np = np
        self.wordpiece = WordPiece(os.path.join(folder, "vocab.txt"))
        with open(os.path.join(folder, "config.json"), "r", encoding="utf-8") as f:
            id2label = json.load(f)["id2label"]
        self.labels = [id2label[str(i)] for i in range(len(id2label))]
        options = ort.SessionOptions()
        self.model_path = model_path
        self.threads = threads or ner_threads()
        options.intra_op_num_threads = self.threads
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self._cache: dict = {}

    def with_threads(self, threads: int) -> "NerTagger":
        """This model with ``threads`` intra-op threads per call ("Chapters checked at once" shares the
        cores out); loaded once per count and kept, this tagger itself for its own count."""
        threads = max(1, int(threads))
        if threads == getattr(self, "threads", threads):
            return self
        key = (getattr(self, "model_path", ""), threads)
        with _tagger_lock:
            tagger = _taggers_by_threads.get(key)
            if tagger is None:
                tagger = _taggers_by_threads[key] = NerTagger(self.model_path, threads)
        return tagger

    def _windows(self, text: str):
        """Word spans with their WordPiece ids, grouped into <= MAX_TOKENS windows at line ends."""
        window, size = [], 0
        for start, end in basic_words(text):
            ids = self.wordpiece.pieces(text[start:end])
            if size + len(ids) > MAX_TOKENS and window:
                yield window
                window, size = [], 0
            window.append((start, end, ids[:MAX_TOKENS]))
            size += len(window[-1][2])
            if size >= MAX_TOKENS:
                yield window
                window, size = [], 0
        if window:
            yield window

    def _run(self, window: list) -> list:
        np = self._np
        ids = [self.wordpiece.cls]
        first_piece = []
        for _start, _end, pieces in window:
            first_piece.append(len(ids))
            ids.extend(pieces)
        ids.append(self.wordpiece.sep)
        feed = {"input_ids": np.array([ids], dtype=np.int64)}
        if "attention_mask" in self.inputs:
            feed["attention_mask"] = np.ones((1, len(ids)), dtype=np.int64)
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros((1, len(ids)), dtype=np.int64)
        # InferenceSession.run is thread-safe: parallel chapters share one session (glossary_name_check)
        logits = self.session.run(None, feed)[0][0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        # one label per word: its first sub-token's (HF aggregation "first"). The score is how sure the model
        # is that the word is part of a name at all (1 - P(O)), as it is often unsure of the type only (a
        # title's "Mirena Falls": person / place / misc, 0.58 at best, but 0.99 a name); a word whose name
        # labels together outweigh "O" takes the best of them ("Tasty Mirena": O 0.40, the rest 0.60)
        outside = self.labels.index("O")
        labels = []
        for i in first_piece:
            name = 1.0 - float(probs[i][outside])
            if name < 0.5:
                labels.append((outside, name))
                continue
            others = probs[i].copy()
            others[outside] = -1.0
            labels.append((int(others.argmax()), name))
        return labels

    def entities(self, text: str, min_score: float = 0.5) -> list:
        """Named entities in ``text`` (character spans), adjacent B-/I- words of one name merged; ``score``
        is the mean probability that its words are part of a name."""
        key = hash(text)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == text:
            return cached[1]
        found = []
        for window in self._windows(text):
            current = None
            for (start, end, _ids), (label_id, score) in zip(window, self._run(window)):
                label = self.labels[label_id]
                if label == "O":
                    if current:
                        found.append(current)
                    current = None
                    continue
                prefix, kind = label.split("-", 1)
                gap = text[current["end"]:start] if current else ""
                # I- continues the open entity across spaces / joining marks ("Si-hyeon", "O'Neil"), whatever type
                # the model gave it ("World Tree El" MISC + "Lamina" I-LOC); B- of its type only across a mark
                if current and (prefix == "I" or (kind == current["label"] and gap in ("-", "'", "’"))) \
                        and gap.strip(" -'’") == "" and "\n" not in gap:
                    current["end"] = end
                    current["scores"].append(score)
                    continue
                if current:
                    found.append(current)
                current = {"label": kind, "start": start, "end": end, "scores": [score], "begins": prefix == "B"}
            if current:
                found.append(current)
        # the hyphen / apostrophe inside a name is its own "O" word ("So-jeong", "Al'shel"): rejoin
        merged = []
        for item in found:
            prev = merged[-1] if merged else None
            if prev and prev["label"] == item["label"] and text[prev["end"]:item["start"]] in ("-", "'", "’"):
                prev["end"] = item["end"]
                prev["scores"].extend(item["scores"])
            else:
                merged.append(item)
        # ...and a part after the joining mark the model labelled "O" or another type ("Yoo So-jeong" came
        # back as "Yoo So", "Ju-in" as "Ju", "Tu Zhi'an" as "Tu Zhi", "Yuki-onna" as "Yuki") is the name's too,
        # unless it is a capitalized word the model starts another entity with ("Lee Da-yun-Style Coping Method")
        starts = {item["start"]: item for item in merged}
        found = []
        for item in merged:
            if found and item["start"] < found[-1]["end"]:
                continue  # the joined part of the name before it
            while True:
                part = _JOINED_PART.match(text, item["end"])
                if part is None or (part.group(0)[0] != "-" and part.group(1).casefold() in _CONTRACTIONS):
                    break
                other = starts.get(part.start(1))
                if other is not None and other["begins"] and part.group(1)[0].isupper():
                    break
                item["end"] = part.end()
            found.append(item)
        out = []
        for item in found:
            score = sum(item["scores"]) / len(item["scores"])
            if item["label"] in _LABEL_TYPES and score >= min_score:
                out.append(Entity(text[item["start"]:item["end"]], item["label"], item["start"], item["end"], score))
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = (text, out)
        return out


_tagger_lock = threading.Lock()
_taggers_by_threads: dict = {}  # (model path, threads) -> NerTagger (``NerTagger.with_threads``)
_tagger: Optional[NerTagger] = None


def get_tagger(log: Optional[Callable[[str], None]] = None, cancel=None, progress=None) -> Optional[NerTagger]:
    """The shared tagger (downloads and loads the model once); None when NER cannot run here."""
    global _tagger
    if not runtime_available():
        return None
    with _tagger_lock:
        if _tagger is None:
            path = ensure_model(log, cancel, progress)
            if not path or not os.path.isfile(path):
                return None
            _tagger = NerTagger(path)
            if log:
                log("🧠 Glossary name check: named-entity model (NER) loaded")
        return _tagger


