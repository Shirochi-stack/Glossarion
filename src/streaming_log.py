"""Lossless stream fragments carried through the app's line-oriented log pipe."""

import json


STREAM_FRAGMENT_PREFIX = "[STREAM_FRAGMENT] "


def encode_stream_fragment(channel, text):
    if channel not in ("content", "reasoning") or not isinstance(text, str):
        raise ValueError("Stream fragments require a content/reasoning channel and text")
    return STREAM_FRAGMENT_PREFIX + json.dumps(
        {"channel": channel, "text": text}, ensure_ascii=False, separators=(",", ":")
    )


def decode_stream_fragment(message):
    if not isinstance(message, str) or not message.startswith(STREAM_FRAGMENT_PREFIX):
        return None
    try:
        fragment = json.loads(message[len(STREAM_FRAGMENT_PREFIX):])
    except (ValueError, TypeError):
        return None
    if (not isinstance(fragment, dict) or fragment.get("channel") not in ("content", "reasoning")
            or not isinstance(fragment.get("text"), str)):
        return None
    return {"channel": fragment["channel"], "text": fragment["text"]}
