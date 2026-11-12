"""Read a GGUF's layer count without loading weights or allocating its tokenizer arrays."""
from __future__ import annotations

import functools
import os
import struct


def gpu_layer_limit(path: str) -> int | None:
    try:
        stat = os.stat(path)
        return _read(path, stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError, struct.error, UnicodeError, KeyError):
        return None


@functools.lru_cache(maxsize=32)
def _read(path: str, size: int, modified: int) -> int | None:
    # GGUF v2/v3 metadata: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
    formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?',
               10: 'Q', 11: 'q', 12: 'd'}
    with open(path, 'rb') as handle:
        limit = min(size, 64 * 1024 * 1024)

        def take(count):
            if count < 0 or handle.tell() + count > limit:
                raise ValueError('Invalid GGUF metadata size')
            data = handle.read(count)
            if len(data) != count:
                raise ValueError('Truncated GGUF metadata')
            return data

        def number(fmt):
            return struct.unpack('<' + fmt, take(struct.calcsize(fmt)))[0]

        def text():
            count = number('Q')
            if count > 1024 * 1024:
                raise ValueError('Oversized GGUF key')
            return take(count).decode('utf8')

        def skip(kind, depth=0):
            if depth > 4:
                raise ValueError('Nested GGUF array')
            if kind == 9:
                inner, count = number('I'), number('Q')
                if count > 2_000_000:
                    raise ValueError('Oversized GGUF array')
                if inner in formats:
                    seek(count * struct.calcsize(formats[inner]))
                else:
                    for _ in range(count):
                        skip(inner, depth + 1)
            else:
                seek(number('Q') if kind == 8 else struct.calcsize(formats[kind]))

        def seek(count):
            if count < 0 or handle.tell() + count > limit:
                raise ValueError('Truncated GGUF metadata')
            handle.seek(count, 1)

        if take(4) != b'GGUF' or number('I') not in (2, 3):
            return None
        number('Q')  # tensors; their data is never read
        count = number('Q')
        if count > 100_000:
            return None
        architecture, blocks = None, {}
        for _ in range(count):
            key, kind = text(), number('I')
            if key == 'general.architecture' and kind == 8:
                architecture = text()
            elif key.endswith('.block_count') and kind in (0, 1, 2, 3, 4, 5, 10, 11):
                blocks[key] = number(formats[kind])
            else:
                skip(kind)
            value = blocks.get(f'{architecture}.block_count') if architecture else None
            if value is not None:
                # llama.cpp can also offload the output layer after all transformer blocks.
                return int(value) + 1 if 0 < value < 10_000 else None
        return None
