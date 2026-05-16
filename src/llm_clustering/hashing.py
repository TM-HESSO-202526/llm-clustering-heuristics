from __future__ import annotations

import hashlib


def stable_hash(text, n=16):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:n]


def stable_seed(*parts):
    h = stable_hash("|".join(map(str, parts)), 16)
    return int(h, 16) % (2**32)
