from __future__ import annotations

import re

FORBIDDEN_PATTERNS = [
    r"sklearn", r"scipy", r"pandas", r"joblib", r"numba",
    r"torch", r"tensorflow", r"jax", r"faiss",
    r"multiprocessing", r"threading", r"concurrent", r"os\.", r"sys\.",
    r"subprocess", r"open\s*\(", r"exec\s*\(", r"eval\s*\(",
    r"__import__", r"KMeans", r"MiniBatchKMeans",
]


def extract_code_block(raw: str) -> str:
    raw = str(raw)
    m = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def parse_name(raw: str, code: str = "") -> str:
    m = re.search(r"#\s*Name\s*:\s*(.+)", str(raw))
    if m:
        return m.group(1).strip()[:120]
    m = re.search(r"class\s+(\w+)", str(code))
    if m:
        return m.group(1)
    return "unnamed"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def reject_forbidden_code(code: str):
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            raise ValueError(f"forbidden code pattern: {pat}")
