"""Yangilik matnidan: qaysi aktivlarga tegishli va qanday voqea turi ekanini aniqlash.

Bu birinchi (tez, tushunarli) qatlam. Ikkinchi qatlam — modelning o'zi: yangilikni
yangilik shoxiga qo'shib, har bir aktiv bashoratidagi o'zgarishni o'lchaymiz (api/impact.py).
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..config import ASSETS, EVENT_TYPES


def _rx(words: tuple[str, ...]) -> re.Pattern | None:
    if not words:
        return None
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])", re.I)


@lru_cache(maxsize=1)
def _asset_patterns():
    return [(a.symbol, _rx(a.keywords + ((a.symbol.lower(),) if len(a.symbol) >= 3 else ()))) for a in ASSETS]


@lru_cache(maxsize=1)
def _event_patterns():
    return [(k, _rx(v)) for k, v in EVENT_TYPES.items() if v]


def tag_assets(text: str) -> list[str]:
    return [sym for sym, rx in _asset_patterns() if rx is not None and rx.search(text)]


def detect_event(text: str) -> str:
    for name, rx in _event_patterns():
        if rx.search(text):
            return name
    return "none"


# Oddiy leksik ton (FinBERT yo'q holatlar uchun zaxira). FinBERT bo'lsa, u ishlatiladi.
_POS = _rx(("surge", "soar", "rally", "beat", "beats", "record high", "jump", "gain", "gains", "rise", "rises",
            "upgrade", "strong", "bullish", "approve", "approved", "cut rates", "rate cut", "stimulus", "boost"))
_NEG = _rx(("plunge", "crash", "slump", "miss", "misses", "fall", "falls", "drop", "drops", "downgrade", "weak",
            "bearish", "ban", "hack", "lawsuit", "war", "sanction", "recession", "default", "hike", "drought", "tariff"))


def lexical_tone(text: str) -> float:
    p = len(_POS.findall(text))
    n = len(_NEG.findall(text))
    return 0.0 if p + n == 0 else (p - n) / (p + n)
