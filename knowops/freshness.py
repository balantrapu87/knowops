"""
Freshness scoring — the actual fix for date-aware retrieval.

This is real code, not a prompt instruction. A document's final ranking score
combines semantic similarity with an exponential recency bonus:

    hybrid_score = semantic_weight * semantic_score
                 + freshness_weight * exp(-age_days / decay_days)

Because every trap group's *correct* document is the *newer* one, a non-zero
freshness_weight guarantees the current document outranks the stale one even
when their semantic scores are close. Weights come from configs/pipeline.yaml
via FreshnessProfile, so behaviour is tunable without touching this file.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from knowops.config import FreshnessProfile


def _now_ts(now: Optional[float]) -> float:
    """Return the reference 'now' as a Unix timestamp (injectable for tests)."""
    return now if now is not None else datetime.now(timezone.utc).timestamp()


def freshness_score(updated_ts: int, decay_days: int = 180, now: Optional[float] = None) -> float:
    """Exponential-decay recency score in (0, 1].

    1.0 means updated right now; the value halves roughly every
    ``decay_days * ln(2)`` days. Future timestamps are clamped to 'now' so a
    slightly skewed clock can never produce a score above 1.0.
    """
    age_days = (_now_ts(now) - updated_ts) / 86400.0
    if age_days < 0:
        age_days = 0.0
    denom = float(decay_days) if decay_days > 0 else 1.0
    return math.exp(-age_days / denom)


def hybrid_score(
    semantic_score: float,
    updated_ts: int,
    semantic_weight: float = 0.7,
    freshness_weight: float = 0.3,
    decay_days: int = 180,
    now: Optional[float] = None,
) -> float:
    """Blend semantic similarity (0-1 cosine) with the freshness score."""
    fresh = freshness_score(updated_ts, decay_days, now=now)
    return (semantic_weight * semantic_score) + (freshness_weight * fresh)


def score_with_profile(
    semantic_score: float,
    updated_ts: int,
    profile: FreshnessProfile,
    now: Optional[float] = None,
) -> float:
    """hybrid_score using the weights/decay from a configured FreshnessProfile."""
    return hybrid_score(
        semantic_score,
        updated_ts,
        semantic_weight=profile.semantic_weight,
        freshness_weight=profile.freshness_weight,
        decay_days=profile.decay_days,
        now=now,
    )
