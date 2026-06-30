import math

import pytest

from knowops.config import get_profile
from knowops.freshness import freshness_score, hybrid_score, score_with_profile

NOW = 1_700_000_000.0
DAY = 86_400


def test_freshness_score_is_one_when_updated_now():
    assert freshness_score(int(NOW), decay_days=180, now=NOW) == 1.0


def test_freshness_score_exponential_decay_after_one_decay_period():
    updated_ts = int(NOW - 180 * DAY)

    assert freshness_score(updated_ts, decay_days=180, now=NOW) == pytest.approx(math.exp(-1))


def test_freshness_score_newer_doc_scores_higher_than_older_doc():
    newer = int(NOW - 10 * DAY)
    older = int(NOW - 90 * DAY)

    assert freshness_score(newer, now=NOW) > freshness_score(older, now=NOW)


def test_freshness_score_future_timestamp_clamps_to_one():
    assert freshness_score(int(NOW + 7 * DAY), decay_days=180, now=NOW) == 1.0


def test_hybrid_score_perfect_semantic_and_freshness_is_one():
    assert hybrid_score(1.0, int(NOW), semantic_weight=0.7, freshness_weight=0.3, now=NOW) == 1.0


def test_hybrid_score_zero_semantic_uses_freshness_weight():
    assert hybrid_score(0.0, int(NOW), semantic_weight=0.7, freshness_weight=0.3, now=NOW) == 0.3


def test_hybrid_score_respects_semantic_and_freshness_weights():
    updated_ts = int(NOW - 180 * DAY)
    expected = (0.25 * 0.8) + (0.75 * math.exp(-1))

    assert hybrid_score(
        0.8,
        updated_ts,
        semantic_weight=0.25,
        freshness_weight=0.75,
        decay_days=180,
        now=NOW,
    ) == pytest.approx(expected)


def test_hybrid_score_higher_freshness_weight_boosts_newer_doc():
    newer = int(NOW)
    older = int(NOW - 180 * DAY)
    low_freshness_gap = hybrid_score(0.8, newer, 0.9, 0.1, 180, now=NOW) - hybrid_score(
        0.8, older, 0.9, 0.1, 180, now=NOW
    )
    high_freshness_gap = hybrid_score(0.8, newer, 0.5, 0.5, 180, now=NOW) - hybrid_score(
        0.8, older, 0.5, 0.5, 180, now=NOW
    )

    assert high_freshness_gap > low_freshness_gap


def test_score_with_profile_matches_hybrid_score_for_profile():
    profile = get_profile("high")
    updated_ts = int(NOW - 30 * DAY)

    assert score_with_profile(0.72, updated_ts, profile, now=NOW) == pytest.approx(
        hybrid_score(
            0.72,
            updated_ts,
            semantic_weight=profile.semantic_weight,
            freshness_weight=profile.freshness_weight,
            decay_days=profile.decay_days,
            now=NOW,
        )
    )
