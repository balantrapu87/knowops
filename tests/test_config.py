import pytest

from knowops.config import SETTINGS, load_settings


def test_load_settings_includes_expected_profiles():
    settings = load_settings()

    assert {"high", "medium", "low"}.issubset(settings.profiles)


def test_high_profile_matches_pipeline_config():
    profile = load_settings().get_profile("high")

    assert profile.semantic_weight == pytest.approx(0.55)
    assert profile.freshness_weight == pytest.approx(0.45)
    assert profile.decay_days == 120
    assert profile.date_filter_days is None


def test_unknown_profile_falls_back_to_default_profile():
    settings = load_settings()

    assert settings.get_profile("unknown") == settings.get_profile(settings.default_time_sensitivity)


def test_pipeline_scalar_settings_match_config():
    settings = load_settings()

    assert settings.relevance_floor_ratio == pytest.approx(0.7)
    assert settings.reranker_top_k == 5
    assert settings.retriever_top_k == 20
    assert settings.freshness_warning_days == 180
    assert settings.default_time_sensitivity == "medium"


def test_settings_llm_offline_when_openrouter_key_absent(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    settings = load_settings()
    assert settings.openrouter_api_key == ""
    assert settings.llm_offline is True
