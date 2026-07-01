"""
Central configuration for the KnowOps agentic pipeline.

Two sources are merged:
  - .env            — secrets + infrastructure endpoints + simple scalars
  - configs/pipeline.yaml — freshness profiles selected per query by the Planner

Everything the pipeline needs is exposed through a single ``SETTINGS`` object so
modules never read ``os.environ`` directly. ``offline`` mode swaps every external
dependency (Milvus, Ollama, OpenRouter) for a deterministic local stand-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PIPELINE_CONFIG = _PROJECT_ROOT / "configs" / "pipeline.yaml"


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ── Freshness profile ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FreshnessProfile:
    """One row of configs/pipeline.yaml — how recency is weighted for a query.

    semantic_weight + freshness_weight should sum to 1.0. ``decay_days`` controls
    how fast the recency bonus falls off. ``date_filter_days`` is an optional SOFT
    cut-off (None = no filter; never drops a doc newer than the cut-off).
    """
    name: str
    semantic_weight: float
    freshness_weight: float
    decay_days: int
    date_filter_days: Optional[int] = None


@dataclass(frozen=True)
class PlannerKeywords:
    """Keyword sets from configs/pipeline.yaml used by the offline Planner.

    In live mode the LLM classifies questions; these sets are the deterministic
    fallback (and are therefore business logic that lives in config, not code).
    """
    recency_words: frozenset
    jira_words: frozenset
    confluence_words: frozenset
    # Maps intent label → frozenset of trigger words (ordered: first match wins)
    intent_words: dict


# ── Settings ─────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    # Infrastructure
    milvus_host: str
    milvus_port: str
    ollama_base_url: str
    database_url: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    # Mode
    offline: bool
    # Pipeline scalars
    retriever_top_k: int
    reranker_top_k: int
    embedding_batch_size: int
    freshness_warning_days: int
    default_time_sensitivity: str
    # Recency reranking only applies among candidates scoring at least
    # relevance_floor_ratio * top_semantic_score — so a fresh-but-irrelevant
    # document can never outrank a relevant one.
    relevance_floor_ratio: float
    # Freshness profiles keyed by time_sensitivity (high|medium|low)
    profiles: dict[str, FreshnessProfile]
    # Offline planner keyword sets (loaded from pipeline.yaml)
    planner_keywords: PlannerKeywords

    @property
    def llm_offline(self) -> bool:
        """LLM agents fall back to deterministic logic when offline or unkeyed."""
        return self.offline or not self.openrouter_api_key

    def get_profile(self, time_sensitivity: Optional[str]) -> FreshnessProfile:
        """Return the freshness profile for a time_sensitivity label, with fallback."""
        key = (time_sensitivity or self.default_time_sensitivity).lower()
        if key in self.profiles:
            return self.profiles[key]
        return self.profiles[self.default_time_sensitivity]


def _load_profiles(raw: dict) -> dict[str, FreshnessProfile]:
    profiles: dict[str, FreshnessProfile] = {}
    for name, p in (raw.get("freshness_profiles") or {}).items():
        fw = float(p["freshness_weight"])
        sw = float(p.get("semantic_weight", round(1.0 - fw, 6)))
        dfd = p.get("date_filter_days", None)
        profiles[name] = FreshnessProfile(
            name=name,
            semantic_weight=sw,
            freshness_weight=fw,
            decay_days=int(p["decay_days"]),
            date_filter_days=(int(dfd) if dfd not in (None, "", "null") else None),
        )
    if not profiles:
        raise ValueError("configs/pipeline.yaml defines no freshness_profiles")
    return profiles


def _load_planner_keywords(raw: dict) -> PlannerKeywords:
    kw = raw.get("planner_keywords") or {}
    intent_raw = kw.get("intent_words") or {}
    return PlannerKeywords(
        recency_words=frozenset(kw.get("recency_words") or []),
        jira_words=frozenset(kw.get("jira_words") or []),
        confluence_words=frozenset(kw.get("confluence_words") or []),
        intent_words={intent: frozenset(words) for intent, words in intent_raw.items()},
    )


def load_settings(pipeline_config_path: Optional[str | Path] = None) -> Settings:
    """Build a Settings object from .env + pipeline.yaml (env wins for scalars)."""
    cfg_path = Path(
        pipeline_config_path
        or os.getenv("PIPELINE_CONFIG", _DEFAULT_PIPELINE_CONFIG)
    )
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path

    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh) or {}

    pipe = raw.get("pipeline", {}) or {}
    profiles = _load_profiles(raw)

    default_ts = os.getenv(
        "DEFAULT_TIME_SENSITIVITY", pipe.get("default_time_sensitivity", "medium")
    )
    if default_ts not in profiles:
        default_ts = next(iter(profiles))

    def _int(env_key: str, yaml_key: str, fallback: int) -> int:
        return int(os.getenv(env_key, pipe.get(yaml_key, fallback)))

    return Settings(
        milvus_host=os.getenv("MILVUS_HOST", "localhost"),
        milvus_port=os.getenv("MILVUS_PORT", "19530"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql://knowops:knowops_dev@localhost:5432/knowops",
        ),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        offline=_as_bool(os.getenv("KNOWOPS_OFFLINE"), default=False),
        retriever_top_k=_int("RETRIEVER_TOP_K", "retriever_top_k", 20),
        reranker_top_k=_int("RERANKER_TOP_K", "reranker_top_k", 5),
        embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", "embedding_batch_size", 8),
        freshness_warning_days=_int(
            "FRESHNESS_WARNING_DAYS", "freshness_warning_days", 180
        ),
        default_time_sensitivity=default_ts,
        relevance_floor_ratio=float(
            os.getenv("RELEVANCE_FLOOR_RATIO", pipe.get("relevance_floor_ratio", 0.7))
        ),
        profiles=profiles,
        planner_keywords=_load_planner_keywords(raw),
    )


# Module-level singleton. Import this everywhere instead of reading os.environ.
SETTINGS = load_settings()


def get_profile(time_sensitivity: Optional[str]) -> FreshnessProfile:
    """Convenience wrapper around SETTINGS.get_profile()."""
    return SETTINGS.get_profile(time_sensitivity)
