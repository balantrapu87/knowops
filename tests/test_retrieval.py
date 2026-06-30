import json
from pathlib import Path

import pytest

from knowops.pipeline import Pipeline, PipelineResult

TRAPS = json.loads((Path(__file__).resolve().parent.parent / "data" / "trap_manifest.json").read_text())


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(offline=True)


@pytest.mark.parametrize("trap", TRAPS, ids=[trap["trap_group"] for trap in TRAPS])
def test_fixed_retrieval_selects_correct_document(pipeline, trap):
    selected, _plan, _debug = pipeline.retrieve_fixed(trap["topic"])

    assert selected
    assert selected[0].doc_id == trap["correct_document_id"]
    assert selected[0].doc_id not in trap["outdated_document_ids"]


@pytest.mark.parametrize("trap", TRAPS, ids=[trap["trap_group"] for trap in TRAPS])
def test_fixed_hybrid_ranking_puts_correct_above_stale_candidates(pipeline, trap):
    _selected, _plan, debug = pipeline.retrieve_fixed(trap["topic"])
    index_by_doc_id = {candidate.doc_id: index for index, candidate in enumerate(debug["candidates"])}

    assert trap["correct_document_id"] in index_by_doc_id
    for outdated_doc_id in trap["outdated_document_ids"]:
        assert outdated_doc_id in index_by_doc_id
        assert index_by_doc_id[trap["correct_document_id"]] < index_by_doc_id[outdated_doc_id]


def test_fixed_path_beats_baseline_overall(pipeline):
    fixed_correct = 0
    baseline_correct = 0

    for trap in TRAPS:
        selected, _plan, _debug = pipeline.retrieve_fixed(trap["topic"])
        baseline = pipeline.retrieve_baseline(trap["topic"])
        fixed_correct += selected[0].doc_id == trap["correct_document_id"]
        baseline_correct += baseline[0].doc_id == trap["correct_document_id"]

    assert fixed_correct == len(TRAPS) == 10
    assert baseline_correct < fixed_correct


def test_baseline_demonstrates_stale_top_one_bug(pipeline):
    stale_top_one_count = 0

    for trap in TRAPS:
        baseline = pipeline.retrieve_baseline(trap["topic"])
        stale_top_one_count += baseline[0].doc_id in trap["outdated_document_ids"]

    assert stale_top_one_count >= 4


def test_pipeline_run_returns_offline_result_with_answer():
    result = Pipeline(offline=True).run("What is the current recommended Milvus index configuration?")

    assert isinstance(result, PipelineResult)
    assert result.mode == "offline"
    assert isinstance(result.answer, str)
    assert result.answer.strip()
