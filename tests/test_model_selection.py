from __future__ import annotations

from ml.training.train import select_best_candidate


def make_candidate(
    name: str,
    validation_f1: float,
    test_f1: float,
    latency: float,
    accepted: bool,
) -> dict[str, object]:
    return {
        "candidate_name": name,
        "validation": {"macro_f1": validation_f1},
        "test": {"macro_f1": test_f1},
        "latency_ms_per_review": latency,
        "passes_acceptance": accepted,
    }


def test_select_best_candidate_prefers_highest_validation_f1_among_accepted() -> None:
    selected = select_best_candidate(
        [
            make_candidate("fast_low", 0.70, 0.80, 0.01, True),
            make_candidate("best_valid", 0.82, 0.78, 0.02, True),
            make_candidate("not_accepted", 0.90, 0.70, 0.01, False),
        ]
    )
    assert selected["candidate_name"] == "best_valid"


def test_select_best_candidate_falls_back_when_none_accepted() -> None:
    selected = select_best_candidate(
        [
            make_candidate("a", 0.71, 0.70, 0.01, False),
            make_candidate("b", 0.73, 0.71, 0.02, False),
        ]
    )
    assert selected["candidate_name"] == "b"

