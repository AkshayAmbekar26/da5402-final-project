from __future__ import annotations

from pathlib import Path

import pytest

from ml.evaluation.check_acceptance import check_acceptance


def write_evaluation(path: Path, macro_f1: float, latency_ms: float, accepted: bool) -> None:
    path.write_text(
        (
            "{"
            f'"accepted": {str(accepted).lower()},'
            f'"latency_ms_per_review": {latency_ms},'
            f'"metrics": {{"macro_f1": {macro_f1}}}'
            "}"
        ),
        encoding="utf-8",
    )


def test_acceptance_gate_passes_for_accepted_model(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    output_path = tmp_path / "acceptance_gate.json"
    write_evaluation(evaluation_path, macro_f1=0.80, latency_ms=10.0, accepted=True)

    report = check_acceptance(evaluation_path, output_path)

    assert report["accepted"] is True
    assert output_path.exists()


def test_acceptance_gate_fails_for_rejected_model(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    output_path = tmp_path / "acceptance_gate.json"
    write_evaluation(evaluation_path, macro_f1=0.40, latency_ms=10.0, accepted=False)

    with pytest.raises(SystemExit):
        check_acceptance(evaluation_path, output_path)

    assert output_path.exists()
