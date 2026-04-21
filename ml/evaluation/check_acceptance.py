from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml.common import REPORTS, read_json, read_params, write_json


def check_acceptance(
    evaluation_path: Path = REPORTS / "evaluation.json",
    output_path: Path = REPORTS / "acceptance_gate.json",
    *,
    fail_on_reject: bool = True,
) -> dict[str, Any]:
    """Fail the lifecycle when the selected model misses quality or latency gates."""
    evaluation = read_json(evaluation_path)
    training_params = read_params("training")
    min_macro_f1 = float(training_params.get("acceptance_test_macro_f1", 0.75))
    max_latency_ms = float(training_params.get("acceptance_latency_ms", 200.0))
    metrics = evaluation.get("metrics", {})
    macro_f1 = float(metrics.get("macro_f1", 0.0) or 0.0)
    latency_ms = float(evaluation.get("latency_ms_per_review", 0.0) or 0.0)
    accepted = bool(evaluation.get("accepted", False)) and macro_f1 >= min_macro_f1 and latency_ms < max_latency_ms
    report = {
        "stage": "acceptance_gate",
        "status": "success" if accepted else "failed",
        "accepted": accepted,
        "macro_f1": macro_f1,
        "latency_ms_per_review": latency_ms,
        "min_macro_f1": min_macro_f1,
        "max_latency_ms": max_latency_ms,
        "reason": "accepted" if accepted else "model did not meet macro F1 and/or latency acceptance gates",
    }
    write_json(output_path, report)
    if fail_on_reject and not accepted:
        raise SystemExit(
            f"Model acceptance failed: macro_f1={macro_f1:.4f} "
            f"(required >= {min_macro_f1:.4f}), latency_ms={latency_ms:.3f} "
            f"(required < {max_latency_ms:.3f})"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail the pipeline if the selected model is not accepted.")
    parser.add_argument("--evaluation", type=Path, default=REPORTS / "evaluation.json")
    parser.add_argument("--output", type=Path, default=REPORTS / "acceptance_gate.json")
    args = parser.parse_args()
    check_acceptance(args.evaluation, args.output)


if __name__ == "__main__":
    main()
