from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.eda.analyze import analyze


def test_analyze_writes_json_markdown_and_figures(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "reviews.csv"
    params_path = tmp_path / "params.yaml"
    report_path = tmp_path / "eda_report.json"
    markdown_path = tmp_path / "eda_report.md"
    figures_dir = tmp_path / "figures"
    pd.DataFrame(
        [
            {"review_id": "n1", "review_text": "Bad broken product with poor packaging.", "rating": 1, "sentiment": "negative", "source": "unit", "ingested_at": "now"},
            {"review_id": "u1", "review_text": "Average product that works as described.", "rating": 3, "sentiment": "neutral", "source": "unit", "ingested_at": "now"},
            {"review_id": "p1", "review_text": "Excellent useful product with fast delivery.", "rating": 5, "sentiment": "positive", "source": "unit", "ingested_at": "now"},
        ]
    ).to_csv(input_path, index=False)
    params_path.write_text(
        """
paths:
  artifacts:
    eda_report: reports/eda_report.json
    class_distribution_figure: reports/figures/class_distribution.png
    rating_distribution_figure: reports/figures/rating_distribution.png
    text_length_distribution_figure: reports/figures/text_length_distribution.png
    top_tokens_by_sentiment_figure: reports/figures/top_tokens_by_sentiment.png
data:
  dataset_name: unit
  dataset_split: train
  min_text_length: 20
  max_text_length: 3000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARAMS_PATH", str(params_path))

    report = analyze(
        input_path=input_path,
        report_output_path=report_path,
        markdown_output_path=markdown_path,
        figures_dir=figures_dir,
    )

    assert report["rows"] == 3
    assert report_path.exists()
    assert markdown_path.exists()
    assert (figures_dir / "class_distribution.png").exists()
    assert (figures_dir / "rating_distribution.png").exists()
