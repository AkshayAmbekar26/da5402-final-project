from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ml.common import DATA_RAW, REPORTS, ROOT, ensure_dirs, read_json, write_json
from ml.monitoring.performance import timed_stage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES_DIR = REPORTS / "figures"


def text_length_summary(lengths: pd.Series) -> dict[str, float | int]:
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
    }


def top_terms(texts: pd.Series, top_k: int = 20, ngram_range: tuple[int, int] = (1, 2)) -> list[dict[str, object]]:
    if texts.empty:
        return []
    vectorizer = CountVectorizer(stop_words="english", ngram_range=ngram_range, max_features=5000)
    matrix = vectorizer.fit_transform(texts.astype(str))
    counts = matrix.sum(axis=0).A1
    features = vectorizer.get_feature_names_out()
    ranked = sorted(zip(features, counts, strict=False), key=lambda item: item[1], reverse=True)[:top_k]
    return [{"term": str(term), "count": int(count)} for term, count in ranked]


def plot_bar(data: dict[str, int], title: str, output_path: Path, color: str = "#17613f") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(list(data.keys()), list(data.values()), color=color)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_histogram(values: pd.Series, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40, color="#3d7ea6", edgecolor="#ffffff")
    plt.title(title)
    plt.xlabel("Review length in characters")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# EDA Report",
        "",
        f"Dataset: `{report['dataset'].get('dataset_name', 'unknown')}`",
        f"Rows: `{report['rows']}`",
        "",
        "## Class Distribution",
        "",
    ]
    for label, count in report["class_distribution"].items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Rating Distribution", ""])
    for rating, count in report["rating_distribution"].items():
        lines.append(f"- `{rating}` stars: {count}")
    lines.extend(
        [
            "",
            "## Text Length",
            "",
            f"- Mean: {report['text_length']['mean']:.2f}",
            f"- Median: {report['text_length']['median']:.2f}",
            f"- P95: {report['text_length']['p95']:.2f}",
            f"- P99: {report['text_length']['p99']:.2f}",
            "",
            "## Data Quality Notes",
            "",
            f"- Missing values: `{report['missing_values']}`",
            f"- Duplicate review IDs: `{report['duplicates']['review_id']}`",
            f"- Duplicate review text: `{report['duplicates']['review_text']}`",
            f"- Mixed-label duplicate text: `{report['duplicates']['mixed_label_text']}`",
            "",
            "## Bias And Limitation Notes",
            "",
            "- Reviews are English-only Amazon-style product reviews.",
            "- Sentiment is derived from star ratings rather than direct human sentiment annotation.",
            "- Three-star reviews are treated as neutral, but they can contain mixed positive and negative language.",
            "- Product categories are not available in the SetFit reduced schema, so category-level bias cannot be measured from this source.",
            "",
            "## Generated Figures",
            "",
            "- `reports/figures/class_distribution.png`",
            "- `reports/figures/rating_distribution.png`",
            "- `reports/figures/text_length_distribution.png`",
            "- `reports/figures/top_tokens_by_sentiment.png`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    input_path: Path = DATA_RAW / "reviews.csv",
    config_path: Path = ROOT / "configs" / "data_config.json",
    report_output_path: Path = REPORTS / "eda_report.json",
    markdown_output_path: Path = REPORTS / "eda_report.md",
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Any]:
    with timed_stage("run_eda") as perf:
        ensure_dirs()
        figures_dir.mkdir(parents=True, exist_ok=True)
        config = read_json(config_path) if config_path.exists() else {}
        df = pd.read_csv(input_path)
        perf["rows_processed"] = int(len(df))
        perf["extra"] = {"figures_generated": 4}
        normalized_text = df["review_text"].astype(str).str.lower().str.strip()
        text_lengths = df["review_text"].astype(str).str.len()
        class_distribution = {
            str(key): int(value) for key, value in df["sentiment"].value_counts().sort_index().to_dict().items()
        }
        rating_distribution = {
            str(key): int(value) for key, value in df["rating"].value_counts().sort_index().to_dict().items()
        }

        top_by_sentiment = {
            sentiment: top_terms(df.loc[df["sentiment"] == sentiment, "review_text"], top_k=15)
            for sentiment in sorted(df["sentiment"].dropna().unique())
        }
        mixed_label_text = int(
            df.assign(_normalized_text=normalized_text).groupby("_normalized_text")["sentiment"].nunique().gt(1).sum()
        )

        plot_bar(class_distribution, "Class distribution", figures_dir / "class_distribution.png")
        plot_bar(rating_distribution, "Rating distribution", figures_dir / "rating_distribution.png", color="#d58a2a")
        plot_histogram(text_lengths, "Review text length distribution", figures_dir / "text_length_distribution.png")
        sentiment_token_counts = {
            sentiment: sum(item["count"] for item in terms[:10]) for sentiment, terms in top_by_sentiment.items()
        }
        plot_bar(
            sentiment_token_counts,
            "Top-token volume by sentiment",
            figures_dir / "top_tokens_by_sentiment.png",
            color="#6457a6",
        )

        report: dict[str, Any] = {
            "stage": "run_eda",
            "status": "success",
            "dataset": {
                "dataset_name": config.get("dataset_name", "unknown"),
                "dataset_split": config.get("dataset_split", "unknown"),
                "source_homepage": config.get("source_homepage", "unknown"),
            },
            "input_path": str(input_path),
            "rows": int(len(df)),
            "missing_values": {str(key): int(value) for key, value in df.isnull().sum().to_dict().items()},
            "duplicates": {
                "review_id": int(df["review_id"].duplicated().sum()),
                "review_text": int(normalized_text.duplicated().sum()),
                "mixed_label_text": mixed_label_text,
            },
            "text_length": text_length_summary(text_lengths),
            "very_short_reviews": int((text_lengths < int(config.get("min_text_length", 20))).sum()),
            "very_long_reviews": int((text_lengths > int(config.get("max_text_length", 3000))).sum()),
            "class_distribution": class_distribution,
            "rating_distribution": rating_distribution,
            "top_terms_overall": top_terms(df["review_text"], top_k=20),
            "top_terms_by_sentiment": top_by_sentiment,
            "figures": {
                "class_distribution": str(figures_dir / "class_distribution.png"),
                "rating_distribution": str(figures_dir / "rating_distribution.png"),
                "text_length_distribution": str(figures_dir / "text_length_distribution.png"),
                "top_tokens_by_sentiment": str(figures_dir / "top_tokens_by_sentiment.png"),
            },
        }
        write_json(report_output_path, report)
        write_markdown_report(report, markdown_output_path)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis on raw reviews.")
    parser.add_argument("--input", type=Path, default=DATA_RAW / "reviews.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data_config.json")
    args = parser.parse_args()
    analyze(args.input, args.config)


if __name__ == "__main__":
    main()
