# Data Card

## Dataset

Primary dataset: `SetFit/amazon_reviews_multi_en`

Source: <https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en>

The dataset is a reduced English Amazon review classification dataset with review text and a five-level rating label. The ingestion pipeline maps it into the project schema and derives three sentiment classes.

## Project Schema

| Column | Description |
| --- | --- |
| `review_id` | Stable source review identifier |
| `review_text` | Product review text used for prediction |
| `rating` | Star rating from 1 to 5 |
| `sentiment` | Derived target label: negative, neutral, positive |
| `source` | Dataset source name |
| `ingested_at` | UTC ingestion timestamp |

## Label Mapping

| Source label | Rating | Project sentiment |
| ---: | ---: | --- |
| `0` | 1 | negative |
| `2` | 3 | neutral |
| `4` | 5 | positive |

## Sampling

The default pipeline uses unambiguous ratings for training labels:

- 1-star reviews as negative
- 3-star reviews as neutral
- 5-star reviews as positive

2-star and 4-star reviews are intentionally excluded from the default supervised training dataset because they often contain mixed or borderline sentiment. This makes the target definition cleaner for a three-class sentiment analyzer and improves metric interpretability.

The default pipeline samples a balanced dataset by final sentiment class:

- `5000` negative reviews
- `5000` neutral reviews
- `5000` positive reviews
- `15000` total reviews by default

The local seed dataset remains only as an offline fallback if Hugging Face loading fails.

## Preprocessing

The preprocessing stage:

- normalizes whitespace
- removes empty reviews
- removes reviews below the configured minimum text length
- removes reviews above the configured maximum text length
- removes exact duplicate review text
- writes rejected rows to `data/interim/rejected_reviews.csv`
- creates deterministic stratified train/validation/test splits

## Known Limitations

- Sentiment is derived from star ratings, not direct human sentiment labels.
- Three-star reviews are treated as neutral, but they may contain mixed sentiment.
- The SetFit reduced schema does not include product category, reviewer ID, or product ID, so category-level fairness and reviewer-level leakage cannot be analyzed.
- The dataset is English-only and may not generalize to multilingual reviews.

## Reproducibility

The dataset configuration is stored in `configs/data_config.json`. DVC tracks the pipeline stages and generated data artifacts. Model metadata records the Git commit, MLflow run ID, and DVC lock hash.
