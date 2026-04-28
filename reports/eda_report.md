# EDA Report

Dataset: `SetFit/amazon_reviews_multi_en`
Rows: `15000`

## Class Distribution

- `negative`: 5000
- `neutral`: 5000
- `positive`: 5000

## Rating Distribution

- `1` stars: 5000
- `3` stars: 5000
- `5` stars: 5000

## Text Length

- Mean: 169.10
- Median: 118.00
- P95: 472.05
- P99: 828.01

## Data Quality Notes

- Missing values: `{'review_id': 0, 'review_text': 0, 'rating': 0, 'sentiment': 0, 'source': 0, 'ingested_at': 0}`
- Duplicate review IDs: `0`
- Duplicate review text: `13`
- Mixed-label duplicate text: `1`

## Bias And Limitation Notes

- Reviews are English-only Amazon-style product reviews.
- Sentiment is derived from star ratings rather than direct human sentiment annotation.
- Three-star reviews are treated as neutral, but they can contain mixed positive and negative language.
- Product categories are not available in the SetFit reduced schema, so category-level bias cannot be measured from this source.

## Generated Figures

- `/Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project/reports/figures/class_distribution.png`
- `/Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project/reports/figures/rating_distribution.png`
- `/Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project/reports/figures/text_length_distribution.png`
- `/Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project/reports/figures/top_tokens_by_sentiment.png`
