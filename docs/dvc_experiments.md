# DVC Reproducible Experimentation

This project uses DVC to reproduce the full model lifecycle and to demonstrate the Assignment 1 concepts inside the final MLOps application.

## Tracked Pipeline

The DVC DAG is defined in `dvc.yaml`:

```text
ingest -> validate -> eda -> preprocess -> featurize -> train -> evaluate -> accept -> drift -> publish
```

The staged workflow tracks:

- raw data in `data/raw/reviews.csv`
- rejected rows in `data/interim/rejected_reviews.csv`
- fixed train, validation, and test splits in `data/processed/`
- drift baselines in `data/baselines/`
- selected model artifacts in `models/`
- metrics and lifecycle reports in `reports/`

## Parameterized Experiments

Experiment parameters live in `params.yaml`.

Data and split parameters:

- `data.max_rows_total`
- `data.min_text_length`
- `data.max_text_length`
- `data.validation_size`
- `data.test_size`
- `data.random_seed`

Training parameters:

- `training.acceptance_test_macro_f1`
- `training.acceptance_latency_ms`
- `training.latency_sample_size`
- `training.candidates`

The current candidate set spans three distinct text-model paths:

- `tfidf_logistic_*`: TF-IDF features with Logistic Regression
- `tfidf_sgd_log_loss`: TF-IDF features with SGDClassifier
- `count_naive_bayes`: CountVectorizer features with Multinomial Naive Bayes

DVC stage parameter dependencies are declared directly in `dvc.yaml`, so changing a preprocessing parameter reruns preprocessing and downstream stages, while changing a training parameter reruns training and downstream stages.

Example commands:

```bash
dvc repro
dvc exp run -S data.max_rows_total=6000
dvc exp run -S training.acceptance_test_macro_f1=0.78
dvc exp run -S training.candidates.2.max_features=60000
```

## Metrics And Plots

Metrics:

```bash
dvc metrics show
dvc metrics diff
```

Tracked metric files include:

- `reports/ingestion_report.json`
- `reports/data_validation.json`
- `reports/preprocessing_report.json`
- `reports/training_metrics.json`
- `reports/model_comparison.json`
- `reports/evaluation.json`
- `reports/final_metrics.json`
- `reports/acceptance_gate.json`
- `reports/drift_report.json`
- `reports/pipeline_report.json`
- `reports/pipeline_performance.json`

DVC-native plot sources:

- `reports/model_comparison_plot.csv`
- `reports/confusion_matrix.csv`

Plot commands:

```bash
dvc plots show
dvc plots diff
```

The model-comparison plot compares candidate validation macro F1, test macro F1, and test accuracy. The confusion-matrix plot records actual/predicted counts for the selected model.

The flat `reports/final_metrics.json` file exists specifically to make `dvc metrics show` easy to present in a demo.

## Recovery And Traceability

The current model can be traced through:

- Git commit hash in `models/model_metadata.json`
- MLflow run ID in `models/model_metadata.json`
- DVC artifact hashes in `dvc.lock`
- model metrics in `reports/evaluation.json`
- model comparison in `reports/model_comparison.json`

To recover an older committed experiment state:

```bash
git checkout <commit>
dvc checkout
dvc repro
```

To confirm the current workspace is reproducible:

```bash
dvc status
dvc dag
dvc repro
```

This repository also includes a helper script that wraps the local DVC cache configuration used in this project:

```bash
./scripts/dvc_repro_check.sh
```

The project also configures a local DVC remote named `local_artifacts` at `dvc_remote/`. This keeps the final demo cloud-free while still demonstrating artifact push/recovery:

```bash
dvc push
dvc pull
dvc checkout
```

It runs:

- `dvc status`
- `dvc dag`
- `dvc repro`
- `dvc metrics show`
- `dvc plots show`
