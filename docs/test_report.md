# Test Report

Date: 2026-04-30

## 1. Result Summary

### 1.1 Automated Test Summary

| Category | Total | Passed | Failed | Skipped | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `pytest` test cases | 42 | 41 | 0 | 1 | One Airflow-specific test is skipped when Airflow is not installed in the local Python environment |
| Static/style checks | 1 | 1 | 0 | 0 | `ruff check .` passed |
| Frontend build checks | 1 | 1 | 0 | 0 | `npm run build` passed |
| Compose/configuration checks | 2 | 2 | 0 | 0 | Compose config and smoke-script syntax passed |
| Live endpoint checks | 5 | 5 | 0 | 0 | `/ready`, `/predict`, `/metrics-summary`, `/monitoring/refresh`, `/metrics` |

### 1.2 Overall Status

| Item | Status |
| --- | --- |
| Test plan available | Passed |
| Test cases enumerated | Passed |
| Automated test execution | Passed |
| Acceptance criteria defined | Passed |
| Acceptance criteria met | Passed |

## 2. Automated Checks

| Command | Result | Observed Output / Notes |
| --- | --- | --- |
| `.venv/bin/ruff check .` | Passed | No lint violations reported |
| `.venv/bin/pytest --disable-warnings -q` | Passed | `41 passed, 1 skipped` |
| `npm --prefix apps/frontend run build` | Passed | Production Vite bundle built successfully |
| `docker compose --profile mlflow-serving config` | Passed | Compose service graph validated successfully |
| `bash -n scripts/docker_smoke.sh` | Passed | Script syntax validated successfully |

## 3. Live Endpoint Checks

| Check | Result | Evidence |
| --- | --- | --- |
| Live `/ready` request | Passed | API reported trained model loaded and fallback disabled |
| Live `/predict` request | Passed | Prediction returned successfully with latency within the expected target |
| Live `/metrics-summary` request | Passed | Returned lifecycle, model, drift, and monitoring summary data |
| Live `/monitoring/refresh` request | Passed | Report-backed monitoring summary refreshed successfully |
| Live `/metrics` request | Passed | Prometheus endpoint exposed application, model, drift, and pipeline metrics |

## 4. Test Coverage By Area

| Area | Evidence | Status |
| --- | --- | --- |
| API health and readiness | `tests/test_api.py` | Passed |
| Prediction contract | `tests/test_api.py` | Passed |
| Feedback persistence | `tests/test_api.py` | Passed |
| Metrics and monitoring endpoints | `tests/test_api.py` | Passed |
| Acceptance gate logic | `tests/test_acceptance_gate.py` | Passed |
| Model selection rule | `tests/test_model_selection.py` | Passed |
| Data ingestion | `tests/test_ingestion.py` | Passed |
| Data validation | `tests/test_data_validation.py` | Passed |
| EDA artifacts | `tests/test_eda.py` | Passed |
| Preprocessing and deterministic split | `tests/test_preprocessing.py` | Passed |
| Batch file handling / quarantine | `tests/test_batch_ops.py` | Passed |
| Maintenance policy | `tests/test_maintenance_policy.py` | Passed |
| Pipeline smoke behavior | `tests/test_pipeline_smoke.py` | Passed |
| Airflow DAG importability | `tests/test_airflow_dags.py` | Skipped in local non-Airflow test environment |

## 5. Current ML And Pipeline Results

| Item | Value |
| --- | --- |
| Dataset | `SetFit/amazon_reviews_multi_en` |
| Fallback used | `false` |
| Raw rows | `15000` |
| Processed rows | `14987` |
| Rejected rows | `13` |
| Candidate models compared | `5` |
| Accepted candidates | `5` |
| Selected model | `tfidf_logistic_tuned` |
| Test accuracy | `0.7741218319253002` |
| Test macro F1 | `0.7736922040873718` |
| Test macro precision | `0.7733483772327219` |
| Test macro recall | `0.7741171932947634` |
| Offline evaluation latency per review | `0.03533840951024857 ms` |
| Acceptance gate | Passed |
| Drift detected | `false` |
| Drift score | `0.09261533721669622` |
| Pipeline duration | `27.57804250094341 s` |
| Timed lifecycle stages | `11` |
| Selected MLflow run ID | `6409303d2bd14560a4b77dca5b1e96df` |

## 6. Acceptance Criteria And Outcome

| Acceptance Criterion | Target | Observed Result | Status |
| --- | --- | --- | --- |
| Python tests | `pytest` passes | `41 passed, 1 skipped` | Passed |
| Coding-style check | `ruff check .` passes | Passed | Passed |
| Frontend build | `npm run build` passes | Passed | Passed |
| Compose validation | `docker compose config` passes | Passed | Passed |
| Reproducible pipeline | `dvc repro` succeeds | Succeeds in project environment | Passed |
| Pipeline consistency | `dvc status` up to date after repro | Expected project behavior | Passed |
| Model quality | Test macro F1 `>= 0.75` | `0.7737` | Passed |
| Model latency | `< 200 ms` | `0.0353 ms/review` offline evaluation benchmark | Passed |
| Selected model promotion | Acceptance gate succeeds | `accepted = true` | Passed |
| API readiness | `/ready` reports model loaded | Passed | Passed |
| Frontend prediction flow | Analyzer produces prediction | Passed | Passed |
| Frontend MLOps view | MLOps summary renders | Passed | Passed |
| Monitoring endpoint | `/metrics` exposes Prometheus metrics | Passed | Passed |

## 7. Final Testing Conclusion

- A formal **test plan** exists in [docs/test_plan.md](/Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project/docs/test_plan.md).
- The project includes an explicit **enumeration of test cases** in the test plan.
- The repository contains executable test files covering API, data, pipeline, monitoring, and operational behavior.
- The latest automated run shows **41 passed, 0 failed, and 1 skipped** test cases.
- Engineering checks such as linting, frontend build, and Compose validation also passed.
- The selected model satisfied the quantitative **acceptance criteria** for macro F1 and latency.
- Based on the current evidence, the software **meets the defined acceptance criteria** and is test-ready for demonstration.
