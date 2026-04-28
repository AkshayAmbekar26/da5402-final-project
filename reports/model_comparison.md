# Model Comparison Report

Choose the candidate with highest validation macro F1 among models where test macro F1 >= 0.75 and latency < 200.0 ms. If no model passes, choose the highest validation macro F1 and mark acceptance false.

| Candidate | Validation Macro F1 | Test Macro F1 | Test Accuracy | Latency ms/review | Accepted | MLflow Run |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| tfidf_logistic_tuned | 0.7750 | 0.7737 | 0.7741 | 0.035 | yes | `da0b72efcc224fe1a700c8fce7d7a677` |
| tfidf_logistic_regularized | 0.7746 | 0.7720 | 0.7723 | 0.023 | yes | `6b8a61f421ee470999aaace703c4b9d3` |
| count_naive_bayes | 0.7743 | 0.7747 | 0.7741 | 0.020 | yes | `573ca7f7c1954d229d29e81245815907` |
| tfidf_sgd_log_loss | 0.7679 | 0.7686 | 0.7697 | 0.033 | yes | `cb0384e0041d413a80e03eaadb2cd2a0` |
| tfidf_logistic_baseline | 0.7650 | 0.7593 | 0.7594 | 0.020 | yes | `5cd8c328c38447ed8a2d2474e68fdd6f` |

Selected candidate: `tfidf_logistic_tuned`
