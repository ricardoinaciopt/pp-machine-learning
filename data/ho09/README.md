# HO9: a teaching collection for basic meta-learning

| OpenML ID (version 1)                 | Dataset                          | Rows | Predictors | Classes | Teaching role                                  |
| ------------------------------------- | -------------------------------- | ---: | ---------: | ------: | ---------------------------------------------- |
| [3](https://www.openml.org/d/3)       | kr-vs-kp                         | 3196 |         36 |       2 | Symbolic chess; categorical interactions       |
| [11](https://www.openml.org/d/11)     | balance-scale                    |  625 |          4 |       3 | Geometric rules and a rare balanced class      |
| [15](https://www.openml.org/d/15)     | breast-w                         |  699 |          9 |       2 | Numeric cytology with explicit missing values  |
| [31](https://www.openml.org/d/31)     | credit-g                         | 1000 |         20 |       2 | Mixed predictors and unequal class frequencies |
| [37](https://www.openml.org/d/37)     | diabetes                         |  768 |          8 |       2 | Numeric measurements and data conventions      |
| [40](https://www.openml.org/d/40)     | sonar                            |  208 |         60 |       2 | Many predictors relative to observations       |
| [50](https://www.openml.org/d/50)     | tic-tac-toe                      |  958 |          9 |       2 | Categorical combinatorial interactions         |
| [54](https://www.openml.org/d/54)     | vehicle                          |  846 |         18 |       4 | Correlated shape descriptors; multiclass task  |
| [61](https://www.openml.org/d/61)     | iris                             |  150 |          4 |       3 | Small balanced geometric reference             |
| [187](https://www.openml.org/d/187)   | wine                             |  178 |         13 |       3 | Chemistry; unequal numeric scales              |
| [1462](https://www.openml.org/d/1462) | banknote-authentication          | 1372 |          4 |       2 | Compact continuous signal descriptors          |
| [1464](https://www.openml.org/d/1464) | blood-transfusion-service-center |  748 |          4 |       2 | Low-dimensional imbalanced binary task         |

`openml_datasets.csv`: ordered IDs, names, versions, exact targets, dimensions, source URLs, SHA-256 digests of the downloaded Parquet bytes, the licence label reported by OpenML, and a per-dataset rationale. ID-based versioning follows the [OpenML dataset documentation](https://docs.openml.org/examples/20_basic/simple_datasets_tutorial/). Consult the linked records and original sources for citation and reuse terms.

## Data dictionary

`Dataset`, `OpenMLID`, and `Source` identify each problem. `NInstances`, `NTrain`, and `NTest` describe raw and partition sizes; `TrainIndexSHA256` and `TestIndexSHA256` identify the split. `CorrectDT` and `CorrectDS` are test correct counts. `AccuracyDT`, `AccuracyDS`, `BalancedAccuracyDT`, `BalancedAccuracyDS`, `AccuracyMajority`, `AccuracyGap_DT_minus_DS`, `Best`, `BestModel`, and `BestModelBinary` are outcomes.

All `mf_` values describe the training partition:

| Feature                     | Definition                                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `mf_n_instances`            | Training rows                                                                                                    |
| `mf_n_features_raw`         | Raw predictors, excluding target                                                                                 |
| `mf_n_features_encoded`     | Training-fitted imputed/one-hot predictor count                                                                  |
| `mf_n_classes`              | Observed training classes                                                                                        |
| `mf_feature_instance_ratio` | Raw predictors / training rows                                                                                   |
| `mf_instance_feature_ratio` | Training rows / raw predictors                                                                                   |
| `mf_numeric_ratio`          | Numeric raw predictors / all raw predictors                                                                      |
| `mf_missing_rate`           | Explicit null predictor cells / all predictor cells, before imputation                                           |
| `mf_class_entropy`          | Shannon entropy in bits of training class proportions                                                            |
| `mf_majority_class_prop`    | Largest training class proportion                                                                                |
| `mf_minority_class_prop`    | Smallest positive training class proportion                                                                      |
| `mf_imbalance_ratio`        | Largest / smallest positive class count                                                                          |
| `mf_mean_abs_feature_corr`  | Mean finite absolute Pearson correlation over distinct pairs of encoded training predictors; 0 if no valid pairs |
