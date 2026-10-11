# Baselines (validation set)

Train rows: 33,920. Validation rows: 6,958. Test set not used.

## Overall

| model | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 |
|---|---|---|---|---|---|---|---|
| Majority class | 0.2910 | 0.0364 | 0.1250 | 0.0564 | 0.0847 | 0.2910 | 0.1312 |
| TF-IDF + LogReg (default) | 0.8525 | 0.8950 | 0.8113 | 0.8470 | 0.8588 | 0.8525 | 0.8524 |

## Majority class

Per-class results:

| class | precision | recall | f1-score | support |
|---|---|---|---|---|
| Hardware | 0.2910 | 1.0000 | 0.4509 | 2025 |
| HR Support | 0.0000 | 0.0000 | 0.0000 | 1572 |
| Access | 0.0000 | 0.0000 | 0.0000 | 1060 |
| Miscellaneous | 0.0000 | 0.0000 | 0.0000 | 1056 |
| Storage | 0.0000 | 0.0000 | 0.0000 | 413 |
| Internal Project | 0.0000 | 0.0000 | 0.0000 | 318 |
| Administrative rights | 0.0000 | 0.0000 | 0.0000 | 264 |
| Purchase | 0.0000 | 0.0000 | 0.0000 | 250 |

## TF-IDF + LogReg (default)

Per-class results:

| class | precision | recall | f1-score | support |
|---|---|---|---|---|
| Hardware | 0.7907 | 0.9067 | 0.8447 | 2025 |
| HR Support | 0.8544 | 0.8696 | 0.8619 | 1572 |
| Access | 0.9249 | 0.8481 | 0.8848 | 1060 |
| Miscellaneous | 0.8295 | 0.8248 | 0.8272 | 1056 |
| Storage | 0.9724 | 0.8523 | 0.9084 | 413 |
| Internal Project | 0.9333 | 0.7484 | 0.8307 | 318 |
| Administrative rights | 0.8978 | 0.6326 | 0.7422 | 264 |
| Purchase | 0.9573 | 0.8080 | 0.8764 | 250 |
