# Model selection

Tuning: GridSearchCV, 5-fold StratifiedGroupKFold on train (33,920 rows), scoring = macro F1, seed = 42. Each tuned model is then refit on all of train and scored once on validation (6,958 rows). Test set not used.
## Results (sorted by validation macro F1)

| model | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | cv_macro_f1 | cv_macro_f1_std |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (tuned) | 0.8594 | 0.8700 | 0.8544 | 0.8616 | 0.8608 | 0.8594 | 0.8597 | 0.8586 | 0.0053 |
| Word + char n-grams + LinearSVC (tuned) | 0.8571 | 0.8542 | 0.8611 | 0.8571 | 0.8587 | 0.8571 | 0.8574 | 0.8540 | 0.0093 |
| TF-IDF + LinearSVC (tuned) | 0.8560 | 0.8531 | 0.8576 | 0.8548 | 0.8576 | 0.8560 | 0.8563 | 0.8562 | 0.0058 |
| TF-IDF + LogReg (default) | 0.8525 | 0.8950 | 0.8113 | 0.8470 | 0.8588 | 0.8525 | 0.8524 | 0.8429 | 0.0059 |
| Majority class | 0.2910 | 0.0364 | 0.1250 | 0.0564 | 0.0847 | 0.2910 | 0.1312 | 0.0550 | 0.0008 |


## Majority class

- Settings tried: 1 x 5 folds; time 0.3 min
- Grid: none (fixed settings)
- Best settings: n/a
- CV macro F1: 0.0550 (+/- 0.0008)
- Validation macro F1: 0.0564

## TF-IDF + LogReg (default)

- Settings tried: 1 x 5 folds; time 0.5 min
- Grid: none (fixed settings)
- Best settings: n/a
- CV macro F1: 0.8429 (+/- 0.0059)
- Validation macro F1: 0.8470

## TF-IDF + LogReg (tuned)

- Settings tried: 12 x 5 folds; time 16.6 min
- Grid: {'tfidf__ngram_range': [(1, 1), (1, 2)], 'clf__C': [1, 5, 20], 'clf__class_weight': [None, 'balanced']}
- Best settings: {'clf__C': 20, 'clf__class_weight': 'balanced', 'tfidf__ngram_range': (1, 2)}
- CV macro F1: 0.8586 (+/- 0.0053)
- Validation macro F1: 0.8616

## TF-IDF + LinearSVC (tuned)

- Settings tried: 12 x 5 folds; time 1.5 min
- Grid: {'tfidf__ngram_range': [(1, 1), (1, 2)], 'clf__C': [0.1, 0.3, 1], 'clf__class_weight': [None, 'balanced']}
- Best settings: {'clf__C': 0.3, 'clf__class_weight': 'balanced', 'tfidf__ngram_range': (1, 2)}
- CV macro F1: 0.8562 (+/- 0.0058)
- Validation macro F1: 0.8548

## Word + char n-grams + LinearSVC (tuned)

- Settings tried: 3 x 5 folds; time 3.1 min
- Grid: {'clf__C': [0.1, 0.3, 1]}
- Best settings: {'clf__C': 0.3}
- CV macro F1: 0.8540 (+/- 0.0093)
- Validation macro F1: 0.8571

## Per-class F1 on validation

| class | support | TF-IDF + LogReg (default) | TF-IDF + LogReg (tuned) | TF-IDF + LinearSVC (tuned) | Word + char n-grams + LinearSVC (tuned) |
|---|---|---|---|---|---|
| Hardware | 2025 | 0.8447 | 0.8504 | 0.8470 | 0.8416 |
| HR Support | 1572 | 0.8619 | 0.8623 | 0.8648 | 0.8706 |
| Access | 1060 | 0.8848 | 0.8892 | 0.8889 | 0.8917 |
| Miscellaneous | 1056 | 0.8272 | 0.8323 | 0.8257 | 0.8289 |
| Storage | 413 | 0.9084 | 0.9104 | 0.9048 | 0.9005 |
| Internal Project | 318 | 0.8307 | 0.8608 | 0.8576 | 0.8682 |
| Administrative rights | 264 | 0.7422 | 0.8023 | 0.7776 | 0.7808 |
| Purchase | 250 | 0.8764 | 0.8852 | 0.8724 | 0.8742 |
