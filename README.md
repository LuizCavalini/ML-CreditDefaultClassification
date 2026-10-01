# Credit Default Classification

Binary classifier to support credit approval decisions, built for
*Introduction to Machine Learning* (EEL891) at UFRJ and submitted to a
class Kaggle competition.

**Result:** 0.6040 accuracy on the Kaggle leaderboard — 4th place in the class.

## Problem

Predict whether a credit applicant will default, from 20,000 historical
applications with known outcomes. Test set has 5,000 applications with
hidden labels. Metric: accuracy.

The target is perfectly balanced (50/50) and no feature correlates
strongly with it (max |r| = 0.12), which puts a low ceiling on
achievable accuracy — most of the work went into extracting weak signal
rather than tuning a strong one.

## Pipeline

1. **EDA** — target distribution, null analysis, correlation study
2. **Preprocessing** — dropped 4 uninformative columns, imputed nulls by
   median/mode/sentinel, converted Y/N flags to binary
3. **Feature engineering** — 14 derived variables: financial ratios,
   geographic aggregates, interactions (age × income), log transforms
4. **Encoding** — LabelEncoder for low-cardinality categoricals,
   target encoding for states and phone area codes
5. **Baselines** — Logistic Regression, Random Forest, LightGBM,
   XGBoost, CatBoost, compared with 5-fold stratified CV
6. **Tuning** — Optuna (Bayesian search), 50 trials on an 80/20 split
7. **Ensemble** — 3 models × 5 seeds = 15 models, soft voting

## Results

| Approach | CV | Kaggle |
|---|---|---|
| Blend 3×5, Optuna 50 trials | 0.607 | **0.6040** |
| Blend 3×5, Optuna 150 trials (5-fold) | 0.610 | 0.6016 |
| Blend 3×5, Optuna 300 trials (5-fold) | 0.611 | 0.5988 |
| Expanded target encoding | 0.636 | 0.5888 |

Two findings worth noting:

**Longer hyperparameter searches made things worse.** 300 Optuna trials
with 5-fold CV scored higher in validation but lower on Kaggle than 50
trials on a simple split — overfitting to the validation process itself.

**Expanded target encoding leaked.** CV jumped to 0.636 while the Kaggle
score dropped to 0.5888, the clearest signal of leakage in the whole
project. Diagnosing that gap was the most useful lesson here.

## Files

| File | Description |
|---|---|
| `trabalho1_eel891.py` | Full pipeline, from loading to Kaggle submission |
| `conjunto_de_treinamento.csv` | Training data (20,000 samples) |
| `conjunto_de_teste.csv` | Test data (5,000 samples) |
| `eda_visualizacoes.png` | Exploratory analysis plots |
| `feature_importance.png` | Feature importance (LightGBM) |
| `eel891_report.pdf` | Full report (Portuguese) |
| `eel891_report.tex` | LaTeX source |

## Running

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            lightgbm xgboost catboost optuna

python trabalho1_eel891.py
```

Generates `submissao_kaggle.csv`, `eda_visualizacoes.png` and
`feature_importance.png`.

## Stack

Python 3.12 · pandas · NumPy · scikit-learn · LightGBM · XGBoost ·
CatBoost · Optuna · Matplotlib · seaborn
