# Heavy Equipment Selling Price Prediction - MLP Project

* Name: Pranay Aggarwal
* Email ID: 24f3004524@ds.study.iitm.ac.in
* Roll No: 24f3004524



This notebook builds a regression pipeline that predicts the selling price of heavy equipment (bulldozers, excavators, wheel loaders, etc.) at auction, based on the **Heavy Equipment Selling Price Prediction Challenge** dataset.

# 🏆 Kaggle Leaderboard: 761/2593

## Overview

- **Task**: Regression — predict `TargetValue` (equipment selling price) for each transaction.
- **Train set**: 138,701 rows × 50 columns
- **Test set**: 15,000 rows × 49 columns
- **Evaluation metric**: Root Mean Squared Logarithmic Error (RMSLE)
- **Final result**: Ridge-blended ensemble of Random Forest, LightGBM, and XGBoost — **RMSLE ≈ 0.2001**

## Data

Input files (read from `/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/`):

| File | Description |
|---|---|
| `train.csv` | Training transactions with target values |
| `test.csv` | Test transactions to predict |
| `sample_submission.csv` | Submission format template |
| `metadata.csv` | Descriptions for anonymized `col1`–`col30` fields |

The dataset mixes 6 numerical and 44 categorical features describing machine specs, operational history, inventory classification, geography, vendor info, and transaction metadata.

## Pipeline

### 1. Imports & Setup
Standard data science stack (`pandas`, `numpy`, `matplotlib`, `seaborn`) plus modeling libraries: `scikit-learn` (pipelines, encoders, metrics), `xgboost`, `lightgbm`, `catboost`.

### 2. Data Loading & Preprocessing
- Loads train/test/metadata files.
- Renames anonymized `colN` columns to human-readable names using `metadata.csv` (e.g. `col9` → `power_output`).

### 3. Exploratory Data Analysis
- **Descriptive statistics**: numeric summary (mean/median/quartiles) and categorical summary (cardinality, top category, frequency).
- **Key findings**:
  - `TargetValue` ranges $7,500–$142,000 (mean $41,522, median $35,000), right-skewed.
  - `ManufactureYear` has an invalid placeholder value (`1001`).
  - Several spec columns are >90% dominated by "None or Unspecified".
  - High-cardinality fields: `Spec_FullDescriptor` (3,505 uniques), `TransactionDate` (3,676 uniques).
  - No duplicate rows.
- **Missing value analysis**: heatmap/bar chart of missingness by column; many spec columns are >80% missing.
- `"None or Unspecified"` categorical values are converted to `NaN` (they carry little information but aren't flagged as missing by default).

### 4. Feature Engineering
- Fixes invalid `ManufactureYear` (1001 → `NaN`).
- Derives from `TransactionDate`: `TransactionYear`, `TransactionMonth`, `TransactionQuarter`.
- `AssetAge` = `TransactionYear` − `ManufactureYear`.
- `Intensity` = `OperationalHoursMeter` / `AssetAge` (infinite values from 0-age assets set to 0).

### 5. Visualization
- Transaction volume by year/month/quarter — peak activity 2008–2012 (esp. 2010).
- Top 20 regions by transaction count — Florida and Texas dominate.
- Target value distribution — right-skewed with a long high-price tail.

### 6. Missing Value Handling
- Drops `TransactionID`, `AssetID`, `TransactionDate` (identifiers / already-encoded date).
- **Note**: Dropping high-missingness (>80%) columns was tried but performed worse on the leaderboard, so all remaining columns were kept.
- Imputation strategy: constant placeholder values (numeric and categorical) rather than mean/median/mode — this outperformed statistical imputation on the leaderboard.

### 7. Outlier Analysis
- IQR-based outlier detection explored for `OperationalHoursMeter`, `ManufactureYear`, `TransactionYear`, `TransactionMonth`, `AssetAge`.
- Outliers were judged to be genuine (older/heavily used machines), so **no outliers were removed** — removal was tested and hurt leaderboard performance.

### 8. Correlation Analysis
- Numeric features show weak linear correlation with `TargetValue`; `AssetAge` is most negatively correlated, `HasVariantModifier`/`ManufactureYear` most positively correlated.
- Categorical features (equipment type, utilization tier) show stronger discriminatory power than numeric ones — e.g., Track Type Tractors, Motor Graders, and Wheel Loaders command the highest average prices; High Utilization assets sell for more than Medium/Low.

### 9. Modeling
- **Target transform**: `log1p(TargetValue)` to reduce right-skew impact.
- **Train/validation split**: 80/20 (`random_state=4524`).
- **Categorical encoding**: Ordinal encoding (target encoding was tried but underperformed).
- **No scaling** — all candidate models are tree-based (scale-invariant).

**Baseline models compared (RMSLE on validation set):**

| Rank | Model | RMSLE |
|---|---|---:|
| 1 | XGBRegressor | 0.2029 |
| 2 | LightGBM | 0.2048 |
| 3 | CatBoostRegressor | 0.2113 |
| 4 | Random Forest | 0.2162 |
| 5 | Decision Tree | 0.3065 |

### 10. Hyperparameter Tuning
- `RandomizedSearchCV` applied to Random Forest, XGBoost, and LightGBM.
- **Result**: Tuned models did **not** beat the manually configured baselines on the leaderboard, so tuning results were discarded in favor of the original hand-set hyperparameters.

### 11. Ensemble
- Combines Random Forest, LightGBM, and XGBoost predictions using a **Ridge regression blender** (non-negative weights, `alpha=0.01`) fit on validation predictions, rather than a simple average.
- **Final ensemble RMSLE: 0.20015** — the best result of any single model or combination tried.

### 12. Feature Importance
- Plots top-20 feature importances for each of the three ensembled models.

### 13. Submission
- Refits all three tree models on the **full training set**.
- Applies the fitted Ridge blender to combine test-set predictions.
- Inverse-transforms from log scale (`expm1`) and clips negative predictions to 0.
- Writes results to `submission.csv`.

## Key Takeaways / Design Decisions

- **Constant-value imputation** beat mean/median/mode imputation.
- **Keeping all columns** (even >80% missing) beat dropping high-missingness columns.
- **No outlier removal** — detected outliers reflect real equipment age/usage variation.
- **Ordinal encoding** beat target encoding for categoricals.
- **Log-transforming the target** helped manage right-skew.
- **A Ridge-blended ensemble** of XGBoost, LightGBM, and Random Forest outperformed every individual model, including hyperparameter-tuned variants.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
```

## How to Run

1. Place `train.csv`, `test.csv`, `metadata.csv`, and `sample_submission.csv` under `/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/` (or update the paths).
2. Run all cells in order.
3. The final predictions are written to `submission.csv` in the working directory.
