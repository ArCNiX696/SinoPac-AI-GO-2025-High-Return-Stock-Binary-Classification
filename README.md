# IA Go Competition

**A binary classification challenge to predict “飆股” (high-return stocks) using traditional ML models.**

---

## 📋 Introduction

## 📋 Introduction

This project was developed for the **2025 SinoPac AI GO Competition – “股神對決”**, organized by **永豐金控 (SinoPac Holdings)** in collaboration with the T-Brain platform. The competition challenges participants to design models capable of identifying **high-return stocks (飆股)** based on rich financial datasets containing technical indicators, time-series trends, and baseline tabular features.

To begin, the dataset was balanced to a 1:1 ratio (2,940 samples) and split into three subsets:


1. **Technical Analysis subset** (~30 features)  
2. **Time Series subset** (~465 features)  
3. **Baseline subset** (~905 features)  

This README focuses on our work with the **Technical Analysis subset** to establish strong baselines before tackling the larger, more challenging subsets.

---

## 🎯 Modeling Strategy

1. **Non-linear tree-based models first**  
   - Decision Tree  
   - Random Forest  

   These capture complex interactions without extensive preprocessing and naturally provide feature-importance scores.

2. **Cross-validation & hyperparameter tuning**  
   - **Decision Tree** optimized with `RandomizedSearchCV`  
   - **Random Forest** optimized with `HalvingRandomSearchCV`

3. **Feature selection & meta-dataset**  
   - Extract top features from each subset  
   - Merge into a lean “meta-dataset” for further modeling

4. **Deep learning exploration**  
   - Once the most predictive features are identified, experiment with feed-forward neural nets on the distilled dataset.

---

## 📈 Current Results

### Decision Tree (RandomizedSearchCV)

- **Hyperparameters**  
  - `max_depth`: 5  
  - `min_samples_split`: 10  
  - `criterion`: gini  

- **Test Set Metrics**  
  | Metric     | Score |
  |------------|:-----:|
  | Accuracy   | 0.81  |
  | Precision  | 0.79  |
  | Recall     | 0.86  |
  | F1-score   | 0.82  |

![Decision Tree Test Metrics](./docs/evaluation_metrics_Decision_Tree_Test.png)

---

### Random Forest (HalvingRandomSearchCV)

- **Hyperparameters**  
  - `n_estimators`: 300  
  - `max_depth`: 12  
  - `min_samples_split`: 8  
  - `min_samples_leaf`: 4  
  - `max_features`: sqrt  
  - `criterion`: entropy  

- **Test Set Metrics**  
  | Metric     | Score |
  |------------|:-----:|
  | Accuracy   | 0.85  |
  | Precision  | 0.82  |
  | Recall     | 0.88  |
  | F1-score   | 0.85  |

![Random Forest Test Metrics](./docs/evaluation_metrics_Random_Forest_Test.png)

---

## 🔍 Feature Importances

Both models agree that **技術指標_乖離率(20日)** (20-day deviation rate) and multi-day deviation features are the strongest predictors.

| Rank | Decision Tree | Random Forest |
|:----:|:-------------:|:-------------:|
| 1    | 技術指標_乖離率(20日)               | 技術指標_乖離率(20日)               |
| 2    | 個股19天乖離率                      | 個股19天乖離率                      |
| 3    | 個股10天乖離率                      | 個股10天乖離率                      |
| …    | …            | …            |
  
![Top 15 Feature Importances (Validation)](./docs/feature_importance_Random_Forest_Validation.png)

---

## 📂 Next Steps

1. **Apply the same workflow** to the Time Series and Baseline subsets (cleaning or imputing missing data first).  
2. **Build a meta-dataset** merging top features across subsets.  
3. **Experiment with non-sequential deep nets** on the distilled feature set.  
4. **Compare performance** and choose the best end-to-end pipeline.

---

_End of Progress Report_  

