# Bank Customer Churn Prediction

Two machine-learning projects on the same dataset, predicting whether a bank
customer will **churn** (leave) or **stay**. One project goes deep on a single
**neural network**; the other goes wide, comparing **11 classification
algorithms** to find the best performer. Together they cover both depth and
breadth on the same business problem.

---

## The problem

Banks lose revenue when customers quietly leave for a competitor. If we can
predict *who* is likely to leave, the bank can step in early with retention
offers. This is a **binary classification** problem: for each customer, predict
`Exited = 1` (left) or `Exited = 0` (stayed).

## The dataset

`Churn_Modelling.csv` — 10,000 bank customers, 14 columns.

| | |
|---|---|
| Target | `Exited` (1 = left, 0 = stayed) |
| Class balance | **~80% stayed / ~20% left** (imbalanced) |
| Features used | CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary |
| Dropped | RowNumber, CustomerId, Surname (identifiers, no predictive value) |

After one-hot encoding `Geography` and `Gender`, the models see **13 numeric
features**. Because the data is imbalanced, **recall** (how many real leavers we
catch) matters as much as accuracy — a model that predicts "nobody leaves" is
already ~80% accurate while catching zero churners.

---

## Projects

### 1. `Bank_Churn_NN.ipynb` — Neural network (depth)

A feed-forward neural network built in **Keras / TensorFlow**, taken end to end:
load → explore → encode → scale → build → train → evaluate → tune with k-fold
cross-validation. The network is `13 → Dense(7, relu) → Dense(6, relu) →
Dense(1, sigmoid)`, trained with the `adam` optimizer and `binary_crossentropy`
loss. Evaluation goes beyond accuracy to a confusion-matrix heatmap and a full
precision / recall / F1 report, and the model is tuned with `GridSearchCV` (via
scikeras).

**Results:** ~0.86 accuracy, ~0.42 recall, ~0.79 precision, ~0.55 F1.
Best grid-search parameters: `batch_size=25, epochs≈20, optimizer='adam'`.

### 2. `Bank_Churn_MultiAlgorithm.ipynb` — Algorithm comparison (breadth)

Prepares the data once, then trains **11 algorithms** in one fair loop and ranks
them on Accuracy, Precision, Recall, F1 and ROC-AUC, with a leaderboard table, a
bar chart, and overlaid ROC curves.

Logistic Regression · K-Nearest Neighbours · Gaussian Naive Bayes · Decision Tree ·
Random Forest · Extra Trees · AdaBoost · Gradient Boosting · XGBoost ·
SVM (RBF) · Neural Network (MLP)

**Leaderboard (test set, by ROC-AUC):**

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Gradient Boosting** | 0.869 | 0.494 | 0.605 | **0.870** |
| Neural Net (MLP) | 0.861 | 0.489 | 0.589 | 0.860 |
| Random Forest | 0.858 | 0.442 | 0.559 | 0.849 |
| AdaBoost | 0.860 | 0.452 | 0.568 | 0.848 |
| Extra Trees | 0.856 | 0.435 | 0.551 | 0.845 |
| Decision Tree | 0.860 | 0.442 | 0.563 | 0.840 |
| XGBoost | 0.851 | 0.491 | 0.573 | 0.839 |
| SVM (RBF) | 0.862 | 0.391 | 0.535 | 0.828 |
| K-Nearest Neighbours | 0.830 | 0.290 | 0.410 | 0.790 |
| Gaussian Naive Bayes | 0.806 | 0.376 | 0.441 | 0.777 |
| Logistic Regression | 0.808 | 0.187 | 0.284 | 0.775 |

**Winner: Gradient Boosting** (ROC-AUC 0.870), with the MLP neural net and Random
Forest close behind. Adding `scale_pos_weight` to handle the imbalance lifted
churner **recall from 0.49 to 0.64**. Top churn drivers by feature importance:
**age, number of products, balance, active membership**.

---

## Repository structure

```
.
├── Churn_Modelling.csv               # dataset (10,000 customers)
├── Bank_Churn_NN.ipynb               # Project 1: neural network (Keras)
├── Bank_Churn_MultiAlgorithm.ipynb   # Project 2: 11-algorithm comparison
└── README.md
```

## How to run

```bash
# Project 1 (neural network)
pip install pandas numpy seaborn scikit-learn tensorflow scikeras

# Project 2 (algorithm comparison)
pip install pandas numpy seaborn scikit-learn xgboost

# then, from the repo folder:
jupyter notebook
```

Open either notebook and run the cells top to bottom (**Cell ▸ Run All**). Keep
`Churn_Modelling.csv` in the same folder.

## Key takeaways

- On imbalanced data, **accuracy is misleading** — every model looks ~80%+
  accurate, so the honest scores are **ROC-AUC, F1 and recall**.
- **Boosted trees and ensembles** (Gradient Boosting, Random Forest) outperform
  the simpler linear and probabilistic baselines on this problem.
- A small **imbalance adjustment** (`scale_pos_weight` / `class_weight='balanced'`)
  meaningfully increases how many real churners the model catches.
- The right metric depends on the business goal: for a retention programme,
  catching leavers (**recall**) is usually worth a little extra cost in false alarms.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow / Keras · scikeras ·
seaborn · matplotlib
