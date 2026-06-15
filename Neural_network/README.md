# Bank Customer Churn Modelling — Introduction to Neural Networks

A neural-network classifier that predicts whether a bank customer will **churn**
(leave) or **stay**, built on the public *Churn_Modelling* dataset (10,000
customers, 14 columns). This repo contains a **corrected and commented** version
of the original PG-AIML project notebook.

## The idea

Banks lose money when customers quietly leave for a competitor. If we can predict
*who* is likely to leave, the bank can step in early. This is a **binary
classification** problem — the answer is one of two classes, `Exited = 1` (left)
or `Exited = 0` (stayed) — and we solve it with a small feed-forward **neural
network** built in Keras/TensorFlow.

## The data

| | |
|---|---|
| Rows | 10,000 customers |
| Target | `Exited` (1 = left, 0 = stayed) |
| Class balance | **~80% stayed / ~20% left** (imbalanced) |
| Features used | CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary |
| Dropped | RowNumber, CustomerId, Surname (pure identifiers) |

After one-hot encoding `Geography` (3 values) and `Gender` (2 values), the model
sees **13 numeric features**.

## What the notebook does, step by step

1. **Read & explore** — load the CSV, check shape, types, and the 80/20 class balance.
2. **Drop ID columns** — identifiers carry no general pattern.
3. **Features / target + encoding** — split into `X` and `y`; one-hot encode the two text columns.
4. **Train/test split** — hold back 20% as an unseen test set (stratified to keep the 80/20 ratio).
5. **Normalize** — `StandardScaler`, fit on train only, applied to test (no leakage).
6. **Build & train** — `13 → Dense(7, relu) → Dense(6, relu) → Dense(1, sigmoid)`, `adam` optimizer, `binary_crossentropy` loss, 30 epochs.
7. **Predict** — turn output probabilities into 0/1 decisions at a 0.5 threshold.
8. **Evaluate** — accuracy, confusion-matrix heatmap, and a full **precision / recall / F1** report.
9. **Tune** — `GridSearchCV` (via **scikeras**) over batch size, epochs, and optimizer with k-fold cross-validation.

## What was fixed vs. the original

The original notebook ran but contained four real bugs. Each fix is marked `# FIX:`
in the code.

| # | Problem in the original | Fix | Why it matters |
|---|--------------------------|-----|----------------|
| 1 | Output layer used `activation='relu'` | Changed to **`sigmoid`** | A binary classifier must output a probability in [0,1]. This fix roughly **doubled recall** on churners (~0.23 → ~0.42). |
| 2 | `precision = TN / (TN + FP)` | Corrected to **`TP / (TP + FP)`** | The original formula computes *specificity*, not precision, so the reported precision and F1 were wrong. |
| 3 | No random seed | Added `set_seed()` | Without a fixed seed the model isn't reproducible — and reproducibility is a basic trust requirement. |
| 4 | Tuning function ignored its `optimizer` argument, called `.fit()` internally, returned the wrong model, and used the removed `keras.wrappers` API with the old `nb_epoch` key | Rewrote with **scikeras**, used the `optimizer` arg, returned the built model, corrected `nb_epoch` → `epochs` | The original grid search could not actually vary the optimizer and would crash on modern TensorFlow. |

Also added: a confusion-matrix heatmap, a `classification_report`, and a closing
model-risk reflection.

## Results (after fixes)

- **Test accuracy ≈ 0.86**
- **Recall (churners) ≈ 0.42** — of the customers who actually left, ~42% were caught
- **Precision (churners) ≈ 0.79**, **F1 ≈ 0.55**
- Best grid-search parameters: `batch_size=25, epochs≈20, optimizer='adam'`

The gap between **0.86 accuracy** and **0.42 recall** is the key lesson: on
imbalanced data, accuracy looks great while the model still misses most of the
customers you actually care about. **Recall** is the honest headline here.

## How to run

```bash
pip install pandas numpy seaborn scikit-learn tensorflow scikeras
jupyter notebook Bank_Churn_NN_FIXED.ipynb
```

Place `Churn_Modelling.csv` in the same folder. Run the cells top to bottom
(**Cell ▸ Run All**).

## Key takeaway

A neural network that scores 86% accuracy can still miss most churners. The value
isn't the headline number — it's judging the model on the metric that matches the
business goal (recall here), making it reproducible, and being honest about its
limits (imbalance, fairness, explainability). Build it, then challenge it.
