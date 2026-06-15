# Bank Customer Churn — Multi-Algorithm Builder Project

Same dataset as the neural-network project (10,000 bank customers), but here we
act as a **model builder** instead of a validator: prepare the data once, then
train a whole lineup of classification algorithms, compare them on equal footing,
pick the winner, and squeeze more out of it.

## The problem

Predict whether a customer will **churn** (`Exited = 1`, left the bank) or **stay**
(`0`). Binary classification on ~80% stayers / ~20% leavers — an imbalanced
dataset, so **recall** (how many real leavers we catch) matters as much as accuracy.

## What the notebook does

1. **Load & explore** — shape, types, and the 80/20 class balance.
2. **Prepare once** — drop ID columns, one-hot encode `Geography` and `Gender`
   (→ 13 features), split 80/20 (stratified), and scale with `StandardScaler`.
3. **Train 11 algorithms** in one fair loop, scoring each on Accuracy, Precision,
   Recall, F1 and ROC-AUC.
4. **Leaderboard** — a sorted results table, a grouped bar chart, and overlaid ROC curves.
5. **Best model** — confusion matrix and full classification report.
6. **Builder move** — handle the imbalance with `scale_pos_weight` to lift recall.
7. **Feature importance** — which inputs drive churn.

## Algorithms compared

Logistic Regression · K-Nearest Neighbours · Gaussian Naive Bayes · Decision Tree ·
Random Forest · Extra Trees · AdaBoost · Gradient Boosting · XGBoost ·
Support Vector Machine (RBF) · Neural Network (MLP)

## Results (test set, sorted by ROC-AUC)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** | 0.869 | 0.782 | 0.494 | 0.605 | **0.870** |
| Neural Net (MLP) | 0.861 | 0.740 | 0.489 | 0.589 | 0.860 |
| Random Forest | 0.858 | 0.759 | 0.442 | 0.559 | 0.849 |
| AdaBoost | 0.860 | 0.763 | 0.452 | 0.568 | 0.848 |
| Extra Trees | 0.856 | 0.753 | 0.435 | 0.551 | 0.845 |
| Decision Tree | 0.860 | 0.776 | 0.442 | 0.563 | 0.840 |
| XGBoost | 0.851 | 0.687 | 0.491 | 0.573 | 0.839 |
| SVM (RBF) | 0.862 | 0.850 | 0.391 | 0.535 | 0.828 |
| K-Nearest Neighbours | 0.830 | 0.702 | 0.290 | 0.410 | 0.790 |
| Gaussian Naive Bayes | 0.806 | 0.533 | 0.376 | 0.441 | 0.777 |
| Logistic Regression | 0.808 | 0.589 | 0.187 | 0.284 | 0.775 |

**Winner: Gradient Boosting** (ROC-AUC 0.870), with the MLP neural net and Random
Forest close behind. The simple linear and Naive Bayes models trail — they're good
honest baselines but can't match the ensembles here.

### Handling the imbalance

Adding `scale_pos_weight` (≈ the 4:1 stay/leave ratio) to XGBoost raised churner
**recall from 0.49 to 0.64** — catching far more real leavers, at the cost of a
little precision. For a retention programme that trade is usually worth it.

## How to run

```bash
pip install pandas numpy seaborn scikit-learn xgboost
jupyter notebook Bank_Churn_MultiAlgorithm.ipynb
```

Put `Churn_Modelling.csv` in the same folder and run all cells.

## Key takeaway

The headline isn't one accuracy number — it's that **boosted trees and ensembles
beat the simpler models**, that **accuracy hides the recall problem on imbalanced
data**, and that a small **imbalance fix** meaningfully improves how many churners
you actually catch. That's the builder's job: try many tools, compare them
honestly, and tune the best one toward the business goal.
