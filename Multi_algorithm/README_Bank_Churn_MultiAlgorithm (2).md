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

## Algorithms compared (quick beginner guide)

New to these? Here's each one in a sentence or two.

**Simple baselines**
- **Logistic Regression** — draws a straight dividing line between "stay" and "leave" and outputs a probability. Fast and easy to explain, but too basic for complex patterns.
- **Gaussian Naive Bayes** — uses probability to ask "given these details, how likely is each outcome?" It's "naive" because it assumes every feature acts independently, so it's quick but rough.
- **K-Nearest Neighbours (KNN)** — to classify a customer, it finds the most similar past customers and takes a majority vote. Like asking the people most like you what they did.

**One decision tree**
- **Decision Tree** — plays 20 questions ("Is age > 40? Is balance high?") branching down to a verdict. Easy to follow, but a single tree is shaky on its own.

**Many trees voting (ensembles)**
- **Random Forest** — builds hundreds of trees on random slices of the data and lets them vote. Many slightly-different trees averaged = far more reliable than one.
- **Extra Trees** — like Random Forest but makes its splits even more randomly, which makes it faster and sometimes more robust.

**Trees that fix each other's mistakes (boosting)**
- **AdaBoost** — builds trees one after another, each paying extra attention to the customers the previous trees got wrong.
- **Gradient Boosting** — same idea, more refined: each new tree is trained to reduce the leftover error of the model so far. *(Winner here.)*
- **XGBoost** — a faster, heavily-optimized version of gradient boosting with built-in safeguards against overfitting. It's the one tuned in the imbalance step.

**Boundary drawer**
- **Support Vector Machine (RBF)** — draws the best possible boundary between the two groups with the widest gap; the "RBF" part lets that boundary curve instead of being a straight line.

**Mini-brain**
- **Neural Network (MLP)** — a small network of connected "neurons" in layers that tunes its connection weights during training to map inputs to the right answer. Good at complex patterns. *(2nd place here.)*

*Mental map: simple baselines → one tree → many trees voting → trees fixing each other → boundary drawer → mini-brain.*

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
