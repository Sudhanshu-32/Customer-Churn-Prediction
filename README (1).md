# E-Commerce Customer Churn Prediction using SVM

A supervised machine learning project that predicts whether an e-commerce customer will churn (stop purchasing), using a Support Vector Machine (SVM) trained on real, imbalanced tabular data.

---

## Problem

E-commerce businesses lose revenue every time a customer stops ordering, and acquiring a new customer typically costs more than retaining an existing one. Without a predictive system, a business only discovers a customer has churned *after* they're already gone — there's no early warning.

This project frames churn prediction as a **binary classification problem**: given a customer's account and behavior data, predict whether they will churn (`1`) or stay (`0`), so retention efforts can be targeted *before* the customer leaves rather than after.

---

## Dataset

- **Source:** [E-commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (Kaggle)
- **Size:** 5,630 rows × 20 columns
- **Target:** `Churn` (1 = churned, 0 = stayed)
- **Class balance:** ~83% stayed vs ~17% churned — a real, imbalanced dataset, not an artificially clean one
- **Features:** tenure, city tier, preferred payment mode, satisfaction score, complaint history, cashback amount, order count, days since last order, marital status, and more

A real dataset was chosen deliberately over a synthetic/toy one, because it comes with the kind of problems a production dataset actually has: missing values, inconsistent category labels, mixed data types, and class imbalance — each of which had to be diagnosed and handled, not assumed away.

---

## Technologies

| Tool | Role |
|---|---|
| Python 3 | Core language |
| Google Colab + Google Drive | Development environment; Drive used for persistent dataset storage, since files uploaded directly to Colab are lost on runtime reset |
| pandas, NumPy | Data loading, cleaning, transformation |
| Matplotlib, Seaborn | EDA visualizations |
| scikit-learn | Preprocessing, SVM model, evaluation metrics |

---

## Workflow

```
Raw data → Clean → EDA → Preprocess → Split → Train (baseline) →
Train (balanced) → Evaluate → Interpret → Business insights
```

Each stage exists to answer a specific question: Is the data trustworthy? What patterns does it already show? Is it in a shape SVM can use? Is the model actually solving the real problem, not just scoring well on paper?

---

## EDA

Key patterns found before any model was built:

- **Tenure is the strongest churn signal** (correlation ≈ -0.34 with `Churn`) — churn is heavily concentrated in customers with under ~5 months of tenure; customers who pass that window rarely churn.
- **Complaints nearly triple churn risk** — 10.9% churn for customers who never complained vs 31.7% for those who did.
- **A counter-intuitive finding:** churn rate *rises* with `SatisfactionScore`, peaking at the highest score (5) rather than the lowest. This isn't explained by the dataset's documentation, so it's reported honestly as an open question rather than forced into a tidy explanation.

EDA wasn't just a formality before modeling — it directly shaped which findings made it into the final business insights, and confirmed (via a permutation importance check) that the model was relying on the same signals the raw data already pointed to.

---

## Preprocessing

Every step below was a deliberate choice, not a default:

- **Missing values** (7 numeric columns, ~250–300 blanks each) were filled with the **median**, not the mean. Several of these columns are skewed by outlier customers (e.g. very long days-since-last-order), and the mean would be pulled toward those extremes, making it a less representative fill value than the median.
- **Duplicate category labels** (e.g. `"Phone"` vs `"Mobile Phone"`, `"CC"` vs `"Credit Card"`) were merged before encoding. Left unfixed, the model would treat them as unrelated categories, splitting real signal across two labels that mean the same thing.
- **`CustomerID` was dropped.** It's a row identifier with no predictive meaning — keeping it risks the model learning a spurious relationship between an arbitrary ID and the outcome.
- **Categorical features were one-hot encoded**, not label-encoded. SVM is a distance-based algorithm and can't interpret text at all, so encoding is required, not optional. One-hot (rather than assigning 1/2/3 to categories) avoids inventing a false order between categories that have no real ranking (e.g. marital status).
- **Numeric features were scaled with `StandardScaler`.** This is the single most important preprocessing decision for an SVM specifically: SVM classifies by measuring distances between points, and features like `CashbackAmount` (hundreds) would otherwise dominate features like `SatisfactionScore` (1–5) purely because of their units, not because they're more predictive. The scaler was fit only on the training data and applied to the test data, so no information from the held-out set leaked into training.
- **Train-test split was stratified (80/20)**, preserving the real ~83/17 churn ratio in both sets — so the test set stays a realistic, representative sample of actual customers.

---

## Models

Two SVMs (RBF kernel) were trained and compared, not just one:

1. **Default SVM** — baseline, no class weighting.
2. **Balanced SVM (`class_weight='balanced'`)** — chosen as the final model.

**Why SVM at all:** it's a strong, well-understood choice for a moderate-sized tabular classification problem with a clear margin-based decision boundary, and it's the algorithm this project was specifically set out to apply.

**Why `class_weight='balanced'` over the default:** with ~83% of customers staying, the default SVM naturally biases toward predicting "stayed," since that minimizes error on the majority class. `class_weight='balanced'` penalizes mistakes on the minority (churn) class more heavily during training, forcing the model to actually learn to recognize churners instead of defaulting to the easy answer.

**A deliberate caution, not just a result:** an earlier `GridSearchCV` pass, tuned on raw accuracy, pushed test accuracy to ~99%. Investigating this (rather than reporting it uncritically) revealed that ~17% of the test rows shared an identical feature combination with a training row — a property of this dataset's low-cardinality features, not a coding error — which let a tightly-fit model partially "memorize" rather than generalize. This is why the model reported below was evaluated on **cross-validated and held-out metrics together**, not on whichever single number looked best.

---

## Evaluation

Accuracy alone is misleading on this dataset: a model that predicts "stayed" for every customer would already score ~83% accuracy while catching **zero** churners. That single fact shaped the entire evaluation strategy — **recall and F1-score on the churn class mattered more than raw accuracy**, because the real cost of a missed churner (lost customer, lost future revenue) is far higher than the cost of a false alarm (an unnecessary retention email or coupon).

Metrics reported, in order of relevance to the business problem:
1. **Recall (churn class)** — of all customers who actually churned, how many did the model catch? This is the headline metric for this problem.
2. **F1-score** — balances recall against precision, since maximizing recall alone (e.g. flagging everyone as a churn risk) is trivial and useless.
3. **Accuracy** — reported for completeness, but explicitly *not* used to select the final model.

---

## Results

Final model: **SVM (RBF kernel, `class_weight='balanced'`)**

| Metric | Class 0 (Stayed) | Class 1 (Churned) |
|---|---|---|
| Recall | 0.89 | 0.93 |
| F1-score | 0.93 | 0.75 |

**Overall accuracy:** 90%

**Confusion matrix:**
```
                Predicted: Stayed   Predicted: Churned
Actual: Stayed        831                  105
Actual: Churned        13                  177
```

The model correctly identifies **93% of customers who actually churn** — only 13 real churners were missed out of 190 in the test set. The trade-off is a higher false-positive rate (105 loyal customers flagged as at-risk), which was accepted deliberately, since a false alarm is far cheaper to the business than a missed churner.

**Business takeaways:**
- Retention effort should be front-loaded toward new customers (first ~5 months), where churn risk is highest.
- A complaint should trigger an automatic retention follow-up, not just a support ticket close-out, given its strong association with churn.
- Retention offers should be targeted at model-flagged at-risk customers rather than sent as blanket campaigns, since targeting is both more effective and cheaper than contacting every customer equally.

---

## How to Run

1. Clone this repository and open `svm_ecommerce_churn.ipynb` in Google Colab.
2. Upload `E_Commerce_Dataset.xlsx` to a folder in your Google Drive (e.g. `SVM_Project/`).
3. In the notebook's first cell, mount your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Update the file path in the data-loading cell to match your Drive folder.
5. Run all cells in order — the notebook proceeds through cleaning, EDA, preprocessing, model training, and evaluation sequentially.

**Requirements** (if running locally instead of Colab):
```
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

## Project Structure

```
.
├── svm.ipynb     # Main notebook: EDA, preprocessing, modeling, evaluation
├── E_Commerce_Dataset.xlsx       # Raw dataset (Kaggle)
├── Ecommerce_Churn_Presentation.pptx   # Project presentation
└── README.md                     # This file
```

---

## Future Improvements

- **Deploy the model** behind a simple scheduled script or API that scores customers regularly, so predictions reach the retention team automatically rather than living in a notebook.
- **Compare against other algorithms** (Random Forest, XGBoost, Logistic Regression) to check whether a different model offers a better precision-recall balance for this specific business trade-off.
- **Add richer features** — browsing behavior, support chat logs, or finer-grained purchase recency — which could catch churn signals earlier than the current account-level snapshot allows.
- **Schedule periodic retraining**, since churn drivers shift over time (seasonality, competitor activity, pricing changes); a model trained once on historical data will gradually lose accuracy without refreshing.
- **Investigate the satisfaction-score paradox** further — it may indicate the business's own satisfaction survey isn't measuring what it's assumed to measure.

---

## Author

**Sudhanshu**
- GitHub: [Sudhanshu-32](https://github.com/Sudhanshu-32)
- LinkedIn: [sudhanshu-kumar-dev](https://linkedin.com/in/sudhanshu-kumar-dev)
