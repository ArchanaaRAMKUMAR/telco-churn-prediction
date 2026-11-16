# Customer Churn Prediction

End-to-end machine learning project that predicts which telecom customers are likely to leave, explains why, and helps decide who to contact with a retention offer. Includes a Streamlit dashboard and single-customer predictor.

**Live demo:** [https://telco-churn-prediction-dashboard.streamlit.app/](https://telco-churn-prediction-dashboard.streamlit.app/)

## Problem

Keeping an existing customer costs far less than acquiring a new one. The goal is to rank customers by churn risk so a retention team with a limited budget can focus on the customers where an offer is most likely to pay off.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data): about 7,000 customers, 21 columns covering demographics, services, contract, billing, and a churn label. About 27% of customers churned, so the classes are imbalanced.

## Approach

1. **Cleaning:** converted `TotalCharges` from text to numeric (blank for new customers), dropped `customerID`, encoded the target.
2. **EDA:** churn is highest in the first months of tenure, among month-to-month contracts, fiber optic users, and customers without tech support.
3. **Feature engineering:** `num_services`, `charges_per_month`, `tenure_bucket`.
4. **Pipeline:** stratified 80/20 split; scikit-learn `ColumnTransformer` (scaling + one-hot encoding) inside a `Pipeline` to avoid data leakage.
5. **Modeling:** compared Logistic Regression, Random Forest, XGBoost, and LightGBM with 5-fold stratified cross-validation scored on PR-AUC, using class weights for imbalance.
6. **Threshold selection:** chose the decision threshold by expected profit instead of defaulting to 0.5.
7. **Explainability:** SHAP for global drivers and per-customer explanations.

## Results

| Model | CV PR-AUC (mean ± std, 5-fold) |
|---|---|
| Logistic Regression | 0.668 ± 0.014 |
| Random Forest | 0.617 ± 0.029 |
| XGBoost | 0.662 ± 0.018 |
| LightGBM | 0.648 ± 0.023 |

Logistic Regression, XGBoost, and LightGBM are statistically close (differences are within one standard deviation), and Random Forest is the weakest. This is common on small tabular datasets with mostly categorical features. XGBoost was chosen as the final model because it performs on par with the simpler baseline while supporting exact SHAP explanations in the app.

Final model: **XGBoost**, evaluated once on the held-out test set (1,409 customers, 374 churners).

| Metric (test set, threshold 0.5) | Value |
|---|---|
| ROC-AUC | 0.84 |
| PR-AUC | 0.656 (baseline = churn rate, about 0.27) |
| Recall (churn class) | 0.78 |
| Precision (churn class) | 0.53 |
| F1 (churn class) | 0.63 |
| Accuracy | 0.76 |

At the default 0.5 threshold the model catches 78% of churners, at the cost of about half of flagged customers being false alarms.

**Business cost assumptions** (editable in the app): offer cost ₹500, retained-customer value ₹5,000, offer success rate 30%.

### Cost-based threshold

Using the assumptions above, expected profit on the test set peaks at a threshold of **0.53**, with an expected profit of about **₹166,000** (1,409 customers). The profit curve is fairly flat between roughly 0.45 and 0.70, so the exact threshold is not critical, and the default 0.5 is close to optimal. For comparison, contacting every customer would lose about ₹143,500 under the same assumptions (374 churners × ₹1,000 net gain, minus 1,035 non-churners × ₹500 wasted offers). Profit turns negative when the threshold is set too low, because too many offers go to customers who would have stayed.

## Key insights

- New customers are the most at risk; churn drops sharply after the first year.
- Month-to-month contracts churn far more than one- or two-year contracts.
- Fiber optic customers and customers without tech support churn more.
- Churners pay higher monthly charges on average.

These are associations in the data, not proven causes.

## Streamlit app

- **Dashboard tab:** filters, KPI cards, exits by gender and contract, and tenure-based churn charts.
- **Predict tab:** enter a customer's details to get churn probability, risk level, expected gain from contacting them, and a SHAP chart of the top reasons.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py`, `churn_pipeline.joblib`, and `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder.

## Project structure

```
app.py                                   Streamlit dashboard and predictor
churn_model.ipynb                        Data cleaning, modeling, evaluation
churn_pipeline.joblib                    Saved trained pipeline
WA_Fn-UseC_-Telco-Customer-Churn.csv     Dataset
requirements.txt                         Dependencies
```

## Limitations and next steps

- Single snapshot of data, so no time-based validation or trend analysis.
- Cost figures are assumptions, not company data.
- Possible next steps: hyperparameter tuning with Optuna, probability calibration, experiment tracking with MLflow, and drift monitoring.

## Tech stack

Python, pandas, scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, Plotly
