import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit as st

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "churn_pipeline.joblib"
YELLOW, ORANGE, GREEN, RED = "#F2B632", "#E8672A", "#22c55e", "#d62728"
TENURE_ORDER = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.banner {background:#2ec08f; color:white; text-align:center; font-size:30px;
         font-weight:700; padding:10px; border-radius:4px; margin-bottom:14px;}
.card {background:white; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,.18);
       padding:16px 6px; text-align:center; margin-bottom:12px;}
.card .v {font-size:30px; font-weight:800; color:#222;}
.card .l {font-size:14px; color:#444;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="banner">CUSTOMER CHURN DASHBOARD</div>', unsafe_allow_html=True)


def card(col, value, label):
    col.markdown(f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


@st.cache_data
def load_data(src):
    df = pd.read_csv(src)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churned"] = np.where(df["Churn"] == "Yes", "Exited", "Retained")
    df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 36, 48, 60, 72],
                                labels=TENURE_ORDER).astype(str)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def style(fig, title, h=300):
    fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=14)), height=h,
                      margin=dict(l=10, r=10, t=45, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def add_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["tenure"] = df["tenure"].clip(0, 72)
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_bucket"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                                 labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    return df


tab1, tab2 = st.tabs(["Dashboard", "Predict a customer"])

# =============================== DASHBOARD ===============================
with tab1:
    data = None
    if os.path.exists(DATA_PATH):
        data = load_data(DATA_PATH)
    else:
        up = st.file_uploader("Upload the Telco churn CSV (or place it next to app.py)", type="csv")
        if up is not None:
            data = load_data(up)

    if data is None:
        st.info("Waiting for the Telco churn CSV.")
    else:
        # ---- filter bar ----
        with st.container(border=True):
            fc = st.columns(5)
            f = data
            for col, label, name in zip(fc, ["Contract", "Internet Service", "Payment Method",
                                             "Tenure Group", "Gender"],
                                        ["Contract", "InternetService", "PaymentMethod",
                                         "tenure_group", "gender"]):
                opts = TENURE_ORDER if name == "tenure_group" else sorted(data[name].unique())
                v = col.selectbox(label, ["All"] + opts)
                if v != "All":
                    f = f[f[name] == v]

        if f.empty:
            st.warning("No customers match these filters.")
            st.stop()

        total = len(f)
        exited = int((f["Churn"] == "Yes").sum())
        retained = total - exited

        # ---- KPI cards + donut/bar ----
        left, right = st.columns([1.1, 1.4])
        with left, st.container(border=True):
            r1 = st.columns(3)
            card(r1[0], f"{total:,}", "Total Customers")
            card(r1[1], f"{exited:,}", "Exit Customers")
            card(r1[2], f"{retained:,}", "Retain Customers")
            r2 = st.columns(4)
            card(r2[0], f"{exited / total:.1%}", "Churn Rate")
            card(r2[1], f"{f['tenure'].mean():.0f} mo", "Avg Tenure")
            card(r2[2], f"₹{f['MonthlyCharges'].mean():,.0f}", "Avg Monthly Charge")
            card(r2[3], f"{f.loc[f['Churn'] == 'Yes', 'MonthlyCharges'].sum() / 1000:,.1f}K",
                 "Monthly Revenue at Risk")

        ex = f[f["Churn"] == "Yes"]
        with right, st.container(border=True):
            c1, c2 = st.columns(2)
            g = ex.groupby("gender").size().reset_index(name="n")
            fig = px.pie(g, names="gender", values="n", hole=0.6, color="gender",
                         color_discrete_map={"Female": YELLOW, "Male": GREEN})
            c1.plotly_chart(style(fig, "EXIT CUSTOMERS BY GENDER"), width="stretch")

            ct = ex.groupby("Contract").size().reset_index(name="n").sort_values("n")
            fig = px.bar(ct, x="n", y="Contract", orientation="h", text="n",
                         color_discrete_sequence=[GREEN])
            fig.update_layout(xaxis_title="Exit Customers", yaxis_title="Contract")
            c2.plotly_chart(style(fig, "EXIT CUSTOMERS BY CONTRACT"), width="stretch")

        # ---- bottom charts ----
        by_t = (f.groupby(["tenure_group", "Churned"]).size().unstack(fill_value=0)
                .reindex(TENURE_ORDER, fill_value=0)
                .reindex(columns=["Retained", "Exited"], fill_value=0))

        b1, b2 = st.columns(2)
        with b1, st.container(border=True):
            fig = go.Figure(go.Scatter(x=TENURE_ORDER, y=by_t["Exited"], mode="lines+markers+text",
                                       text=by_t["Exited"], textposition="top center",
                                       line=dict(color=RED, dash="dot", width=3),
                                       marker=dict(size=8)))
            fig.update_layout(xaxis_title="Tenure group (months)", yaxis_title="Exited customers")
            st.plotly_chart(style(fig, "COUNT OF EXITED BY TENURE GROUP"), width="stretch")

        with b2, st.container(border=True):
            fig = go.Figure([
                go.Bar(name="Retained", x=TENURE_ORDER, y=by_t["Retained"], marker_color=YELLOW,
                       text=by_t["Retained"]),
                go.Bar(name="Exited", x=TENURE_ORDER, y=by_t["Exited"], marker_color=ORANGE,
                       text=by_t["Exited"]),
            ])
            fig.update_layout(barmode="group", xaxis_title="Tenure group (months)")
            st.plotly_chart(style(fig, "TOTAL CUSTOMERS BY TENURE GROUP"), width="stretch")

# =============================== PREDICTION ===============================
with tab2:
    try:
        pipe = load_model()
    except FileNotFoundError:
        st.error(f"Could not find `{MODEL_PATH}` in this folder. Save it from your notebook first.")
        st.stop()

    s1, s2, s3 = st.columns(3)
    threshold = s1.slider("Churn threshold", 0.05, 0.95, 0.50, 0.01)
    offer_cost = s2.number_input("Retention offer cost (₹)", 0, 100000, 500, 50)
    customer_value = s3.number_input("Value of retained customer (₹)", 0, 1000000, 5000, 500)
    save_rate = 0.30

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
        security = st.selectbox("Online security", opts)
        backup = st.selectbox("Online backup", opts)
    with c3:
        protection = st.selectbox("Device protection", opts)
        support = st.selectbox("Tech support", opts)
        tv = st.selectbox("Streaming TV", opts)
        movies = st.selectbox("Streaming movies", opts)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    d1, d2, d3 = st.columns(3)
    paperless = d1.selectbox("Paperless billing", ["Yes", "No"])
    payment = d2.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = d3.number_input("Monthly charges", 0.0, 200.0, 70.0, 0.5)
    total_c = d3.number_input("Total charges", 0.0, 10000.0, float(round(monthly * max(tenure, 1), 2)), 1.0)

    if st.button("Predict churn risk", type="primary"):
        row = add_features(pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total_c}]))
        p = float(pipe.predict_proba(row)[:, 1][0])
        gain = p * save_rate * customer_value - offer_cost
        tier = "High" if p >= max(0.7, threshold) else "Medium" if p >= threshold else "Low"

        k = st.columns(3)
        card(k[0], f"{p:.1%}", "Churn Probability")
        card(k[1], tier, "Risk Level")
        card(k[2], f"₹{gain:,.0f}", "Expected Gain if Contacted")

        if p >= threshold and gain > 0:
            st.error("Recommended: contact this customer with a retention offer.")
        elif p >= threshold:
            st.warning("At risk, but the offer is not expected to pay off at these settings.")
        else:
            st.success("Low risk at the current threshold.")

        try:
            Xt = pipe.named_steps["pre"].transform(row)
            names = [n.split("__", 1)[-1] for n in pipe.named_steps["pre"].get_feature_names_out()]
            sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
            sv = np.array(sv[1] if isinstance(sv, list) else sv)[0]
            s = pd.Series(sv, index=names)
            s = s.reindex(s.abs().sort_values(ascending=False).index).head(8)[::-1]
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.barh(s.index, s.values, color=[RED if v > 0 else GREEN for v in s.values])
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel("Impact on churn (red raises risk, green lowers it)")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP explanation needs a tree model. ({e})")
