import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
st.markdown("""
<style>

/* FULL PAGE BACKGROUND FIX */
html, body, [data-testid="stAppViewContainer"] {
    background: #0e1117 !important;
    height: 100%;
    width: 100%;
}

/* REMOVE WHITE OUTER SPACE */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* APP */
.stApp {
    background: #0e1117;
}

/* HEADER */
.main-header {
    background: #111827;
    padding: 18px 22px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    margin-bottom: 18px;
}

.main-header h1 {
    margin: 0;
    font-size: 24px;
    color: #ffffff;
}

.main-header p {
    margin: 0;
    color: #d1d5db;
    font-size: 13px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: #d1d5db !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #38bdf8 !important;
}

/* KPI CARDS */
[data-testid="stMetric"] {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-weight: 700;
    font-size: 30px !important;
}

[data-testid="stMetricLabel"] {
    color: #d1d5db !important;
    font-weight: 600;
}

/* TEXT */
h1, h2, h3 {
    color: #ffffff !important;
}

p, label, span {
    color: #d1d5db !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    background-color: #111827 !important;
    border-radius: 10px;
    border: 1px solid #1f2937;
}

/* INPUTS */
div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

/* TAGS */
span[data-baseweb="tag"] {
    background-color: #374151 !important;
    color: #ffffff !important;
}

/* BUTTONS */
.stButton > button {
    background: #1f2937;
    color: #ffffff;
    border: 1px solid #374151;
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #2563eb;
    color: white;
}

/* SLIDER */
.stSlider > div div div div {
    background-color: #38bdf8 !important;
}

/* CHARTS */
[data-testid="stPlotlyChart"] {
    background-color: #111827;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}

/* PLOTLY TEXT */
.js-plotly-plot text {
    fill: #ffffff !important;
    font-weight: 600 !important;
}

/* AXES */
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text {
    fill: #ffffff !important;
}

/* GRID */
.js-plotly-plot .gridlayer path {
    stroke: #374151 !important;
}

/* LEGEND */
.js-plotly-plot .legend text {
    fill: #ffffff !important;
}

/* TABLES */
table {
    color: #ffffff !important;
}

/* SUCCESS / INFO / WARNING */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

model = joblib.load("churn_model.pkl")

df_raw = pd.read_csv("Sales - Marketing customer dataset.csv")
df_clean = pd.read_csv("final_customer_data.csv")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Home", "Data Overview", "Dashboard", "EDA", "Prediction", "Insights"]
)

# =========================
# HOME
# =========================
if page == "Home":
    st.title("Customer Churn Prediction")

    st.markdown("""
    ## Project Overview
    - Data Cleaning  
    - Feature Engineering  
    - EDA  
    - ML Model  
    - Streamlit App  
    """)

# =========================
# DATA OVERVIEW
# =========================
elif page == "Data Overview":

    st.title("Data Overview")

    st.subheader("Raw Dataset")
    st.dataframe(df_raw, use_container_width=True)
    st.write("Shape:", df_raw.shape)

    st.subheader("Cleaned Dataset")
    st.dataframe(df_clean, use_container_width=True)
    st.write("Shape:", df_clean.shape)

# =========================
# DASHBOARD
# =========================
elif page == "Dashboard":

    st.title("Business Dashboard")

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.header("Filters")

    min_age, max_age = st.sidebar.slider(
        "Age Range",
        int(df_clean["age"].min()),
        int(df_clean["age"].max()),
        (20, 60)
    )

    if "country" in df_clean.columns:
        country_filter = st.sidebar.multiselect(
            "Country",
            df_clean["country"].unique(),
            default=df_clean["country"].unique()
        )
    else:
        country_filter = df_clean["country"].unique()

    churn_filter = st.sidebar.multiselect(
        "Churn",
        df_clean["churn"].unique(),
        default=df_clean["churn"].unique()
    )

    # =========================
    # APPLY FILTERS
    # =========================
    df_filtered = df_clean[
        (df_clean["age"] >= min_age) &
        (df_clean["age"] <= max_age) &
        (df_clean["churn"].isin(churn_filter))
    ]

    if "country" in df_clean.columns:
        df_filtered = df_filtered[df_filtered["country"].isin(country_filter)]

    # =========================
    # METRICS
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", len(df_filtered))
    c2.metric("Churn Rate", round(df_filtered["churn"].mean() * 100, 2))
    c3.metric("Avg Total Spent", round(df_filtered["total_spent"].mean(), 2))
    c4.metric("Avg Satisfaction", round(df_filtered["satisfaction_score"].mean(), 2))

    st.divider()

    # =========================
    # CHARTS
    # =========================

    fig1 = px.histogram(
        df_filtered,
        x="age",
        nbins=30,
        title="Age Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(
        df_filtered,
        names="churn",
        title="Churn Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "country" in df_filtered.columns:
        country_churn = df_filtered.groupby("country")["churn"].mean().sort_values(ascending=False)

        fig3 = px.bar(
            country_churn,
            title="Churn Rate by Country"
        )
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(
        df_filtered,
        x="churn",
        y="lifetime_value",
        title="Lifetime Value vs Churn"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # =========================
    # FIXED LINE CHART ONLY
    # =========================
    if "signup_month" in df_filtered.columns:

        trend = (
            df_filtered.groupby("signup_month")["churn"]
            .mean()
            .reset_index()
            .sort_values("signup_month")
        )

        fig = px.line(
            trend,
            x="signup_month",
            y="churn",
            markers=True,
            title="Churn Trend Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

# EDA
# =========================
elif page == "EDA":

    st.title("Exploratory Data Analysis")

    def safe_hist(x_col, title):
        if x_col in df_clean.columns:
            fig = px.histogram(
                df_clean,
                x=x_col,
                color="churn",
                barmode="group",
                title=title
            )
            st.plotly_chart(fig, use_container_width=True)

    def safe_box(x, y, title):
        if x in df_clean.columns and y in df_clean.columns:
            fig = px.box(df_clean, x=x, y=y, title=title)
            st.plotly_chart(fig, use_container_width=True)

    # Histograms
    safe_hist("age", "Age Distribution")
    safe_hist("gender", "Gender vs Churn")
    safe_hist("subscription_type", "Subscription Type vs Churn")
    safe_hist("payment_method", "Payment Method vs Churn")

    # Boxplots
    safe_box("churn", "age", "Age vs Churn")
    safe_box("churn", "total_spent", "Total Spent vs Churn")
    safe_box("churn", "satisfaction_score", "Satisfaction Score vs Churn")
    safe_box("churn", "lifetime_value", "Lifetime Value vs Churn")

    # =========================
    # CORRELATION MATRIX (FIXED)
    # =========================
    numeric_df = df_clean.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    if corr.shape[0] > 15:
        corr = corr.iloc[:15, :15]

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# PREDICTION
# =========================
elif page == "Prediction":

    st.title("Customer Churn Prediction")

    age = st.number_input("Age", 18, 100, 35)
    total_visits = st.number_input("Total Visits", 0, 500, 50)
    avg_session_time = st.number_input("Avg Session Time", 0.0, 1000.0, 30.0)
    total_spent = st.number_input("Total Spent", 0.0, 50000.0, 500.0)
    support_tickets = st.number_input("Support Tickets", 0, 100, 2)
    satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)

    if st.button("Predict"):

        sample = df_clean.drop("churn", axis=1).iloc[0:1].copy()

        sample["age"] = age
        sample["total_visits"] = total_visits
        sample["avg_session_time"] = avg_session_time
        sample["total_spent"] = total_spent
        sample["support_tickets"] = support_tickets
        sample["satisfaction_score"] = satisfaction_score

        prediction = model.predict(sample)[0]

        if prediction == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is likely to stay")

 # =========================
# INSIGHTS PAGE
# =========================
elif page == "Insights":

    st.title("Business Insights")

    st.markdown("### 1. High Value Customers are more loyal")

    if "total_spent" in df_clean.columns:

        insight1 = df_clean.groupby("churn")["total_spent"].mean().reset_index()

        fig1 = px.bar(
            insight1,
            x="churn",
            y="total_spent",
            color="churn",
            text_auto=True,
            title="Average Spending by Churn"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.info("Customers who spend more are less likely to churn.")

    st.markdown("---")

    st.markdown("### 2. Engagement affects churn")

    if "engagement_score" in df_clean.columns:

        fig2 = px.box(
            df_clean,
            x="churn",
            y="engagement_score",
            title="Engagement Score vs Churn"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.success("Low engagement customers are at higher risk of churn.")

    st.markdown("---")

    st.markdown("### 3. Satisfaction impact")

    if "satisfaction_score" in df_clean.columns:

        fig3 = px.box(
            df_clean,
            x="churn",
            y="satisfaction_score",
            title="Satisfaction vs Churn"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.warning("Lower satisfaction strongly increases churn probability.")