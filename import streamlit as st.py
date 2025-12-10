import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from datetime import datetime

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NovaMart Analytics Dashboard",
    layout="wide"
)

st.sidebar.title("NovaMart Dashboard Navigation")

pages = [
    "Executive Overview",
    "Campaign Analytics",
    "Customer Insights",
    "Product Performance",
    "Geographic Analysis",
    "Attribution & Funnel",
    "ML Model Evaluation"
]

page = st.sidebar.radio("Go to", pages)

# =========================================================
# 2. DATA LOADER (CACHED)
# =========================================================
@st.cache_data
def load_data():
    data = {
        "campaign": pd.read_csv("campaign_performance.csv"),
        "customers": pd.read_csv("customer_data.csv"),
        "product": pd.read_csv("product_sales.csv"),
        "lead": pd.read_csv("lead_scoring_results.csv"),
        "feature": pd.read_csv("feature_importance.csv"),
        "learning": pd.read_csv("learning_curve.csv"),
        "geo": pd.read_csv("geographic_data.csv"),
        "attribution": pd.read_csv("channel_attribution.csv"),
        "funnel": pd.read_csv("funnel_data.csv"),
        "journey": pd.read_csv("customer_journey.csv"),
        "corr": pd.read_csv("correlation_matrix.csv")
    }
    return data

data = load_data()

campaign = data["campaign"]
customers = data["customers"]
product = data["product"]
lead = data["lead"]
feature = data["feature"]
learning = data["learning"]
geo = data["geo"]
attrib = data["attribution"]
funnel = data["funnel"]
corr = data["corr"]

# =========================================================
# 3. PAGE 1 – EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":
    st.title("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${campaign['revenue'].sum():,.0f}")
    col2.metric("Total Conversions", f"{campaign['conversions'].sum():,}")
    col3.metric("Avg ROAS", f"{campaign['roas'].mean():.2f}")
    col4.metric("Total Customers", f"{customers.shape[0]:,}")

    # Line Chart (Revenue Trend)
    st.subheader("Revenue Trend Over Time")
    fig = px.line(campaign, x="date", y="revenue", title="Daily Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Channel Performance Bar Chart
    st.subheader("Channel Revenue Comparison")
    channel_rev = campaign.groupby("channel")["revenue"].sum().reset_index()
    fig2 = px.bar(channel_rev, x="revenue", y="channel", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 4. PAGE 2 – CAMPAIGN ANALYTICS
# =========================================================
if page == "Campaign Analytics":
    st.title("Campaign Analytics")

    # Line Chart
    st.subheader("Revenue Trend")
    fig = px.line(campaign, x="date", y="revenue", color="channel")
    st.plotly_chart(fig, use_container_width=True)

    # Grouped Bar Chart (Region vs Quarter)
    st.subheader("Regional Revenue by Quarter")
    campaign["quarter"] = pd.to_datetime(campaign["date"]).dt.to_period("Q")
    grouped = campaign.groupby(["region", "quarter"])["revenue"].sum().reset_index()
    fig3 = px.bar(grouped, x="quarter", y="revenue", color="region", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

    # Stacked Bar (Campaign Type)
    st.subheader("Monthly Spend by Campaign Type")
    campaign["month"] = pd.to_datetime(campaign["date"]).dt.to_period("M")
    stack = campaign.groupby(["month", "campaign_type"])["spend"].sum().reset_index()
    fig4 = px.bar(stack, x="month", y="spend", color="campaign_type")
    st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# 5. PAGE 3 – CUSTOMER INSIGHTS
# =========================================================
if page == "Customer Insights":
    st.title("Customer Insights")

    # Histogram
    st.subheader("Customer Age Distribution")
    fig = px.histogram(customers, x="age", nbins=40)
    st.plotly_chart(fig, use_container_width=True)

    # Box Plot (LTV by Segment)
    st.subheader("LTV by Customer Segment")
    fig2 = px.box(customers, x="segment", y="ltv")
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter Plot (Income vs LTV)
    st.subheader("Income vs Lifetime Value")
    fig3 = px.scatter(customers, x="income", y="ltv", color="segment")
    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# 6. PAGE 4 – PRODUCT PERFORMANCE
# =========================================================
if page == "Product Performance":
    st.title("Product Performance")

    st.subheader("Treemap: Product Hierarchy")
    fig = px.treemap(
        product,
        path=["category", "subcategory", "product"],
        values="sales",
        color="profit_margin"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 7. PAGE 5 – GEOGRAPHIC ANALYSIS
# =========================================================
if page == "Geographic Analysis":
    st.title("Geographical Performance")

    st.subheader("Revenue by State (Choropleth)")
    fig = px.choropleth(
        geo,
        geojson="https://raw.githubusercontent.com/varun-singhh/india-geojson/master/india_states.geojson",
        featureidkey="properties.st_nm",
        locations="state",
        color="revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 8. PAGE 6 – ATTRIBUTION & FUNNEL
# =========================================================
if page == "Attribution & Funnel":
    st.title("Attribution & Funnel Analysis")

    # Donut Chart
    st.subheader("Attribution Model Comparison")
    model = st.selectbox("Choose Attribution Model", ["first_touch", "last_touch", "linear"])
    fig = px.pie(attrib, values=model, names="channel", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Funnel Chart
    st.subheader("Marketing Funnel")
    fig2 = go.Figure(go.Funnel(
        y=funnel["stage"],
        x=funnel["value"],
        textinfo="value+percent initial"
    ))
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 9. PAGE 7 – ML MODEL EVALUATION
# =========================================================
if page == "ML Model Evaluation":
    st.title("ML Model Evaluation")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(lead["actual_converted"], lead["predicted_class"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(lead["actual_converted"], lead["predicted_probability"])
    auc = roc_auc_score(lead["actual_converted"], lead["predicted_probability"])
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random"))
    roc_fig.update_layout(title=f"AUC = {auc:.2f}")
    st.plotly_chart(roc_fig, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance")
    fig3 = px.bar(feature, x="importance", y="feature", orientation="h",
                  error_x="std")
    st.plotly_chart(fig3, use_container_width=True)
