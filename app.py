"""RetailLens — an interactive dashboard backed by the reproducible pipeline."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "artifacts" / "dashboard"
COLORS = ["#a3e635", "#38bdf8", "#a78bfa", "#fb923c", "#2dd4bf", "#f472b6"]
st.set_page_config(page_title="RetailLens · Customer intelligence", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background: #0b1119;}
.block-container {padding-top: 4.3rem; padding-bottom: 3rem; max-width: 1480px;}
[data-testid="stSidebar"] {background: #101923; border-right: 1px solid #223043;}
h1,h2,h3 {letter-spacing: -.035em;}
h1 {font-size: 2.5rem !important; font-weight: 750 !important;}
.eyebrow {color:#a3e635; font-size: .72rem; letter-spacing:.16em; font-weight:700; text-transform:uppercase;}
.intro {color:#98a9bd; max-width: 760px; line-height:1.65; margin-bottom:1.5rem;}
.brand {font-size:1.7rem; font-weight:750; letter-spacing:-.05em; margin-bottom:.15rem;}
.brand span {color:#a3e635;}
.brand-caption {color:#8da2ba; font-size:.8rem; margin-bottom:2rem;}
.status {border:1px solid #2c4228; color:#c5e6a5; background:#15231a; border-radius:20px; padding:.3rem .7rem; display:inline-block; font-size:.72rem; margin: .5rem 0 1rem;}
[data-testid="stMetric"] {background:#131d29; border:1px solid #283748; border-radius:14px; padding:1.15rem 1.3rem; min-height:118px;}
[data-testid="stMetricValue"] {font-size:1.85rem; letter-spacing:-.04em;}
[data-testid="stMetricLabel"] {color:#a9b7c8; font-size:.83rem;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#253447 !important; border-radius:14px !important;}
div[data-testid="stDataFrame"] {border-radius:10px; overflow:hidden;}
.small-note {font-size:.8rem; line-height:1.55; color:#97a9bf; margin-top:1.2rem;}
.callout {background:#17251e; border-left:3px solid #a3e635; border-radius:8px; padding:1rem 1.15rem; color:#d5e7d8; line-height:1.6;}
@media(max-width:700px) {
  .block-container {padding:3.8rem 1rem 2rem;}
  h1 {font-size:1.9rem !important;}
  [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
  [data-testid="stHorizontalBlock"]>[data-testid="stColumn"] {flex:1 1 240px; min-width: min(240px,100%) !important;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])>[data-testid="stColumn"] {flex:1 1 145px; min-width:min(145px,100%) !important;}
  [data-testid="stMetric"] {padding:.85rem; min-height:100px;}
  [data-testid="stMetricValue"] {font-size:1.35rem;}
  [data-testid="stMetricLabel"] {font-size:.75rem;}
}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_data(signature: tuple) -> tuple:
    del signature  # File mtimes invalidate cache after a fresh pipeline run.
    transactions = pd.read_csv(DATA / "transactions.csv", dtype={"CustomerID": "string", "InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    customers = pd.read_csv(DATA / "customers.csv", dtype={"CustomerID": str})
    documents = {name: json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
                 for name in ["summary", "quality", "segmentation", "evaluation"]}
    cohorts = pd.read_csv(DATA / "cohorts.csv")
    return transactions, customers, documents, cohorts


def chart_style(fig, height: int = 320):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=COLORS, height=height, margin=dict(l=5, r=10, t=25, b=10),
                      font=dict(family="Arial, sans-serif", color="#a9b9cd", size=12),
                      legend=dict(orientation="h", y=-0.18), hoverlabel=dict(bgcolor="#172330"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#233044", zeroline=False)
    return fig


def plot(fig, height: int = 320):
    st.plotly_chart(chart_style(fig, height), width="stretch", config={"displayModeBar": False})


required = ["transactions.csv", "customers.csv", "cohorts.csv", "summary.json", "quality.json", "segmentation.json", "evaluation.json"]
if any(not (DATA / name).exists() for name in required):
    st.title("RetailLens")
    st.info("Generate the analytics artifacts to start exploring the public retail dataset.")
    st.code("python -m retaillens.pipeline\nstreamlit run app.py", language="bash")
    st.stop()

sales, customers, documents, cohorts = load_data(tuple((name, (DATA / name).stat().st_mtime_ns) for name in required))
summary, quality, segmentation, evaluation = [documents[key] for key in ["summary", "quality", "segmentation", "evaluation"]]

with st.sidebar:
    st.markdown('<div class="brand"><span>◈</span> RetailLens</div><div class="brand-caption">Customer intelligence / public-data case study</div>', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "Customer segments", "Purchase predictions", "Model lab", "Data & methodology"], label_visibility="collapsed")
    st.divider()
    st.caption("ANALYSIS FILTER")
    country = st.selectbox("Market", ["All markets", *sorted(sales["Country"].unique())])
    st.caption("Market filters apply to sales, segments and prediction lists. Model evaluation remains global.")
    st.divider()
    st.markdown('<div class="status">● UCI public dataset</div>', unsafe_allow_html=True)
    st.caption(f"{summary['start']} → {summary['end']}\n\nCurrency: GBP · historical data")
    st.link_button("View data source ↗", "https://archive.ics.uci.edu/dataset/352/online+retail", width="stretch")

filtered_sales = sales if country == "All markets" else sales.loc[sales["Country"] == country]
filtered_customers = customers if country == "All markets" else customers.loc[customers["Country"] == country]
st.markdown(f'<div class="eyebrow">Retail intelligence / {html.escape(country)}</div>', unsafe_allow_html=True)

if page == "Overview":
    st.title("A clearer view of your customers.")
    st.markdown('<div class="intro">Explore the sales behind the numbers, understand who keeps coming back, and turn customer behavior into testable business decisions.</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    revenue = filtered_sales["Revenue"].sum()
    orders = filtered_sales["InvoiceNo"].nunique()
    columns[0].metric("Gross positive sales", f"£{revenue:,.0f}")
    columns[1].metric("Orders", f"{orders:,}")
    columns[2].metric("Identified customers", f"{filtered_sales['CustomerID'].nunique():,}")
    columns[3].metric("Average order value", f"£{revenue / orders:,.2f}" if orders else "—")
    st.caption("Full observed period · excludes returns and cancellations · positive sales, not net revenue")
    left, right = st.columns([1.7, 1])
    with left, st.container(border=True):
        st.subheader("Sales over time")
        st.caption("Monthly positive sales · December 2011 contains only nine days")
        monthly = filtered_sales.assign(month=filtered_sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()).groupby("month")["Revenue"].sum().reset_index()
        fig = px.area(monthly, x="month", y="Revenue", color_discrete_sequence=[COLORS[0]], labels={"month": "", "Revenue": "Sales (£)"})
        fig.update_traces(line=dict(width=2.5), fillcolor="rgba(163,230,53,.10)")
        plot(fig, 335)
    with right, st.container(border=True):
        st.subheader("Customer mix")
        st.caption(f"Identified buyers in the final {evaluation['lookback_days']}-day window")
        if filtered_customers.empty:
            st.info("No identified buyers for this market in the final customer window.")
        else:
            mix = filtered_customers.groupby("segment").size().rename("customers").reset_index()
            fig = px.pie(mix, values="customers", names="segment", hole=.73, color_discrete_sequence=COLORS)
            fig.update_traces(textinfo="none", marker=dict(line=dict(color="#0b1119", width=3)))
            fig.add_annotation(text=f"<b>{len(filtered_customers):,}</b><br>customers", showarrow=False, font=dict(size=19, color="#edf4fa"))
            plot(fig, 335)
    left, right = st.columns([1.7, 1])
    with left, st.container(border=True):
        st.subheader("Items driving sales")
        products = filtered_sales.groupby("Description")["Revenue"].sum().nlargest(7).sort_values().reset_index()
        products["label"] = products["Description"].map(lambda value: value if len(value) <= 30 else value[:27] + "…")
        fig = px.bar(products, x="Revenue", y="label", orientation="h", hover_name="Description", color_discrete_sequence=[COLORS[1]], labels={"Revenue": "Sales (£)", "label": ""})
        plot(fig, 300)
        st.caption("Source descriptions include postage and service charges.")
    with right, st.container(border=True):
        st.subheader("From insight to experiment")
        st.markdown("**01 · Reward loyalty**\n\nTest a loyalty offer with frequent, recent buyers.")
        st.markdown("**02 · Reconnect thoughtfully**\n\nTry a small reactivation campaign with dormant customers.")
        st.markdown("**03 · Measure what changes**\n\nUse a randomized holdout to measure incremental purchases and margin.")
        st.caption("Proposed experiments. Campaign impact has not been measured in this dataset.")

elif page == "Customer segments":
    st.title("Different customers. Different approaches.")
    st.markdown(f'<div class="intro">Behavioral groups from recency, purchase frequency and spend. Snapshot: {summary["snapshot"]} · preceding 180 days · globally fitted clusters.</div>', unsafe_allow_html=True)
    if filtered_customers.empty:
        st.info("No identified customers in this market within the final 180-day window.")
        st.stop()
    profile = filtered_customers.groupby("segment").agg(customers=("CustomerID", "count"), spend=("monetary", "sum"),
            median_recency=("recency", "median"), median_orders=("frequency", "median"), median_spend=("monetary", "median")).reset_index()
    columns = st.columns(3)
    columns[0].metric("Customers in window", f"{len(filtered_customers):,}")
    columns[1].metric("Global cluster count", segmentation["selected_k"])
    columns[2].metric("Global silhouette", f"{segmentation['silhouette']:.3f}")
    left, right = st.columns([1.45, 1])
    with left, st.container(border=True):
        st.subheader("Recency meets value")
        fig = px.scatter(filtered_customers, x="recency", y="monetary", color="segment", size="frequency",
                         size_max=28, log_y=True, opacity=.6, color_discrete_sequence=COLORS,
                         hover_data={"CustomerID": True, "frequency": True, "monetary": ":,.2f"},
                         labels={"recency": "Days since last purchase", "monetary": "180-day spend (£, log scale)", "segment": ""})
        plot(fig, 380)
    with right, st.container(border=True):
        st.subheader("Spend by segment")
        plot(px.bar(profile.sort_values("spend"), x="spend", y="segment", orientation="h", color="segment", color_discrete_sequence=COLORS,
                    labels={"spend": "180-day sales (£)", "segment": ""}).update_layout(showlegend=False), 380)
    st.subheader("Segment profiles")
    st.dataframe(profile, hide_index=True, width="stretch", column_config={
        "segment": "Segment", "customers": "Customers", "spend": st.column_config.NumberColumn("Total spend", format="£%.2f"),
        "median_recency": st.column_config.NumberColumn("Median recency (days)", format="%.1f"),
        "median_orders": "Median orders", "median_spend": st.column_config.NumberColumn("Median customer spend", format="£%.2f")})
    st.caption("Names are descriptive analyst interpretations of cluster medians. Cluster selection maximizes sampled silhouette over k=2…6; it does not prove business usefulness.")
    st.subheader("Observed-cohort retention · all markets")
    st.caption("Share of each first-observed-purchase cohort returning in a later month. Blank cells are unobserved; December 2011 is partial.")
    matrix = cohorts.set_index("cohort")
    fig = px.imshow(matrix, aspect="auto", color_continuous_scale=[[0, "#142030"], [1, "#a3e635"]], zmin=0, zmax=1,
                    labels={"x": "Months since first observed purchase", "y": "Cohort", "color": "Retention"})
    fig.update_traces(hovertemplate="Cohort %{y}<br>Month %{x}<br>Retention %{z:.1%}<extra></extra>")
    plot(fig, 400)

elif page == "Purchase predictions":
    st.title("Who is likely to return?")
    st.markdown(f'<div class="intro">Estimated purchase probability over the next 30 days, using behavior in the prior 180 days. Scored at {summary["snapshot"]}; future outcomes are unavailable for this final snapshot.</div>', unsafe_allow_html=True)
    chosen_segment = st.selectbox("Customer segment", ["All segments", *sorted(filtered_customers["segment"].unique())])
    minimum = st.slider("Minimum estimated probability", 0, 100, 0, step=5, format="%d%%")
    selected = filtered_customers.loc[filtered_customers["repeat_probability"] >= minimum / 100].copy()
    if chosen_segment != "All segments":
        selected = selected.loc[selected["segment"] == chosen_segment]
    columns = st.columns(3)
    columns[0].metric("Matching customers", f"{len(selected):,}")
    columns[1].metric("Median estimated probability", f"{selected['repeat_probability'].median():.0%}" if len(selected) else "—")
    columns[2].metric("Selected model", evaluation["selected_model"])
    st.caption("Model estimates are not separately calibrated. This ranking identifies likely purchasers, not proven campaign responders.")
    if selected.empty:
        st.info("No customers match these filters. Lower the probability threshold or select another market or segment.")
    else:
        display = selected.sort_values("repeat_probability", ascending=False)[["CustomerID", "Country", "segment", "repeat_probability", "recency", "frequency", "monetary"]].copy()
        display["repeat_probability"] *= 100
        st.download_button("Download filtered customer list ↓", data=display.to_csv(index=False).encode("utf-8"), file_name="retaillens_customers.csv", mime="text/csv")
        st.dataframe(display, hide_index=True, width="stretch", column_config={
            "CustomerID": "Customer", "Country": "Market", "segment": "Segment",
            "repeat_probability": st.column_config.ProgressColumn("Estimated probability (%)", min_value=0, max_value=100, format="%.1f%%"),
            "recency": st.column_config.NumberColumn("Recency (days)", format="%.1f"),
            "frequency": "Orders / 180 days", "monetary": st.column_config.NumberColumn("Spend / 180 days", format="£%.2f")})

elif page == "Model lab":
    st.title("Performance, with the evidence attached.")
    st.markdown('<div class="intro">A prior baseline, logistic regression and gradient boosting compete on chronological validation. The final test snapshot is held out from model selection.</div>', unsafe_allow_html=True)
    chosen = evaluation["models"][evaluation["selected_model"]]["test"]
    columns = st.columns(4)
    columns[0].metric("Test average precision", f"{chosen['average_precision']:.3f}")
    columns[1].metric("Test ROC AUC", f"{chosen['roc_auc']:.3f}")
    columns[2].metric("Top 10% lift", f"{chosen['lift_at_10pct']:.2f}×")
    columns[3].metric("Test Brier score ↓", f"{chosen['brier_score']:.3f}")
    st.caption(f"Selected: {evaluation['selected_model']} · test purchase rate: {chosen['positive_rate']:.1%} · global evaluation")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Precision–recall comparison")
        fig = go.Figure()
        for name, curve in evaluation["precision_recall_curves"].items():
            fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], name=name, mode="lines"))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1.02])
        plot(fig, 340)
    with right, st.container(border=True):
        st.subheader("What helps the ranking?")
        importance = pd.DataFrame(evaluation["permutation_importance"]).sort_values("mean")
        fig = px.bar(importance, x="mean", y="feature", error_x="std", orientation="h", color_discrete_sequence=[COLORS[0]],
                     labels={"mean": "Drop in average precision after shuffling", "feature": ""})
        plot(fig, 340)
        st.caption("Held-out permutation importance describes this model; it is not causal evidence. Correlated features can share importance.")
    comparison = []
    for name, metrics in evaluation["models"].items():
        comparison.append({"Model": name, "Validation AP": metrics["validation"]["average_precision"],
                           "Test AP": metrics["test"]["average_precision"], "Test ROC AUC": metrics["test"]["roc_auc"],
                           "Test Brier": metrics["test"]["brier_score"], "Test top-10% lift": metrics["test"]["lift_at_10pct"]})
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(comparison).round(4), hide_index=True, width="stretch")
    st.subheader("Chronological split")
    st.dataframe(pd.DataFrame(evaluation["split"]).T.reset_index(names="split"), hide_index=True, width="stretch")
    st.markdown('<div class="callout">Every feature uses purchases before its snapshot. Training labels mature before validation, and validation labels mature before test. Customers may recur across snapshots: this measures prediction for existing customers.</div>', unsafe_allow_html=True)

else:
    st.title("Transparent data. Reproducible decisions.")
    st.markdown('<div class="intro">The dataset, assumptions and quality checks behind every chart and estimate.</div>', unsafe_allow_html=True)
    st.subheader("Source & attribution")
    st.markdown("[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) · Chen, D. (2015) · [DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33) · CC BY 4.0")
    st.caption(f"Observed period: {summary['start']} to {summary['end']}. One UK retailer, including wholesale customers.")
    st.subheader("Data quality audit")
    st.dataframe(pd.DataFrame(quality["steps"]), hide_index=True, width="stretch",
                 column_config={"reason": "Cleaning step", "removed": "Rows removed", "remaining": "Rows remaining"})
    st.info(f"{quality['rows_without_customer_id']:,} positive-sales lines have no customer ID. Sales charts include them; segmentation and prediction exclude them.")
    st.subheader("Definitions that matter")
    st.markdown("""
    - **Revenue:** positive quantity × positive unit price, in GBP. Returns and cancellations are excluded; this is not net revenue or profit.
    - **Orders:** unique invoice numbers. Average order value = positive sales / orders.
    - **Recency:** days between the snapshot and the most recent purchase in the preceding 180 days.
    - **Frequency / spend:** unique invoices / total positive sales within that same window.
    - **Customer market:** country on the customer's latest eligible purchase; a customer can have sales in other countries.
    - **Repeat purchase:** at least one positive purchase during the next 30 days. This is not a permanent churn label.
    - **Tenure:** days since the first purchase observed within the lookback window, not lifetime tenure.
    """)
    st.subheader("Limitations & reproducibility")
    st.markdown("""
    Historical data cannot establish current market behavior. December 2011 is partial.
    Exact duplicates are removed as a heuristic, and product codes may contain service charges.
    Cohorts refer to first **observed** purchases. Predictions are not individually calibrated,
    and campaign uplift needs a randomized experiment. Segment and country filters do not retrain models.

    Run the pipeline with seed **42** to regenerate audits, segments, chronological evaluation,
    customer scores and model artifacts. See the repository README and generated findings report.
    """)
    st.code("python -m retaillens.pipeline\npython -m pytest\nstreamlit run app.py", language="bash")
    st.download_button("Download evaluation report ↓", json.dumps(evaluation, indent=2), file_name="retaillens_evaluation.json", mime="application/json")

st.markdown('<div class="small-note">RetailLens · An independent data science portfolio project · Public historical data, real computed results.</div>', unsafe_allow_html=True)
