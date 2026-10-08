"""Download, audit, segment, evaluate and export a reproducible retail case study.

Run from the repository root: python -m retaillens.pipeline
All prediction features precede their snapshot cutoff. Labels occupy the next
30 days. Model selection uses validation average precision, never the test set.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve,
    roc_auc_score, silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
FEATURES = ["recency", "frequency", "monetary", "average_order_value",
            "tenure_days", "unique_products", "units"]
LOOKBACK = 180
HORIZON = 30
SEED = 42


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def download_dataset(destination: Path) -> dict:
    """Download only the expected XLSX member; do not extract arbitrary zip paths."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = Request(SOURCE_URL, headers={"User-Agent": "RetailLens/1.0 (educational case study)"})
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == "Online Retail.xlsx"]
            if len(matches) != 1:
                raise ValueError("The UCI archive did not contain the expected Online Retail.xlsx.")
            destination.write_bytes(archive.read(matches[0]))
    return {
        "name": "UCI Online Retail", "url": SOURCE_URL,
        "landing_page": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "citation": "Chen, D. (2015). Online Retail. UCI. https://doi.org/10.24432/C5BW33",
        "license": "CC BY 4.0", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                "UnitPrice", "CustomerID", "Country"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = raw[required].copy()
    audit = {"raw_rows": len(frame), "steps": []}

    def retain(mask: pd.Series, reason: str) -> None:
        nonlocal frame
        before = len(frame)
        frame = frame.loc[mask].copy()
        audit["steps"].append({"reason": reason, "removed": before - len(frame), "remaining": len(frame)})

    retain(~frame.duplicated(), "Exact duplicate rows (documented heuristic)")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    for column in ["Quantity", "UnitPrice"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    retain(frame["InvoiceDate"].notna(), "Invalid transaction timestamp")
    retain(frame["InvoiceNo"].notna() & frame["StockCode"].notna(), "Missing invoice or product identifier")
    frame["InvoiceNo"] = frame["InvoiceNo"].astype(str).str.strip()
    frame["StockCode"] = frame["StockCode"].astype(str).str.strip()
    retain(~frame["InvoiceNo"].str.upper().str.startswith("C"), "Cancelled invoices")
    retain(np.isfinite(frame["Quantity"]) & np.isfinite(frame["UnitPrice"])
           & (frame["Quantity"] > 0) & (frame["UnitPrice"] > 0), "Invalid / non-positive quantity or unit price")
    frame["CustomerID"] = pd.to_numeric(frame["CustomerID"], errors="coerce").astype("Int64").astype("string")
    frame["Country"] = frame["Country"].fillna("Unknown").astype(str)
    frame["Description"] = frame["Description"].fillna("Unspecified product").astype(str)
    frame["Revenue"] = frame["Quantity"] * frame["UnitPrice"]
    frame = frame.sort_values("InvoiceDate").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid positive sales remained after cleaning.")
    audit.update({
        "clean_sales_rows": len(frame),
        "rows_without_customer_id": int(frame["CustomerID"].isna().sum()),
        "customer_analysis_rows": int(frame["CustomerID"].notna().sum()),
        "revenue_definition": "Gross positive sales after cleaning; cancellations and returns excluded, not net revenue.",
    })
    return frame, audit


def customer_features(sales: pd.DataFrame, cutoff: pd.Timestamp, lookback: int = LOOKBACK) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    eligible = sales.loc[sales["CustomerID"].notna() & (sales["InvoiceDate"] < cutoff)
                         & (sales["InvoiceDate"] >= cutoff - pd.Timedelta(days=lookback))].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["CustomerID", *FEATURES, "Country"])
    eligible = eligible.sort_values("InvoiceDate")
    grouped = eligible.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"), first_purchase=("InvoiceDate", "min"),
        frequency=("InvoiceNo", "nunique"), monetary=("Revenue", "sum"),
        unique_products=("StockCode", "nunique"), units=("Quantity", "sum"), Country=("Country", "last"),
    ).reset_index()
    grouped["recency"] = (cutoff - grouped.pop("last_purchase")).dt.total_seconds() / 86400
    grouped["tenure_days"] = (cutoff - grouped.pop("first_purchase")).dt.total_seconds() / 86400
    grouped["average_order_value"] = grouped["monetary"] / grouped["frequency"]
    return grouped[["CustomerID", *FEATURES, "Country"]]


def labelled_snapshot(sales: pd.DataFrame, cutoff: pd.Timestamp,
                      observation_end: pd.Timestamp, lookback: int = LOOKBACK,
                      horizon: int = HORIZON) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    label_end = cutoff + pd.Timedelta(days=horizon)
    if label_end > pd.Timestamp(observation_end):
        raise ValueError("Incomplete label window: future purchases are not fully observable.")
    features = customer_features(sales, cutoff, lookback)
    future = sales.loc[(sales["InvoiceDate"] >= cutoff) & (sales["InvoiceDate"] < label_end), "CustomerID"]
    features["repeat_purchase"] = features["CustomerID"].isin(future.dropna()).astype(int)
    features["snapshot"] = cutoff
    features["label_end"] = label_end
    return features


def temporal_split(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoffs = sorted(snapshots["snapshot"].unique())
    if len(cutoffs) < 3:
        raise ValueError("Need at least three complete monthly snapshots for train, validation and test.")
    train = snapshots.loc[snapshots["snapshot"].isin(cutoffs[:-2])].copy()
    validation = snapshots.loc[snapshots["snapshot"] == cutoffs[-2]].copy()
    test = snapshots.loc[snapshots["snapshot"] == cutoffs[-1]].copy()
    if train["label_end"].max() > validation["snapshot"].min():
        raise ValueError("Training labels overlap the validation decision time.")
    if validation["label_end"].max() > test["snapshot"].min():
        raise ValueError("Validation labels overlap the test decision time.")
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        if split["repeat_purchase"].nunique() < 2:
            raise ValueError(f"The {name} split needs both positive and negative labels.")
    return train, validation, test


def segment_customers(customers: pd.DataFrame) -> tuple[pd.DataFrame, dict, object]:
    """Separate descriptive clustering from supervised prediction."""
    if len(customers) < 3:
        raise ValueError("At least three customers are needed for segmentation.")
    features = ["recency", "frequency", "monetary"]
    matrix = np.log1p(customers[features])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    candidates = []
    fitted = {}
    for k in range(2, min(6, len(customers) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(scaled, labels, sample_size=min(2000, len(customers)), random_state=SEED))
        candidates.append({"k": k, "silhouette": score})
        fitted[k] = model
    if not candidates:
        raise ValueError("Customer features have insufficient variation for clustering.")
    best = max(candidates, key=lambda item: item["silhouette"])
    model = fitted[best["k"]]
    result = customers.copy()
    result["cluster"] = model.labels_
    profile = result.groupby("cluster")[features].median()
    names = {}
    for cluster, row in profile.iterrows():
        if row["recency"] <= 30 and row["frequency"] >= 3:
            name = "Active loyalists"
        elif row["recency"] <= 30:
            name = "Recent shoppers"
        elif row["recency"] > 90:
            name = "Dormant shoppers"
        else:
            name = "Occasional shoppers"
        names[cluster] = f"{name} · {cluster + 1:02d}"
    result["segment"] = result["cluster"].map(names)
    return result, {"selected_k": best["k"], "silhouette": best["silhouette"],
                    "candidates": candidates, "method": "K-means on standardized log1p(RFM), descriptive only",
                    "names": {str(k): v for k, v in names.items()}}, {"scaler": scaler, "model": model}


def model_metrics(y: pd.Series, probability: np.ndarray) -> dict:
    count = max(1, int(np.ceil(len(y) * 0.1)))
    order = np.argsort(-probability, kind="stable")[:count]
    precision_top = float(np.asarray(y)[order].mean())
    prevalence = float(y.mean())
    return {"average_precision": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "precision_at_10pct": precision_top, "lift_at_10pct": precision_top / prevalence,
            "positive_rate": prevalence, "rows": len(y)}


def train_models(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, object, pd.DataFrame]:
    candidates = {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                                             l2_regularization=5, random_state=SEED),
    }
    fitted = {}
    validation_scores = {}
    for name, model in candidates.items():
        model.fit(train[FEATURES], train["repeat_purchase"])
        fitted[name] = model
        validation_scores[name] = model_metrics(validation["repeat_purchase"], model.predict_proba(validation[FEATURES])[:, 1])
    selected = max(validation_scores, key=lambda name: validation_scores[name]["average_precision"])
    # Refit only after choosing the model on validation, with all labels matured by test cutoff.
    combined = pd.concat([train, validation], ignore_index=True)
    comparison = {}
    curves = {}
    for name, model in fitted.items():
        model.fit(combined[FEATURES], combined["repeat_purchase"])
        probability = model.predict_proba(test[FEATURES])[:, 1]
        comparison[name] = {"validation": validation_scores[name], "test": model_metrics(test["repeat_purchase"], probability)}
        precision, recall, _ = precision_recall_curve(test["repeat_purchase"], probability)
        indices = np.unique(np.linspace(0, len(precision) - 1, min(160, len(precision))).astype(int))
        curves[name] = {"precision": precision[indices].tolist(), "recall": recall[indices].tolist()}
    chosen = fitted[selected]
    predictions = test[["CustomerID", "snapshot", "repeat_purchase"]].copy()
    predictions["probability"] = chosen.predict_proba(test[FEATURES])[:, 1]
    importance = permutation_importance(chosen, test[FEATURES], test["repeat_purchase"],
                                         scoring="average_precision", n_repeats=5, random_state=SEED)
    report = {"selected_model": selected, "selection_metric": "Validation average precision",
              "models": comparison, "precision_recall_curves": curves,
              "permutation_importance": [{"feature": name, "mean": float(mean), "std": float(std)}
                                         for name, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
              "feature_columns": FEATURES, "lookback_days": LOOKBACK, "horizon_days": HORIZON,
              "split": {name: {"rows": len(data), "customers": int(data["CustomerID"].nunique()),
                                "first_snapshot": str(data["snapshot"].min().date()),
                                "last_snapshot": str(data["snapshot"].max().date()),
                                "last_label_end": str(data["label_end"].max().date())}
                        for name, data in [("train", train), ("validation", validation), ("test", test)]},
              "prediction_scope": "Previously observed customers who bought in the preceding 180 days; positive purchase in the next 30 days.",
              "customer_overlap": "Customers can recur across chronological snapshots; this evaluates existing-customer prediction, not unseen-customer generalization.",
              "probability_note": "Model estimates are not separately calibrated; ranking quality and Brier score are reported.",
              "seed": SEED}
    return report, chosen, predictions


def cohort_retention(sales: pd.DataFrame) -> pd.DataFrame:
    known = sales.dropna(subset=["CustomerID"]).copy()
    known["month"] = known["InvoiceDate"].dt.to_period("M")
    known["cohort"] = known.groupby("CustomerID")["month"].transform("min")
    known["age"] = [(month.year - cohort.year) * 12 + month.month - cohort.month
                    for month, cohort in zip(known["month"], known["cohort"])]
    counts = known.groupby(["cohort", "age"])["CustomerID"].nunique().unstack()
    # Zero means an observed month with no returns; NaN means not yet observable.
    last_month = known["month"].max()
    first_cohort = known["cohort"].min()
    max_age = (last_month.year - first_cohort.year) * 12 + last_month.month - first_cohort.month
    counts = counts.reindex(columns=range(max_age + 1))
    for cohort in counts.index:
        observed_age = (last_month.year - cohort.year) * 12 + last_month.month - cohort.month
        observed = [age for age in counts.columns if age <= observed_age]
        counts.loc[cohort, observed] = counts.loc[cohort, observed].fillna(0)
    retention = counts.div(counts[0], axis=0)
    retention.index = retention.index.astype(str)
    return retention.reset_index()


def create_findings(summary: dict, audit: dict, segmentation: dict, evaluation: dict,
                    segments: pd.DataFrame, destination: Path) -> None:
    selected = evaluation["models"][evaluation["selected_model"]]["test"]
    baseline = evaluation["models"]["Prior baseline"]["test"]
    top = segments.sort_values("monetary", ascending=False).iloc[0]
    text = f"""# RetailLens — findings from the executed pipeline

This historical case study uses UCI Online Retail, not live business data.

## Data and revenue

- Raw transaction lines: **{audit['raw_rows']:,}**; cleaned positive-sales lines: **{audit['clean_sales_rows']:,}**.
- Gross positive sales: **£{summary['revenue']:,.2f}**, across **{summary['orders']:,}** invoices.
- Customers with identifiable IDs: **{summary['customers']:,}**. **{audit['rows_without_customer_id']:,}** sales lines lack a customer ID; they contribute to sales totals but not customer modeling.
- Returns, cancellations and non-positive prices/quantities are excluded. Sales are not net revenue or profit.

## Customer segmentation

- Segmentation uses the final {LOOKBACK}-day window, ending at {summary['snapshot']}.
- Selected **{segmentation['selected_k']} clusters**, silhouette **{segmentation['silhouette']:.3f}**, from k=2…6 candidates.
- Highest aggregate spend segment: **{top['segment']}**, with **{int(top['customers']):,}** customers and **£{top['monetary']:,.2f}** in that window.
- Segment labels describe observed behavior. They are analyst interpretations, not ground-truth personas.

## Held-out prediction results

- Selected on validation: **{evaluation['selected_model']}**.
- Test average precision: **{selected['average_precision']:.3f}**; prior baseline: **{baseline['average_precision']:.3f}**.
- Test ROC AUC: **{selected['roc_auc']:.3f}**; Brier score: **{selected['brier_score']:.3f}**.
- Top 10% precision: **{selected['precision_at_10pct']:.1%}**, compared with **{selected['positive_rate']:.1%}** test prevalence; lift **{selected['lift_at_10pct']:.2f}×**.
- This predicts observed purchase within 30 days, not a verified permanent-churn label.

## Proposed business experiments

1. Compare a loyalty offer against a no-offer control for active, frequent buyers.
2. Test a modest reactivation message for dormant shoppers, with a randomized holdout.
3. Use repeat-purchase ranking to prioritize a limited outreach budget, and measure incremental response and margin in an experiment.

These are proposed experiments, not measured campaign effects. Low purchase probability does not establish who will benefit from an intervention.

## Limits

The dataset covers one UK retailer in 2010–2011, including wholesale shoppers. December 2011 is a partial month. Exact-row deduplication is a heuristic and can remove legitimate repeated lines. Product identifiers may include non-merchandise charges. Cohorts start at the first observed purchase, not necessarily the customer's first-ever purchase. Final December scores have no complete future labels; performance numbers come only from the earlier held-out test snapshot. Country filters restrict a globally fitted segmentation and model, rather than retraining them.
"""
    destination.write_text(text, encoding="utf-8")


def run(input_path: Path | None = None) -> None:
    output = ROOT / "artifacts" / "dashboard"
    reports = ROOT / "reports"
    models = ROOT / "artifacts" / "models"
    for directory in [output, reports, models]:
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = input_path or ROOT / "data" / "raw" / "Online Retail.xlsx"
    if input_path is None:
        source = download_dataset(raw_path)
    else:
        source = {"name": "User-provided Online Retail-compatible data", "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  "license": "Verify the input dataset's own license"}
    print("Reading and auditing source transactions…", flush=True)
    raw = pd.read_excel(raw_path, engine="openpyxl") if raw_path.suffix.lower() == ".xlsx" else pd.read_csv(raw_path)
    sales, audit = clean_transactions(raw)
    observation_end = sales["InvoiceDate"].max()
    final_cutoff = observation_end.normalize() + pd.Timedelta(days=1)
    print("Building customer segments and chronological labels…", flush=True)
    customers = customer_features(sales, final_cutoff)
    customers, segmentation, segment_model = segment_customers(customers)
    cutoffs = pd.date_range(sales["InvoiceDate"].min().normalize() + pd.Timedelta(days=LOOKBACK),
                            observation_end - pd.Timedelta(days=HORIZON), freq="MS")
    snapshots = pd.concat([labelled_snapshot(sales, cutoff, observation_end) for cutoff in cutoffs], ignore_index=True)
    train, validation, test = temporal_split(snapshots)
    print("Training baseline, logistic regression and gradient boosting…", flush=True)
    evaluation, chosen, predictions = train_models(train, validation, test)
    customers["repeat_probability"] = chosen.predict_proba(customers[FEATURES])[:, 1]
    # Final scores are prospective; do not evaluate against unavailable future labels.
    customers.to_csv(output / "customers.csv", index=False)
    predictions.to_csv(output / "customer_predictions.csv", index=False)
    sales[["InvoiceNo", "InvoiceDate", "StockCode", "Description", "Country", "Quantity", "Revenue", "CustomerID"]].to_csv(output / "transactions.csv", index=False)
    monthly = sales.assign(month=sales["InvoiceDate"].dt.to_period("M").astype(str)).groupby("month").agg(
        revenue=("Revenue", "sum"), orders=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique")
    ).reset_index()
    monthly.to_csv(output / "monthly.csv", index=False)
    cohort_retention(sales).to_csv(output / "cohorts.csv", index=False)
    segments = customers.groupby("segment").agg(customers=("CustomerID", "nunique"), monetary=("monetary", "sum"),
                                                 median_recency=("recency", "median"), median_frequency=("frequency", "median"),
                                                 median_probability=("repeat_probability", "median")).reset_index()
    segments.to_csv(output / "segments.csv", index=False)
    summary = {"revenue": float(sales["Revenue"].sum()), "orders": int(sales["InvoiceNo"].nunique()),
               "customers": int(sales["CustomerID"].nunique()), "rows": len(sales),
               "start": str(sales["InvoiceDate"].min().date()), "end": str(observation_end.date()),
               "snapshot": str(final_cutoff.date()), "currency": "GBP", "source": source,
               "partial_month": str(observation_end.to_period("M")),
               "generated_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    for filename, payload in [("summary.json", summary), ("quality.json", audit),
                               ("segmentation.json", segmentation), ("evaluation.json", evaluation)]:
        save_json(output / filename, payload)
    joblib.dump({"model": chosen, "features": FEATURES, "trained_through": evaluation["split"]["validation"]["last_label_end"],
                 "lookback_days": LOOKBACK, "horizon_days": HORIZON}, models / "repeat_purchase.joblib")
    joblib.dump(segment_model, models / "segmentation.joblib")
    create_findings(summary, audit, segmentation, evaluation, segments, reports / "findings.md")
    save_json(reports / "evaluation.json", evaluation)
    save_json(reports / "data_quality.json", audit)
    print(f"Done. Selected: {evaluation['selected_model']}; test AP={evaluation['models'][evaluation['selected_model']]['test']['average_precision']:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional local XLSX or CSV with the Online Retail schema")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
