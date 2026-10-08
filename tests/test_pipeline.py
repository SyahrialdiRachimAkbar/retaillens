import numpy as np
import pandas as pd
import pytest

from retaillens.pipeline import clean_transactions, customer_features, labelled_snapshot, temporal_split, cohort_retention


def sample_sales():
    return pd.DataFrame({
        "InvoiceNo": ["1", "2", "3", "4", "5", "6"], "StockCode": ["A"] * 6,
        "Description": ["Test product"] * 6, "Quantity": [1] * 6, "UnitPrice": [10.0] * 6,
        "InvoiceDate": pd.to_datetime(["2020-12-01", "2021-01-01", "2021-01-31", "2021-02-01", "2021-02-15", "2021-03-03"]),
        "CustomerID": [101, 102, 101, 102, 103, 104], "Country": ["UK"] * 6,
    })


def test_cleaning_audit_reconciles_and_anonymous_sales_survive():
    raw = sample_sales()
    raw.loc[0, "CustomerID"] = np.nan
    raw.loc[1, "InvoiceNo"] = "C2"
    raw.loc[2, "UnitPrice"] = 0
    raw.loc[3, "Quantity"] = -1
    raw.loc[4, "UnitPrice"] = np.inf
    raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)
    clean, audit = clean_transactions(raw)
    assert len(clean) == 2
    assert audit["raw_rows"] - sum(step["removed"] for step in audit["steps"]) == len(clean)
    assert audit["rows_without_customer_id"] == 1
    assert clean["Revenue"].sum() == 20


def test_features_exclude_cutoff_and_future_and_count_unique_invoices():
    clean, _ = clean_transactions(sample_sales())
    cutoff = pd.Timestamp("2021-02-01")
    features = customer_features(clean, cutoff).set_index("CustomerID")
    assert features.loc["101", "frequency"] == 2
    assert features.loc["102", "frequency"] == 1  # Purchase exactly at cutoff is excluded.
    assert "103" not in features.index
    changed = clean.copy()
    changed.loc[changed["InvoiceDate"] >= cutoff, "Revenue"] = 99999
    pd.testing.assert_frame_equal(customer_features(clean, cutoff), customer_features(changed, cutoff))


def test_label_window_includes_start_excludes_end_and_rejects_censoring():
    raw = sample_sales()
    # 30-day window from Feb 1 ends at March 3: last purchase must not count.
    raw.loc[5, "CustomerID"] = 101
    clean, _ = clean_transactions(raw)
    labels = labelled_snapshot(clean, pd.Timestamp("2021-02-01"), pd.Timestamp("2021-03-04")).set_index("CustomerID")
    assert labels.loc["102", "repeat_purchase"] == 1
    assert labels.loc["101", "repeat_purchase"] == 0
    with pytest.raises(ValueError, match="Incomplete label"):
        labelled_snapshot(clean, pd.Timestamp("2021-02-01"), pd.Timestamp("2021-02-20"))


def test_temporal_split_rejects_training_labels_that_are_not_yet_known():
    snapshots = pd.DataFrame({"snapshot": pd.to_datetime(["2021-01-01"] * 2 + ["2021-02-01"] * 2 + ["2021-03-01"] * 2),
                              "label_end": pd.to_datetime(["2021-02-02"] * 2 + ["2021-03-01"] * 2 + ["2021-03-31"] * 2),
                              "repeat_purchase": [0, 1] * 3})
    with pytest.raises(ValueError, match="overlap"):
        temporal_split(snapshots)
    snapshots.loc[:1, "label_end"] = pd.Timestamp("2021-02-01")
    train, validation, test = temporal_split(snapshots)
    assert train["label_end"].max() <= validation["snapshot"].min()
    assert validation["label_end"].max() <= test["snapshot"].min()


def test_required_schema_is_validated():
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_transactions(pd.DataFrame({"Quantity": [1]}))


def test_multiple_product_lines_do_not_inflate_order_frequency():
    raw = sample_sales()
    extra = raw.iloc[[0]].copy()
    extra["StockCode"] = "B"
    raw = pd.concat([raw, extra], ignore_index=True)
    clean, _ = clean_transactions(raw)
    features = customer_features(clean, pd.Timestamp("2021-02-01")).set_index("CustomerID")
    assert features.loc["101", "frequency"] == 2
    assert features.loc["101", "unique_products"] == 2
    assert features.loc["101", "monetary"] == 30


def test_lookback_window_boundaries():
    clean, _ = clean_transactions(sample_sales())
    features = customer_features(clean, pd.Timestamp("2021-02-01"), lookback=31).set_index("CustomerID")
    assert features.loc["101", "frequency"] == 1  # December purchase falls outside window.
    assert features.loc["102", "frequency"] == 1  # January 1 is included exactly at boundary.


def test_cohort_zero_retention_is_distinct_from_unobserved_future():
    clean, _ = clean_transactions(sample_sales())
    retention = cohort_retention(clean).set_index("cohort")
    assert retention.loc["2020-12", 2] == 0  # February observed, no December-cohort return.
    assert pd.isna(retention.loc["2021-03", 1])  # April is not observed.
