# RetailLens — findings from the executed pipeline

This historical case study uses UCI Online Retail, not live business data.

## Data and revenue

- Raw transaction lines: **541,909**; cleaned positive-sales lines: **524,878**.
- Gross positive sales: **£10,642,110.80**, across **19,960** invoices.
- Customers with identifiable IDs: **4,338**. **132,186** sales lines lack a customer ID; they contribute to sales totals but not customer modeling.
- Returns, cancellations and non-positive prices/quantities are excluded. Sales are not net revenue or profit.

## Customer segmentation

- Segmentation uses the final 180-day window, ending at 2011-12-10.
- Selected **2 clusters**, silhouette **0.430**, from k=2…6 candidates.
- Highest aggregate spend segment: **Active loyalists · 02**, with **1,222** customers and **£4,222,028.44** in that window.
- Segment labels describe observed behavior. They are analyst interpretations, not ground-truth personas.

## Held-out prediction results

- Selected on validation: **Gradient boosting**.
- Test average precision: **0.659**; prior baseline: **0.384**.
- Test ROC AUC: **0.720**; Brier score: **0.206**.
- Top 10% precision: **84.4%**, compared with **38.4%** test prevalence; lift **2.20×**.
- This predicts observed purchase within 30 days, not a verified permanent-churn label.

## Proposed business experiments

1. Compare a loyalty offer against a no-offer control for active, frequent buyers.
2. Test a modest reactivation message for dormant shoppers, with a randomized holdout.
3. Use repeat-purchase ranking to prioritize a limited outreach budget, and measure incremental response and margin in an experiment.

These are proposed experiments, not measured campaign effects. Low purchase probability does not establish who will benefit from an intervention.

## Limits

The dataset covers one UK retailer in 2010–2011, including wholesale shoppers. December 2011 is a partial month. Exact-row deduplication is a heuristic and can remove legitimate repeated lines. Product identifiers may include non-merchandise charges. Cohorts start at the first observed purchase, not necessarily the customer's first-ever purchase. Final December scores have no complete future labels; performance numbers come only from the earlier held-out test snapshot. Country filters restrict a globally fitted segmentation and model, rather than retraining them.
