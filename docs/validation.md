# Verification record

Verified locally on **8 October 2026**, using Windows, Python 3.14, and the library versions in `requirements-lock.txt`.

## Analytical correctness

- Full UCI pipeline completed: 541,909 raw rows → 524,878 cleaned positive-sales rows.
- Sales totals reconcile to £10,642,110.80 and 19,960 unique invoices.
- **8 automated tests passed** without warnings: cleaning reconciliation, invalid values,
  retained anonymous sales, unique-invoice counting, feature-window boundaries,
  future-feature exclusion, label-window boundaries/censoring, chronological overlap and cohort zero vs missing behavior.
- The analytical notebook passed nbformat validation; all **7 code cells** executed successfully
  in the project interpreter, with regenerated text, tabular and Plotly outputs.
- App source, pipeline and tests passed Python compilation.

## Dashboard behavior

Streamlit AppTest checks passed for all five pages:
Overview, Customer segments, Purchase predictions, Model lab, Data & methodology.
Changing the market to France changed the real sales totals. Setting minimum purchase
probability to 100% produced the intended zero-result state without an exception.

Playwright browser checks verified:

- Navigation through all five pages without a displayed Python exception.
- Market search/select interaction; France yielded 71 final-window eligible customers.
- Customer CSV download, with expected filename `retaillens_customers.csv`.
- Desktop **1440 × 1100** and mobile **390 × 844** layouts, including mobile sidebar collapse.
- No body-level horizontal overflow at either viewport.
- No JavaScript console errors during the checked flows.

Streamlit 1.52.2 emits repeated theme warnings mentioning empty derived sidebar colors
(`widgetBackgroundColor`, `widgetBorderColor`, `skeletonBackgroundColor`). These warnings
remain visible in the console; the sidebar, controls and charts rendered with the configured
colors. AppTest also emits its standard missing-ScriptRunContext warning in bare mode.
Neither warning produced an application exception in these checks.

Screenshots: [desktop](dashboard-desktop.png), [mobile](dashboard-mobile.png).

## Evaluation guardrails

- Training snapshots: June–September 2011; last training labels mature 1 October.
- Validation snapshot: 1 October; its labels mature 31 October.
- Held-out test snapshot: 1 November; its labels mature 1 December.
- Final 10 December snapshot is scored without claiming observed future outcomes.
- Model selection uses validation average precision. Test diagnostics do not change the selected model.
- Customer-level source/export files and model binaries are excluded from Git; aggregate reports are included.

These checks establish reproducibility and the checked analytical/UI behavior. They do not
establish current-market performance, calibrated individual probabilities or causal campaign impact.
