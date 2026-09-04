# OTT Viewer Drop-Off Risk Model

Diagnosing why viewers disengage during Season 1 of OTT shows, and building a model that flags high-risk episodes from their content design — with SHAP-based explainability and content segmentation to turn the findings into product recommendations.

## Business Problem

OTT platforms often see strong launch viewership, but a significant share of viewers disengage during Season 1 — before a show has the chance to build a loyal audience. This hurts lifetime value and makes content renewal decisions harder to justify. This project uses episode-level content-design and viewer-behavior data to answer three questions:

1. Why do viewers drop off, and where in the season does it happen?
2. Which content-design factors carry the most predictive weight?
3. Can episodes be segmented into distinct risk profiles, and what should the platform do about the riskiest one?

## Dataset

- 33,171 episode-level records across 489 shows, Season 1 only
- Content-design features (duration, pacing, hook strength, cognitive load, dialogue density, visual intensity, genre, platform) and viewer-behavior features (average watch %, pause count, rewind count)
- Data-quality issues identified and handled explicitly: 1,024 true duplicate-key rows (same show/episode/platform with contradictory values) removed, and one implausible outlier episode (1,225-minute runtime) excluded
- Overall drop-off rate: **14.5%**

## Methodology

**1. Leakage analysis.** The raw data includes three outcome-adjacent columns — `drop_off`, `drop_off_probability`, `retention_risk` — that are all derived from the same underlying score, plus a behavioral column (`avg_watch_percentage`) that correlates at -0.96 with the target. All four were excluded from the feature set; a side-by-side comparison confirms including `avg_watch_percentage` inflates ROC-AUC to a perfect 1.000 and dominates feature importance, versus 0.991 without it — evidence for the exclusion, not just a claim.

**2. Group-aware validation.** Train/test and cross-validation splits are grouped by `show_id`, since episodes from the same show share content style and would otherwise leak across folds.

**3. Modeling.** Logistic Regression baseline and XGBoost (class-weighted for the ~14.5% positive rate), validated with 5-fold grouped cross-validation.

**4. Explainability.** SHAP values rank feature importance on the trained model.

**5. Segmentation.** K-Means clustering (k=4, chosen for interpretability despite k=3 having a marginally higher silhouette score) groups episodes into distinct content profiles.

## Key Results

- **0.99 ROC-AUC** (0.990 ± 0.001 across 5-fold grouped cross-validation), using only content-design and real-time behavioral-friction features — no outcome-adjacent columns
- **Top 3 SHAP drivers:** hook-load balance, cognitive load, and pacing
- **Episode 1 is uniquely sticky** (3.7% drop-off vs. ~15–18% from episode 2 onward) — this coincides with a sharp drop in hook strength (7.6 in episode 1 vs. 5.5 in episodes 2–10), while cognitive load and pacing are essentially flat across the same episodes, ruling those two out as the explanation for this specific pattern
- **Four content segments identified**, with one at-risk profile (long duration, slow pacing, high cognitive load) showing a **43% drop-off rate** — roughly 3x the platform average

## Business Recommendations

1. Investigate what makes episode 1's hook work, and whether that strength can be sustained into episode 2, where risk steps up to its baseline level
2. Target the highest-risk content segment specifically for pacing/editing review, rather than a platform-wide change
3. Use the model's predicted probability as a continuous retention-risk score to prioritize pre-release editorial QA
4. Re-validate all findings against a subsequent season before wider rollout — these results are correlational and drawn from a single Season 1 window

## Repository Contents

| File | Description |
|---|---|
| `ott_dropoff_analysis.ipynb` | Full analysis: data cleaning, leakage handling, EDA, feature engineering, modeling, SHAP, clustering, recommendations |
| `OTT_Viewer_Retention_Deck.pptx` | Business-facing summary deck (problem, drivers, segments, recommendations, product workflow) |
| `ott_viewer_dropoff_retention_us_v1.0.csv` | Source dataset |

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
jupyter nbconvert --to notebook --execute --inplace ott_dropoff_analysis.ipynb
```

## A Note on the Data

This project originated as a case-study dataset (Consulting & Analytics Club, IIT Guwahati Winter Consulting capstone). The very high ROC-AUC reflects that the dataset's engagement score was likely constructed from these same content-design features — it demonstrates a correctly-built, leakage-aware modeling pipeline, not a claim that real-world viewer behavior is this predictable.
