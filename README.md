# SHAP Analysis in Motor Insurance Pricing

Interpretable machine learning framework for claim severity prediction using LightGBM with SHAP (SHapley Additive exPlanations) explainability, combined with traditional actuarial diagnostics (calibration, Lorenz curve, Gini coefficient).

---

## Overview

This project demonstrates **interpretable machine learning in insurance** by combining:

1. **LightGBM severity model** — high-performance gradient boosting for claim amount prediction
2. **SHAP values** — game-theoretic explanations of individual predictions
3. **Actuarial diagnostics** — Lorenz curve and Gini coefficient for discriminatory power

The workflow bridges the gap between black-box ML and regulatory requirements (model interpretability, fairness, explainability).

---
## Data

**Source:** [freMTPL dataset](https://www.kaggle.com/datasets/karansarpal/fremtpl-french-motor-tpl-insurance-claims) (French Motor Third-Party Liability)

---

## Key Insights

### 1. Feature Importance (SHAP)

**Most impactful:**
- **Density** — unpredictable; requires further analysis
- **DriverAge** — interpretable; young drivers consistently high-risk
- **CarAge** — vehicle age affects repair costs monotonically

**Less impactful:**
- Regional and brand categorical features (small SHAP values)

### 2. Model Behavior

**Strengths:**
- Clear monotonic relationships (age effects)
- Reasonable predictions for central cases
- Stable under subsampling

**Weaknesses:**
- Gini=0.189 indicates limited discriminatory power
- High variance in severity (inherent to the target)
- Potential non-linear interactions

---
