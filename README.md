# Student Early Warning System — Streamlit dashboard

Interactive advisor-facing demo for the dropout-prediction model. Runs on the
precomputed artifacts from the HelioCampus notebook, so the trained model is
not needed at runtime (no xgboost or shap required).

## Files in this repo
- `app.py` — the dashboard
- `ews_artifacts.joblib` — exported arrays from the notebook
- `requirements.txt` — pinned libraries

## What it shows
- Metric cards (flagged count, dropout recall, dropout precision, accuracy) that update with the threshold
- Ranked risk list, filterable by subgroup
- Fairness audit (recall gap by subgroup) that recomputes as the threshold moves
- Per-student SHAP explanation with the recommended advisor action

## Export the artifacts (run once in the notebook)
```python
import joblib
joblib.dump({
    "X_test": X_test, "X_test_processed": X_test_processed, "y_test": y_test,
    "y_proba": y_proba, "shap_dropout": shap_dropout,
    "all_feature_names": all_feature_names, "display_names": display_names,
    "interpretable_features": interpretable_features,
    "class_names": list(le.classes_), "X_test_orig": X_test_orig,
}, "ews_artifacts.joblib")
```
Note `class_names` is stored as a plain list, so the app needs no scikit-learn.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repo and upload `app.py`, `ews_artifacts.joblib`, `requirements.txt`
   (Add file → Upload files → Commit).
2. Go to share.streamlit.io, sign in with GitHub.
3. Create app → pick the repo, branch `main`, main file `app.py` → Deploy.
4. Wait ~1–2 minutes for the build, then open and test the public URL.
5. Community Cloud sleeps idle apps; open the URL a few minutes before presenting.

Data note: uses the public UCI Portuguese-student dataset (held-out test split),
not real institutional records.
