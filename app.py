"""
Student Early Warning System — Streamlit dashboard
===================================================
Runs on the precomputed artifacts exported from the HelioCampus notebook
(ews_artifacts.joblib). The trained model is NOT needed at runtime, so this
app does not require xgboost or shap installed.

Local test:   streamlit run app.py
Deploy:        push app.py + ews_artifacts.joblib + requirements.txt to a
               GitHub repo, then deploy on share.streamlit.io.

Expected keys in ews_artifacts.joblib:
  X_test, X_test_processed, y_test, y_proba, shap_dropout,
  all_feature_names, display_names, interpretable_features,
  class_names (list, e.g. ['Dropout','Enrolled','Graduate']), X_test_orig

Class order: 0 = Dropout, 1 = Enrolled, 2 = Graduate
Runs on the 885 held-out TEST students (not a live SIS feed).
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# palette
NAVY, RED, BLUE, GREEN, AMBER, MUTED = "#1C2A5E", "#B5402F", "#3E7CB1", "#2E7D46", "#C97A12", "#5A6072"
DROPOUT, ENROLLED, GRADUATE = 0, 1, 2

st.set_page_config(page_title="Student Early Warning System", layout="wide")


# ----------------------------------------------------------------------------
@st.cache_data
def load_artifacts(path="ews_artifacts.joblib"):
    a = joblib.load(path)
    a["X_test_processed"] = (
        np.asarray(a["X_test_processed"].todense())
        if hasattr(a["X_test_processed"], "todense")
        else np.asarray(a["X_test_processed"])
    )
    a["y_test"] = np.asarray(a["y_test"])
    a["y_proba"] = np.asarray(a["y_proba"])
    return a


A = load_artifacts()
yt = A["y_test"]
proba = A["y_proba"]
p_dropout = proba[:, DROPOUT]
shap_dropout = A["shap_dropout"]
Xp = A["X_test_processed"]
all_feature_names = list(A["all_feature_names"])
display_names = A["display_names"]
interpretable_features = list(A["interpretable_features"])
class_names = list(A["class_names"])
X_test_orig = A["X_test_orig"].reset_index(drop=False).rename(columns={"index": "student_id"})
student_ids = list(A["X_test"].index)
n_students = len(yt)
n_actual_dropout = int(np.sum(yt == DROPOUT))

SUBGROUPS = {
    "Financial risk": ("financial_risk", {0: "No financial risk", 1: "Has financial risk"}),
    "Gender": ("Gender", {0: "Female", 1: "Male"}),
    "Scholarship": ("Scholarship holder", {0: "No scholarship", 1: "Has scholarship"}),
    "Attendance": ("Daytime/evening attendance", {0: "Evening", 1: "Daytime"}),
}
FILTER_OPTIONS = ["All students"] + [lbl for _, (_, m) in SUBGROUPS.items() for lbl in m.values()]
_label_to_filter = {lbl: (col, code) for _, (col, m) in SUBGROUPS.items() for code, lbl in m.items()}


# ----------------------------------------------------------------------------
def preds_at(thr):
    is_dropout = p_dropout >= thr
    other = np.argmax(proba[:, 1:], axis=1) + 1
    return np.where(is_dropout, DROPOUT, other)


def dropout_scores(pred):
    tp = np.sum((yt == DROPOUT) & (pred == DROPOUT))
    fn = np.sum((yt == DROPOUT) & (pred != DROPOUT))
    fp = np.sum((yt != DROPOUT) & (pred == DROPOUT))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = np.mean(pred == yt)
    return recall, precision, accuracy, int(np.sum(pred == DROPOUT)), int(tp), int(fp)


def subgroup_recall(pred, col, mapping):
    out = {}
    codes = X_test_orig[col].values
    for code, label in mapping.items():
        denom = np.sum((codes == code) & (yt == DROPOUT))
        num = np.sum((codes == code) & (yt == DROPOUT) & (pred == DROPOUT))
        out[label] = (num / denom if denom else np.nan, int(denom))
    return out


# ----------------------------------------------------------------------------
# header + global control
st.markdown(
    f"<h1 style='color:{NAVY};margin-bottom:0'>Student Early Warning System</h1>"
    f"<p style='color:{MUTED};font-size:16px;margin-top:4px'>"
    f"Interactive demo on {n_students} held-out students. Move the decision threshold to watch the "
    f"flagged population, the model's accuracy, and the fairness gap change together.</p>",
    unsafe_allow_html=True,
)

thr = st.slider("Dropout decision threshold", 0.10, 0.79, 0.33, 0.01,
                help="Flag a student as Dropout when their predicted P(Dropout) is at or above this value.")
pred = preds_at(thr)
recall, precision, accuracy, flagged, tp, fp = dropout_scores(pred)

# ---- Panel 5: metric cards (top) ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students flagged", f"{flagged}", help=f"of {n_students} students")
c2.metric("Dropout recall", f"{recall:.1%}", help=f"caught {tp} of {n_actual_dropout} real dropouts")
c3.metric("Dropout precision", f"{precision:.1%}", help=f"{tp} real of {flagged} flagged")
c4.metric("Overall accuracy", f"{accuracy:.1%}")

st.divider()
left, right = st.columns([1.15, 1])

# ---- Panel 1: ranked risk table (+ filter) ----
with left:
    st.markdown(f"<h3 style='color:{NAVY}'>Ranked risk list</h3>", unsafe_allow_html=True)
    filt = st.selectbox("Show", FILTER_OPTIONS, index=0)
    order = np.argsort(-p_dropout)
    if filt != "All students":
        col, code = _label_to_filter[filt]
        order = order[X_test_orig[col].values[order] == code]
    top_n = st.slider("How many to show", 10, 50, 20, 5)
    order = order[:top_n]
    table = pd.DataFrame({
        "Rank": np.arange(1, len(order) + 1),
        "Student ID": [student_ids[p] for p in order],
        "P(Dropout)": [round(float(p_dropout[p]), 3) for p in order],
        "Flagged": ["YES" if pred[p] == DROPOUT else "—" for p in order],
        "Actual outcome": [class_names[yt[p]] for p in order],
    })

    def _highlight(row):
        return [f"background-color:#FBE8E4" if row["Flagged"] == "YES" else "" for _ in row]

    st.dataframe(table.style.apply(_highlight, axis=1), hide_index=True,
                 use_container_width=True, height=460)

# ---- Panel 4: fairness, moves with threshold ----
with right:
    st.markdown(f"<h3 style='color:{NAVY}'>Fairness audit</h3>", unsafe_allow_html=True)
    attr = st.selectbox("Audit recall by", list(SUBGROUPS.keys()), index=0)
    col, mapping = SUBGROUPS[attr]
    res = subgroup_recall(pred, col, mapping)
    labels = list(res.keys())
    vals = [res[l][0] * 100 for l in labels]
    ns = [res[l][1] for l in labels]
    bar_colors = [RED, GREEN] if len(labels) == 2 else [BLUE] * len(labels)
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=bar_colors[:len(labels)],
        text=[f"{v:.1f}%<br>(n={n})" for v, n in zip(vals, ns)],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        yaxis=dict(title="Dropout recall (%)", range=[0, 110]),
        height=360, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        font=dict(color=NAVY),
    )
    if len(vals) >= 2 and not any(np.isnan(vals)):
        gap = max(vals) - min(vals)
        fig.update_layout(title=dict(text=f"Recall gap at threshold {thr:.2f}: {gap:.1f} points", font=dict(size=15)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower the threshold and watch the gap between groups narrow.")

st.divider()

# ---- Panel 3: per-student SHAP explanation ----
st.markdown(f"<h3 style='color:{NAVY}'>Why was this student flagged?</h3>", unsafe_allow_html=True)
order_all = np.argsort(-p_dropout)
options = {f"Rank {i+1}  ·  ID {student_ids[p]}  ·  P(Dropout)={p_dropout[p]:.3f}": int(p)
           for i, p in enumerate(order_all[:50])}
choice = st.selectbox("Select a student (top 50 by risk)", list(options.keys()))
pos = options[choice]

df = pd.DataFrame({"f": all_feature_names, "shap": shap_dropout[pos], "val": Xp[pos]})
df = df[df["f"].isin(interpretable_features)].copy()
df["name"] = df["f"].map(display_names).fillna(df["f"])
df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(8).iloc[::-1]

scol, icol = st.columns([1.4, 1])
with scol:
    fig2 = go.Figure(go.Bar(
        x=df["shap"], y=df["name"], orientation="h",
        marker_color=[RED if v > 0 else BLUE for v in df["shap"]],
        text=[f"{v:+.2f}" for v in df["shap"]], textposition="outside", cliponaxis=False,
    ))
    fig2.update_layout(
        height=380, margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY),
        xaxis=dict(title="SHAP value  (red = toward Dropout, blue = away)", zeroline=True,
                   zerolinecolor="black", zerolinewidth=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

with icol:
    pd_, pe_, pg_ = proba[pos, DROPOUT], proba[pos, ENROLLED], proba[pos, GRADUATE]
    actual, predicted = class_names[yt[pos]], class_names[pred[pos]]
    st.markdown(
        f"<div style='font-family:Arial;color:{NAVY};font-size:15px;line-height:1.7'>"
        f"<b>Student {student_ids[pos]}</b><br>"
        f"P(Dropout) <b style='color:{RED}'>{pd_:.3f}</b><br>"
        f"P(Enrolled) {pe_:.3f}<br>"
        f"P(Graduate) <b style='color:{GREEN}'>{pg_:.3f}</b><br>"
        f"Actual outcome: <b>{actual}</b><br>"
        f"Predicted at {thr:.2f}: <b>{predicted}</b></div>",
        unsafe_allow_html=True,
    )
    action = {
        "Tuition Fees Current": "Financial Aid Office — same day",
        "Financial Risk Flag": "Coordinated case management — within 2 weeks",
        "Semester 2 Pass Rate": "Academic Advisor — this week",
        "Semester 1 Pass Rate": "Academic Advisor — this week",
        "Age at Enrollment": "Student Support Services — this week",
    }
    top_pos = df[df["shap"] > 0].sort_values("shap", ascending=False)
    if len(top_pos):
        drv = top_pos.iloc[0]["name"]
        st.info(f"**Top driver:** {drv}\n\n**Recommended:** {action.get(drv, 'Academic Advisor — this week')}")

st.caption("Runs on held-out test data from the UCI Portuguese-student dataset, not a live student information system.")
