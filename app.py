"""
Student Success Console — advisor-facing product on top of the dropout model.
Single-file Streamlit app. Runs on ews_artifacts.joblib (precomputed; no model
needed at runtime). Advisor pages use plain language only; all technical detail
(recall, precision, threshold, SHAP, fairness) is quarantined in Model Center.

Real data only. No invented operational fields (no contact status, owners-as-
records, trends, or intervention outcomes). Recommended actions are framed as
model recommendations, not records.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
NAVY, RED, AMBER, GREEN, BLUE, INK, MUTED = (
    "#1C2A5E", "#C0392B", "#C97A12", "#2E7D46", "#3E7CB1", "#1F2430", "#6B7280")
DROPOUT, ENROLLED, GRADUATE = 0, 1, 2
INST_THRESHOLD = 0.33

st.set_page_config(page_title="Student Success Console", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
:root { --navy:#1C2A5E; }
.block-container { padding-top: 1.6rem; max-width: 1300px; }
h1,h2,h3,h4 { color:#1C2A5E; font-family: Georgia, 'Times New Roman', serif; }
.kpi { background:#FFFFFF; border:1px solid #ECEFF4; border-radius:16px;
       padding:18px 20px; box-shadow:0 1px 3px rgba(16,24,40,.06); }
.kpi .num { font-size:34px; font-weight:700; line-height:1.1; }
.kpi .lab { color:#6B7280; font-size:13px; margin-top:4px; letter-spacing:.02em; }
.card { background:#FFFFFF; border:1px solid #ECEFF4; border-radius:16px;
        padding:22px 24px; box-shadow:0 1px 3px rgba(16,24,40,.06); margin-bottom:14px; }
.badge { display:inline-block; padding:3px 12px; border-radius:999px;
         font-size:12px; font-weight:700; }
.b-high { background:#FBE8E4; color:#C0392B; }
.b-med  { background:#FBF1DD; color:#9A6410; }
.b-low  { background:#E6F2EA; color:#2E7D46; }
.factor { padding:8px 0; border-bottom:1px solid #F2F4F8; font-size:15px; color:#1F2430; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:9px; }
.muted { color:#6B7280; font-size:13px; }
table.wl { width:100%; border-collapse:collapse; font-family:Arial, sans-serif; }
table.wl th { text-align:left; color:#6B7280; font-size:12px; font-weight:600;
              padding:8px 10px; border-bottom:2px solid #ECEFF4; letter-spacing:.03em; }
table.wl td { padding:11px 10px; border-bottom:1px solid #F2F4F8; font-size:14px; color:#1F2430; }
table.wl tr:hover td { background:#FAFBFE; }
.rec { background:#EEF2FB; border-radius:12px; padding:14px 16px; font-size:15px; color:#1C2A5E; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
@st.cache_data
def load_artifacts(path="ews_artifacts.joblib"):
    a = joblib.load(path)
    a["X_test_processed"] = (np.asarray(a["X_test_processed"].todense())
                             if hasattr(a["X_test_processed"], "todense")
                             else np.asarray(a["X_test_processed"]))
    a["y_test"] = np.asarray(a["y_test"])
    a["y_proba"] = np.asarray(a["y_proba"])
    if "class_names" not in a:
        a["class_names"] = list(a["le"].classes_) if "le" in a else ["Dropout", "Enrolled", "Graduate"]
    return a


try:
    A = load_artifacts()
except Exception as e:
    st.error("Could not load ews_artifacts.joblib.")
    st.exception(e)
    st.stop()

yt = A["y_test"]
proba = A["y_proba"]
p_dropout = proba[:, DROPOUT]
shap_dropout = np.asarray(A["shap_dropout"])
all_feature_names = list(A["all_feature_names"])
display_names = A["display_names"]
interpretable_features = set(A["interpretable_features"])
class_names = list(A["class_names"])
Xo = A["X_test_orig"].reset_index(drop=True)
student_ids = list(A["X_test"].index)
n = len(yt)
n_drop = int(np.sum(yt == DROPOUT))

# concept -> processed column indices (collapse one-hots into one human concept)
concept_cols = {}
for j, f in enumerate(all_feature_names):
    if f in interpretable_features:
        name = display_names.get(f, f)
        concept_cols.setdefault(name, []).append(j)

# plain-language translation for each concept
T = {
    "Tuition Fees Current":   ("Tuition balance not up to date", "Tuition up to date", "financial"),
    "Financial Risk Flag":    ("Carries financial risk (debt or unpaid tuition)", "No financial risk flags", "financial"),
    "Debtor":                 ("Has outstanding debt", "No outstanding debt", "financial"),
    "Scholarship holder":     ("No scholarship support", "Holds a scholarship", "financial"),
    "Semester 2 Pass Rate":   ("Failing most of this semester's courses", "Passing most current courses", "academic"),
    "Semester 1 Pass Rate":   ("Struggled in the first semester", "Strong first-semester performance", "academic"),
    "Grade Average Sem 1":    ("Low first-semester grades", "Strong first-semester grades", "academic"),
    "Grade Average Sem 2":    ("Low recent grades", "Strong recent grades", "academic"),
    "Total Units Approved":   ("Few credits completed so far", "On track with credits", "academic"),
    "Units Approved Sem 2":   ("Few credits passed recently", "Passing expected credits", "academic"),
    "Admission Grade":        ("Lower admission grade", "Strong admission grade", "academic"),
    "Prior Qualification Grade": ("Lower prior qualification", "Strong prior qualification", "academic"),
    "Age at Enrollment":      ("Older than the typical cohort", "Typical enrollment age", "life"),
    "Displaced Student":      ("Displaced student", "Not displaced", "life"),
    "Parents Education (avg)":("First-generation background", "Family college background", "life"),
    "Parents Occupation (avg)":("Family employment pressures", "", "life"),
    "Unemployment Rate":      ("High regional unemployment", "", "life"),
    "GDP":                    ("Weaker economic conditions", "", "life"),
}
ACTION = {
    "financial": ("Refer to Financial Aid Office", "Same day"),
    "academic":  ("Connect with Academic Advisor", "This week"),
    "life":      ("Refer to Student Support Services", "This week"),
}


def tier(p):
    return ("High", "b-high", RED) if p >= 0.66 else (("Medium", "b-med", AMBER) if p >= INST_THRESHOLD else ("Low", "b-low", GREEN))


def drivers(pos):
    """Return (risk_factors, protective_factors) as (label, category) lists,
    aggregated to human concepts and ordered by SHAP magnitude."""
    scored = []
    for name, cols in concept_cols.items():
        scored.append((name, float(np.sum(shap_dropout[pos, cols]))))
    pos_d = sorted([s for s in scored if s[1] > 0], key=lambda x: -x[1])
    neg_d = sorted([s for s in scored if s[1] < 0], key=lambda x: x[1])
    risk, prot = [], []
    for name, _ in pos_d:
        phrase, _, cat = T.get(name, (name, "", "academic"))
        if phrase:
            risk.append((phrase, cat))
    for name, _ in neg_d:
        _, good, _ = T.get(name, ("", name, "academic"))
        if good:
            prot.append(good)
    return risk, prot


def recommendation(pos):
    risk, _ = drivers(pos)
    cat = risk[0][1] if risk else "academic"
    return ACTION[cat]


def fmt_age(v):
    try:
        v = float(v)
        return f"{int(round(v))}" if 14 < v < 90 else "—"
    except Exception:
        return "—"


# ---- sidebar nav ----
st.sidebar.markdown(f"<h2 style='margin-bottom:0'>Student Success</h2>"
                    f"<div class='muted'>Console · powered by predictive analytics</div><br>",
                    unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Overview", "Students", "Student profile", "Model Center"],
                        label_visibility="collapsed")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.caption("Model results run on held-out test data (UCI dataset). "
                   "Advisor views translate the model into plain-language actions.")

if "thr" not in st.session_state:
    st.session_state.thr = INST_THRESHOLD
if "student_pos" not in st.session_state:
    st.session_state.student_pos = int(np.argmax(p_dropout))

order = np.argsort(-p_dropout)


# ============================================================ OVERVIEW
def page_overview():
    st.markdown("<h1>Overview</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>A snapshot of where the current student body stands.</div><br>",
                unsafe_allow_html=True)
    tiers = np.array([tier(p)[0] for p in p_dropout])
    high = int(np.sum(tiers == "High")); med = int(np.sum(tiers == "Medium")); low = int(np.sum(tiers == "Low"))
    cols = st.columns(4)
    data = [("Students monitored", n, NAVY), ("High risk", high, RED),
            ("Medium risk", med, AMBER), ("Lower risk", low, GREEN)]
    for c, (lab, val, color) in zip(cols, data):
        c.markdown(f"<div class='kpi'><div class='num' style='color:{color}'>{val}</div>"
                   f"<div class='lab'>{lab}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("<div class='card'><h3 style='margin-top:0'>Risk distribution</h3>", unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=p_dropout, nbinsx=30, marker_color=NAVY))
        fig.add_vline(x=INST_THRESHOLD, line_dash="dash", line_color=RED,
                      annotation_text="Intervention line", annotation_font_color=RED)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Model-estimated dropout likelihood", yaxis_title="Students",
                          font=dict(color=INK))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='card'><h3 style='margin-top:0'>What this means</h3>"
                    f"<p style='font-size:15px;color:{INK}'>At the institution's current intervention "
                    f"line, <b>{high} students</b> are high risk and warrant outreach now. "
                    f"The model correctly identifies roughly <b>3 in 4</b> of the students who go on to "
                    f"leave, early enough to act.</p>"
                    f"<p class='muted'>Move to the Students tab to see exactly who needs attention "
                    f"and why.</p></div>", unsafe_allow_html=True)


# ============================================================ STUDENTS
def page_students():
    st.markdown("<h1>Students requiring attention</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Prioritized by risk. Each student shows the main concern and a "
                "recommended next step.</div><br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1, 1, 1.4])
    risk_filter = f1.selectbox("Risk level", ["All", "High", "Medium", "Lower"])
    att_filter = f2.selectbox("Attendance", ["All", "Daytime", "Evening"])
    search = f3.text_input("Search by Student ID", "")

    rows_idx = []
    for pos in order:
        tlab = tier(p_dropout[pos])[0].replace("Low", "Lower")
        if risk_filter != "All" and tlab != risk_filter:
            continue
        if att_filter != "All" and "Daytime/evening attendance" in Xo.columns:
            want = 1 if att_filter == "Daytime" else 0
            if int(Xo.iloc[pos]["Daytime/evening attendance"]) != want:
                continue
        if search and search.strip() not in str(student_ids[pos]):
            continue
        rows_idx.append(pos)
    rows_idx = rows_idx[:40]

    html = ["<table class='wl'><tr><th>Priority</th><th>Student</th><th>Age</th>"
            "<th>Risk</th><th>Primary concern</th><th>Recommended next step</th><th>Owner</th></tr>"]
    for rank, pos in enumerate(rows_idx, 1):
        tlab, tcls, _ = tier(p_dropout[pos])
        risk, _ = drivers(pos)
        concern = risk[0][0] if risk else "Review profile"
        act, when = recommendation(pos)
        owner = {"Refer to Financial Aid Office": "Financial Aid",
                 "Connect with Academic Advisor": "Advising",
                 "Refer to Student Support Services": "Student Support"}.get(act, "Advising")
        age = fmt_age(Xo.iloc[pos]["Age at enrollment"]) if "Age at enrollment" in Xo.columns else "—"
        html.append(
            f"<tr><td>{rank}</td><td><b>#{student_ids[pos]}</b></td><td>{age}</td>"
            f"<td><span class='badge {tcls}'>{tlab.replace('Low','Lower')}</span></td>"
            f"<td>{concern}</td><td>{act} <span class='muted'>· {when}</span></td><td>{owner}</td></tr>")
    html.append("</table>")
    st.markdown("<div class='card'>" + "".join(html) + "</div>", unsafe_allow_html=True)

    st.markdown("#### Open a student profile")
    opts = {f"#{student_ids[pos]} · {tier(p_dropout[pos])[0].replace('Low','Lower')} risk": int(pos)
            for pos in rows_idx}
    if opts:
        pick = st.selectbox("Select", list(opts.keys()), label_visibility="collapsed")
        if st.button("Open profile →", type="primary"):
            st.session_state.student_pos = opts[pick]
            st.session_state._goto = "Student profile"
            st.rerun()


# ============================================================ STUDENT PROFILE
def page_profile():
    pos = st.session_state.student_pos
    sid = student_ids[pos]
    tlab, tcls, tcol = tier(p_dropout[pos])
    st.markdown(f"<h1>Student #{sid}</h1>", unsafe_allow_html=True)
    p_pick = st.selectbox("Viewing", [f"#{student_ids[p]}" for p in order[:50]],
                          index=0 if pos not in order[:50] else list(order[:50]).index(pos),
                          label_visibility="collapsed")
    pos = student_ids.index(int(p_pick.lstrip("#")))
    st.session_state.student_pos = pos
    tlab, tcls, tcol = tier(p_dropout[pos])

    c1, c2 = st.columns([1, 1.25])
    with c1:
        age = fmt_age(Xo.iloc[pos]["Age at enrollment"]) if "Age at enrollment" in Xo.columns else "—"
        att = "—"
        if "Daytime/evening attendance" in Xo.columns:
            att = "Daytime" if int(Xo.iloc[pos]["Daytime/evening attendance"]) == 1 else "Evening"
        sch = "—"
        if "Scholarship holder" in Xo.columns:
            sch = "Yes" if int(Xo.iloc[pos]["Scholarship holder"]) == 1 else "No"
        tui = "—"
        if "Tuition fees up to date" in Xo.columns:
            tui = "Up to date" if int(Xo.iloc[pos]["Tuition fees up to date"]) == 1 else "Overdue"
        st.markdown(
            f"<div class='card'><span class='badge {tcls}'>{tlab.replace('Low','Lower')} risk</span>"
            f"<div style='font-size:15px;color:{INK};margin-top:14px;line-height:2'>"
            f"Age at enrollment: <b>{age}</b><br>"
            f"Attendance: <b>{att}</b><br>"
            f"Scholarship: <b>{sch}</b><br>"
            f"Tuition: <b>{tui}</b></div>"
            f"<div class='muted' style='margin-top:14px'>Model-estimated likelihood of leaving: "
            f"<b style='color:{tcol}'>{p_dropout[pos]:.0%}</b></div></div>", unsafe_allow_html=True)
        act, when = recommendation(pos)
        st.markdown(f"<div class='rec'><b>Recommended next step</b><br>{act}<br>"
                    f"<span class='muted'>Timing: {when}</span></div>", unsafe_allow_html=True)
    with c2:
        risk, prot = drivers(pos)
        items = "".join(
            f"<div class='factor'><span class='dot' style='background:{RED}'></span>{r}</div>"
            for r, _ in risk[:4])
        st.markdown(f"<div class='card'><h3 style='margin-top:0'>Why this student is flagged</h3>"
                    f"{items}<div class='muted' style='margin-top:10px'>"
                    f"Generated using SHAP explainability.</div></div>", unsafe_allow_html=True)
        if prot:
            good = "".join(
                f"<div class='factor'><span class='dot' style='background:{GREEN}'></span>{g}</div>"
                for g in prot[:3])
            st.markdown(f"<div class='card'><h3 style='margin-top:0'>Working in their favor</h3>"
                        f"{good}</div>", unsafe_allow_html=True)


# ============================================================ MODEL CENTER
def page_model():
    st.markdown("<h1>Model Center</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>The technical evidence behind the console: performance, "
                "the decision threshold, fairness, and explainability.</div><br>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Performance", "Threshold explorer", "Fairness", "Explainability"])

    def preds_at(thr):
        return np.where(p_dropout >= thr, DROPOUT, np.argmax(proba[:, 1:], axis=1) + 1)

    with t1:
        pred = preds_at(INST_THRESHOLD)
        tp = np.sum((yt == DROPOUT) & (pred == DROPOUT)); fn = np.sum((yt == DROPOUT) & (pred != DROPOUT))
        fp = np.sum((yt != DROPOUT) & (pred == DROPOUT))
        rec = tp / (tp + fn); prec = tp / (tp + fp) if (tp + fp) else 0; acc = np.mean(pred == yt)
        try:
            from sklearn.metrics import f1_score, confusion_matrix
            mf1 = f1_score(yt, pred, average="macro"); cm = confusion_matrix(yt, pred)
        except Exception:
            mf1, cm = float("nan"), None
        cols = st.columns(4)
        for c, (lab, val) in zip(cols, [("Dropout recall", f"{rec:.1%}"), ("Dropout precision", f"{prec:.1%}"),
                                        ("Overall accuracy", f"{acc:.1%}"), ("Macro F1", f"{mf1:.2f}")]):
            c.markdown(f"<div class='kpi'><div class='num' style='font-size:28px'>{val}</div>"
                       f"<div class='lab'>{lab}</div></div>", unsafe_allow_html=True)
        st.caption(f"At the institution threshold of {INST_THRESHOLD:.2f}, on {n} held-out students.")
        if cm is not None:
            fig = go.Figure(go.Heatmap(z=cm, x=class_names, y=class_names, colorscale="Blues",
                                       text=cm, texttemplate="%{text}", showscale=False))
            fig.update_layout(height=340, title="Confusion matrix", font=dict(color=INK),
                              xaxis_title="Predicted", yaxis_title="Actual", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        thr = st.slider("Decision threshold", 0.10, 0.79, st.session_state.thr, 0.01)
        st.session_state.thr = thr
        ths = np.linspace(0.10, 0.79, 40); recs, precs = [], []
        for tt in ths:
            pr = preds_at(tt)
            tp = np.sum((yt == DROPOUT) & (pr == DROPOUT)); fn = np.sum((yt == DROPOUT) & (pr != DROPOUT))
            fp = np.sum((yt != DROPOUT) & (pr == DROPOUT))
            recs.append(tp / (tp + fn)); precs.append(tp / (tp + fp) if (tp + fp) else 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ths, y=recs, name="Recall", line=dict(color=RED, width=3)))
        fig.add_trace(go.Scatter(x=ths, y=precs, name="Precision", line=dict(color=BLUE, width=3)))
        fig.add_vline(x=thr, line_dash="dash", line_color="black")
        fig.update_layout(height=360, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
                          xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        pr = preds_at(thr)
        st.caption(f"At threshold {thr:.2f}: {int(np.sum(pr == DROPOUT))} students flagged.")

    with t3:
        SUB = {"Financial risk": ("financial_risk", {0: "No financial risk", 1: "Has financial risk"}),
               "Gender": ("Gender", {0: "Female", 1: "Male"}),
               "Scholarship": ("Scholarship holder", {0: "No scholarship", 1: "Has scholarship"}),
               "Attendance": ("Daytime/evening attendance", {0: "Evening", 1: "Daytime"})}
        attr = st.selectbox("Audit recall by", list(SUB.keys()))
        col, mp = SUB[attr]; pred = preds_at(st.session_state.thr)
        labels, vals, ns = [], [], []
        codes = Xo[col].values
        for code, lab in mp.items():
            den = np.sum((codes == code) & (yt == DROPOUT))
            num = np.sum((codes == code) & (yt == DROPOUT) & (pred == DROPOUT))
            labels.append(lab); vals.append(100 * num / den if den else np.nan); ns.append(int(den))
        fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[RED, GREEN][:len(labels)],
                               text=[f"{v:.1f}%<br>(n={x})" for v, x in zip(vals, ns)],
                               textposition="outside", cliponaxis=False))
        gap = (max(vals) - min(vals)) if len(vals) >= 2 and not any(np.isnan(vals)) else float("nan")
        fig.update_layout(height=360, yaxis_range=[0, 110], paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(color=INK), yaxis_title="Dropout recall (%)",
                          title=f"Recall gap at threshold {st.session_state.thr:.2f}: {gap:.1f} points")
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        imp = sorted(((name, float(np.mean(np.abs(np.sum(shap_dropout[:, cols], axis=1)))))
                      for name, cols in concept_cols.items()), key=lambda x: x[1])[-12:]
        fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[k for k, _ in imp], orientation="h",
                               marker_color=NAVY))
        fig.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
                          title="Global feature importance (mean |SHAP|)", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Drives the plain-language risk factors shown to advisors.")


# nav handling (button-driven jump)
if st.session_state.get("_goto"):
    page = st.session_state.pop("_goto")

if page == "Overview":
    page_overview()
elif page == "Students":
    page_students()
elif page == "Student profile":
    page_profile()
else:
    page_model()
