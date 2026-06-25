"""
Student Success Console — advisor-facing product on the dropout model.
Single-file Streamlit app on ews_artifacts.joblib (precomputed; no model at runtime).
Advisor pages: plain language only. Technical detail quarantined in Model Center.
Real data only — no fabricated operational fields.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# palette ---------------------------------------------------------------------
NAVY, RED, AMBER, GREEN, BLUE, INK, MUTED = (
    "#1B2A5E", "#C0392B", "#B7791F", "#2E7D46", "#3E7CB1", "#1F2430", "#6B7280")
PAGE_BG = "#F4F6FB"
DROPOUT, ENROLLED, GRADUATE = 0, 1, 2
INST_THRESHOLD = 0.33

st.set_page_config(page_title="Student Success Console", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; }
.stApp { background:#F4F6FB; }
.block-container { padding-top:1.4rem; max-width:1320px; }
h1,h2,h3,h4 { font-family:'Inter',sans-serif; color:#1B2A5E; font-weight:700; letter-spacing:-.01em; }
section[data-testid="stSidebar"] { background:#10193A; }
section[data-testid="stSidebar"] * { color:#C7D0EA !important; }
section[data-testid="stSidebar"] h2 { color:#FFFFFF !important; }

.kpi { position:relative; background:#FFFFFF; border-radius:18px; padding:20px 22px 18px 24px;
       box-shadow:0 4px 16px rgba(20,30,70,.07); overflow:hidden;
       transition:transform .15s ease, box-shadow .15s ease; }
.kpi:hover { transform:translateY(-3px); box-shadow:0 10px 26px rgba(20,30,70,.12); }
.kpi:before { content:""; position:absolute; left:0; top:0; bottom:0; width:5px; }
.kpi .num { font-size:38px; font-weight:800; line-height:1; letter-spacing:-.02em; }
.kpi .lab { color:#6B7280; font-size:11px; margin-top:8px; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }

.card { background:#FFFFFF; border-radius:18px; padding:24px 26px;
        box-shadow:0 4px 16px rgba(20,30,70,.07); margin-bottom:16px; }
.scard { background:#FFFFFF; border-radius:16px; padding:18px 20px; height:100%;
         box-shadow:0 3px 12px rgba(20,30,70,.06); border-left:5px solid #ccc;
         transition:transform .15s ease, box-shadow .15s ease; }
.scard:hover { transform:translateY(-3px); box-shadow:0 10px 24px rgba(20,30,70,.12); }
.scard .sid { font-weight:700; font-size:16px; color:#1B2A5E; }
.scard .meta { color:#6B7280; font-size:12px; margin-bottom:10px; }
.scard .concern { font-size:14px; color:#1F2430; margin:8px 0; }
.scard .rec { font-size:13px; color:#3E7CB1; font-weight:600; }

.badge { display:inline-block; padding:4px 13px; border-radius:999px; font-size:11px;
         font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.b-high { background:#FBE3DE; color:#C0392B; } .b-med{ background:#FaeEDc; color:#9A6410; } .b-low{ background:#E2F1E8; color:#2E7D46; }
.factor { padding:10px 0; border-bottom:1px solid #F1F3F8; font-size:15px; color:#1F2430; display:flex; align-items:center; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:11px; flex:none; }
.muted { color:#6B7280; font-size:13px; }
.rec-box { background:linear-gradient(135deg,#EEF2FC,#E7ECFA); border-radius:14px; padding:16px 18px; color:#1B2A5E; }
.conf { font-size:13px; color:#6B7280; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_artifacts(path="ews_artifacts.joblib"):
    a = joblib.load(path)
    a["X_test_processed"] = (np.asarray(a["X_test_processed"].todense())
                             if hasattr(a["X_test_processed"], "todense")
                             else np.asarray(a["X_test_processed"]))
    a["y_test"] = np.asarray(a["y_test"]); a["y_proba"] = np.asarray(a["y_proba"])
    if "class_names" not in a:
        a["class_names"] = list(a["le"].classes_) if "le" in a else ["Dropout", "Enrolled", "Graduate"]
    return a


try:
    A = load_artifacts()
except Exception as e:
    st.error("Could not load ews_artifacts.joblib."); st.exception(e); st.stop()

yt = A["y_test"]; proba = A["y_proba"]; p_dropout = proba[:, DROPOUT]
shap_dropout = np.asarray(A["shap_dropout"])
all_feature_names = list(A["all_feature_names"]); display_names = A["display_names"]
interpretable_features = set(A["interpretable_features"]); class_names = list(A["class_names"])
Xo = A["X_test_orig"].reset_index(drop=True); student_ids = list(A["X_test"].index)
n = len(yt); n_drop = int(np.sum(yt == DROPOUT))

concept_cols = {}
for j, f in enumerate(all_feature_names):
    if f in interpretable_features:
        concept_cols.setdefault(display_names.get(f, f), []).append(j)

T = {
    "Tuition Fees Current": ("Tuition balance not up to date", "Tuition up to date", "financial"),
    "Financial Risk Flag": ("Carries financial risk (debt or unpaid tuition)", "No financial risk flags", "financial"),
    "Debtor": ("Has outstanding debt", "No outstanding debt", "financial"),
    "Scholarship holder": ("No scholarship support", "Holds a scholarship", "financial"),
    "Semester 2 Pass Rate": ("Failing most of this semester's courses", "Passing most current courses", "academic"),
    "Semester 1 Pass Rate": ("Struggled in the first semester", "Strong first-semester performance", "academic"),
    "Grade Average Sem 1": ("Low first-semester grades", "Strong first-semester grades", "academic"),
    "Grade Average Sem 2": ("Low recent grades", "Strong recent grades", "academic"),
    "Total Units Approved": ("Few credits completed so far", "On track with credits", "academic"),
    "Units Approved Sem 2": ("Few credits passed recently", "Passing expected credits", "academic"),
    "Admission Grade": ("Lower admission grade", "Strong admission grade", "academic"),
    "Prior Qualification Grade": ("Lower prior qualification", "Strong prior qualification", "academic"),
    "Age at Enrollment": ("Older than the typical cohort", "Typical enrollment age", "life"),
    "Displaced Student": ("Displaced student", "Not displaced", "life"),
    "Parents Education (avg)": ("First-generation background", "Family college background", "life"),
    "Parents Occupation (avg)": ("Family employment pressures", "", "life"),
    "Unemployment Rate": ("High regional unemployment", "", "life"),
    "GDP": ("Weaker economic conditions", "", "life"),
}
ACTION = {"financial": ("Refer to Financial Aid Office", "Same day", "Financial Aid"),
          "academic": ("Connect with Academic Advisor", "This week", "Advising"),
          "life": ("Refer to Student Support Services", "This week", "Student Support")}


def tier(p):
    if p >= 0.66: return ("High", "b-high", RED, "Needs immediate outreach")
    if p >= INST_THRESHOLD: return ("Medium", "b-med", AMBER, "Monitor and check in")
    return ("Lower", "b-low", GREEN, "On track")


def drivers(pos):
    scored = [(name, float(np.sum(shap_dropout[pos, cols]))) for name, cols in concept_cols.items()]
    risk = [(T[nm][0], T[nm][2])
            for nm, v in sorted([s for s in scored if s[1] > 0], key=lambda x: -x[1])
            if nm in T and T[nm][0]]
    # protective: only concepts pushing AWAY from dropout AND with a real positive plain meaning
    prot = [T[nm][1]
            for nm, v in sorted([s for s in scored if s[1] < 0], key=lambda x: x[1])
            if nm in T and T[nm][1]]
    return risk, prot


def recommendation(pos):
    risk, _ = drivers(pos)
    return ACTION[risk[0][1] if risk else "academic"]


def style_fig(fig, h=320):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=INK, size=13),
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(gridcolor="#EEF1F7", zeroline=False),
                      yaxis=dict(gridcolor="#EEF1F7", zeroline=False),
                      hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"))
    return fig


def age_of(pos):
    if "Age at enrollment" in Xo.columns:
        try:
            v = float(Xo.iloc[pos]["Age at enrollment"])
            if 14 < v < 90: return f"Age {int(round(v))}"
        except Exception: pass
    return ""


# sidebar ---------------------------------------------------------------------
st.sidebar.markdown("<h2 style='margin-bottom:2px'>◆ Student Success</h2>"
                    "<div style='font-size:12px;opacity:.7'>Predictive retention console</div><br>",
                    unsafe_allow_html=True)
PAGES = {"Dashboard": "Dashboard", "Students": "Students", "Student profile": "Student profile",
         "Model Center": "Model Center"}
page = st.sidebar.radio("nav", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.caption("Model results run on held-out test data (UCI dataset). Advisor views "
                   "translate the model into plain-language actions.")

if "student_pos" not in st.session_state:
    st.session_state.student_pos = int(np.argmax(p_dropout))
if "thr" not in st.session_state:
    st.session_state.thr = INST_THRESHOLD
order = np.argsort(-p_dropout)


def kpi(col, value, label, color):
    col.markdown(f"<div class='kpi' style='border-top:0'><div class='num' style='color:{color}'>{value}</div>"
                 f"<div class='lab'>{label}</div></div>"
                 f"<style>.kpi:before{{background:{color}}}</style>", unsafe_allow_html=True)


# ============ DASHBOARD ============
def page_overview():
    st.markdown("<h1>Retention snapshot</h1>"
                "<div class='muted'>Where the current student body stands, and who needs attention.</div><br>",
                unsafe_allow_html=True)
    tiers = np.array([tier(p)[0] for p in p_dropout])
    high = int(np.sum(tiers == "High")); med = int(np.sum(tiers == "Medium")); low = int(np.sum(tiers == "Lower"))
    cols = st.columns(4)
    for c, (v, l, col) in zip(cols, [(n, "Students monitored", NAVY), (high, "High risk", RED),
                                     (med, "Medium risk", AMBER), (low, "Lower risk", GREEN)]):
        c.markdown(f"<div class='kpi'><div class='num' style='color:{col}'>{v}</div>"
                   f"<div class='lab'>{l}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        labels = ["High risk", "Medium risk", "Lower risk"]
        counts = [high, med, low]
        colors = [RED, AMBER, GREEN]
        fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h",
                               marker_color=colors, width=0.62,
                               text=[f"{c} ({c/n:.0%})" for c in counts],
                               textposition="outside", cliponaxis=False, textfont=dict(size=13)))
        style_fig(fig, 320)
        fig.update_layout(title="Students by risk level",
                          xaxis=dict(range=[0, max(counts) * 1.32], showgrid=True, gridcolor="#EEF1F7"),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown(f"<div class='card'><h3 style='margin-top:0'>What this means</h3>"
                    f"<p style='font-size:15px;color:{INK};line-height:1.6'>At the institution's current "
                    f"intervention line, <b>{high} students</b> are high risk and warrant outreach now, "
                    f"<b>{med}</b> are worth monitoring, and <b>{low}</b> look on track.</p>"
                    f"<p style='font-size:15px;color:{INK};line-height:1.6'>The model identifies roughly "
                    f"<b>3 in 4</b> of the students who go on to leave, early enough to act.</p>"
                    f"<p class='muted'>Open <b>Students</b> to see who needs attention and why.</p></div>",
                    unsafe_allow_html=True)


# ============ STUDENTS (cards) ============
def page_students():
    st.markdown("<h1>Students requiring attention</h1>"
                "<div class='muted'>Prioritized by risk. Each card shows the main concern and a recommended "
                "next step.</div><br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1, 1, 1.5])
    rf = f1.selectbox("Risk level", ["All", "High", "Medium", "Lower"])
    af = f2.selectbox("Attendance", ["All", "Daytime", "Evening"])
    sr = f3.text_input("Search by Student ID", "")
    sel = []
    for pos in order:
        tl = tier(p_dropout[pos])[0]
        if rf != "All" and tl != rf: continue
        if af != "All" and "Daytime/evening attendance" in Xo.columns:
            if int(Xo.iloc[pos]["Daytime/evening attendance"]) != (1 if af == "Daytime" else 0): continue
        if sr and sr.strip() not in str(student_ids[pos]): continue
        sel.append(pos)
    sel = sel[:24]
    st.caption(f"{len(sel)} students shown")
    for r0 in range(0, len(sel), 3):
        cols = st.columns(3)
        for c, pos in zip(cols, sel[r0:r0 + 3]):
            tl, tc, tcol, sub = tier(p_dropout[pos])
            risk, _ = drivers(pos)
            concern = risk[0][0] if risk else "Review profile"
            act, when, owner = recommendation(pos)
            meta = " · ".join(x for x in [age_of(pos)] if x)
            c.markdown(
                f"<div class='scard' style='border-left-color:{tcol}'>"
                f"<span class='badge {tc}'>{tl}</span>"
                f"<div class='sid' style='margin-top:10px'>Student #{student_ids[pos]}</div>"
                f"<div class='meta'>{meta}</div>"
                f"<div class='concern'><b>Concern:</b> {concern}</div>"
                f"<div class='rec'>→ {act} · {when}</div>"
                f"<div class='conf' style='margin-top:8px'>Model confidence {p_dropout[pos]:.0%}</div></div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Open a full profile")
    opts = {f"#{student_ids[pos]} · {tier(p_dropout[pos])[0]} risk": int(pos) for pos in sel}
    if opts:
        pick = st.selectbox("sel", list(opts.keys()), label_visibility="collapsed")
        if st.button("Open profile →", type="primary"):
            st.session_state.student_pos = opts[pick]; st.session_state._goto = "Student profile"; st.rerun()


# ============ PROFILE ============
def page_profile():
    if st.button("← Back to Students"):
        st.session_state._goto = "Students"; st.rerun()
    pos = st.session_state.student_pos
    p_pick = st.selectbox("Viewing", [f"#{student_ids[p]}" for p in order[:50]],
                          index=list(order[:50]).index(pos) if pos in order[:50] else 0,
                          label_visibility="collapsed")
    pos = student_ids.index(int(p_pick.lstrip("#"))); st.session_state.student_pos = pos
    tl, tc, tcol, sub = tier(p_dropout[pos])
    st.markdown(f"<h1 style='margin-bottom:2px'>Student #{student_ids[pos]}</h1>"
                f"<span class='badge {tc}'>{tl} risk</span> "
                f"<span class='muted'>· {sub} · model confidence {p_dropout[pos]:.0%}</span><br><br>",
                unsafe_allow_html=True)
    risk, prot = drivers(pos)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        items = "".join(f"<div class='factor'><span class='dot' style='background:{RED}'></span>{r}</div>"
                        for r, _ in risk[:4])
        st.markdown(f"<div class='card'><h3 style='margin-top:0'>Why this student is flagged</h3>{items}"
                    f"<div class='muted' style='margin-top:10px'>Generated using SHAP explainability.</div></div>",
                    unsafe_allow_html=True)
        act, when, owner = recommendation(pos)
        st.markdown(f"<div class='card'><h3 style='margin-top:0'>Recommended next steps</h3>"
                    f"<div class='rec-box'><b>{act}</b><br><span class='muted'>Refer to: {owner} · {when}</span></div>"
                    f"</div>", unsafe_allow_html=True)
    with c2:
        rows = []
        if "Age at enrollment" in Xo.columns: rows.append(("Age at enrollment", age_of(pos).replace("Age ", "") or "—"))
        if "Daytime/evening attendance" in Xo.columns:
            rows.append(("Attendance", "Daytime" if int(Xo.iloc[pos]["Daytime/evening attendance"]) == 1 else "Evening"))
        if "Scholarship holder" in Xo.columns:
            rows.append(("Scholarship", "Yes" if int(Xo.iloc[pos]["Scholarship holder"]) == 1 else "No"))
        if "Tuition fees up to date" in Xo.columns:
            rows.append(("Tuition", "Up to date" if int(Xo.iloc[pos]["Tuition fees up to date"]) == 1 else "Overdue"))
        snap = "".join(f"<div class='factor'><span style='color:{MUTED};width:130px;display:inline-block'>{k}</span>"
                       f"<b>{v}</b></div>" for k, v in rows)
        st.markdown(f"<div class='card'><h3 style='margin-top:0'>Student snapshot</h3>{snap}</div>",
                    unsafe_allow_html=True)
        if prot:
            good = "".join(f"<div class='factor'><span class='dot' style='background:{GREEN}'></span>{g}</div>"
                           for g in prot[:3])
            st.markdown(f"<div class='card'><h3 style='margin-top:0'>Working in their favor</h3>{good}</div>",
                        unsafe_allow_html=True)


# ============ MODEL CENTER ============
def page_model():
    st.markdown("<h1>Model Center</h1><div class='muted'>The technical evidence behind the console.</div><br>",
                unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Performance", "Threshold explorer", "Fairness", "Explainability"])
    def preds_at(thr): return np.where(p_dropout >= thr, DROPOUT, np.argmax(proba[:, 1:], axis=1) + 1)
    with t1:
        pred = preds_at(st.session_state.thr)
        tp = np.sum((yt == DROPOUT) & (pred == DROPOUT)); fn = np.sum((yt == DROPOUT) & (pred != DROPOUT))
        fp = np.sum((yt != DROPOUT) & (pred == DROPOUT))
        rec = tp / (tp + fn); prec = tp / (tp + fp) if (tp + fp) else 0; acc = np.mean(pred == yt)
        # macro F1 computed manually so it never depends on sklearn
        f1s = []
        K = len(class_names)
        cm = np.zeros((K, K), dtype=int)
        for a, p in zip(yt, pred):
            cm[int(a), int(p)] += 1
        for k in range(K):
            tpk = np.sum((yt == k) & (pred == k))
            fpk = np.sum((yt != k) & (pred == k))
            fnk = np.sum((yt == k) & (pred != k))
            pk = tpk / (tpk + fpk) if (tpk + fpk) else 0.0
            rk = tpk / (tpk + fnk) if (tpk + fnk) else 0.0
            f1s.append(2 * pk * rk / (pk + rk) if (pk + rk) else 0.0)
        mf1 = float(np.mean(f1s))
        cs = st.columns(4)
        for c, (l, v) in zip(cs, [("Dropout recall", f"{rec:.1%}"), ("Dropout precision", f"{prec:.1%}"),
                                  ("Accuracy", f"{acc:.1%}"), ("Macro F1", f"{mf1:.2f}")]):
            c.markdown(f"<div class='kpi'><div class='num' style='font-size:28px;color:{NAVY}'>{v}</div>"
                       f"<div class='lab'>{l}</div></div>", unsafe_allow_html=True)
        st.caption(f"At threshold {st.session_state.thr:.2f}, on {n} held-out students. Adjust it in the Threshold explorer tab.")
        # diagonal = correct (green), off-diagonal = mistake (red). Percentages are ROW-based
        # (out of the actual group), so the diagonal reads as recall. Hover sentence leads with the prediction.
        K = len(class_names)
        VERB_ACT = {"Dropout": "had actually dropped out", "Enrolled": "were actually still enrolled",
                    "Graduate": "had actually graduated"}
        def actual_phrase(c): return VERB_ACT.get(c, "were actually " + c.lower())
        row_tot = cm.sum(axis=1, keepdims=True)  # totals per actual class
        color = np.zeros((K, K)); hover = [["" for _ in range(K)] for _ in range(K)]
        celltext = [["" for _ in range(K)] for _ in range(K)]
        for i in range(K):      # actual (row)
            for j in range(K):  # predicted (column)
                cnt = int(cm[i, j]); pct = (cnt / row_tot[i, 0] * 100) if row_tot[i, 0] else 0
                color[i, j] = pct if i == j else -pct
                celltext[i][j] = f"{cnt}<br>{pct:.0f}%"
                if i == j:
                    hover[i][j] = (f"The model predicted <b>{class_names[j]}</b> for <b>{cnt} students</b> "
                                   f"who {actual_phrase(class_names[i])} — a correct call.<br>"
                                   f"That is {pct:.0f}% of all students who {actual_phrase(class_names[i])}.")
                else:
                    miss = " The model missed these at-risk students." if class_names[i] == "Dropout" else ""
                    hover[i][j] = (f"The model predicted <b>{class_names[j]}</b> for <b>{cnt} students</b> "
                                   f"who {actual_phrase(class_names[i])}.{miss}<br>"
                                   f"That is {pct:.0f}% of all students who {actual_phrase(class_names[i])}.")
        diverging = [[0.0, "#C0392B"], [0.5, "#F6DCD7"], [0.5, "#DCEEE1"], [1.0, "#1E7A3D"]]
        fig = go.Figure(go.Heatmap(
            z=color, x=class_names, y=class_names, zmin=-100, zmax=100,
            colorscale=diverging, showscale=False,
            text=celltext, texttemplate="%{text}", textfont=dict(size=15, color="#111111"),
            customdata=hover, hovertemplate="%{customdata}<extra></extra>",
            xgap=3, ygap=3))
        style_fig(fig, 360)
        fig.update_layout(title="Confusion matrix — green is correct, red is a mistake",
                          xaxis_title="Predicted (what the model said)",
                          yaxis_title="Actual (what really happened)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Each cell: number of students and the percentage within that actual-outcome row "
                   "(so the green diagonal is recall). Hover any cell for a plain-language reading.")
    with t2:
        thr = st.slider("Decision threshold", 0.10, 0.79, st.session_state.thr, 0.01); st.session_state.thr = thr
        ths = np.linspace(0.10, 0.79, 40); recs, precs = [], []
        for tt in ths:
            pr = preds_at(tt); tp = np.sum((yt == DROPOUT) & (pr == DROPOUT)); fn = np.sum((yt == DROPOUT) & (pr != DROPOUT))
            fp = np.sum((yt != DROPOUT) & (pr == DROPOUT)); recs.append(tp / (tp + fn)); precs.append(tp / (tp + fp) if (tp + fp) else 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ths, y=recs, name="Recall", line=dict(color=RED, width=3)))
        fig.add_trace(go.Scatter(x=ths, y=precs, name="Precision", line=dict(color=BLUE, width=3)))
        fig.add_vline(x=thr, line_dash="dash", line_color=INK)
        style_fig(fig, 360); fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"At threshold {thr:.2f}: {int(np.sum(preds_at(thr) == DROPOUT))} students flagged.")
    with t3:
        SUB = {"Financial risk": ("financial_risk", {0: "No financial risk", 1: "Has financial risk"}),
               "Gender": ("Gender", {0: "Female", 1: "Male"}),
               "Scholarship": ("Scholarship holder", {0: "No scholarship", 1: "Has scholarship"}),
               "Attendance": ("Daytime/evening attendance", {0: "Evening", 1: "Daytime"})}
        attr = st.selectbox("Audit recall by", list(SUB.keys())); col, mp = SUB[attr]; pred = preds_at(st.session_state.thr)
        labels, vals, ns = [], [], []; codes = Xo[col].values
        for code, lab in mp.items():
            den = np.sum((codes == code) & (yt == DROPOUT)); num = np.sum((codes == code) & (yt == DROPOUT) & (pred == DROPOUT))
            labels.append(lab); vals.append(100 * num / den if den else np.nan); ns.append(int(den))
        fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[RED, GREEN][:len(labels)],
                               text=[f"{v:.1f}%<br>(n={x})" for v, x in zip(vals, ns)], textposition="outside", cliponaxis=False))
        gap = (max(vals) - min(vals)) if len(vals) >= 2 and not any(np.isnan(vals)) else float("nan")
        style_fig(fig, 360); fig.update_layout(yaxis_range=[0, 110], yaxis_title="Dropout recall (%)",
                                               title=f"Recall gap at threshold {st.session_state.thr:.2f}: {gap:.1f} points")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with t4:
        imp = sorted(((nm, float(np.mean(np.abs(np.sum(shap_dropout[:, cols], axis=1)))))
                      for nm, cols in concept_cols.items()), key=lambda x: x[1])[-12:]
        fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[k for k, _ in imp], orientation="h", marker_color=NAVY))
        style_fig(fig, 420); fig.update_layout(title="Global feature importance (mean |SHAP|)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Drives the plain-language risk factors shown to advisors.")


if st.session_state.get("_goto"):
    page = st.session_state.pop("_goto")
{"Dashboard": page_overview, "Students": page_students,
 "Student profile": page_profile, "Model Center": page_model}[page]()
