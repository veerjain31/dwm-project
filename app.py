import os
import sqlite3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
                             silhouette_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

from data_gen import create_dataset
from mining import NUM_FEATURES, CAT_COLS, NUMERIC_COLS, clean_data, apriori_rules, tree_path, root_split_gain

st.set_page_config(page_title="Placement Mining Lab", page_icon="🎓", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "student_placement_60000.csv")
DB_PATH = os.path.join(BASE, "student_placement.db")

INK, PLACED_C, NOT_C = "#1B2540", "#1F9D8B", "#E4572E"
J48_C, NB_C, GREY = "#3A59D1", "#F2A33A", "#8892A6"
LABELS = {"CGPA": "CGPA", "Tenth_Percent": "10th %", "Twelfth_Percent": "12th %", "Backlogs": "Backlogs",
          "Internships": "Internships", "Projects": "Projects", "Certifications": "Certifications",
          "Communication_Score": "Communication", "Aptitude_Score": "Aptitude"}
DEFAULTS = {"CGPA": 7.5, "Tenth_Percent": 78, "Twelfth_Percent": 74, "Backlogs": 0, "Internships": 1,
            "Projects": 2, "Certifications": 1, "Communication_Score": 65, "Aptitude_Score": 62}
RANGES = {"CGPA": (4, 10), "Tenth_Percent": (35, 100), "Twelfth_Percent": (35, 100), "Backlogs": (0, 10),
          "Internships": (0, 10), "Projects": (0, 15), "Certifications": (0, 15),
          "Communication_Score": (0, 100), "Aptitude_Score": (0, 100)}
REG_FEATS = ["CGPA", "Internships", "Projects", "Communication_Score", "Aptitude_Score"]
CLASS_NAMES = ["Not Placed", "Placed"]

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .22, "axes.titleweight": "bold",
                     "axes.titlesize": 10.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
                     "axes.edgecolor": "#B8BFCE", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

# ======================================================================== styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label { font-family:'Manrope', 'Segoe UI', sans-serif; }
h1, h2, h3, h4 { font-family:'Space Grotesk', 'Segoe UI', sans-serif !important; color:#1B2540; letter-spacing:-.01em; }
.block-container { padding-top:1.4rem; max-width:1320px; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:#F5F6FA !important; }
[data-testid="stMarkdownContainer"] *, [data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"] *,
.stTabs [data-baseweb="tab"] *, [data-testid="stExpander"] summary * { color:#1B2540; }
[data-testid="stExpander"] { background:#fff; border-radius:8px; }
pre, code { color:#1B2540 !important; background:#EEF0F7 !important; }
[data-testid="stSidebar"] { background:#1B2540; }
[data-testid="stSidebar"] * { color:#E6EAF4 !important; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap:.15rem; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { padding:.55rem .8rem; border-radius:8px; width:100%; cursor:pointer; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background:#27345C; }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background:#F2A33A; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * { color:#1B2540 !important; font-weight:700; }
.side-brand { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; line-height:1.2; }
.side-sub { font-size:.78rem; opacity:.7; margin-bottom:.9rem; }
.side-note { font-size:.76rem; opacity:.75; line-height:1.5; border-top:1px solid #34426C; padding-top:.7rem; margin-top:1rem; }
.hero, .hero * { color:#fff !important; }
.chips span { color:#D5DBEC !important; }
.hero { background:#1B2540; color:#fff; border-radius:14px; padding:1.1rem 1.5rem; margin-bottom:1.1rem;
        border-left:8px solid #F2A33A; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; }
.hero-sub { opacity:.78; font-size:.92rem; margin:.15rem 0 .7rem; }
.chips span { display:inline-block; font-size:.76rem; padding:.18rem .65rem; margin:0 .35rem .25rem 0;
              border:1px solid #4A5A8C; border-radius:999px; color:#D5DBEC; }
.page-title { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; margin:.2rem 0 0; }
.page-blurb { color:#56607A; margin-bottom:.8rem; font-size:.95rem; }
.kpi { background:#fff; border:1px solid #E1E5EF; border-left:5px solid #1B2540; border-radius:10px;
       padding:.7rem .95rem; margin-bottom:.6rem; }
.kpi-label { font-size:.78rem; color:#68728D; }
.kpi-value { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:#1B2540; line-height:1.25; }
.kpi-note { font-size:.74rem; color:#8892A6; }
.defcard, .defcard * { color:#1B2540 !important; }
.defcard { background:#fff; border:1px solid #E1E5EF; border-top:6px solid #3A59D1; border-radius:10px;
           padding:1rem 1.2rem; height:100%; }
.defcard h4 { margin:0 0 .4rem; }
.defcard p, .defcard li { font-size:.9rem; line-height:1.55; }
.defcard ul { padding-left:1.1rem; margin:.2rem 0 .5rem; }
.formula { background:#F0F2F9; border-radius:8px; padding:.5rem .8rem; font-family:'Space Grotesk',monospace; font-size:.88rem; margin:.4rem 0; }
.verdict, .verdict * { color:#1B2540 !important; }
.verdict { background:#fff; border:1px solid #E1E5EF; border-left:6px solid #1F9D8B; border-radius:10px; padding:.8rem 1.1rem; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
div[data-testid="stDataFrame"] { border:1px solid #E1E5EF; border-radius:8px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ======================================================================== helpers
def page_header(title, blurb):
    st.markdown(f"<div class='page-title'>{title}</div><div class='page-blurb'>{blurb}</div>", unsafe_allow_html=True)


def kpi(col, label, value, note="", color=INK):
    col.markdown(f"<div class='kpi' style='border-left-color:{color}'><div class='kpi-label'>{label}</div>"
                 f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>",
                 unsafe_allow_html=True)


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def mono_cmap(color):
    return LinearSegmentedColormap.from_list("m", ["#FFFFFF", color])


def plot_cm(ax, yte, pred, color, title=""):
    cm = confusion_matrix(yte, pred)
    ax.imshow(cm, cmap=mono_cmap(color))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n({v / cm.sum() * 100:.1f}%)", ha="center", va="center", fontsize=9,
                color="white" if v > cm.max() * .55 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES); ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title); ax.grid(False)


def prediction_form(prefix, features):
    cols = st.columns(3)
    vals = []
    for i, f in enumerate(features):
        lo, hi = RANGES[f]
        vals.append(cols[i % 3].number_input(LABELS[f], min_value=float(lo), max_value=float(hi),
                                             value=float(DEFAULTS[f]), step=.1 if f == "CGPA" else 1.0,
                                             key=f"{prefix}_{f}"))
    return pd.DataFrame([vals], columns=features)


def display_raw(df):
    return df.astype(object).where(df.notna(), "NaN").astype(str)


# ======================================================================== data layer
@st.cache_data(show_spinner="Loading the 60,000-student database ...")
def load_raw():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return create_dataset(60000)


@st.cache_data(show_spinner="Cleaning data ...")
def get_clean():
    return clean_data(load_raw())


def split_cls(data, features=NUM_FEATURES):
    X = data[features]
    y = (data["Placed"] == "Placed").astype(int)
    return train_test_split(X, y, test_size=.2, random_state=42, stratify=y)


def cls_metrics(yte, pred, proba):
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return {"Accuracy (%)": accuracy_score(yte, pred) * 100, "Precision (%)": precision_score(yte, pred) * 100,
            "Recall (%)": recall_score(yte, pred) * 100, "Specificity (%)": tn / (tn + fp) * 100,
            "F1 Score (%)": f1_score(yte, pred) * 100, "ROC-AUC": roc_auc_score(yte, proba),
            "Matthews Corr.": matthews_corrcoef(yte, pred)}


@st.cache_resource(show_spinner="Training classifier ...")
def train_cls(kind, depth=5, min_leaf=20, criterion="entropy"):
    data, _ = get_clean()
    Xtr, Xte, ytr, yte = split_cls(data)
    if kind == "j48":
        model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_leaf=min_leaf, random_state=42)
    else:
        model = GaussianNB()
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); pred_t = time.perf_counter() - t0
    proba = model.predict_proba(Xte)[:, 1]
    cv = cross_val_score(model, data[NUM_FEATURES], (data["Placed"] == "Placed").astype(int),
                         cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
    train_acc = accuracy_score(ytr, model.predict(Xtr)) * 100
    return dict(model=model, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, pred=pred, proba=proba, cv=cv * 100,
                fit_time=fit_t, pred_time=pred_t, train_acc=train_acc, metrics=cls_metrics(yte, pred, proba))


@st.cache_resource(show_spinner="Scanning tree depths ...")
def depth_curve(min_leaf, criterion):
    data, _ = get_clean()
    Xtr, Xte, ytr, yte = split_cls(data)
    rows = []
    for d in range(1, 15):
        m = DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_leaf=min_leaf, random_state=42).fit(Xtr, ytr)
        rows.append((d, m.score(Xtr, ytr) * 100, m.score(Xte, yte) * 100, m.get_n_leaves()))
    return pd.DataFrame(rows, columns=["depth", "train", "test", "leaves"])


@st.cache_resource(show_spinner="Building learning curves ...")
def learning_curves(depth, min_leaf):
    data, _ = get_clean()
    Xtr, Xte, ytr, yte = split_cls(data)
    rows = []
    for s in [500, 1000, 2500, 5000, 10000, 20000, len(Xtr)]:
        tr = DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_leaf=min_leaf, random_state=42).fit(Xtr[:s], ytr[:s])
        nb = GaussianNB().fit(Xtr[:s], ytr[:s])
        rows.append((s, tr.score(Xte, yte) * 100, nb.score(Xte, yte) * 100))
    return pd.DataFrame(rows, columns=["n", "J48", "NB"])


@st.cache_resource(show_spinner="Training regressors ...")
def train_reg(depth):
    data, _ = get_clean()
    rd = data[(data.Placed == "Placed")].dropna(subset=["Package_LPA"])
    Xtr, Xte, ytr, yte = train_test_split(rd[REG_FEATS], rd.Package_LPA, test_size=.2, random_state=42)
    lr = LinearRegression().fit(Xtr, ytr)
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(Xtr, ytr)
    out = {"n": len(rd), "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "lr": lr, "dt": dt}
    for k, m in [("lr", lr), ("dt", dt)]:
        p = m.predict(Xte)
        out[k + "_pred"] = p
        out[k + "_m"] = {"R² Score": r2_score(yte, p), "MAE (LPA)": mean_absolute_error(yte, p),
                         "RMSE (LPA)": float(np.sqrt(mean_squared_error(yte, p)))}
    return out


@st.cache_resource(show_spinner="Running Apriori ...")
def get_rules(min_sup, min_conf):
    data, _ = get_clean()
    items = pd.DataFrame({
        "Branch": data.Branch, "Tier": data.College_Tier,
        "CGPA": pd.cut(data.CGPA, [0, 6.5, 8, 10], labels=["Low", "Medium", "High"]).astype(str),
        "Internship": np.where(data.Internships > 0, "Yes", "No"),
        "Backlog": np.where(data.Backlogs > 0, "Yes", "No"),
        "Aptitude": pd.cut(data.Aptitude_Score, [0, 55, 70, 100], labels=["Low", "Medium", "High"]).astype(str),
        "Status": data.Placed})
    return apriori_rules(pd.get_dummies(items.astype(str)), min_sup, min_conf)


@st.cache_resource(show_spinner="Scanning k for K-Means ...")
def kmeans_scan(feats):
    data, _ = get_clean()
    Xs = StandardScaler().fit_transform(data[list(feats)])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs)
        rows.append((k, km.inertia_, silhouette_score(Xs, km.labels_, sample_size=5000, random_state=42)))
    return pd.DataFrame(rows, columns=["k", "Inertia", "Silhouette"])


@st.cache_resource(show_spinner="Fitting K-Means ...")
def kmeans_fit(feats, k):
    data, _ = get_clean()
    Xs = StandardScaler().fit_transform(data[list(feats)])
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs)
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    return dict(labels=km.labels_, Xs=Xs, coords=pca.transform(Xs), centers=pca.transform(km.cluster_centers_),
                var=pca.explained_variance_ratio_, inertia=km.inertia_,
                sil=silhouette_score(Xs, km.labels_, sample_size=5000, random_state=42))


# ======================================================================== schematic diagrams
def _box(ax, x, y, text, fc, tc="white", fs=8.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec="none"))


def _arrow(ax, p, q, label=None, color=GREY):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
    if label:
        ax.text((p[0] + q[0]) / 2 + .15, (p[1] + q[1]) / 2, label, fontsize=7.5, color=color)


def draw_j48_schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)
    _box(ax, 5, 5.2, "CGPA ≤ 7.2 ?", J48_C)
    _box(ax, 2.5, 3.2, "Backlogs = 0 ?", J48_C)
    _box(ax, 7.5, 3.2, "Internships ≥ 1 ?", J48_C)
    for x, t, c in [(1.2, "Not\nPlaced", NOT_C), (3.8, "Placed", PLACED_C), (6.2, "Not\nPlaced", NOT_C), (8.8, "Placed", PLACED_C)]:
        _box(ax, x, 1.1, t, c)
    _arrow(ax, (4.6, 4.9), (2.9, 3.55), "yes"); _arrow(ax, (5.4, 4.9), (7.1, 3.55), "no")
    _arrow(ax, (2.2, 2.9), (1.4, 1.5), "no"); _arrow(ax, (2.8, 2.9), (3.6, 1.5), "yes")
    _arrow(ax, (7.2, 2.9), (6.4, 1.5), "no"); _arrow(ax, (7.8, 2.9), (8.6, 1.5), "yes")
    ax.set_title("J48: ask yes/no questions, follow a path to a leaf", fontsize=10)


def draw_nb_schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)
    _box(ax, 5, 5.2, "Class\nPlaced / Not Placed", NB_C, tc=INK)
    feats = ["CGPA", "Aptitude", "Intern-\nships", "Backlogs", "Commu-\nnication"]
    for i, f in enumerate(feats):
        x = 1 + i * 2
        _box(ax, x, 2.6, f, GREY, fs=8)
        _arrow(ax, (5, 4.7), (x, 3.1))
    ax.text(5, .9, r"$P(C\mid x)\ \propto\ P(C)\,\prod_i P(x_i\mid C)$", ha="center", fontsize=11)
    ax.text(5, .2, "each feature votes independently", ha="center", fontsize=8, color=GREY)
    ax.set_title("Naive Bayes: every feature votes on its own", fontsize=10)


def draw_entropy(ax, p_now=None):
    p = np.linspace(.001, .999, 300)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    ax.plot(p, h, color=J48_C, lw=2)
    ax.fill_between(p, h, alpha=.1, color=J48_C)
    if p_now is not None:
        hp = -p_now * np.log2(p_now) - (1 - p_now) * np.log2(1 - p_now)
        ax.scatter([p_now], [hp], color=NOT_C, zorder=5)
        ax.annotate(f"our data\nP(placed)={p_now:.2f}\nH={hp:.2f}", (p_now, hp), (p_now - .38, hp - .35), fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_xlabel("P(Placed) at a node"); ax.set_ylabel("Entropy H (bits)"); ax.set_title("Entropy: how mixed is a node?")


# ======================================================================== MODULES
def render_dashboard():
    page_header("Dashboard", "A one-page summary of the cleaned placement database.")
    data, log = get_clean()
    placed = data[data.Placed == "Placed"]
    c = st.columns(5)
    kpi(c[0], "Students (after cleaning)", f"{len(data):,}", f"{log['raw_rows']:,} raw records", INK)
    kpi(c[1], "Placement rate", f"{(data.Placed == 'Placed').mean() * 100:.1f}%", f"{len(placed):,} placed", PLACED_C)
    kpi(c[2], "Not placed", f"{(data.Placed != 'Placed').sum():,}", "", NOT_C)
    kpi(c[3], "Average package", f"₹{placed.Package_LPA.mean():.2f} LPA", f"median ₹{placed.Package_LPA.median():.2f}", NB_C)
    kpi(c[4], "Highest package", f"₹{placed.Package_LPA.max():.1f} LPA", "", J48_C)

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        vc = data.Placed.value_counts()
        ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=[PLACED_C if i == "Placed" else NOT_C for i in vc.index],
               wedgeprops=dict(width=.42, edgecolor="white"), startangle=90, textprops=dict(fontsize=9))
        ax.set_title("Placement status"); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        r = data.groupby("Branch").Placed.apply(lambda s: (s == "Placed").mean() * 100).sort_values()
        ax.barh(r.index, r.values, color=PLACED_C)
        for i, v in enumerate(r.values):
            ax.text(v + .6, i, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Placement rate (%)"); ax.set_title("Placement rate by branch"); ax.set_xlim(0, 100); show(fig)

    a, b, c3 = st.columns(3)
    with a:
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        r = data.groupby("College_Tier").Placed.apply(lambda s: (s == "Placed").mean() * 100)
        ax.bar(r.index, r.values, color=[J48_C, NB_C, GREY][:len(r)])
        for i, v in enumerate(r.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5)
        ax.set_ylim(0, 100); ax.set_ylabel("Placement rate (%)"); ax.set_title("By college tier"); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        for lab, col in [("Placed", PLACED_C), ("Not Placed", NOT_C)]:
            ax.hist(data[data.Placed == lab].CGPA, bins=30, alpha=.6, color=col, label=lab)
        ax.set_xlabel("CGPA"); ax.set_ylabel("Students"); ax.set_title("CGPA by status"); ax.legend(); show(fig)
    with c3:
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        ax.hist(placed.Package_LPA.dropna(), bins=40, color=NB_C)
        ax.axvline(placed.Package_LPA.median(), color=INK, ls="--", lw=1)
        ax.set_xlabel("Package (LPA)"); ax.set_ylabel("Students"); ax.set_title("Package distribution"); show(fig)

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        cols = NUM_FEATURES + ["Placed_bin"]
        corr = data.assign(Placed_bin=(data.Placed == "Placed").astype(int))[cols].corr()
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        names = [LABELS.get(c, "Placed") for c in cols]
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(names); ax.grid(False)
        for (i, j), v in np.ndenumerate(corr.values):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > .6 else INK)
        fig.colorbar(im, ax=ax, shrink=.8); ax.set_title("Correlation heatmap"); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        r = data.groupby(data.Internships.clip(upper=4)).Placed.apply(lambda s: (s == "Placed").mean() * 100)
        ax.plot(r.index, r.values, marker="o", color=J48_C, lw=2)
        r2 = data.groupby(data.Backlogs.clip(upper=4)).Placed.apply(lambda s: (s == "Placed").mean() * 100)
        ax.plot(r2.index, r2.values, marker="s", color=NOT_C, lw=2)
        ax.legend(["Internships (4 = 4+)", "Backlogs (4 = 4+)"])
        ax.set_xlabel("Count"); ax.set_ylabel("Placement rate (%)"); ax.set_title("Internships help, backlogs hurt"); show(fig)


def render_dataset():
    page_header("Dataset: 60,000-record database", "The raw (uncleaned) table exactly as stored, plus a SQL window onto the SQLite copy.")
    raw = load_raw()
    c = st.columns(4)
    kpi(c[0], "Rows", f"{len(raw):,}", "includes duplicate records")
    kpi(c[1], "Columns", f"{raw.shape[1]}", "")
    kpi(c[2], "Cells with missing values", f"{int(raw.isna().sum().sum()):,}", "", NOT_C)
    kpi(c[3], "Unique Student_IDs", f"{raw.Student_ID.nunique():,}", "", PLACED_C)
    t1, t2, t3, t4 = st.tabs(["🔎 Browse records", "🧾 Schema & statistics", "🕳️ Missing-value map", "🛢️ SQL explorer"])
    with t1:
        a, b = st.columns([1, 3])
        n = a.slider("Rows to show", 20, 500, 100, key="ds_rows")
        sel = b.multiselect("Filter branch (raw spelling)", sorted(raw.Branch.dropna().astype(str).unique()), key="ds_branch")
        view = raw[raw.Branch.isin(sel)] if sel else raw
        st.dataframe(display_raw(view.head(n)), height=420)
        st.download_button("⬇ Download raw CSV", raw.to_csv(index=False).encode(), "student_placement_60000.csv", "text/csv")
    with t2:
        info = pd.DataFrame({"dtype as loaded": raw.dtypes.astype(str), "missing": raw.isna().sum(),
                             "missing %": (raw.isna().mean() * 100).round(2), "unique values": raw.nunique()})
        st.dataframe(info)
        st.caption("Numeric columns load as text because junk tokens such as '?' and 'unknown' are hidden inside them.")
        num = raw[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
        st.dataframe(num.describe().T.round(2))
    with t3:
        sample = raw.head(300)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.imshow(sample.isna().to_numpy().T, aspect="auto", cmap=mono_cmap(NOT_C), interpolation="nearest")
        ax.set_yticks(range(raw.shape[1])); ax.set_yticklabels(raw.columns, fontsize=8); ax.grid(False)
        ax.set_xlabel("first 300 records"); ax.set_title("Missing-value map (orange = missing)"); show(fig)
        fig, ax = plt.subplots(figsize=(11, 3))
        m = (raw.isna().mean() * 100).sort_values(ascending=False)
        ax.bar(m.index, m.values, color=NOT_C); ax.set_ylabel("Missing %"); plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_title("Missing values per column (whole table)"); show(fig)
    with t4:
        if not os.path.exists(DB_PATH):
            st.info("student_placement.db not found. Run `python data_gen.py` once to create it.")
        else:
            examples = {
                "Placement rate by branch (clean)": "SELECT Branch, COUNT(*) AS students,\n  ROUND(100.0*SUM(Placed='Placed')/COUNT(*),1) AS placement_pct,\n  ROUND(AVG(Package_LPA),2) AS avg_lpa\nFROM students_clean GROUP BY Branch ORDER BY placement_pct DESC",
                "Dirty Branch spellings (raw)": "SELECT Branch, COUNT(*) AS n FROM students_raw GROUP BY Branch ORDER BY n DESC",
                "Average package: tier x branch": "SELECT College_Tier, Branch, ROUND(AVG(Package_LPA),2) AS avg_lpa, COUNT(*) AS placed\nFROM students_clean WHERE Package_LPA IS NOT NULL\nGROUP BY College_Tier, Branch ORDER BY College_Tier, avg_lpa DESC",
                "Top 20 packages": "SELECT Name, Branch, College_Tier, CGPA, Package_LPA FROM students_clean\nWHERE Package_LPA IS NOT NULL ORDER BY Package_LPA DESC LIMIT 20",
            }
            pick = st.selectbox("Example query", list(examples), key="sql_pick")
            q = st.text_area("SQL (SELECT only). Tables: students_raw, students_clean", examples[pick], height=140, key=f"sql_{pick}")
            if q.strip().lower().startswith("select") and ";" not in q.strip().rstrip(";"):
                try:
                    with sqlite3.connect(DB_PATH) as con:
                        res = pd.read_sql_query(q.strip().rstrip(";"), con)
                    st.dataframe(res)
                    if res.shape[1] >= 2 and res.shape[0] <= 40 and pd.api.types.is_numeric_dtype(res.iloc[:, 1]):
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.bar(res.iloc[:, 0].astype(str), res.iloc[:, 1], color=J48_C)
                        ax.set_ylabel(res.columns[1]); plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
                        ax.set_title("Quick chart of the result"); show(fig)
                except Exception as e:
                    st.error(f"Query failed: {e}")
            else:
                st.warning("Only a single SELECT statement is allowed.")


def render_cleaning():
    page_header("Data cleaning & preprocessing", "Every problem injected into the raw table, how it was fixed, and proof with before/after charts.")
    raw = load_raw()
    clean, log = get_clean()
    c = st.columns(5)
    kpi(c[0], "Raw rows", f"{log['raw_rows']:,}")
    kpi(c[1], "Duplicates removed", f"{log['duplicates_removed']:,}", "", NOT_C)
    kpi(c[2], "Unknown label dropped", f"{log['label_missing_dropped']:,}", "", NOT_C)
    kpi(c[3], "Values imputed", f"{sum(log['missing_imputed'].values()):,}", "median / mode", NB_C)
    kpi(c[4], "Clean rows", f"{log['final_rows']:,}", "", PLACED_C)

    t1, t2, t3, t4 = st.tabs(["🧭 Pipeline & issue counts", "🕳️ Missing values", "🔤 Categories & outliers", "📋 Log & download"])
    with t1:
        steps = pd.DataFrame([
            ["1", "Standardise text", "cse / C.S.E / Computer Science → CSE; Y / yes → Placed", f"{sum(log['text_fixed'].values()):,} cells"],
            ["2", "Force numeric columns", "'?', 'unknown', '-' become NaN", f"{sum(log['junk_tokens'].values()):,} tokens"],
            ["3", "Remove impossible values", "CGPA 15, % 150, backlogs −1, package 999 become NaN", f"{sum(log['invalid_values'].values()):,} values"],
            ["4", "Drop duplicates", "same Student_ID kept once", f"{log['duplicates_removed']:,} rows"],
            ["5", "Drop unknown label", "target 'Placed' cannot be guessed", f"{log['label_missing_dropped']:,} rows"],
            ["6", "Fix package logic", "unplaced student cannot have a salary", f"{log['package_contradictions']:,} rows"],
            ["7", "Impute", "median for numbers, mode for categories", f"{sum(log['missing_imputed'].values()):,} cells"]],
            columns=["Step", "Action", "Example", "Fixed"])
        st.dataframe(steps, hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        names = ["Inconsistent text", "Junk tokens", "Impossible values", "Missing (imputed)", "Duplicates", "Unknown label", "Package contradictions"]
        vals = [sum(log["text_fixed"].values()), sum(log["junk_tokens"].values()), sum(log["invalid_values"].values()),
                sum(log["missing_imputed"].values()), log["duplicates_removed"], log["label_missing_dropped"], log["package_contradictions"]]
        ax.barh(names[::-1], vals[::-1], color=[NOT_C, NB_C, J48_C, PLACED_C, GREY, INK, "#9B5DE5"][::-1])
        for i, v in enumerate(vals[::-1]):
            ax.text(v + max(vals) * .01, i, f"{v:,}", va="center", fontsize=8.5)
        ax.set_xlabel("Cells / rows affected"); ax.set_title("Dirty-data issues found in the raw table"); show(fig)
    with t2:
        a, b = st.columns(2)
        cols = [c for c in raw.columns if c not in ("Student_ID", "Name", "Package_LPA")]
        before = raw[cols].isna().sum()
        after = clean[cols].isna().sum()
        with a:
            fig, ax = plt.subplots(figsize=(6, 5))
            y = np.arange(len(cols))
            ax.barh(y - .2, before.values, .4, color=NOT_C, label="before")
            ax.barh(y + .2, after.values, .4, color=PLACED_C, label="after")
            ax.set_yticks(y); ax.set_yticklabels(cols); ax.invert_yaxis(); ax.legend()
            ax.set_xlabel("Missing cells"); ax.set_title("Missing values before vs after"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(6, 5))
            imp = pd.Series(log["missing_imputed"]).sort_values()
            ax.barh(imp.index, imp.values, color=NB_C)
            ax.set_xlabel("Cells imputed"); ax.set_title("What was imputed (all numeric = median, categorical = mode)"); show(fig)
        st.success("Package_LPA is intentionally left empty for unplaced students; only placed students with a valid salary are used by the regression module.")
    with t3:
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(6, 4))
            cats = ["Branch", "Gender", "College_Tier", "Placed"]
            x = np.arange(len(cats))
            ax.bar(x - .2, [raw[c].nunique() for c in cats], .4, color=NOT_C, label="raw spellings")
            ax.bar(x + .2, [clean[c].nunique() for c in cats], .4, color=PLACED_C, label="after cleaning")
            for i, c in enumerate(cats):
                ax.text(i - .2, raw[c].nunique() + .5, raw[c].nunique(), ha="center", fontsize=8)
                ax.text(i + .2, clean[c].nunique() + .5, clean[c].nunique(), ha="center", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(cats); ax.legend(); ax.set_title("Distinct category labels"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = raw.Branch.astype(str).str.strip().value_counts().head(14)
            ax.barh(vc.index[::-1], vc.values[::-1], color=NOT_C)
            ax.set_title("Raw Branch spellings (top 14)"); ax.set_xlabel("Records"); show(fig)
        feats = ["CGPA", "Twelfth_Percent", "Aptitude_Score", "Backlogs"]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
        for ax, f in zip(axes, feats):
            r = pd.to_numeric(raw[f], errors="coerce").dropna()
            bp = ax.boxplot([r, clean[f]], widths=.55, patch_artist=True, showfliers=True,
                            flierprops=dict(marker=".", markersize=3, alpha=.5))
            for patch, col in zip(bp["boxes"], [NOT_C, PLACED_C]):
                patch.set_facecolor(col); patch.set_alpha(.6)
            ax.set_xticks([1, 2]); ax.set_xticklabels(["raw", "clean"]); ax.set_title(LABELS[f])
        fig.suptitle("Outliers removed: box-plots before vs after", fontweight="bold", fontsize=11); show(fig)
    with t4:
        rows = []
        for k in ["text_fixed", "junk_tokens", "invalid_values", "missing_imputed"]:
            for col, v in log[k].items():
                rows.append({"issue": k.replace("_", " "), "column": col, "count": v})
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=360)
        st.download_button("⬇ Download cleaned CSV", clean.to_csv(index=False).encode(), "student_placement_clean.csv", "text/csv")
        st.markdown("**Cleaned sample**")
        st.dataframe(clean.head(15))


def render_j48():
    page_header("J48 Decision Tree (C4.5)", "A tree of yes/no questions learned from the data. Tune depth and pruning, then inspect the real tree.")
    a, b, c = st.columns(3)
    depth = a.slider("Maximum depth", 2, 12, 5, key="j48_depth")
    min_leaf = b.slider("Minimum samples per leaf (pruning)", 1, 200, 20, key="j48_leaf")
    crit = c.selectbox("Split criterion", ["entropy", "gini"], key="j48_crit",
                       help="J48/C4.5 uses entropy (information gain). Gini is the CART alternative.")
    r = train_cls("j48", depth, min_leaf, crit)
    m, model = r["metrics"], r["model"]
    k = st.columns(5)
    kpi(k[0], "Accuracy", f"{m['Accuracy (%)']:.2f}%", "", J48_C); kpi(k[1], "Precision", f"{m['Precision (%)']:.2f}%")
    kpi(k[2], "Recall", f"{m['Recall (%)']:.2f}%"); kpi(k[3], "F1 score", f"{m['F1 Score (%)']:.2f}%")
    kpi(k[4], "Tree size", f"{model.get_n_leaves()} leaves", f"depth {model.get_depth()}", NB_C)

    t1, t2, t3, t4, t5 = st.tabs(["🌳 Tree diagram", "📈 Performance", "🧠 How it splits", "✂️ Pruning curve", "🎯 Predict"])
    with t1:
        sd = st.slider("Levels to draw", 1, 5, 3, key="j48_show")
        width = {1: 8, 2: 12, 3: 18, 4: 26, 5: 36}[sd]
        fig, ax = plt.subplots(figsize=(width, 3 + 1.6 * sd))
        plot_tree(model, feature_names=[LABELS[f] for f in NUM_FEATURES], class_names=CLASS_NAMES, filled=True,
                  rounded=True, max_depth=sd, fontsize=8 if sd < 4 else 7, ax=ax, impurity=True)
        ax.set_title(f"J48 decision tree (top {sd} levels of {model.get_depth()})"); st.pyplot(fig); plt.close(fig)
        with st.expander("Same tree as IF-THEN text rules"):
            st.code(export_text(model, feature_names=[LABELS[f] for f in NUM_FEATURES], max_depth=4), language="text")
    with t2:
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5, 4)); plot_cm(ax, r["yte"], r["pred"], J48_C, "Confusion matrix"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(r["yte"], r["proba"])
            ax.plot(fpr, tpr, color=J48_C, lw=2, label=f"J48 (AUC = {m['ROC-AUC']:.3f})"); ax.fill_between(fpr, tpr, alpha=.08, color=J48_C)
            ax.plot([0, 1], [0, 1], "--", color=GREY, label="random guess")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curve"); ax.legend(); show(fig)
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            p, rc, _ = precision_recall_curve(r["yte"], r["proba"])
            ax.plot(rc, p, color=J48_C, lw=2); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"Precision-recall (AP = {average_precision_score(r['yte'], r['proba']):.3f})"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.bar(["5-fold CV mean", "Test set", "Train set"], [r["cv"].mean(), m["Accuracy (%)"], r["train_acc"]], color=[GREY, J48_C, NB_C])
            for i, v in enumerate([r["cv"].mean(), m["Accuracy (%)"], r["train_acc"]]):
                ax.text(i, v + .5, f"{v:.1f}", ha="center", fontsize=9)
            ax.set_ylim(50, 100); ax.set_ylabel("Accuracy (%)"); ax.set_title("Train vs test (gap = overfitting)"); show(fig)
    with t3:
        a, b = st.columns(2)
        with a:
            imp = pd.Series(model.feature_importances_, index=[LABELS[f] for f in NUM_FEATURES]).sort_values()
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.barh(imp.index, imp.values, color=J48_C); ax.set_xlabel("Importance (total impurity reduction)")
            ax.set_title("Feature importance"); show(fig)
        with b:
            p_now = float(r["ytr"].mean()); fig, ax = plt.subplots(figsize=(5.5, 4)); draw_entropy(ax, p_now); show(fig)
        g = root_split_gain(model)
        st.markdown(f"**Root split:** `{LABELS[NUM_FEATURES[g['feature']]]} ≤ {g['threshold']:.2f}`")
        st.dataframe(pd.DataFrame({
            "Node": ["Parent (all training data)", "Left child (condition true)", "Right child (condition false)", "Weighted children", "Gain = parent − children"],
            "Samples": [g["left_n"] + g["right_n"], g["left_n"], g["right_n"], "", ""],
            ("Entropy" if crit == "entropy" else "Gini"): [round(g["parent"], 4), round(g["left"], 4), round(g["right"], 4),
                                                         round(g["weighted_children"], 4), round(g["gain"], 4)]}), hide_index=True)
        st.caption("Gain is the drop in impurity after the question is asked; the tree always asks the question with the biggest drop.")
    with t4:
        dc = depth_curve(min_leaf, crit)
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.plot(dc.depth, dc.train, marker="o", color=NB_C, label="train"); ax.plot(dc.depth, dc.test, marker="o", color=J48_C, label="test")
            ax.axvline(depth, color=GREY, ls="--"); ax.set_xlabel("Max depth"); ax.set_ylabel("Accuracy (%)")
            ax.set_title("Overfitting: train keeps rising, test flattens"); ax.legend(); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.plot(dc.depth, dc.leaves, marker="s", color=PLACED_C); ax.set_yscale("log")
            ax.set_xlabel("Max depth"); ax.set_ylabel("Leaves (log scale)"); ax.set_title("Tree size grows with depth"); show(fig)
        st.info(f"Best test accuracy in this scan: **{dc.test.max():.2f}%** at depth **{int(dc.loc[dc.test.idxmax(), 'depth'])}**.")
    with t5:
        row = prediction_form("j48", NUM_FEATURES)
        res = model.predict(row)[0]; pr = model.predict_proba(row)[0]
        (st.success if res else st.error)(f"Prediction: **{CLASS_NAMES[res]}** (leaf probability of placement {pr[1] * 100:.1f}%)")
        st.markdown("**Decision path taken:**")
        for i, s in enumerate(tree_path(model, row, [LABELS[f] for f in NUM_FEATURES]), 1):
            st.write(f"{i}. {s}")


def nb_params(model):
    var = getattr(model, "var_", None)
    if var is None:
        var = model.sigma_
    return model.theta_, var


def render_nb():
    page_header("Naive Bayes", "A probabilistic classifier: every feature votes independently through a bell curve per class.")
    r = train_cls("nb")
    m, model = r["metrics"], r["model"]
    k = st.columns(5)
    kpi(k[0], "Accuracy", f"{m['Accuracy (%)']:.2f}%", "", NB_C); kpi(k[1], "Precision", f"{m['Precision (%)']:.2f}%")
    kpi(k[2], "Recall", f"{m['Recall (%)']:.2f}%"); kpi(k[3], "F1 score", f"{m['F1 Score (%)']:.2f}%")
    kpi(k[4], "Training time", f"{r['fit_time'] * 1000:.1f} ms", "one pass over the data", PLACED_C)
    mu, var = nb_params(model)
    t1, t2, t3, t4 = st.tabs(["🔔 Class-conditional curves", "📈 Performance", "🧮 Parameters & priors", "🎯 Predict & explain"])
    with t1:
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        Xtr, ytr = r["Xtr"], r["ytr"]
        for i, (ax, f) in enumerate(zip(axes.ravel(), NUM_FEATURES)):
            xs = np.linspace(Xtr[f].min(), Xtr[f].max(), 250)
            for c, col in [(0, NOT_C), (1, PLACED_C)]:
                ax.hist(Xtr[ytr == c][f], bins=25, density=True, alpha=.18, color=col)
                ax.plot(xs, np.exp(-(xs - mu[c, i]) ** 2 / (2 * var[c, i])) / np.sqrt(2 * np.pi * var[c, i]), color=col, lw=2,
                        label=CLASS_NAMES[c])
            ax.set_title(LABELS[f]); ax.set_yticks([])
            if i == 0:
                ax.legend()
        fig.suptitle("What Naive Bayes learned: one Gaussian per class per feature (bars = real data)", fontweight="bold", fontsize=11)
        show(fig)
        st.caption("Where the two curves are far apart, the feature is informative. Where the bars do not follow the curve (e.g. Backlogs), the Gaussian assumption is weak.")
    with t2:
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5, 4)); plot_cm(ax, r["yte"], r["pred"], NB_C, "Confusion matrix"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(r["yte"], r["proba"])
            ax.plot(fpr, tpr, color=NB_C, lw=2, label=f"Naive Bayes (AUC = {m['ROC-AUC']:.3f})"); ax.fill_between(fpr, tpr, alpha=.1, color=NB_C)
            ax.plot([0, 1], [0, 1], "--", color=GREY); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC curve"); ax.legend(); show(fig)
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            for c, col in [(0, NOT_C), (1, PLACED_C)]:
                ax.hist(r["proba"][r["yte"].to_numpy() == c], bins=30, alpha=.6, color=col, label=f"actually {CLASS_NAMES[c]}")
            ax.set_xlabel("Predicted P(Placed)"); ax.set_ylabel("Students"); ax.set_title("Posterior probability spread"); ax.legend(); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            bins = np.linspace(0, 1, 11); idx = np.digitize(r["proba"], bins) - 1
            xs, ys = [], []
            for bi in range(10):
                sel = idx == bi
                if sel.sum() > 20:
                    xs.append(r["proba"][sel].mean()); ys.append(r["yte"].to_numpy()[sel].mean())
            ax.plot([0, 1], [0, 1], "--", color=GREY, label="perfectly calibrated"); ax.plot(xs, ys, marker="o", color=NB_C, label="Naive Bayes")
            ax.set_xlabel("Predicted probability"); ax.set_ylabel("Actual placement rate"); ax.set_title("Calibration curve"); ax.legend(); show(fig)
    with t3:
        a, b = st.columns([1, 2])
        with a:
            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            ax.bar(CLASS_NAMES, model.class_prior_, color=[NOT_C, PLACED_C])
            for i, v in enumerate(model.class_prior_):
                ax.text(i, v + .01, f"{v:.3f}", ha="center")
            ax.set_ylim(0, 1); ax.set_title("Class priors P(C)"); show(fig)
        with b:
            mdf = pd.DataFrame(mu, index=CLASS_NAMES, columns=[LABELS[f] for f in NUM_FEATURES]).T
            norm = (mdf - mdf.min()) / (mdf.max() - mdf.min())
            fig, ax = plt.subplots(figsize=(7, 3.6))
            norm.plot(kind="bar", ax=ax, color=[NOT_C, PLACED_C], width=.75)
            ax.set_ylabel("Normalised mean"); ax.set_title("Mean of each feature per class"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); show(fig)
        tbl = pd.concat({"Mean": pd.DataFrame(mu, index=CLASS_NAMES, columns=NUM_FEATURES).T,
                         "Std dev": pd.DataFrame(np.sqrt(var), index=CLASS_NAMES, columns=NUM_FEATURES).T}, axis=1).round(2)
        tbl.columns = [f"{a} ({b})" for a, b in tbl.columns]
        st.dataframe(tbl)
    with t4:
        row = prediction_form("nb", NUM_FEATURES)
        pr = model.predict_proba(row)[0]
        (st.success if pr[1] >= .5 else st.error)(f"Prediction: **{'Placed' if pr[1] >= .5 else 'Not Placed'}** ({pr[1] * 100:.1f}% probability of placement)")
        x = row.to_numpy()[0]
        ll = -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)
        contrib = ll[1] - ll[0]
        prior = np.log(model.class_prior_[1] / model.class_prior_[0])
        names = ["Prior"] + [LABELS[f] for f in NUM_FEATURES]
        vals = np.concatenate([[prior], contrib])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(names[::-1], vals[::-1], color=[PLACED_C if v > 0 else NOT_C for v in vals[::-1]])
        ax.axvline(0, color=INK, lw=.8)
        ax.set_xlabel("log-odds contribution  (right = pushes to Placed, left = pushes to Not Placed)")
        ax.set_title(f"Why this prediction?  total log-odds = {vals.sum():+.2f}"); show(fig)


# ======================================================================== 4-way comparison helpers
REG_C, CLU_C = "#9B5DE5", "#17A589"
CMP_MODELS = [("J48 Decision Tree", J48_C), ("Naive Bayes", NB_C), ("Linear Regression", REG_C), ("K-Means Clustering", CLU_C)]


@st.cache_resource(show_spinner="Training linear regression on the 0/1 placement target ...")
def train_lin_cls():
    """Linear regression used as a classifier: fit Placed (0/1), predict a score, threshold at 0.5."""
    data, _ = get_clean()
    Xtr, Xte, ytr, yte = split_cls(data)
    t0 = time.perf_counter(); model = LinearRegression().fit(Xtr, ytr); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); score = model.predict(Xte); pred = (score >= .5).astype(int); pred_t = time.perf_counter() - t0
    X_all, y_all = data[NUM_FEATURES], (data["Placed"] == "Placed").astype(int)
    cv = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X_all, y_all):
        m = LinearRegression().fit(X_all.iloc[a], y_all.iloc[a])
        cv.append(accuracy_score(y_all.iloc[b], (m.predict(X_all.iloc[b]) >= .5).astype(int)))
    return dict(model=model, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, pred=pred, proba=score, cv=np.array(cv) * 100,
                fit_time=fit_t, pred_time=pred_t, r2=r2_score(yte, score),
                mae=mean_absolute_error(yte, score), metrics=cls_metrics(yte, pred, score))


def km_score(kd, X):
    """Closeness to the 'placed' cluster (0..1): a soft score built from centroid distances."""
    d = kd["km"].transform(kd["scaler"].transform(X))
    return d[:, 1 - kd["placed_c"]] / (d.sum(axis=1) + 1e-12)


@st.cache_resource(show_spinner="Clustering students with K-Means (k = 2) ...")
def train_km_cls():
    """Cluster-then-label: K-Means (k=2) on features only; each cluster is named after its majority class in the training set."""
    data, _ = get_clean()
    Xtr, Xte, ytr, yte = split_cls(data)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    t0 = time.perf_counter(); km = KMeans(n_clusters=2, n_init=5, random_state=42).fit(Xtr_s); fit_t = time.perf_counter() - t0
    lab = km.labels_; yv = ytr.values
    placed_c = int(np.argmax([yv[lab == c].mean() for c in range(2)]))
    kd = dict(km=km, scaler=sc, placed_c=placed_c)
    t0 = time.perf_counter(); pred = (km.predict(Xte_s) == placed_c).astype(int); pred_t = time.perf_counter() - t0
    score = km_score(kd, Xte)
    X_all, y_all = data[NUM_FEATURES], (data["Placed"] == "Placed").astype(int).values
    cv = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X_all, y_all):
        s2 = StandardScaler().fit(X_all.iloc[a]); k2 = KMeans(n_clusters=2, n_init=3, random_state=42).fit(s2.transform(X_all.iloc[a]))
        pc = int(np.argmax([y_all[a][k2.labels_ == c].mean() for c in range(2)]))
        cv.append(accuracy_score(y_all[b], (k2.predict(s2.transform(X_all.iloc[b])) == pc).astype(int)))
    sil = silhouette_score(Xte_s, km.predict(Xte_s), sample_size=5000, random_state=42)
    return dict(kd=kd, Xtr=Xtr, Xte=Xte, Xte_s=Xte_s, ytr=ytr, yte=yte, pred=pred, proba=score, cv=np.array(cv) * 100,
                fit_time=fit_t, pred_time=pred_t, sil=sil, inertia=km.inertia_,
                metrics=cls_metrics(yte, pred, score), **kd)


def draw_reg_schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)
    for i, f in enumerate(["CGPA", "Aptitude", "Internships", "Backlogs"]):
        y = 5 - i * 1.15
        _box(ax, 1.2, y, f, GREY, fs=8)
        _arrow(ax, (2.1, y), (4.1, 3.0), f"w{i + 1}")
    _box(ax, 5, 3.0, "Σ wᵢ·xᵢ + b", REG_C, fs=9)
    _arrow(ax, (5.9, 3.0), (7.0, 3.0))
    _box(ax, 7.6, 3.0, "score ≥ 0.5 ?", J48_C, fs=8.5)
    _box(ax, 9.0, 4.6, "Placed", PLACED_C, fs=8); _box(ax, 9.0, 1.4, "Not\nPlaced", NOT_C, fs=8)
    _arrow(ax, (8.0, 3.3), (8.8, 4.3)); _arrow(ax, (8.0, 2.7), (8.8, 1.7))
    ax.text(5, .5, r"$\hat{y} = b + w_1x_1 + w_2x_2 + \dots + w_nx_n$", ha="center", fontsize=10.5)
    ax.set_title("Linear Regression: weighted sum of features, then a cut-off", fontsize=10)


def draw_km_schematic(ax):
    rng = np.random.default_rng(3)
    a = rng.normal([3, 3.6], .75, (45, 2)); b = rng.normal([7, 2.4], .75, (45, 2))
    ax.scatter(*a.T, s=22, color=CLU_C, alpha=.65); ax.scatter(*b.T, s=22, color=NB_C, alpha=.65)
    ax.scatter([3, 7], [3.6, 2.4], marker="X", s=220, c="white", edgecolors=INK, linewidths=1.8, zorder=5)
    ax.plot([5, 5], [.6, 5.4], ls="--", color=GREY, lw=1.2)
    ax.text(3, 5.2, "Cluster 0", ha="center", fontsize=9, color=CLU_C, fontweight="bold")
    ax.text(7, 5.2, "Cluster 1", ha="center", fontsize=9, color=NB_C, fontweight="bold")
    ax.text(5, .15, "1) pick k centroids   2) assign nearest   3) move centroid to mean   4) repeat", ha="center", fontsize=7.5, color=GREY)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)
    ax.set_title("K-Means: group students around the nearest centre (no labels used)", fontsize=10)


def draw_nb_tree(ax, model):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off"); ax.grid(False)
    mu, var = nb_params(model); prior = model.class_prior_
    sep = np.abs(mu[1] - mu[0]) / np.sqrt((var[0] + var[1]) / 2); top = list(np.argsort(-sep)[:3])
    _box(ax, 5, 5.6, "Student", INK, fs=9)
    for c, x, col in [(0, 2.5, NOT_C), (1, 7.5, PLACED_C)]:
        _box(ax, x, 3.7, f"{CLASS_NAMES[c]}\nprior = {prior[c]:.2f}", col, fs=8.5)
        _arrow(ax, (5, 5.25), (x, 4.2))
        for dx, f in zip([-1.75, 0, 1.75], top):
            fx = x + dx
            _box(ax, fx, 1.5, f"{LABELS[NUM_FEATURES[f]]}\nμ={mu[c, f]:.1f}\nσ={np.sqrt(var[c, f]):.1f}", GREY, fs=6.8)
            _arrow(ax, (x, 3.25), (fx, 2.05))
    ax.set_title("Naive Bayes as a one-level probability tree (top 3 features)", fontsize=10)


# ======================================================================== 4-way comparison page
def render_comparison():
    page_header("⚖️ Comparison: J48 vs Naive Bayes vs Regression vs Clustering",
                "All four techniques on one page: definitions, data, metrics, graphs, diagrams and trees. Same cleaned data, same 80/20 split, same random seed.")
    with st.expander("Model settings used in this comparison", expanded=False):
        a, b = st.columns(2)
        a.slider("J48 maximum depth", 2, 12, 5, key="cmp_depth")
        b.slider("J48 minimum samples per leaf", 1, 200, 20, key="cmp_leaf")
        st.slider("Regression-tree depth (for the tree diagram)", 2, 8, 4, key="cmp_rdepth")
    depth = st.session_state.get("cmp_depth", 5); leaf = st.session_state.get("cmp_leaf", 20)
    rdepth = st.session_state.get("cmp_rdepth", 4)
    J, N = train_cls("j48", depth, leaf, "entropy"), train_cls("nb")
    L, K = train_lin_cls(), train_km_cls()
    R = train_reg(rdepth)
    M = [J, N, L, K]; names = [m[0] for m in CMP_MODELS]; cols = [m[1] for m in CMP_MODELS]
    metrics = [m["metrics"] for m in M]

    st.info("**How four different techniques are made comparable.** J48 and Naive Bayes are classifiers. "
            "**Linear Regression** is fitted on the 0/1 placement target and a score ≥ 0.5 means *Placed*. "
            "**K-Means** (k = 2) never sees the labels while clustering; afterwards each cluster is named after the majority class of its training students. "
            "Accuracy, precision, recall and F1 are then computed the same way for all four on the same test students.")

    # ---------------------------------------------------------------- 1 definitions
    st.markdown("## 1. Definitions")
    defs = [
        (J48_C, "J48 Decision Tree", "Classification (supervised)",
         "J48 is Weka's implementation of C4.5. It splits the data top-down on the attribute with the best gain ratio and ends in leaves that name a class. "
         "A student is classified by following yes/no answers from the root to a leaf.",
         "Entropy H(S) = − Σ pᵢ log₂ pᵢ<br>Gain(S,A) = H(S) − Σ (|Sᵥ|/|S|)·H(Sᵥ)",
         "Readable IF-THEN rules, captures interactions. Overfits if too deep."),
        (NB_C, "Naive Bayes", "Classification (supervised)",
         "Applies Bayes' theorem and assumes features are independent given the class. It multiplies the class prior by each feature's likelihood "
         "and predicts the class with the larger posterior (Gaussian variant here).",
         "P(C|x) = P(x|C)·P(C) / P(x)<br>P(x|C) = Π P(xᵢ|C)",
         "Very fast, needs little data, gives probabilities. Independence assumption is often wrong."),
        (REG_C, "Linear Regression", "Regression (supervised, numeric output)",
         "Fits a straight-line (hyperplane) relationship between the features and a numeric target by least squares. "
         "Here it predicts the 0/1 placement score (and, in the Regression module, the salary package).",
         "ŷ = b + w₁x₁ + … + wₙxₙ<br>minimise Σ (yᵢ − ŷᵢ)²",
         "Simple, interpretable coefficients. Cannot bend to non-linear patterns; scores can fall outside 0..1."),
        (CLU_C, "K-Means Clustering", "Clustering (unsupervised, no labels)",
         "Groups students into k clusters so that students in a cluster are close to its centroid. It discovers structure on its own; "
         "labels are used here only afterwards to name the clusters.",
         "minimise Σₖ Σ ‖x − μₖ‖²<br>Silhouette s = (b − a) / max(a, b)",
         "Finds natural profiles without labels. Needs k, assumes round clusters, ignores the target so accuracy is lower.")]
    for row in (defs[:2], defs[2:]):
        c = st.columns(2)
        for col, (color, title, kind, text, formula, note) in zip(c, row):
            col.markdown(f"""<div class='defcard' style='border-top-color:{color}'><h4>{title}</h4>
<p><i>{kind}</i></p><p><b>Definition.</b> {text}</p><div class='formula'>{formula}</div><p><b>In short:</b> {note}</p></div>""", unsafe_allow_html=True)
    st.markdown("#### Side-by-side properties")
    st.dataframe(pd.DataFrame([
        ["Task type", "Classification", "Classification", "Regression (used as classifier here)", "Clustering"],
        ["Learning style", "Supervised", "Supervised", "Supervised", "Unsupervised"],
        ["Output", "Class label", "Class + posterior probability", "Numeric score", "Cluster number"],
        ["Core idea", "Split on highest gain ratio", "Prior × per-feature likelihoods", "Least-squares weighted sum", "Nearest centroid, iterate"],
        ["Key assumption", "None on distribution", "Independent features, Gaussian", "Linear relationship", "Compact, round clusters; k known"],
        ["Feature interactions", "Yes", "No", "No (unless added)", "Through distance only"],
        ["Needs scaling", "No", "No", "Helpful", "Yes"],
        ["Overfitting risk", "High unless pruned", "Low", "Low", "Low"],
        ["Interpretability", "Very high (tree/rules)", "Medium", "High (coefficients)", "Medium (profiles)"],
        ["Typical metrics", "Accuracy, F1, AUC", "Accuracy, F1, AUC", "R², MAE, RMSE", "Silhouette, inertia"],
    ], columns=["Aspect"] + names), hide_index=True)

    # ---------------------------------------------------------------- 2 data
    st.markdown("---"); st.markdown("## 2. Data used")
    data, _ = get_clean()
    nrows = [f"{len(m['Xtr']):,} train / {len(m['Xte']):,} test" for m in M]
    st.dataframe(pd.DataFrame([
        ["Input features", "9 numeric: " + ", ".join(LABELS[f] for f in NUM_FEATURES[:4]) + " …"] + ["same 9 features"] * 2 + ["same 9 features (standardised)"],
        ["Target / label", "Placed (0/1)", "Placed (0/1)", "Placed (0/1) as a number; salary in the Regression module", "None while clustering; Placed used only to name clusters"],
        ["Rows (cleaned data)"] + nrows,
        ["Labels needed for training?", "Yes", "Yes", "Yes", "No"],
        ["Pre-processing", "None", "None", "None (scaling optional)", "StandardScaler (distance based)"],
        ["Settings", f"depth {depth}, min leaf {leaf}, entropy", "GaussianNB defaults", "Ordinary least squares", "k = 2, n_init = 5"],
    ], columns=[""] + names), hide_index=True)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        pr = (data.Placed == "Placed").mean() * 100
        ax.bar(["Placed", "Not Placed"], [pr, 100 - pr], color=[PLACED_C, NOT_C])
        for i, v in enumerate([pr, 100 - pr]):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylim(0, 100); ax.set_ylabel("Students (%)"); ax.set_title("Class balance of the data all four models see"); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for c_, lab, col in [(0, "Not Placed", NOT_C), (1, "Placed", PLACED_C)]:
            ax.hist(data.loc[(data.Placed == "Placed") == bool(c_), "CGPA"], bins=35, alpha=.55, color=col, label=lab)
        ax.set_xlabel("CGPA"); ax.set_ylabel("Students"); ax.set_title("CGPA by outcome: the main signal in the data"); ax.legend(); show(fig)

    # ---------------------------------------------------------------- 3 metrics
    st.markdown("---"); st.markdown("## 3. Accuracy, precision, recall and F1 score")
    c = st.columns(4)
    for col, nm, colr, m in zip(c, names, cols, metrics):
        kpi(col, nm, f"{m['Accuracy (%)']:.2f}%", f"F1 {m['F1 Score (%)']:.2f}%", colr)
    keys = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    table = pd.DataFrame(metrics, index=names)[keys + ["Specificity (%)", "ROC-AUC"]]
    table["5-fold CV acc. (%)"] = [m["cv"].mean() for m in M]
    table["CV std (%)"] = [m["cv"].std() for m in M]
    table["Train time (ms)"] = [m["fit_time"] * 1000 for m in M]
    table["Predict time (ms)"] = [m["pred_time"] * 1000 for m in M]
    st.dataframe(table.style.format("{:.2f}").highlight_max(axis=0, subset=keys + ["Specificity (%)", "ROC-AUC", "5-fold CV acc. (%)"], color="#CDEEE8")
                 .highlight_min(axis=0, subset=["Train time (ms)", "Predict time (ms)", "CV std (%)"], color="#CDEEE8"))
    st.caption("Green cells mark the best value in each column.")
    a, b = st.columns([3, 2])
    with a:
        fig, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(len(keys)); w = .2
        for i, (nm, colr, m) in enumerate(zip(names, cols, metrics)):
            vals = [m[k] for k in keys]
            ax.bar(x + (i - 1.5) * w, vals, w, color=colr, label=nm)
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 1.5) * w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([k.replace(" (%)", "") for k in keys]); ax.set_ylim(0, 110); ax.legend(ncol=2, loc="upper center")
        ax.set_title("Accuracy, precision, recall, F1 for all four"); show(fig)
    with b:
        st.markdown("**Each model's own native yardstick**")
        st.dataframe(pd.DataFrame([
            ["J48 Decision Tree", "Leaves / depth", f"{J['model'].get_n_leaves()} / {J['model'].get_depth()}"],
            ["Naive Bayes", "Class prior (Placed)", f"{N['model'].class_prior_[1]:.3f}"],
            ["Linear Regression", "R² on 0/1 target", f"{L['r2']:.3f}"],
            ["Linear Regression", "R² on salary package", f"{R['lr_m']['R² Score']:.3f}"],
            ["Linear Regression", "MAE on salary", f"₹{R['lr_m']['MAE (LPA)']:.2f} LPA"],
            ["K-Means", "Silhouette (test)", f"{K['sil']:.3f}"],
            ["K-Means", "Inertia (train)", f"{K['inertia']:,.0f}"]],
            columns=["Model", "Measure", "Value"]), hide_index=True)
    ranks = []
    for key, label in [("Accuracy (%)", "accuracy"), ("Precision (%)", "precision"), ("Recall (%)", "recall"), ("F1 Score (%)", "F1 score")]:
        v = [m[key] for m in metrics]; ranks.append(f"- **{names[int(np.argmax(v))]}** leads on {label} ({max(v):.2f}%), **{names[int(np.argmin(v))]}** is lowest ({min(v):.2f}%)")
    ft = [m["fit_time"] for m in M]; ranks.append(f"- **{names[int(np.argmin(ft))]}** trains fastest ({min(ft) * 1000:.1f} ms)")
    st.markdown("<div class='verdict'><b>Verdict</b></div>", unsafe_allow_html=True)
    st.markdown("\n".join(ranks) + "\n\nThe supervised models use the placement labels while learning, so they are expected to beat K-Means, which only groups similar students. "
                "Linear Regression is competitive when the pattern is mostly additive; the tree gains when combinations of features matter.")

    # ---------------------------------------------------------------- 4 graphs
    st.markdown("---"); st.markdown("## 4. Graphs")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, nm, colr, m in zip(axes, names, cols, M):
        plot_cm(ax, m["yte"], m["pred"], colr, nm)
    fig.suptitle("Confusion matrices", fontweight="bold"); show(fig)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4))
        for nm, colr, m in zip(names, cols, M):
            fpr, tpr, _ = roc_curve(m["yte"], m["proba"]); ax.plot(fpr, tpr, color=colr, lw=2, label=f"{nm} (AUC {m['metrics']['ROC-AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color=GREY); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves"); ax.legend(fontsize=7.5); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4))
        for nm, colr, m in zip(names, cols, M):
            p, rc, _ = precision_recall_curve(m["yte"], m["proba"]); ax.plot(rc, p, color=colr, lw=2, label=nm)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves"); ax.legend(fontsize=7.5); show(fig)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for nm, colr, m in zip(names, cols, M):
            ax.plot(range(1, 6), m["cv"], marker="o", color=colr, lw=2, label=nm)
        ax.set_xticks(range(1, 6)); ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy (%)"); ax.set_title("5-fold cross-validation stability"); ax.legend(fontsize=7.5); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.6)); tms = [m["fit_time"] * 1000 for m in M]
        ax.bar([n_.replace(" ", "\n", 1) for n_ in names], tms, color=cols); ax.set_yscale("log"); ax.set_ylabel("Training time (ms, log scale)")
        for i, v in enumerate(tms):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_title("Speed: training"); show(fig)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.scatter(R["yte"], R["lr_pred"], s=7, alpha=.3, color=REG_C); lims = [R["yte"].min(), R["yte"].max()]; ax.plot(lims, lims, "--", color=INK)
        ax.set_xlabel("Actual package (LPA)"); ax.set_ylabel("Predicted (LPA)"); ax.set_title(f"Regression: actual vs predicted salary (R² {R['lr_m']['R² Score']:.2f})"); show(fig)
    with b:
        from sklearn.decomposition import PCA as _PCA
        idx = np.random.default_rng(1).choice(len(K["Xte_s"]), 4000, replace=False)
        co = _PCA(n_components=2, random_state=42).fit(K["Xte_s"][idx]); xy = co.transform(K["Xte_s"][idx]); cen = co.transform(K["km"].cluster_centers_)
        cl = K["km"].predict(K["Xte_s"][idx]); act = K["yte"].values[idx]
        fig, axes = plt.subplots(1, 2, figsize=(7, 4.2), sharey=True)
        axes[0].scatter(xy[:, 0], xy[:, 1], c=[CLU_C if v == K["placed_c"] else NB_C for v in cl], s=5, alpha=.5); axes[0].set_title("K-Means clusters")
        axes[1].scatter(xy[:, 0], xy[:, 1], c=[PLACED_C if v else NOT_C for v in act], s=5, alpha=.5); axes[1].set_title("Actual outcome")
        for ax in axes:
            ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=170, c="white", edgecolors=INK, linewidths=1.6, zorder=5); ax.set_xlabel("PC1"); ax.grid(False)
        axes[0].set_ylabel("PC2"); fig.suptitle("Clustering vs reality (PCA view)", fontweight="bold"); show(fig)

    # ---------------------------------------------------------------- 5 diagrams
    st.markdown("---"); st.markdown("## 5. Diagrams: how each technique works")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.8)); draw_j48_schematic(ax); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.8)); draw_nb_schematic(ax); show(fig)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.8)); draw_reg_schematic(ax); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.8)); draw_km_schematic(ax); show(fig)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.4)); draw_entropy(ax, float(J["ytr"].mean())); show(fig)
    with b:
        co_ = pd.Series(L["model"].coef_, index=[LABELS[f] for f in NUM_FEATURES]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.4)); ax.barh(co_.index, co_.values, color=[PLACED_C if v > 0 else NOT_C for v in co_.values]); ax.axvline(0, color=INK, lw=.8)
        ax.set_xlabel("Change in placement score per +1 unit"); ax.set_title("Linear regression coefficients"); show(fig)

    # ---------------------------------------------------------------- 6 trees
    st.markdown("---"); st.markdown("## 6. Trees")
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(J["model"], feature_names=[LABELS[f] for f in NUM_FEATURES], class_names=CLASS_NAMES, filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("J48 decision tree (top 3 levels)"); st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(18, 6.5))
    plot_tree(R["dt"], feature_names=[LABELS[f] for f in REG_FEATS], filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("Regression tree (leaf value = average package in LPA, top 3 levels)"); st.pyplot(fig); plt.close(fig)
    a, b = st.columns(2)
    with a:
        from scipy.cluster.hierarchy import dendrogram, linkage
        sidx = np.random.default_rng(7).choice(len(K["Xte_s"]), 80, replace=False)
        Z = linkage(K["Xte_s"][sidx], method="ward")
        fig, ax = plt.subplots(figsize=(6.5, 4.2)); dendrogram(Z, truncate_mode="lastp", p=16, leaf_rotation=90, leaf_font_size=8, ax=ax, color_threshold=0.6 * Z[:, 2].max())
        ax.set_xlabel("Merged groups (number of students)"); ax.set_ylabel("Merge distance"); ax.grid(False)
        ax.set_title("Clustering tree: dendrogram of 80 students"); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6.5, 4.2)); draw_nb_tree(ax, N["model"]); show(fig)
    st.caption("Only J48 and the regression tree are true trees. K-Means has no tree itself, so a hierarchical dendrogram is drawn to show how students merge into groups. "
               "Naive Bayes is shown as a one-level probability tree: class priors, then per-feature means and spreads.")

    # ---------------------------------------------------------------- 7 predictor
    st.markdown("---"); st.markdown("## 7. Head-to-head predictor")
    st.caption("Enter one student's profile and see all four models side by side.")
    row = prediction_form("cmp", NUM_FEATURES)
    ps = [J["model"].predict_proba(row)[0, 1], N["model"].predict_proba(row)[0, 1],
          float(np.clip(L["model"].predict(row)[0], 0, 1)), float(km_score(K, row)[0])]
    c = st.columns(4)
    notes = ["P(placed)", "P(placed)", "score (cut-off 0.5)", "closeness to Placed cluster"]
    for col, nm, colr, p, nt in zip(c, names, cols, ps, notes):
        kpi(col, f"{nm} says", "Placed" if p >= .5 else "Not Placed", f"{nt} = {p * 100:.1f}%", colr)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(names[::-1], [p * 100 for p in ps][::-1], color=cols[::-1]); ax.axvline(50, color=GREY, ls="--"); ax.set_xlim(0, 100); ax.set_xlabel("Placement score (%)"); show(fig)
    with st.expander("J48 decision path for this student"):
        for i, s in enumerate(tree_path(J["model"], row, [LABELS[f] for f in NUM_FEATURES]), 1):
            st.write(f"{i}. {s}")


def render_regression():
    page_header("Regression: predicting the salary package", "Only placed students with a valid package are used. Linear Regression vs a Regression Tree.")
    depth = st.slider("Regression tree depth", 2, 12, 5, key="reg_depth")
    r = train_reg(depth)
    c = st.columns(4)
    kpi(c[0], "Training population", f"{r['n']:,}", "placed with valid package")
    kpi(c[1], "Linear R²", f"{r['lr_m']['R² Score']:.3f}", f"MAE ₹{r['lr_m']['MAE (LPA)']:.2f} LPA", J48_C)
    kpi(c[2], "Tree R²", f"{r['dt_m']['R² Score']:.3f}", f"MAE ₹{r['dt_m']['MAE (LPA)']:.2f} LPA", NB_C)
    kpi(c[3], "Better model", "Linear" if r["lr_m"]["R² Score"] >= r["dt_m"]["R² Score"] else "Tree", "by R²", PLACED_C)
    t1, t2, t3, t4 = st.tabs(["📈 Accuracy of predictions", "🌳 Regression tree", "📐 Coefficients", "🎯 Predict"])
    with t1:
        st.dataframe(pd.DataFrame({"Linear Regression": r["lr_m"], "Decision Tree Regressor": r["dt_m"]}).T.round(3))
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for k, name, col in [("lr", "Linear", J48_C), ("dt", "Tree", NB_C)]:
                ax.scatter(r["yte"], r[k + "_pred"], s=7, alpha=.25, color=col, label=name)
            lims = [r["yte"].min(), r["yte"].max()]; ax.plot(lims, lims, "--", color=INK, label="perfect")
            ax.set_xlabel("Actual package (LPA)"); ax.set_ylabel("Predicted (LPA)"); ax.set_title("Actual vs predicted"); ax.legend(); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.hist(r["yte"] - r["lr_pred"], bins=40, alpha=.6, color=J48_C, label="Linear"); ax.hist(r["yte"] - r["dt_pred"], bins=40, alpha=.6, color=NB_C, label="Tree")
            ax.axvline(0, color=INK, lw=1); ax.set_xlabel("Residual (LPA)"); ax.set_ylabel("Students"); ax.set_title("Residual distribution"); ax.legend(); show(fig)
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            em = pd.DataFrame({"Linear": r["lr_m"], "Tree": r["dt_m"]}).T[["MAE (LPA)", "RMSE (LPA)"]]
            em.plot(kind="bar", ax=ax, color=[NOT_C, NB_C], width=.7); plt.setp(ax.get_xticklabels(), rotation=0); ax.set_ylabel("Error (LPA)"); ax.set_title("Error comparison"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ds = list(range(1, 13)); r2s = [r2_score(r["yte"], DecisionTreeRegressor(max_depth=d, random_state=42).fit(r["Xtr"], r["ytr"]).predict(r["Xte"])) for d in ds]
            ax.plot(ds, r2s, marker="o", color=NB_C); ax.axhline(r["lr_m"]["R² Score"], color=J48_C, ls="--", label="Linear R²")
            ax.set_xlabel("Tree depth"); ax.set_ylabel("Test R²"); ax.set_title("Tree R² vs depth"); ax.legend(); show(fig)
    with t2:
        sd = st.slider("Levels to draw", 1, 4, 3, key="reg_show")
        fig, ax = plt.subplots(figsize=({1: 8, 2: 12, 3: 18, 4: 26}[sd], 3 + 1.6 * sd))
        plot_tree(r["dt"], feature_names=[LABELS[f] for f in REG_FEATS], filled=True, rounded=True, max_depth=sd, fontsize=8, ax=ax)
        ax.set_title("Regression tree (leaf value = average package in LPA)"); st.pyplot(fig); plt.close(fig)
    with t3:
        co = pd.Series(r["lr"].coef_, index=[LABELS[f] for f in REG_FEATS]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(co.index, co.values, color=[PLACED_C if v > 0 else NOT_C for v in co.values]); ax.axvline(0, color=INK, lw=.8)
        ax.set_xlabel("Change in package (LPA) per +1 unit"); ax.set_title("Linear regression coefficients"); show(fig)
        st.write(f"Intercept: **{r['lr'].intercept_:.2f} LPA**")
    with t4:
        row = prediction_form("reg", REG_FEATS)
        st.success(f"Linear: **₹{r['lr'].predict(row)[0]:.2f} LPA**   |   Tree: **₹{r['dt'].predict(row)[0]:.2f} LPA**")


def render_rules():
    page_header("Association rules (Apriori)", "Which combinations of student traits go together with being placed or not placed? Apriori is implemented from scratch.")
    a, b = st.columns(2)
    ms = a.slider("Minimum support", .02, .30, .08, .01, key="ar_sup")
    mc = b.slider("Minimum confidence", .40, .95, .60, .05, key="ar_conf")
    items, rules = get_rules(ms, mc)
    c = st.columns(3)
    kpi(c[0], "Frequent itemsets", f"{len(items):,}"); kpi(c[1], "Rules found", f"{len(rules):,}", "", J48_C)
    kpi(c[2], "Max lift", f"{rules.lift.max():.2f}" if len(rules) else "n/a", "", NB_C)
    if not len(rules):
        st.warning("No rules at these thresholds. Lower the support or confidence."); return
    t1, t2, t3 = st.tabs(["📊 Charts", "📋 Rule table", "🎯 Rules about placement"])
    with t1:
        x, y = st.columns(2)
        with x:
            top = items.sort_values("support", ascending=False).head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.2, 5)); ax.barh(top.itemset.str.replace("_", "=").str.slice(0, 48), top.support, color=PLACED_C)
            ax.set_xlabel("Support"); ax.set_title("Top frequent itemsets"); show(fig)
        with y:
            fig, ax = plt.subplots(figsize=(6.2, 5))
            sc = ax.scatter(rules.support, rules.confidence, c=rules.lift, s=20 + 60 * rules.lift, cmap="viridis", alpha=.75)
            fig.colorbar(sc, ax=ax, label="lift"); ax.set_xlabel("Support"); ax.set_ylabel("Confidence"); ax.set_title("Support vs confidence (colour/size = lift)"); show(fig)
        top = rules.sort_values("lift", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 4.8))
        lab = (top.antecedents.str.replace("_", "=") + "  ⇒  " + top.consequents.str.replace("_", "=")).str.slice(0, 95)
        ax.barh(lab, top.lift, color=J48_C); ax.axvline(1, color=NOT_C, ls="--", lw=1); ax.set_xlabel("Lift (> 1 means a real positive association)")
        ax.set_title("Strongest rules by lift"); show(fig)
    with t2:
        st.dataframe(rules.sort_values("lift", ascending=False).head(50).round(3), hide_index=True)
    with t3:
        for tag, col in [("Status_Placed", PLACED_C), ("Status_Not Placed", NOT_C)]:
            sub = rules[rules.consequents == tag].sort_values("lift", ascending=False).head(8)
            st.markdown(f"**Rules leading to `{tag.replace('_', ' = ')}`**")
            if len(sub):
                fig, ax = plt.subplots(figsize=(10, 3.2))
                ax.barh((sub.antecedents.str.replace("_", "=")).str.slice(0, 80).iloc[::-1], sub.confidence.iloc[::-1], color=col)
                ax.set_xlim(0, 1); ax.set_xlabel("Confidence"); show(fig)
            else:
                st.caption("No rule at the current thresholds.")


def render_clustering():
    page_header("K-Means clustering", "Unsupervised grouping of students into profiles. No labels are used.")
    feats = tuple(st.multiselect("Features", NUM_FEATURES, default=REG_FEATS, key="km_feats") or REG_FEATS)
    scan = kmeans_scan(feats)
    best = int(scan.loc[scan.Silhouette.idxmax(), "k"])
    k = st.select_slider("Number of clusters (k): drag to change", options=list(range(2, 9)), value=4, key="km_k")
    st.caption(f"The silhouette score peaks at k = {best} because student attributes form one smooth spectrum rather than "
               "sharply separated groups. k = 3 to 5 gives more useful student profiles, so the slider starts at 4.")
    r = kmeans_fit(feats, k)
    c = st.columns(3)
    kpi(c[0], "Silhouette (this k)", f"{r['sil']:.3f}", f"highest silhouette is at k = {best}", J48_C)
    kpi(c[1], "Inertia", f"{r['inertia']:,.0f}", "", NB_C); kpi(c[2], "PCA variance shown", f"{r['var'].sum() * 100:.1f}%", "2 components", PLACED_C)
    t1, t2, t3 = st.tabs(["🧭 Choosing k", "🗺️ Cluster map", "👥 Cluster profiles"])
    with t1:
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(6, 3.8)); ax.plot(scan.k, scan.Inertia, marker="o", color=J48_C); ax.axvline(k, color=GREY, ls="--")
            ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow method"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(6, 3.8)); ax.plot(scan.k, scan.Silhouette, marker="o", color=PLACED_C); ax.axvline(k, color=GREY, ls="--")
            ax.set_xlabel("k"); ax.set_ylabel("Silhouette"); ax.set_title("Silhouette score"); show(fig)
        st.dataframe(scan.set_index("k").round(3))
    data, _ = get_clean()
    prof = data.assign(Cluster=r["labels"])
    with t2:
        cmap = plt.get_cmap("tab10"); fig, ax = plt.subplots(figsize=(8, 5.4))
        idx = np.random.default_rng(1).choice(len(r["coords"]), 7000, replace=False)
        ax.scatter(r["coords"][idx, 0], r["coords"][idx, 1], c=[cmap(v) for v in r["labels"][idx]], s=6, alpha=.5)
        ax.scatter(r["centers"][:, 0], r["centers"][:, 1], marker="X", s=220, c="white", edgecolors=INK, linewidths=1.8, zorder=5)
        for i, (x, y) in enumerate(r["centers"]):
            ax.text(x, y, str(i), ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)
        ax.set_xlabel(f"PC1 ({r['var'][0] * 100:.0f}%)"); ax.set_ylabel(f"PC2 ({r['var'][1] * 100:.0f}%)"); ax.set_title("Clusters projected with PCA (X = centroid)"); show(fig)
    with t3:
        a, b = st.columns(2)
        with a:
            means = prof.groupby("Cluster")[list(feats)].mean(); z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
            fig, ax = plt.subplots(figsize=(6, 4)); im = ax.imshow(z.T, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
            ax.set_xticks(range(k)); ax.set_xticklabels([f"C{i}" for i in range(k)]); ax.set_yticks(range(len(feats))); ax.set_yticklabels([LABELS[f] for f in feats]); ax.grid(False)
            for (i, j2), v in np.ndenumerate(means.T.values):
                ax.text(j2, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=.8, label="z-score"); ax.set_title("Cluster profile heat-map (numbers = real means)"); show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(6, 4))
            rate = prof.groupby("Cluster").Placed.apply(lambda s: (s == "Placed").mean() * 100)
            ax.bar([f"C{i}" for i in rate.index], rate.values, color=[plt.get_cmap("tab10")(i) for i in rate.index])
            for i, v in enumerate(rate.values):
                ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
            ax.set_ylim(0, 105); ax.set_ylabel("Placement rate (%)"); ax.set_title("Placement rate per cluster (not used for clustering)"); show(fig)
        size = prof.Cluster.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 2.8)); ax.bar([f"C{i}" for i in size.index], size.values, color=[plt.get_cmap("tab10")(i) for i in size.index])
        ax.set_ylabel("Students"); ax.set_title("Cluster sizes"); show(fig)
        st.dataframe(means.round(2).assign(Students=size.values, Placement_pct=rate.round(1).values))


def render_summary():
    page_header("Project summary", "Headline result of every technique, on one page.")
    j, n = train_cls("j48", 5, 20, "entropy"), train_cls("nb")
    r = train_reg(5)
    scan = kmeans_scan(REG_FEATS); best = int(scan.loc[scan.Silhouette.idxmax(), "k"])
    items, rules = get_rules(.08, .60)
    summ = pd.DataFrame([
        ["J48 Decision Tree", "Classification", "Accuracy", f"{j['metrics']['Accuracy (%)']:.2f}%", f"F1 {j['metrics']['F1 Score (%)']:.2f}%, AUC {j['metrics']['ROC-AUC']:.3f}"],
        ["Naive Bayes", "Classification", "Accuracy", f"{n['metrics']['Accuracy (%)']:.2f}%", f"F1 {n['metrics']['F1 Score (%)']:.2f}%, AUC {n['metrics']['ROC-AUC']:.3f}"],
        ["Linear Regression", "Regression", "R²", f"{r['lr_m']['R² Score']:.3f}", f"MAE {r['lr_m']['MAE (LPA)']:.2f} LPA"],
        ["Decision Tree Regressor", "Regression", "R²", f"{r['dt_m']['R² Score']:.3f}", f"MAE {r['dt_m']['MAE (LPA)']:.2f} LPA"],
        ["K-Means", "Clustering", "Best silhouette", f"{scan.Silhouette.max():.3f}", f"best k = {best}"],
        ["Apriori", "Association rules", "Rules found", str(len(rules)), f"{len(items)} frequent itemsets"]],
        columns=["Technique", "Task", "Primary metric", "Score", "Additional"])
    st.dataframe(summ, hide_index=True)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6.2, 4))
        names = ["J48\naccuracy", "NB\naccuracy", "Linear\nR²", "Tree\nR²"]
        vals = [j["metrics"]["Accuracy (%)"] / 100, n["metrics"]["Accuracy (%)"] / 100, r["lr_m"]["R² Score"], r["dt_m"]["R² Score"]]
        bars = ax.bar(names, vals, color=[J48_C, NB_C, PLACED_C, "#9B5DE5"])
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, max(v, 0) + .015, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_title("Primary scores (0 to 1)"); show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6.2, 4)); ax.plot(scan.k, scan.Silhouette, marker="o", color=PLACED_C)
        ax.set_xlabel("k"); ax.set_ylabel("Silhouette"); ax.set_title("K-Means: silhouette by k"); show(fig)
    st.download_button("⬇ Download summary CSV", summ.to_csv(index=False).encode(), "project_summary.csv", "text/csv")


# ======================================================================== navigation
MODULES = {"🏠  Dashboard": render_dashboard,
           "⚖️  Compare: J48, NB, Regression, Clustering": render_comparison,
           "🗄️  Dataset (60,000)": render_dataset,
           "🧹  Data cleaning": render_cleaning,
           "🌳  J48 Decision Tree": render_j48,
           "📊  Naive Bayes": render_nb,
           "📈  Regression": render_regression,
           "🔗  Association rules": render_rules,
           "🎯  Clustering": render_clustering,
           "📋  Project summary": render_summary}


def main():
    with st.sidebar:
        st.markdown("<div class='side-brand'>Placement<br>Mining Lab</div><div class='side-sub'>Data Warehousing &amp; Mining project</div>", unsafe_allow_html=True)
        choice = st.radio("Module", list(MODULES), label_visibility="collapsed", key="nav")
        st.markdown("<div class='side-note'>60,000 raw student records, cleaned live in the app. "
                    "Every model shares one 80/20 split (random_state = 42).</div>", unsafe_allow_html=True)
    st.markdown("""<div class='hero'><div class='hero-title'>Student Placement Prediction &amp; Data Mining System</div>
<div class='hero-sub'>From a messy 60,000-row database to models you can compare.</div>
<div class='chips'><span>Raw records</span><span>Cleaning</span><span>Classification</span><span>Regression</span><span>Association rules</span><span>Clustering</span><span>4-way model comparison</span></div></div>""",
                unsafe_allow_html=True)
    MODULES[choice]()


main()