"""Data-cleaning pipeline, Apriori and small helpers shared by the Streamlit app."""
import re
from itertools import combinations
import numpy as np
import pandas as pd

NUM_FEATURES = ["CGPA", "Tenth_Percent", "Twelfth_Percent", "Backlogs", "Internships",
                "Projects", "Certifications", "Communication_Score", "Aptitude_Score"]
CAT_COLS = ["Gender", "Branch", "College_Tier"]
NUMERIC_COLS = ["Age"] + NUM_FEATURES + ["Package_LPA"]
COUNT_COLS = ["Age", "Backlogs", "Internships", "Projects", "Certifications"]

VALID_RANGES = {"Age": (17, 40), "CGPA": (3, 10), "Tenth_Percent": (33, 100), "Twelfth_Percent": (33, 100),
                "Backlogs": (0, 15), "Internships": (0, 12), "Projects": (0, 15), "Certifications": (0, 15),
                "Communication_Score": (0, 100), "Aptitude_Score": (0, 100), "Package_LPA": (1, 100)}


def _key(v):
    return re.sub(r"[^a-z0-9&]", "", str(v).lower())


BRANCH_LOOKUP = {"cse": "CSE", "computerscience": "CSE", "it": "IT", "infotech": "IT",
                 "informationtechnology": "IT", "ai&ds": "AI&DS", "aids": "AI&DS", "ece": "ECE",
                 "electronics": "ECE", "eee": "EEE", "electrical": "EEE", "mechanical": "Mechanical",
                 "mech": "Mechanical", "civil": "Civil", "chemical": "Chemical", "chem": "Chemical"}
GENDER_LOOKUP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
TIER_LOOKUP = {"tier1": "Tier 1", "t1": "Tier 1", "tier2": "Tier 2", "t2": "Tier 2",
               "tier3": "Tier 3", "t3": "Tier 3"}
PLACED_LOOKUP = {"placed": "Placed", "yes": "Placed", "y": "Placed", "1": "Placed",
                 "notplaced": "Not Placed", "no": "Not Placed", "n": "Not Placed",
                 "unplaced": "Not Placed", "0": "Not Placed"}


def _standardise(series, lookup):
    return series.map(lambda v: lookup.get(_key(v), np.nan) if isinstance(v, str) else np.nan)


def clean_data(raw):
    """Full cleaning pipeline.  Returns (clean_df, log_dict)."""
    df = raw.copy()
    log = {"raw_rows": len(df)}

    # 1) standardise text categories ------------------------------------------------
    text_fixed = {}
    for col, lookup in [("Branch", BRANCH_LOOKUP), ("Gender", GENDER_LOOKUP),
                        ("College_Tier", TIER_LOOKUP), ("Placed", PLACED_LOOKUP)]:
        new = _standardise(df[col], lookup)
        text_fixed[col] = int((df[col].notna() & (df[col].astype(object) != new.astype(object))).sum())
        df[col] = new
    df["Name"] = df["Name"].map(lambda v: " ".join(v.split()).title() if isinstance(v, str) else v)
    log["text_fixed"] = text_fixed

    # 2) force numeric columns to numbers (junk tokens -> NaN) ------------------------
    tokens = {}
    for col in NUMERIC_COLS:
        before = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        tokens[col] = int(df[col].isna().sum()) - before
    log["junk_tokens"] = tokens

    # 3) impossible values -> NaN -------------------------------------------------------
    invalid = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    log["invalid_values"] = invalid

    # 4) duplicates (same Student_ID) ---------------------------------------------------
    n0 = len(df)
    df = df.drop_duplicates(subset="Student_ID", keep="first")
    log["duplicates_removed"] = n0 - len(df)

    # 5) records whose target label is unknown cannot be used -----------------------------
    n0 = len(df)
    df = df[df["Placed"].notna()].copy()
    log["label_missing_dropped"] = n0 - len(df)

    # 6) Package_LPA logic: only placed students have a package -------------------------
    contradict = df["Package_LPA"].notna() & (df["Placed"] == "Not Placed")
    log["package_contradictions"] = int(contradict.sum())
    df.loc[contradict, "Package_LPA"] = np.nan
    log["placed_without_package"] = int(((df["Placed"] == "Placed") & df["Package_LPA"].isna()).sum())

    # 7) impute: median for numbers, mode for categories ---------------------------------
    imputed = {}
    for col in [c for c in NUMERIC_COLS if c != "Package_LPA"]:
        imputed[col] = int(df[col].isna().sum())
        df[col] = df[col].fillna(df[col].median())
        if col in COUNT_COLS:
            df[col] = df[col].round().astype(int)
    for col in CAT_COLS:
        imputed[col] = int(df[col].isna().sum())
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    log["missing_imputed"] = imputed
    log["final_rows"] = len(df)
    return df.reset_index(drop=True), log


# ------------------------------------------------------------------- Apriori (from scratch)
def apriori_rules(onehot, min_support=.08, min_conf=.60, max_len=3):
    """Level-wise Apriori on a one-hot boolean DataFrame.  Returns (itemsets_df, rules_df)."""
    X = onehot.to_numpy(dtype=bool)
    cols = list(onehot.columns)
    sup = {frozenset([i]): s for i, s in enumerate(X.mean(axis=0)) if s >= min_support}
    allf, current, k = dict(sup), list(sup), 2
    while current and k <= max_len:
        cur = set(current)
        cands = set()
        for a, b in combinations(current, 2):
            u = a | b
            if len(u) == k and all(frozenset(s) in cur for s in combinations(u, k - 1)):
                cands.add(u)
        new = {}
        for c in cands:
            s = X[:, sorted(c)].all(axis=1).mean()
            if s >= min_support:
                new[c] = s
        allf.update(new)
        current, k = list(new), k + 1

    items = pd.DataFrame([{"itemset": ", ".join(cols[i] for i in sorted(s)), "size": len(s), "support": v}
                          for s, v in allf.items()])
    rows = []
    for s, v in allf.items():
        if len(s) < 2:
            continue
        for r in range(1, len(s)):
            for ante in combinations(sorted(s), r):
                a = frozenset(ante)
                c = s - a
                conf = v / allf[a]
                if conf >= min_conf:
                    rows.append({"antecedents": ", ".join(cols[i] for i in sorted(a)),
                                 "consequents": ", ".join(cols[i] for i in sorted(c)),
                                 "support": v, "confidence": conf, "lift": conf / allf[c]})
    rules = pd.DataFrame(rows, columns=["antecedents", "consequents", "support", "confidence", "lift"])
    return items, rules


# ---------------------------------------------------------------------------- tree helpers
def tree_path(model, row, names):
    """Human-readable decision path for one row (DataFrame with one row)."""
    leaf = model.apply(row)[0]
    steps = []
    for nid in model.decision_path(row).indices:
        if nid == leaf:
            break
        f, th = model.tree_.feature[nid], model.tree_.threshold[nid]
        v = row.iloc[0, f]
        steps.append(f"{names[f]} = {v:g}  ≤  {th:.2f}" if v <= th else f"{names[f]} = {v:g}  >  {th:.2f}")
    return steps


def root_split_gain(model):
    t = model.tree_
    l, r = t.children_left[0], t.children_right[0]
    n, nl, nr = t.weighted_n_node_samples[[0, l, r]]
    parent = t.impurity[0]
    children = (nl * t.impurity[l] + nr * t.impurity[r]) / n
    return {"feature": int(t.feature[0]), "threshold": float(t.threshold[0]),
            "parent": float(parent), "left": float(t.impurity[l]), "right": float(t.impurity[r]),
            "left_n": int(nl), "right_n": int(nr), "weighted_children": float(children),
            "gain": float(parent - children)}
