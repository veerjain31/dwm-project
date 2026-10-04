"""
Synthetic "Student Placement" database generator  (60,000 records, deliberately dirty)

Run once:   python data_gen.py
Creates :   student_placement_60000.csv   (raw / uncleaned)
            student_placement.db          (SQLite: students_raw + students_clean)

Dirty-data problems injected on purpose (so the cleaning module has real work to do):
  * missing values in numeric, categorical and even the target column
  * non-numeric tokens inside numeric columns ('?', 'unknown', '-')
  * impossible values (CGPA 15, percentage 150, negative backlogs, package 999 ...)
  * inconsistent spellings ('cse', 'C.S.E', 'Computer Science', 'Y', 'not placed' ...)
  * extra spaces / odd capitalisation in names
  * duplicate records (some exact, some with formatting differences)
  * contradictions (unplaced student with a salary, placed student without one)
"""
import os
import sqlite3
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "student_placement_60000.csv")
DB_PATH = os.path.join(BASE, "student_placement.db")

BRANCHES = ["CSE", "IT", "AI&DS", "ECE", "EEE", "Mechanical", "Civil", "Chemical"]
BRANCH_P = [.22, .14, .14, .15, .09, .12, .08, .06]
BRANCH_BONUS = {"CSE": .55, "IT": .45, "AI&DS": .60, "ECE": .20, "EEE": .05,
                "Mechanical": -.20, "Civil": -.35, "Chemical": -.25}
TIERS = ["Tier 1", "Tier 2", "Tier 3"]
TIER_P = [.20, .45, .35]
TIER_BONUS = {"Tier 1": .70, "Tier 2": .10, "Tier 3": -.55}

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Rohan", "Karan", "Ishaan", "Rahul", "Amit",
         "Neha", "Priya", "Ananya", "Isha", "Kavya", "Sneha", "Pooja", "Riya", "Shruti", "Meera",
         "Rajesh", "Suresh", "Vikram", "Nikhil", "Siddharth", "Harsh", "Tanvi", "Divya", "Aditi", "Nisha",
         "Manish", "Yash", "Omkar", "Swapnil", "Prathamesh", "Gauri", "Rutuja", "Sakshi", "Mayuri", "Pranav"]
LAST = ["Sharma", "Patil", "Deshmukh", "Kulkarni", "Joshi", "Jadhav", "Shinde", "More", "Pawar", "Gaikwad",
        "Iyer", "Nair", "Reddy", "Rao", "Gupta", "Verma", "Singh", "Khan", "Mehta", "Shah",
        "Desai", "Bhatt", "Chavan", "Kale", "Naik", "Bose", "Das", "Mishra", "Yadav", "Thakur",
        "Menon", "Pillai", "Agarwal", "Kapoor", "Malhotra", "Bhosale", "Salunkhe", "Phadke", "Ghule", "Wagh"]

BRANCH_VARIANTS = {
    "CSE": ["cse", " CSE", "CSE ", "C.S.E", "Computer Science"],
    "IT": ["it", "I.T.", "IT ", "Info Tech", "Information Technology"],
    "AI&DS": ["ai&ds", "AI & DS", "AIDS", "AI&DS "],
    "ECE": ["ece", "E.C.E", "Electronics", "ECE "],
    "EEE": ["eee", "Electrical", "EEE "],
    "Mechanical": ["mechanical", "Mech", "MECHANICAL", " Mechanical"],
    "Civil": ["civil", "CIVIL", "Civil "],
    "Chemical": ["chemical", "Chem", "CHEMICAL"],
}
GENDER_VARIANTS = {"Male": ["M", "male", "MALE", " Male"], "Female": ["F", "female", "FEMALE", "Female "]}
TIER_VARIANTS = {"Tier 1": ["tier 1", "T1", "Tier-1", "Tier1", "TIER 1"],
                 "Tier 2": ["tier 2", "T2", "Tier-2", "Tier2", "TIER 2"],
                 "Tier 3": ["tier 3", "T3", "Tier-3", "Tier3", "TIER 3"]}
PLACED_VARIANTS = {"Placed": ["placed", "PLACED", "Yes", "Y", "1", "Placed "],
                   "Not Placed": ["not placed", "NOT PLACED", "No", "N", "Unplaced", "0", "NotPlaced"]}

MISSING_RATE = {"Age": .008, "Gender": .02, "Branch": .02, "College_Tier": .02, "CGPA": .04,
                "Tenth_Percent": .03, "Twelfth_Percent": .035, "Backlogs": .025, "Internships": .03,
                "Projects": .03, "Certifications": .03, "Communication_Score": .045,
                "Aptitude_Score": .04, "Placed": .004}
BAD_TOKENS = ["?", "unknown", "-", "NA", "n/a"]
INVALID = {"Age": [-1, 5, 150, 0], "CGPA": [0, 10.8, 12, 15, 78.5, -1], "Tenth_Percent": [101, 105, 150, -5, 0, 950],
           "Twelfth_Percent": [101, 110, 150, -5, 0, 920], "Backlogs": [-1, -2, 25, 99],
           "Internships": [-1, 30], "Communication_Score": [-10, 120, 150, 999],
           "Aptitude_Score": [-10, 120, 150, 999]}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _clean_population(n, rng):
    """Create n *clean* students with realistic relationships between the attributes."""
    branch = rng.choice(BRANCHES, n, p=BRANCH_P)
    tier = rng.choice(TIERS, n, p=TIER_P)
    b_bonus = pd.Series(branch).map(BRANCH_BONUS).to_numpy()
    t_bonus = pd.Series(tier).map(TIER_BONUS).to_numpy()
    ability = rng.normal(0, 1, n)

    cgpa = np.clip(7.2 + .75 * ability + .15 * t_bonus + rng.normal(0, .45, n), 4.5, 10)
    tenth = np.clip(76 + 8 * ability + rng.normal(0, 7, n), 40, 99)
    twelfth = np.clip(73 + 8 * ability + rng.normal(0, 7.5, n), 40, 99)
    backlogs = np.minimum(rng.poisson(np.clip(.5 - .45 * ability, .02, 3), n), 8)
    internships = np.minimum(rng.poisson(np.clip(1.0 + .35 * ability + .2 * t_bonus, .1, 4), n), 6)
    projects = np.minimum(rng.poisson(np.clip(2.2 + .5 * ability, .3, 5), n), 10)
    certs = np.minimum(rng.poisson(np.clip(1.6 + .4 * ability, .2, 5), n), 8)
    comm = np.clip(63 + 5 * ability + rng.normal(0, 11, n), 20, 100)
    apt = np.clip(60 + 9 * ability + rng.normal(0, 12, n), 15, 100)

    logit = (1.1 * (cgpa - 7.2) + .035 * (apt - 60) + .025 * (comm - 62) + .55 * internships
             + .25 * projects + .15 * certs - 1.1 * (backlogs > 0) - .35 * backlogs
             + .02 * (twelfth - 73) + .01 * (tenth - 76) + b_bonus + t_bonus
             + .8 * ((cgpa > 8.5) & (internships >= 2)) - 1.0 * (cgpa < 6.0)
             + rng.normal(0, .9, n))
    lo, hi = -30, 30                       # bisection: choose intercept so ~62 % get placed
    for _ in range(40):
        mid = (lo + hi) / 2
        if _sigmoid(logit - mid).mean() > .62:
            lo = mid
        else:
            hi = mid
    placed = rng.random(n) < _sigmoid(logit - (lo + hi) / 2)

    base = (3.2 + 1.1 * (cgpa - 6) + .035 * (apt - 50) + .45 * internships + .2 * projects
            + 1.2 * t_bonus + 2.0 * b_bonus)
    pkg = np.clip(np.exp(rng.normal(np.log(np.maximum(base, 2.5)), .25)), 2.4, 45).round(2)

    first = rng.integers(0, len(FIRST), n)
    last = rng.integers(0, len(LAST), n)
    return pd.DataFrame({
        "Name": [f"{FIRST[a]} {LAST[b]}" for a, b in zip(first, last)],
        "Gender": rng.choice(["Male", "Female"], n, p=[.58, .42]),
        "Age": rng.choice([20, 21, 22, 23, 24, 25], n, p=[.08, .27, .38, .20, .05, .02]),
        "Branch": branch, "College_Tier": tier,
        "CGPA": cgpa.round(2), "Tenth_Percent": tenth.round(1), "Twelfth_Percent": twelfth.round(1),
        "Backlogs": backlogs, "Internships": internships, "Projects": projects, "Certifications": certs,
        "Communication_Score": comm.round(1), "Aptitude_Score": apt.round(1),
        "Placed": np.where(placed, "Placed", "Not Placed"),
        "Package_LPA": np.where(placed, pkg, np.nan),
    })


def _inject_dirt(df, rng):
    df = df.astype(object).copy()
    n = len(df)

    def pick(frac):
        return np.flatnonzero(rng.random(n) < frac)

    # 1) package contradictions / invalid salaries (needs the true label, so do it first)
    placed = (df["Placed"] == "Placed").to_numpy()
    idx = np.flatnonzero(placed & (rng.random(n) < .010))
    df.loc[idx, "Package_LPA"] = np.nan                                  # placed but no salary
    idx = np.flatnonzero(placed & (rng.random(n) < .004))
    df.loc[idx, "Package_LPA"] = rng.choice([-5, 0, 999], len(idx))     # impossible salary
    idx = np.flatnonzero(~placed & (rng.random(n) < .003))
    df.loc[idx, "Package_LPA"] = rng.uniform(3, 12, len(idx)).round(2)   # unplaced but has salary

    # 2) impossible values
    for col, bad in INVALID.items():
        idx = pick(.005)
        df.loc[idx, col] = rng.choice(bad, len(idx))

    # 3) inconsistent spellings
    for col, variants, frac in [("Branch", BRANCH_VARIANTS, .18), ("Gender", GENDER_VARIANTS, .12),
                                ("College_Tier", TIER_VARIANTS, .15), ("Placed", PLACED_VARIANTS, .10)]:
        idx = pick(frac)
        df.loc[idx, col] = [variants[v][rng.integers(len(variants[v]))] for v in df.loc[idx, col]]

    # 4) missing values and junk tokens
    for col, frac in MISSING_RATE.items():
        df.loc[pick(frac), col] = np.nan
    for col in ["Age", "CGPA", "Tenth_Percent", "Twelfth_Percent", "Backlogs", "Internships",
                "Projects", "Certifications", "Communication_Score", "Aptitude_Score"]:
        idx = pick(.004)
        df.loc[idx, col] = rng.choice(BAD_TOKENS, len(idx))

    # 5) messy names
    idx = pick(.05)
    mode = rng.integers(0, 4, len(idx))
    for i, m in zip(idx, mode):
        v = df.at[i, "Name"]
        df.at[i, "Name"] = [v.upper(), v.lower(), v + "  ", v.replace(" ", "   ")][m]
    return df


def create_dataset(n=60000, seed=42, dirty=True):
    """Return a DataFrame with exactly n rows (about 2 % of them are duplicate records)."""
    rng = np.random.default_rng(seed)
    dup_n = int(n * .02) if dirty else 0
    base_n = n - dup_n
    df = _clean_population(base_n, rng)
    df.insert(0, "Student_ID", [f"S{i:06d}" for i in range(1, base_n + 1)])
    if not dirty:
        return df
    df = _inject_dirt(df, rng)
    dups = df.sample(dup_n, random_state=seed).copy()
    half = rng.random(dup_n) < .5                      # half of the duplicates differ only by spacing
    dups.loc[half, "Name"] = dups.loc[half, "Name"].map(lambda v: v + " " if isinstance(v, str) else v)
    out = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def main():
    from mining import clean_data
    print("Generating 60,000 student records ...")
    raw = create_dataset(60000)
    raw.to_csv(CSV_PATH, index=False)
    print(f"  CSV  -> {CSV_PATH}  ({len(raw):,} rows)")
    clean, _ = clean_data(pd.read_csv(CSV_PATH))
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with sqlite3.connect(DB_PATH) as con:
        pd.read_csv(CSV_PATH, dtype=str).to_sql("students_raw", con, index=False)
        clean.to_sql("students_clean", con, index=False)
        con.execute("CREATE INDEX idx_clean_branch ON students_clean(Branch)")
        con.execute("CREATE INDEX idx_clean_placed ON students_clean(Placed)")
    print(f"  SQLite -> {DB_PATH}  (tables: students_raw, students_clean)")


if __name__ == "__main__":
    main()
