#!/usr/bin/env python3
"""
STEP 1 - Automatic data profiling  (Metric 04.01.05, action step 27)

Reads every business table, and for every column measures: null rate, distinct values, min/max,
value ranges, most common values, and the most common FORMAT PATTERN (A = letter, 9 = digit).
It then prints the suspicious findings and saves the full profile to dq_profile.csv.
The next step (rule suggestion) will use this profile as its input.

NOTE: error_log_ground_truth is deliberately NOT profiled - it holds the answers and is only
used at the end to score the tool.

Run against Oracle:      python dq_step1_profile.py
Run against the CSVs:    python dq_step1_profile.py --csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- same values you used on the oracledb.connect(...) line of load_to_oracle.py
DB_USER = "YOUR_USERNAME"
DB_PASSWORD = "YOUR_PASSWORD"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "XE"          # or "XEPDB1" - whichever worked for the loader

DATA_DIR = "."             # only used with --csv
AS_OF = os.getenv("AS_OF")  # optional 'YYYY-MM-DD'; default = today (used for "future date" checks)
TODAY = pd.Timestamp(AS_OF) if AS_OF else pd.Timestamp.today().normalize()

TABLES = ["customer_master", "account_master", "loan_accounts", "transactions", "branch_master",
          "pincode_reference", "batch_log"]
DATE_COLS = {"opened_date", "dob", "kyc_date", "customer_since", "load_date", "open_date", "disbursement_date",
             "maturity_date", "txn_date"}
EXPECT_FUTURE = {"maturity_date"}          # future dates are normal here (checked against disbursement in a later step)
NUM_COLS = {"annual_income", "balance", "tenure_months", "interest_rate", "emi_amount", "outstanding_principal",
            "overdue_days", "amount", "balance_after", "duration_sec", "row_count", "disbursement_amount"}


# ------------------------------------------------------------------ load
def load_tables():
    if "--csv" in sys.argv:
        for t in TABLES:
            df = pd.read_csv(os.path.join(DATA_DIR, t + ".csv"), dtype=str, keep_default_na=False, na_values=[""])
            for c in df.columns:
                if c in DATE_COLS:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                elif c in NUM_COLS:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            yield t, df
    else:
        import oracledb
        conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                                service_name=DB_SERVICE)
        cur = conn.cursor()
        cur.arraysize = 50_000
        for t in TABLES:
            cur.execute(f"SELECT * FROM {t}")
            cols = [d[0].lower() for d in cur.description]
            yield t, pd.DataFrame(cur.fetchall(), columns=cols)
        conn.close()


# ------------------------------------------------------------------ profile one column
def profile_col(t, s):
    n = len(s)
    nn = int(s.isna().sum())
    nonnull = s.dropna()
    nd = int(nonnull.nunique())
    rec = {"table": t, "column": s.name, "rows": n, "null_count": nn, "null_pct": round(100 * nn / max(n, 1), 2),
           "distinct_count": nd, "distinct_pct": round(100 * nd / max(len(nonnull), 1), 2)}
    if len(nonnull) == 0:
        rec["kind"] = "empty"
        return rec
    if pd.api.types.is_datetime64_any_dtype(s):
        rec.update(kind="date", min=str(nonnull.min().date()), max=str(nonnull.max().date()),
                   future_dates=0 if s.name in EXPECT_FUTURE else int((nonnull > TODAY).sum()),
                   dates_before_1900=int((nonnull < pd.Timestamp("1900-01-01")).sum()))
    elif pd.api.types.is_numeric_dtype(s):
        rec.update(kind="number", min=round(float(nonnull.min()), 2), max=round(float(nonnull.max()), 2),
                   mean=round(float(nonnull.mean()), 2), median=round(float(nonnull.median()), 2),
                   negative_count=int((nonnull < 0).sum()), zero_count=int((nonnull == 0).sum()))
    else:
        v = nonnull.astype(str)
        ln = v.str.len()
        pat = v.str.replace(r"[A-Za-z]", "A", regex=True).str.replace(r"\d", "9", regex=True)
        if ln.max() > 20:                                    # long free text: collapse runs
            pat = pat.str.replace(r"A+", "A", regex=True).str.replace(r"9+", "9", regex=True)
        vc = pat.value_counts()
        rec.update(kind="text", min_len=int(ln.min()), max_len=int(ln.max()), n_patterns=int(len(vc)),
                   top_pattern=vc.index[0], top_pattern_pct=round(100 * vc.iloc[0] / len(v), 2),
                   whitespace_issues=int((v != v.str.strip().str.replace(r"\s+", " ", regex=True)).sum()))
        if nd <= 60:
            tv = v.value_counts().head(6)
            rec["top_values"] = " | ".join(f"{k} ({100 * c / len(v):.1f}%)" for k, c in tv.items())
            rec["code_variants"] = int(nd - v.str.strip().str.lower().nunique())
    return rec


# ------------------------------------------------------------------ report
def show(title, df, cols):
    print(f"\n=== {title} ===")
    print("(none)" if df.empty else df[cols].to_string(index=False))


def main():
    recs, sizes = [], {}
    for t, df in load_tables():
        sizes[t] = len(df)
        print(f"profiling {t:20s} rows={len(df):>9,} columns={len(df.columns)}")
        recs += [profile_col(t, df[c]) for c in df.columns]
    p = pd.DataFrame(recs)
    p.to_csv("dq_profile.csv", index=False)
    print("\nfull profile saved to dq_profile.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    show("1. COMPLETENESS: columns with blanks (null_pct > 0.5%)",
         p[p.null_pct > 0.5].sort_values("null_pct", ascending=False), ["table", "column", "null_count", "null_pct"])
    txt = p[(p.kind == "text") & (p.max_len <= 20) & (p.n_patterns > 1) & (p.top_pattern_pct < 99.95)
            & p.top_pattern.fillna("").str.contains("9")]
    show("2. FORMAT: code columns (PIN, mobile, PAN, IFSC...) with more than one pattern (9 = digit, A = letter)",
         txt.sort_values("top_pattern_pct"), ["table", "column", "top_pattern", "top_pattern_pct", "n_patterns"])
    show("3. STANDARDISATION: code columns with case/spelling variants",
         p[p.get("code_variants", pd.Series(0, index=p.index)).fillna(0) > 0],
         ["table", "column", "code_variants", "top_values"])
    show("4. VALIDITY: negative numbers", p[p.get("negative_count", pd.Series(0, index=p.index)).fillna(0) > 0],
         ["table", "column", "negative_count", "min"])
    dts = p[(p.kind == "date") & ((p.future_dates > 0) | (p.dates_before_1900 > 0))]
    show(f"5. VALIDITY: dates after {TODAY.date()} or before 1900", dts,
         ["table", "column", "future_dates", "dates_before_1900", "min", "max"])
    ws = p[p.get("whitespace_issues", pd.Series(0, index=p.index)).fillna(0) > 0]
    show("6. FORMAT: values with leading/trailing/double spaces", ws, ["table", "column", "whitespace_issues"])
    print("\nNext step: use dq_profile.csv to suggest data-quality rules (action step 28).")


if __name__ == "__main__":
    main()
