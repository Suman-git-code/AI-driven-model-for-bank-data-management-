#!/usr/bin/env python3
"""
END-TO-END AI DATA QUALITY PIPELINE  (generalised version of Steps 27-44)

This is a SELF-LEARNING pipeline. Give it a bank dataset (a table with a
customer/account/loan style structure) and it will, without being told
anything about the specific columns in advance:

  1. LEARN THE RULES  - study the data itself and work out what "normal"
     looks like for every column (typical format, valid categories,
     numeric/date ranges) - satisfies "auto-generate data quality rules"
     (action step 28), but data-driven rather than hand written.

  2. DETECT ERRORS    - turn every rule violation into a feature, then
     train a Random Forest classifier - VALIDATED with a proper
     train/validation/test split (60/20/20) and 5-fold cross-validation,
     not just run once and trusted.

  3. SUGGEST CORRECTIONS - for records flagged as errors, propose a fix
     using general heuristics (not hard-coded to any one country's phone
     format): canonicalise noisy category spellings, strip/pad ID-like
     fields whose length is off by a small, explainable amount, and
     otherwise defer to manual review rather than guess. Each suggestion
     carries a confidence score.

HOW TO USE THIS ON YOUR OWN DATA:
    from dq_ml_pipeline import DQPipeline
    pipe = DQPipeline(id_col="customer_id")
    pipe.fit(historical_df, label_col=None)     # label_col optional - see below
    report = pipe.transform(new_df)             # -> violations + suggestions + confidence

If you have a column of known-bad record IDs (e.g. from error_log_ground_truth,
or from last month's confirmed corrections), pass it as `known_bad_ids` to
fit() to train and VALIDATE the classifier with real accuracy numbers. Without
it, the pipeline still learns rules and flags violations, just without a
trained classifier layer on top (rules alone, like Steps 28-30).

Run directly against Oracle:  python dq_ml_pipeline.py
Run on a CSV instead:         python dq_ml_pipeline.py --csv customer_master.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

DB_USER = "system"
DB_PASSWORD = "suman_123456"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "xe"


# ============================================================
# THE PIPELINE
# ============================================================
class DQPipeline:
    def __init__(self, id_col, as_of=None):
        self.id_col = id_col
        self.as_of = as_of or pd.Timestamp.today().normalize()
        self.rules = {}
        self.model = None
        self.feature_columns = None

    # ---------------- Phase 1: LEARN THE RULES ----------------
    def _looks_like_date(self, s):
        parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        return parsed.notna().mean() > 0.95, parsed

    def learn_rules(self, df):
        rules = {}
        for col in df.columns:
            if col == self.id_col:
                continue
            s = df[col].dropna()
            if len(s) == 0:
                continue
            null_rate = df[col].isna().mean()

            is_date, parsed_all = self._looks_like_date(df[col])
            if is_date:
                valid = parsed_all.dropna()
                lo, hi = valid.quantile(0.01), min(valid.quantile(0.99), self.as_of)
                span = hi - lo
                rules[col] = {"type": "date", "null_rate": round(null_rate, 4),
                             "lower": lo - span * 0.5, "upper": min(hi + span * 0.5, self.as_of)}
                continue

            nunique = s.nunique()
            lengths = s.str.len()
            fixed_width = int(lengths.mode()[0])
            fixed_width_pct = (lengths == fixed_width).mean()
            s_num = pd.to_numeric(s, errors="coerce")
            is_true_numeric = s_num.notna().mean() > 0.98 and fixed_width_pct < 0.90

            if nunique <= 20 and not is_true_numeric:
                norm = s.str.strip().str.upper()
                freq = norm.value_counts(normalize=True)
                valid_categories = freq[freq >= 0.003]
                cats = list(valid_categories.index)
                # Build a "canonical spelling" map, SAFELY: only merge two
                # categories when the shorter one is a genuine PREFIX of the
                # longer one (e.g. "M" is a prefix of "MALE") AND the shorter
                # one is no more than 3 characters (i.e. it looks like a
                # short code/abbreviation, not a real word). This recovers
                # things like MALE -> M or FEMALE -> F purely from the data,
                # while correctly leaving genuinely distinct categories that
                # merely share a first letter (e.g. "Kochi" vs "Kolkata", or
                # "Student" vs "Self_Employed") untouched. A naive "same
                # first letter" grouping was tested and found to wrongly
                # merge unrelated categories - do not simplify this back.
                canon_map = {c: c for c in cats}
                for a in cats:
                    if len(a) > 3:
                        continue
                    for b in cats:
                        if a != b and b.startswith(a) and valid_categories[a] >= valid_categories[b]:
                            canon_map[b] = a
                rules[col] = {"type": "categorical", "null_rate": round(null_rate, 4),
                             "valid_categories": set(valid_categories.index), "canon_map": canon_map}
            elif is_true_numeric:
                q1, q3 = s_num.quantile([0.25, 0.75])
                iqr = q3 - q1
                neg_rate = (s_num < 0).mean()
                rules[col] = {"type": "numeric", "null_rate": round(null_rate, 4),
                             "lower": q1 - 3 * iqr, "upper": q3 + 3 * iqr, "rarely_negative": neg_rate < 0.005}
            else:
                pattern = s.str.replace(r"[A-Za-z]", "A", regex=True).str.replace(r"[0-9]", "9", regex=True)
                pc = pattern.value_counts(normalize=True)
                rules[col] = {"type": "pattern", "null_rate": round(null_rate, 4),
                             "dominant_pattern": pc.index[0], "dominant_pct": pc.iloc[0],
                             "typical_length": fixed_width}
        self.rules = rules
        return rules

    # ---------------- Phase 2a: turn rules into features ----------------
    def build_features(self, df):
        feats = pd.DataFrame(index=df.index)
        for col, r in self.rules.items():
            if col not in df.columns:
                continue
            v = df[col]
            is_null = v.isna()
            feats[f"{col}__null"] = is_null.astype(int)
            if r["type"] == "categorical":
                norm = v.str.strip().str.upper()
                feats[f"{col}__invalid_cat"] = (~norm.isin(r["valid_categories"]) & ~is_null).astype(int)
                feats[f"{col}__noncanonical"] = (norm.map(r["canon_map"]) != norm).fillna(False).astype(int)
            elif r["type"] == "numeric":
                num = pd.to_numeric(v, errors="coerce")
                feats[f"{col}__outlier"] = ((num < r["lower"]) | (num > r["upper"])).fillna(False).astype(int)
            elif r["type"] == "date":
                parsed = pd.to_datetime(v, format="%Y-%m-%d", errors="coerce")
                feats[f"{col}__bad_date_format"] = (parsed.isna() & ~is_null).astype(int)
                feats[f"{col}__out_of_range"] = ((parsed < r["lower"]) | (parsed > r["upper"])).fillna(False).astype(int)
            else:
                if r["dominant_pct"] < 0.5:
                    continue
                vs = v.fillna("")
                pat = vs.str.replace(r"[A-Za-z]", "A", regex=True).str.replace(r"[0-9]", "9", regex=True)
                feats[f"{col}__bad_pattern"] = ((pat != r["dominant_pattern"]) & ~is_null).astype(int)
                feats[f"{col}__wrong_length"] = ((vs.str.len() != r["typical_length"]) & ~is_null).astype(int)
        feats["total_violations"] = feats.filter(like="__").sum(axis=1)
        return feats

    # ---------------- Phase 2b: TRAIN + VALIDATE the classifier ----------------
    def fit(self, df, known_bad_ids=None):
        """known_bad_ids: an iterable of id_col values known to contain an
        error (e.g. from error_log_ground_truth, or last quarter's confirmed
        corrections). If provided, this trains and validates a classifier.
        If omitted, the pipeline still works using rules alone."""
        self.learn_rules(df)

        if known_bad_ids is None:
            print("No labelled errors supplied - rules learned, but no classifier trained. "
                 "Detection will rely on rule violations alone (like steps 28-30).")
            return self

        y = df[self.id_col].isin(set(known_bad_ids)).astype(int)
        print(f"Labelled error rate in training data: {y.mean():.1%}")

        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=y)
        ytr_full = y.loc[train_df.index]
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                           stratify=y.loc[temp_df.index])
        yval, yte = y.loc[val_df.index], y.loc[test_df.index]
        print(f"Split -> train: {len(train_df):,}  validation: {len(val_df):,}  test: {len(test_df):,}")

        # IMPORTANT: rules are learned ONLY from the training split, then
        # applied unchanged to validation/test, exactly as they would be
        # applied to genuinely new incoming data.
        self.learn_rules(train_df)
        Xtr = self.build_features(train_df)
        self.feature_columns = Xtr.columns
        Xval = self.build_features(val_df).reindex(columns=self.feature_columns, fill_value=0)
        Xte = self.build_features(test_df).reindex(columns=self.feature_columns, fill_value=0)

        self.model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42,
                                            class_weight="balanced", n_jobs=-1)
        self.model.fit(Xtr, ytr_full)

        cv = cross_val_score(self.model, Xtr, ytr_full, cv=5, scoring="roc_auc")
        print(f"\n5-fold cross-validation AUC on training data: {cv.round(3)}  mean={cv.mean():.3f}")

        for name, X_, y_ in [("VALIDATION SET", Xval, yval), ("TEST SET (final, held out)", Xte, yte)]:
            proba = self.model.predict_proba(X_)[:, 1]
            pred = (proba >= 0.5).astype(int)
            print(f"\n=== {name} ===")
            print(classification_report(y_, pred, target_names=["No Error", "Has Error"]))
            print(f"AUC: {roc_auc_score(y_, proba):.3f}")
            print(f"Confusion matrix (rows=actual, cols=predicted):\n{confusion_matrix(y_, pred)}")

        imp = pd.Series(self.model.feature_importances_, index=self.feature_columns).sort_values(ascending=False)
        print("\nTop 10 signals the model relies on most:")
        print(imp.head(10).to_string())

        # refit the rules and model on the FULL dataset for deployment
        self.learn_rules(df)
        Xall = self.build_features(df)
        self.feature_columns = Xall.columns
        self.model.fit(Xall, y)
        return self

    # ---------------- Save / load the trained pipeline ----------------
    def save(self, path="dq_pipeline_trained.joblib"):
        import joblib
        joblib.dump(self, path)
        print(f"Trained pipeline saved to {path}")

    @staticmethod
    def load(path="dq_pipeline_trained.joblib"):
        import joblib
        return joblib.load(path)

    # ---------------- Phase 3: SUGGEST CORRECTIONS ----------------
    def _suggest_for_row(self, col, r, value):
        if pd.isna(value):
            return None, 0.0, "Missing value - no safe way to infer it automatically"
        if r["type"] == "categorical":
            norm = value.strip().upper()
            canon = r["canon_map"].get(norm)
            if canon and canon != norm:
                share = 0.9   # canonicalisation within an already-common category is low risk
                return canon, share, f"Standardise non-canonical spelling '{value}' to '{canon}'"
            if norm not in r["valid_categories"]:
                return None, 0.3, f"'{value}' is not a recognised category - needs manual review"
        elif r["type"] == "numeric":
            num = pd.to_numeric(value, errors="coerce")
            if pd.notna(num) and num < 0 and r.get("rarely_negative"):
                return abs(num), 0.6, "Value is negative in a field that is almost never negative - sign may be flipped"
            return None, 0.3, "Statistical outlier - could be genuine, needs manual review"
        elif r["type"] == "date":
            return None, 0.2, "Date is missing/out of plausible range - cannot be safely inferred"
        else:  # pattern / ID-like field
            typ_len = r["typical_length"]
            L = len(str(value))
            if L == typ_len + 2:
                candidate = str(value)[-typ_len:]
                if candidate.isdigit() or not candidate.isalpha():
                    return candidate, 0.85, f"Length is {L}, {typ_len} expected - likely a 2-character prefix; suggest stripping it"
            if " " in str(value) and L - str(value).count(" ") == typ_len:
                return str(value).replace(" ", ""), 0.9, "Contains embedded spaces - suggest removing them"
            if L == typ_len - 1:
                return None, 0.35, f"One character short of the expected length ({typ_len}) - cannot safely guess the missing character"
        return None, 0.3, "Flagged by rule violation - needs manual review"

    def transform(self, df):
        """Apply learned rules (+ trained model, if fitted) to NEW data.
        Returns one row per (record, violated column) with a suggestion.

        PERFORMANCE NOTE: this only computes a suggestion for rows that
        actually violate a rule for a given column - not every row of the
        table. On a table where most records are clean (typical for real
        data, including this project's ~23% error rate), that is a small
        fraction of the total, which is what makes this fast even on
        100,000+ row tables. An earlier version of this method looped
        over every row with repeated .iloc lookups and took several
        minutes on a 100k-row table; this version finishes in seconds."""
        feats = self.build_features(df)
        if self.feature_columns is not None:
            feats = feats.reindex(columns=self.feature_columns, fill_value=0)

        model_proba = None
        if self.model is not None:
            model_proba = pd.Series(self.model.predict_proba(feats)[:, 1], index=df.index)

        violation_cols = [c for c in feats.columns if "__" in c]
        # which base column each violation feature belongs to
        by_col = {}
        for vc in violation_cols:
            by_col.setdefault(vc.split("__")[0], []).append(vc)

        out_frames = []
        for col, vcols in by_col.items():
            r = self.rules.get(col)
            if r is None:
                continue
            mask = (feats[vcols].sum(axis=1) > 0)
            if not mask.any():
                continue
            idx = df.index[mask]
            sub = df.loc[idx, [self.id_col, col]].copy()
            results = sub[col].apply(lambda v: self._suggest_for_row(col, r, v))
            sub["suggested_value"] = results.apply(lambda t: t[0])
            sub["rule_confidence"] = results.apply(lambda t: t[1])
            sub["reason"] = results.apply(lambda t: t[2])
            sub = sub.rename(columns={self.id_col: "record_id", col: "original_value"})
            sub["column_name"] = col
            if model_proba is not None:
                sub["model_error_probability"] = model_proba.loc[idx].round(3).values
            else:
                sub["model_error_probability"] = None
            out_frames.append(sub[["record_id", "column_name", "original_value", "suggested_value",
                                   "rule_confidence", "model_error_probability", "reason"]])

        if not out_frames:
            return pd.DataFrame(columns=["record_id", "column_name", "original_value", "suggested_value",
                                         "rule_confidence", "model_error_probability", "reason"])
        return pd.concat(out_frames, ignore_index=True)


# ============================================================
# RUNNABLE DEMO / ENTRY POINT
# ============================================================
def main():
    if "--csv" in sys.argv:
        path = sys.argv[sys.argv.index("--csv") + 1]
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        gt_path = path.replace("customer_master", "error_log_ground_truth")
        known_bad = None
        try:
            gt = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
            known_bad = set(gt[(gt.table_name == "customer_master") & (gt.error_type != "DUPLICATE_CUSTOMER")]
                            .record_id)
            print(f"Loaded {len(known_bad):,} known-bad record IDs for training/validation.")
        except FileNotFoundError:
            print("No ground-truth file found alongside this CSV - training will proceed without labels.")
    else:
        import oracledb
        conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                                service_name=DB_SERVICE)
        df = pd.read_sql("SELECT * FROM customer_master", conn)
        df.columns = df.columns.str.lower()
        # Oracle returns NUMBER columns as actual int/float, and DATE columns as
        # datetime - but every rule in this pipeline is written assuming text
        # (it applies .str operations and pattern-matching to every column).
        # Cast everything to plain strings here, the same way the CSV path
        # already arrives as strings, so the two input paths behave identically.
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")
            elif pd.api.types.is_numeric_dtype(df[col]):
                # A NUMBER column with any nulls comes back as float64 (e.g.
                # 9876543210.0), which would corrupt every ID-like pattern
                # check below. Use the nullable Int64 type when every value
                # is a whole number, so it prints as "9876543210", not
                # "9876543210.0"; fall back to plain string only if the
                # column genuinely has decimals.
                if df[col].dropna().mod(1).eq(0).all():
                    df[col] = df[col].astype("Int64").astype(str).replace("<NA>", None)
                else:
                    df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
            else:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
        try:
            gt = pd.read_sql("SELECT record_id, error_type FROM error_log_ground_truth "
                            "WHERE UPPER(table_name)='CUSTOMER_MASTER' AND error_type <> 'DUPLICATE_CUSTOMER'", conn)
            gt.columns = gt.columns.str.lower()
            known_bad = set(gt.record_id)
            print(f"Loaded {len(known_bad):,} known-bad record IDs from Oracle for training/validation.")
        except Exception:
            known_bad = None
        conn.close()

    pipe = DQPipeline(id_col="customer_id", as_of=pd.Timestamp("2026-09-23"))
    pipe.fit(df, known_bad_ids=known_bad)

    print("\n\n=== APPLYING THE PIPELINE TO (SIMULATED) NEW INCOMING DATA ===")
    sample = df.sample(min(2000, len(df)), random_state=1)
    report = pipe.transform(sample)
    print(f"Generated {len(report):,} correction suggestions from {len(sample):,} sampled records.")
    print("\nSample of suggestions:")
    print(report.sort_values("rule_confidence", ascending=False).head(15).to_string(index=False))

    out_path = "dq_ml_suggestions.csv"
    report.to_csv(out_path, index=False)
    print(f"\nFull suggestion report written to {out_path}")

    pipe.save("dq_pipeline_trained.joblib")
    print("Use dq_check_new_data.py with this saved file to check NEW data later, without retraining.")


if __name__ == "__main__":
    main()
