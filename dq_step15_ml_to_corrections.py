#!/usr/bin/env python3
"""
INTEGRATION - feed the ML pipeline's suggestions into the SAME review queue
built in Step 45 (dq_corrections), so a data steward reviews suggestions
from the rule-based system (Step 42) and the trained ML pipeline in one
place, rather than two separate outputs.

This script:
  1. Runs the ML pipeline (or reads an already-generated dq_ml_suggestions.csv)
  2. Translates each suggestion's plain-English "reason" into a short,
     consistent error_type label (so it groups sensibly alongside the
     SQL-rule corrections already in the table)
  3. Skips anything already inserted from a previous run (checked by
     table_name + record_id + column_name + correction_method), so
     rerunning this script doesn't create duplicate queue entries
  4. Inserts everything else into dq_corrections with status='PENDING',
     correction_method='ML_PIPELINE', ready for the same approve/reject
     workflow as before

Run:  python dq_step15_ml_to_corrections.py
"""
import pandas as pd
import oracledb

from dq_ml_pipeline1 import DQPipeline

DB_USER = "system"
DB_PASSWORD = "suman_123456"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "xe"

TABLE_NAME = "CUSTOMER_MASTER"
ID_COL = "customer_id"

REASON_TO_ERROR_TYPE = [
    ("Missing value", "NULL_VALUE"),
    ("Standardise non-canonical", "NONSTANDARD_CATEGORY"),
    ("not a recognised category", "INVALID_CATEGORY"),
    ("Statistical outlier", "NUMERIC_OUTLIER"),
    ("negative in a field", "UNEXPECTED_NEGATIVE"),
    ("Date is missing", "DATE_OUT_OF_RANGE"),
    ("extra prefix", "EXTRA_PREFIX"),
    ("embedded spaces", "EMBEDDED_SPACE"),
    ("short of the expected length", "TOO_SHORT"),
]


def to_error_type(reason):
    for phrase, label in REASON_TO_ERROR_TYPE:
        if phrase in reason:
            return label
    return "ML_RULE_VIOLATION"


def cast_oracle_types(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dropna().mod(1).eq(0).all():
                df[col] = df[col].astype("Int64").astype(str).replace("<NA>", None)
            else:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
        else:
            df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
    return df


def main():
    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                            service_name=DB_SERVICE)

    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    df.columns = df.columns.str.lower()
    df = cast_oracle_types(df)

    gt = pd.read_sql(f"SELECT record_id FROM error_log_ground_truth "
                     f"WHERE UPPER(table_name)='{TABLE_NAME}' AND error_type <> 'DUPLICATE_CUSTOMER'", conn)
    gt.columns = gt.columns.str.lower()
    known_bad = set(gt.record_id)

    pipe = DQPipeline(id_col=ID_COL, as_of=pd.Timestamp("2026-09-23"))
    pipe.fit(df, known_bad_ids=known_bad)
    report = pipe.transform(df)
    print(f"\nML pipeline generated {len(report):,} suggestions across the full table.")

    # skip anything already pushed to the queue in a previous run
    existing = pd.read_sql(f"""
        SELECT record_id, column_name FROM dq_corrections
         WHERE table_name = '{TABLE_NAME}' AND correction_method = 'ML_PIPELINE'
    """, conn)
    existing.columns = existing.columns.str.lower()
    already_seen = set(zip(existing.record_id, existing.column_name))
    report = report[~report.apply(lambda r: (r.record_id, r.column_name) in already_seen, axis=1)]
    print(f"{len(report):,} are new (not already in the queue from a previous run).")

    if report.empty:
        print("Nothing new to insert.")
        conn.close()
        return

    report["error_type"] = report["reason"].apply(to_error_type)
    impact = report.groupby(["column_name", "error_type"]).size()

    cur = conn.cursor()
    cur.execute("SELECT NVL(MAX(correction_id), 0) FROM dq_corrections")
    next_id = cur.fetchone()[0] + 1

    for i, (_, r) in enumerate(report.iterrows()):
        n_affected = impact[(r["column_name"], r["error_type"])]
        business_impact = (f"ML pipeline: affects {n_affected} record(s) with this "
                           f"{r['column_name']}/{r['error_type']} pattern. "
                           f"Model's overall error-probability for this record: {r['model_error_probability']}")
        cur.execute("""
            INSERT INTO dq_corrections (correction_id, table_name, record_id, column_name, error_type,
                                        original_value, suggested_value, correction_method,
                                        confidence_score, business_impact, status)
            VALUES (:1, :2, :3, :4, :5, :6, :7, 'ML_PIPELINE', :8, :9, 'PENDING')
        """, [next_id + i, TABLE_NAME, r["record_id"], r["column_name"], r["error_type"],
              None if pd.isna(r["original_value"]) else str(r["original_value"]),
              None if pd.isna(r["suggested_value"]) else str(r["suggested_value"]),
              float(r["rule_confidence"]), business_impact])

    conn.commit()
    print(f"Inserted {len(report):,} ML-generated suggestions into dq_corrections (status=PENDING).")

    print("\nBreakdown of what was added, by error type:")
    print(report.error_type.value_counts().to_string())

    conn.close()


if __name__ == "__main__":
    main()
