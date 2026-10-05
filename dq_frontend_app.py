#!/usr/bin/env python3
"""
FRONT END for the data quality pipeline - a local web app, no command line
arguments needed. Upload a file (or pull recent Oracle rows), see the
quality report as a dashboard, and push suggestions straight into the
review queue with one click.

Setup (PyCharm terminal):
    pip install streamlit

Run (NOT with the normal "Run" button - Streamlit needs its own command):
    streamlit run dq_frontend_app.py

This opens a tab in your browser at http://localhost:8501
"""
import pandas as pd
import streamlit as st

from dq_ml_pipeline1 import DQPipeline

DB_USER = "system"
DB_PASSWORD = "suman_123456"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "xe"
MODEL_PATH = "dq_pipeline_trained.joblib"

st.set_page_config(page_title="Bank Data Quality Checker", layout="wide")


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


@st.cache_resource
def load_pipeline():
    return DQPipeline.load(MODEL_PATH)


def push_to_review_queue(report, table_name="CUSTOMER_MASTER"):
    import oracledb
    REASON_TO_ERROR_TYPE = [
        ("Missing value", "NULL_VALUE"), ("Standardise non-canonical", "NONSTANDARD_CATEGORY"),
        ("not a recognised category", "INVALID_CATEGORY"), ("Statistical outlier", "NUMERIC_OUTLIER"),
        ("negative in a field", "UNEXPECTED_NEGATIVE"), ("Date is missing", "DATE_OUT_OF_RANGE"),
        ("extra prefix", "EXTRA_PREFIX"), ("embedded spaces", "EMBEDDED_SPACE"),
        ("short of the expected length", "TOO_SHORT"),
    ]

    def to_error_type(reason):
        for phrase, label in REASON_TO_ERROR_TYPE:
            if phrase in reason:
                return label
        return "ML_RULE_VIOLATION"

    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                            service_name=DB_SERVICE)
    cur = conn.cursor()
    existing = pd.read_sql(f"""SELECT record_id, column_name FROM dq_corrections
                              WHERE table_name = '{table_name}' AND correction_method = 'ML_PIPELINE'""", conn)
    existing.columns = existing.columns.str.lower()
    already_seen = set(zip(existing.record_id, existing.column_name))
    new_rows = report[~report.apply(lambda r: (r.record_id, r.column_name) in already_seen, axis=1)].copy()
    new_rows["error_type"] = new_rows["reason"].apply(to_error_type)

    cur.execute("SELECT NVL(MAX(correction_id), 0) FROM dq_corrections")
    next_id = cur.fetchone()[0] + 1
    for i, (_, r) in enumerate(new_rows.iterrows()):
        cur.execute("""
            INSERT INTO dq_corrections (correction_id, table_name, record_id, column_name, error_type,
                                        original_value, suggested_value, correction_method,
                                        confidence_score, business_impact, status)
            VALUES (:1, :2, :3, :4, :5, :6, :7, 'ML_PIPELINE', :8,
                   'Submitted via the Data Quality Checker app', 'PENDING')
        """, [next_id + i, table_name, r["record_id"], r["column_name"], r["error_type"],
              None if pd.isna(r["original_value"]) else str(r["original_value"]),
              None if pd.isna(r["suggested_value"]) else str(r["suggested_value"]),
              float(r["rule_confidence"])])
    conn.commit()
    conn.close()
    return len(new_rows)


# ============================================================
# PAGE LAYOUT
# ============================================================
st.title("Bank Data Quality Checker")
st.caption("Upload new data, or pull recent records from Oracle, and check them against the trained model.")

try:
    pipe = load_pipeline()
    st.success(f"Trained model loaded from {MODEL_PATH}")
except FileNotFoundError:
    st.error(f"No trained model found at {MODEL_PATH}. Run dq_ml_pipeline.py first to train and save one.")
    st.stop()

source = st.radio("Where is the data to check?", ["Upload a CSV file", "Pull recent rows from Oracle"],
                  horizontal=True)

df = None
if source == "Upload a CSV file":
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded, dtype=str, keep_default_na=False, na_values=[""])
else:
    days = st.slider("How many days back?", 1, 365, 30)
    if st.button("Pull from Oracle"):
        import oracledb
        conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                                service_name=DB_SERVICE)
        df = pd.read_sql(f"SELECT * FROM customer_master WHERE load_date >= TRUNC(SYSDATE) - {days}", conn)
        df.columns = df.columns.str.lower()
        df = cast_oracle_types(df)
        conn.close()

if df is not None:
    st.write(f"Loaded **{len(df):,}** records.")

    with st.spinner("Checking data quality..."):
        report = pipe.transform(df)

    n_total = len(df)
    n_flagged = report["record_id"].nunique() if len(report) else 0
    score = 100 * (1 - n_flagged / n_total) if n_total else 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Records checked", f"{n_total:,}")
    col2.metric("Records with an issue", f"{n_flagged:,}", f"{100*n_flagged/max(n_total,1):.1f}%")
    col3.metric("Estimated quality score", f"{score:.2f}%")

    if len(report):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Issues by column")
            st.bar_chart(report["column_name"].value_counts())
        with c2:
            st.subheader("Issues by confidence tier")
            tier = pd.cut(report["rule_confidence"], [-0.01, 0.5, 0.8, 1.01],
                          labels=["Low (<0.5)", "Medium (0.5-0.8)", "High (>=0.8)"])
            st.bar_chart(tier.value_counts())

        st.subheader("All suggestions")
        st.dataframe(report.sort_values("rule_confidence", ascending=False), use_container_width=True)

        st.download_button("Download full report as CSV", report.to_csv(index=False),
                           "data_quality_report.csv", "text/csv")

        st.subheader("Send to review queue")
        st.write("This adds every suggestion above to dq_corrections with status PENDING. "
                "Nothing is applied to production data - a steward still has to approve each one.")
        if st.button("Push all suggestions to the review queue"):
            with st.spinner("Writing to dq_corrections..."):
                n_new = push_to_review_queue(report)
            st.success(f"Added {n_new:,} new suggestions to the review queue (duplicates skipped).")
    else:
        st.success("No issues found in this data.")

#streamlit run dq_frontend_app.py