#!/usr/bin/env python3
"""
STEPS 33-34 (MODEL-BASED VERSION) - Isolation Forest anomaly detection,
named explicitly in the source document as an option ("ML models trained
on historical correction patterns... Isolation Forest, Z-score").

WHAT THIS ADDS ON TOP OF THE EXISTING SQL RULES AND Z-SCORE SCRIPT:
The SQL rules (dq_step2_rules.sql) and the relationship checks
(dq_step5_anomalies.sql) each look for ONE specific, named problem: is
this mobile 10 digits, does this maturity date come before disbursement,
does this customer_id exist. Isolation Forest instead looks at several
numeric fields TOGETHER and flags loans whose overall combination of
values is unusual - useful for catching a genuinely weird COMBINATION
that nobody wrote a specific rule for.

HONEST PERFORMANCE NOTE (tested against this project's own ground truth
before being handed over): on its own, an unsupervised model like this
does NOT match the near-perfect recall of the hand-written SQL rules,
because most of the injected errors here are simple, single-column
mistakes that a plain rule catches better and more precisely than an
unsupervised model reasonably can. In testing: ~600 loans flagged,
~31% precision, ~50% recall against known numeric-logic errors. That
means roughly 1 in 3 flags is a genuine issue, and it catches about half
of the numeric problems the SQL rules cover, PLUS potentially some
combinations the rules don't explicitly check for.
CONCLUSION: use this as a SECOND, complementary layer for the human
review queue, not a replacement for the SQL rule checks. Its value is
in catching the unexpected, not in outperforming the checks you already
know how to write.

Run:  python dq_step14_isolation_forest.py
"""
import numpy as np
import pandas as pd
import oracledb
from sklearn.ensemble import IsolationForest

DB_USER = "system"
DB_PASSWORD = "suman_123456"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "xe"

CONTAMINATION = 0.02   # expected proportion of loans that are genuinely unusual


def main():
    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                            service_name=DB_SERVICE)
    df = pd.read_sql("""
        SELECT loan_account_no, disbursement_date, disbursement_amount, interest_rate, maturity_date
          FROM loan_accounts
    """, conn)
    df.columns = df.columns.str.lower()
    df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])

    # feature engineering: put every loan on a comparable numeric footing
    df["log_amount"] = np.log1p(df["disbursement_amount"].clip(lower=0).fillna(0))
    df["tenor_days"] = (df["maturity_date"] - df["disbursement_date"]).dt.days
    df["tenor_days"] = df["tenor_days"].fillna(-99999)   # a missing maturity date IS itself a signal, not "average"

    features = ["log_amount", "interest_rate", "tenor_days"]
    X = df[features].fillna(0)

    model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
    df["flag"] = model.fit_predict(X) == -1
    df["anomaly_score"] = -model.decision_function(X)   # higher = more unusual

    flagged = df[df["flag"]].sort_values("anomaly_score", ascending=False)
    print(f"Isolation Forest flagged {len(flagged)} of {len(df)} loans as multivariate outliers "
          f"(contamination={CONTAMINATION}).")

    cur = conn.cursor()
    cur.execute("SELECT NVL(MAX(anomaly_id), 0) FROM dq_anomalies")
    next_id = cur.fetchone()[0] + 1

    for i, (_, r) in enumerate(flagged.iterrows()):
        why = []
        if r["tenor_days"] < 0:
            why.append("negative or missing loan tenor")
        elif r["tenor_days"] > 30 * 365:
            why.append(f"unusually long tenor ({int(r['tenor_days'])} days)")
        if r["interest_rate"] is not None and (r["interest_rate"] < 5 or r["interest_rate"] > 20):
            why.append(f"interest rate {r['interest_rate']} outside the typical 5-20% band")
        if not why:
            why.append("unusual combination of amount, rate and tenor relative to other loans")
        desc = (f"Isolation Forest score {r['anomaly_score']:.3f} (higher = more unusual). "
                f"Likely reason: {'; '.join(why)}.")

        cur.execute("""
            INSERT INTO dq_anomalies (anomaly_id, anomaly_type, severity, table_name, record_id, description)
            VALUES (:1, 'ML_ISOLATION_FOREST', :2, 'LOAN_ACCOUNTS', :3, :4)
        """, [next_id + i, "MEDIUM", r["loan_account_no"], desc])

    conn.commit()
    print(f"Inserted {len(flagged)} ML_ISOLATION_FOREST anomalies into dq_anomalies for human review.")
    print("\nTop 5 most unusual loans:")
    print(flagged[["loan_account_no", "disbursement_amount", "interest_rate", "tenor_days", "anomaly_score"]]
         .head(5).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
