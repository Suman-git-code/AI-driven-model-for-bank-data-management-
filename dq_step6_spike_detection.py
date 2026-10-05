#!/usr/bin/env python3
"""
STEP 35 - Statistical spike detection (action step 35)
"Configure anomaly detection rules and sensitivity thresholds per data
 domain... flag if average loan disbursement amount spikes 5x above the
 90-day rolling average."

Reads transactions and loan_accounts from Oracle, calculates a ROBUST
rolling Z-score per day (median + MAD instead of mean + std dev, because
amounts are highly skewed - a plain average is thrown off by the skew
itself). Any day whose Z-score is far outside the recent normal range is
written into dq_anomalies as an AMOUNT_SPIKE anomaly.

Why debit-only for transactions: monthly salary credits are large and
land on the 1st-5th of every month - that is a real, repeating pattern,
not an anomaly, so it is excluded to avoid false alarms. Loans use ALL
disbursements (no similar recurring pattern to exclude).

Run:  python dq_step6_spike_detection.py
"""
import numpy as np
import pandas as pd
import oracledb

DB_USER = "system"
DB_PASSWORD = "suman_123456"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "xe"

Z_THRESHOLD_TXN = 5.0          # robust Z-score cutoff for transactions
Z_THRESHOLD_LOAN = 3.5         # robust Z-score cutoff for loans
MIN_ROWS_PER_DAY_TXN = 30      # ignore days with too few transactions to average meaningfully
MIN_ROWS_PER_DAY_LOAN = 8
WINDOW = 60                     # rolling window, in days, used as the "recent normal" baseline
MIN_PERIODS = 20


def robust_zscore(daily_avg):
    """Median + MAD based rolling Z-score - robust to the natural skew in money amounts."""
    roll_median = daily_avg.rolling(WINDOW, min_periods=MIN_PERIODS).median().shift(1)
    mad = (daily_avg - roll_median).abs().rolling(WINDOW, min_periods=MIN_PERIODS).median().shift(1)
    return 0.6745 * (daily_avg - roll_median) / mad.replace(0, np.nan)


def detect_transaction_spikes(conn):
    df = pd.read_sql("""
        SELECT txn_id, account_number, txn_date, amount
          FROM transactions
         WHERE txn_type = 'DEBIT' AND channel <> 'ACH'
    """, conn)
    df.columns = df.columns.str.lower()
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    daily = df.groupby("txn_date").agg(avg_amount=("amount", "mean"), n=("txn_id", "count"))
    daily = daily[daily.n >= MIN_ROWS_PER_DAY_TXN]
    daily["z"] = robust_zscore(daily["avg_amount"])
    baseline = daily["avg_amount"].rolling(WINDOW, min_periods=MIN_PERIODS).median().shift(1)
    spike_days = daily[daily["z"].abs() > Z_THRESHOLD_TXN]

    rows = []
    for d, r in spike_days.iterrows():
        base = baseline.loc[d]
        mult = round(r["avg_amount"] / base, 1) if base and base > 0 else None
        rows.append(("AMOUNT_SPIKE", "HIGH" if r["z"] > 8 else "MEDIUM", "TRANSACTIONS",
                     d.strftime("%Y-%m-%d"), None,
                     f"Average debit transaction amount on {d:%Y-%m-%d} was Rs.{r['avg_amount']:,.0f} "
                     f"({mult}x the recent normal of Rs.{base:,.0f}), robust Z-score {r['z']:.1f}, "
                     f"based on {int(r['n'])} transactions"))
    return rows


def detect_loan_spikes(conn):
    df = pd.read_sql("""
        SELECT loan_account_no, disbursement_date, disbursement_amount
          FROM loan_accounts
         WHERE disbursement_amount IS NOT NULL
    """, conn)
    df.columns = df.columns.str.lower()
    df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])
    daily = df.groupby("disbursement_date").agg(avg_amount=("disbursement_amount", "mean"),
                                                 n=("loan_account_no", "count"))
    daily = daily[daily.n >= MIN_ROWS_PER_DAY_LOAN]
    daily["z"] = robust_zscore(daily["avg_amount"])
    baseline = daily["avg_amount"].rolling(WINDOW, min_periods=MIN_PERIODS).median().shift(1)
    spike_days = daily[daily["z"].abs() > Z_THRESHOLD_LOAN]

    rows = []
    for d, r in spike_days.iterrows():
        base = baseline.loc[d]
        mult = round(r["avg_amount"] / base, 1) if base and base > 0 else None
        rows.append(("AMOUNT_SPIKE", "HIGH" if r["z"] > 6 else "MEDIUM", "LOAN_ACCOUNTS",
                     d.strftime("%Y-%m-%d"), None,
                     f"Average loan disbursement on {d:%Y-%m-%d} was Rs.{r['avg_amount']:,.0f} "
                     f"({mult}x the recent normal of Rs.{base:,.0f}), robust Z-score {r['z']:.1f}, "
                     f"based on {int(r['n'])} loans"))
    return rows


def main():
    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                            service_name=DB_SERVICE)
    txn_rows = detect_transaction_spikes(conn)
    loan_rows = detect_loan_spikes(conn)
    all_rows = txn_rows + loan_rows
    print(f"Found {len(txn_rows)} transaction-spike day(s), {len(loan_rows)} loan-spike day(s)")

    cur = conn.cursor()
    cur.execute("SELECT NVL(MAX(anomaly_id), 0) FROM dq_anomalies")
    next_id = cur.fetchone()[0] + 1
    for i, (atype, sev, table, rec_id, rel, desc) in enumerate(all_rows):
        cur.execute("""
            INSERT INTO dq_anomalies (anomaly_id, anomaly_type, severity, table_name, record_id,
                                      related_table, description)
            VALUES (:1, :2, :3, :4, :5, :6, :7)
        """, [next_id + i, atype, sev, table, rec_id, rel, desc])
    conn.commit()
    print(f"Inserted {len(all_rows)} spike anomalies into dq_anomalies")

    print("\n--- details ---")
    for r in all_rows:
        print(f"[{r[1]:6s}] {r[2]:16s} {r[3]}  {r[5]}")
    conn.close()


if __name__ == "__main__":
    main()
