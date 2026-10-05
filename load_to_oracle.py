#!/usr/bin/env python3
"""
Load the bank CSV files into Oracle from Python (no SQL Developer import wizard needed).

Setup (PyCharm terminal):   pip install oracledb
Run:                        python load_to_oracle.py            -> creates tables + loads all CSVs
                            python load_to_oracle.py --dry-run  -> only checks the CSVs, no database

Fill in the 3 connection settings below (or set the environment variables
ORA_USER / ORA_PASSWORD / ORA_DSN). The SQL Developer connection screen shows the same values:
right-click your connection -> Properties -> host, port, service name (or SID).
"""
import csv
import datetime as dt
import decimal
import os
import sys

# ------------------------------------------------------------------ settings
DB_USER = os.getenv("ORA_USER", "YOUR_USERNAME")
DB_PASSWORD = os.getenv("ORA_PASSWORD", "YOUR_PASSWORD")
# Service-name style:  "localhost:1521/XEPDB1"   (Oracle XE 18c/21c)  or  "localhost:1521/ORCLPDB1"
# SID style:           use oracledb.makedsn("localhost", 1521, sid="XE") instead
DB_DSN = os.getenv("ORA_DSN", "localhost:1521/XEPDB1")

DATA_DIR = os.getenv("DATA_DIR", ".")           # folder that holds the CSVs + 01_create_tables_oracle.sql
DROP_EXISTING = True                            # True = DROP the tables below first (old data is lost!)
BATCH_SIZE = 10_000

# load order (small reference tables first, biggest last)
TABLES = ["pincode_reference", "branch_master", "customer_master", "account_master", "loan_accounts",
          "batch_log", "correction_history", "error_log_ground_truth", "transactions"]

DATE_COLS = {"opened_date", "dob", "kyc_date", "customer_since", "load_date", "open_date", "disbursement_date",
             "maturity_date", "txn_date", "reviewed_on"}
NUM_COLS = {"annual_income", "balance", "tenure_months", "interest_rate", "emi_amount", "outstanding_principal",
            "overdue_days", "amount", "balance_after", "duration_sec", "row_count", "confidence_score",
            "disbursement_amount"}


def convert(col, v):
    """CSV text -> Python value. Empty cell -> NULL."""
    if v == "":
        return None
    if col in DATE_COLS:
        return dt.datetime.strptime(v, "%Y-%m-%d")
    if col in NUM_COLS:
        return decimal.Decimal(v)
    return v


def read_batches(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        yield header
        batch = []
        for row in rd:
            batch.append([convert(c, v) for c, v in zip(header, row)])
            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch


def dry_run():
    for t in TABLES:
        path = os.path.join(DATA_DIR, t + ".csv")
        total = 0
        it = read_batches(path)
        header = next(it)
        for b in it:
            total += len(b)
        print(f"{t:26s} columns={len(header):2d} rows={total:>9,}  OK")


def run_ddl(cur):
    if DROP_EXISTING:
        for t in reversed(TABLES):
            try:
                cur.execute(f"DROP TABLE {t} PURGE")
                print(f"dropped {t}")
            except Exception as e:                       # ORA-00942: table does not exist -> fine
                if "ORA-00942" not in str(e):
                    raise
    with open(os.path.join(DATA_DIR, "01_create_tables_oracle.sql"), encoding="utf-8") as fh:
        sql = "\n".join(l for l in fh.read().splitlines() if not l.strip().startswith("--"))
    for stmt in sql.split(";"):
        if stmt.strip():
            cur.execute(stmt)
    print("tables created")


def load():
    import oracledb                                         # pip install oracledb (thin mode, no client needed)
    conn = oracledb.connect(user="system", password="suman_123456", host="localhost", port=1521, service_name="XE")
    print("connected to", DB_DSN, "as", DB_USER)
    cur = conn.cursor()
    run_ddl(cur)
    for t in TABLES:
        it = read_batches(os.path.join(DATA_DIR, t + ".csv"))
        header = next(it)
        sql = f"INSERT INTO {t} ({','.join(header)}) VALUES ({','.join(f':{i + 1}' for i in range(len(header)))})"
        total = bad = 0
        for batch in it:
            cur.executemany(sql, batch, batcherrors=True)
            errs = cur.getbatcherrors()
            bad += len(errs)
            for e in errs[:3]:
                print(f"   ! {t} row {total + e.offset + 2}: {e.message.strip()}")
            total += len(batch)
            print(f"   {t}: {total:,} rows read", end="\r")
        conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"{t:26s} loaded={cur.fetchone()[0]:>9,}  rejected={bad}          ")
    conn.close()
    print("done")


if __name__ == "__main__":
    dry_run() if "--dry-run" in sys.argv else load()
