# AI-Driven Bank Data Quality Management System

An end-to-end data quality platform for banking data, built on Oracle and Python. It combines **rule-based SQL checks**, **statistical anomaly detection**, and **trained, validated machine learning models** to automatically profile data, detect errors, diagnose root causes, and suggest corrections — all routed through a human-approval queue with a full audit trail.

Built against Metric 04.01.05 — *AI-Driven Modules for Data Management* (automatic rule generation, AI-assisted anomaly detection, AI-assisted correction workflows).

> All data used in this project is **synthetic**. No real customer, account, or transaction data is used anywhere.

---

## What this project does

1. **Generates a realistic synthetic bank dataset** (~100k customers, 130k accounts, 30k loans, 480k transactions) with deliberately injected data-quality errors and a hidden ground-truth log, so detection accuracy can be objectively measured.
2. **Profiles the data and auto-generates data-quality rules**, scored and approved through a review workflow.
3. **Detects anomalies**: broken relationships between tables, impossible date logic, and statistical spikes in transaction/loan amounts (robust rolling Z-score).
4. **Diagnoses root cause**: distinguishes a one-off data-entry mistake from a whole batch/feed failure, by tracing anomalies back to their source system and load date.
5. **Trains and validates real ML models**:
   - A **Random Forest classifier** that studies a table's own data, learns its own rules (no hard-coded formats), and detects erroneous records — with a proper train/validation/test split and cross-validation.
   - A **Logistic Regression classifier** that predicts how likely a reviewer is to approve a given correction, trained on historical approval decisions.
   - An **Isolation Forest** model for multivariate anomaly detection on loans, evaluated honestly as a complementary (not primary) detection layer.
6. **Suggests corrections** with a confidence score and a plain-English reason, routed into a **human review queue** — nothing is applied to production data without approval.
7. **Applies approved fixes** through a staged workflow (staging → validation → production) with a complete **audit trail** of every change.
8. **Retrains monthly**, recalculating confidence scores from the latest approve/reject decisions.
9. Ships a **Streamlit web app** so new data can be checked from a browser, no code required.

---

## Architecture

```
                 ┌─────────────────────┐
                 │   Oracle Database    │
                 │  (customer/account/  │
                 │  loan/transaction)    │
                 └─────────┬────────────┘
                           │
      ┌────────────────────┼────────────────────┐
      ▼                                          ▼
┌─────────────┐                         ┌──────────────────┐
│  SQL/PL-SQL │                         │   Python / ML     │
│   pipeline   │                         │     pipeline       │
│  (rules,     │                         │ (self-learning     │
│  scores,     │                         │  rules, trained    │
│  anomalies,  │                         │  classifiers,       │
│  root cause) │                         │  Isolation Forest)  │
└──────┬───────┘                         └─────────┬─────────┘
       │                                            │
       └───────────────┬────────────────────────────┘
                        ▼
              ┌───────────────────┐
              │   dq_corrections    │   ← shared human review queue
              │  (PENDING/APPROVED/ │
              │     REJECTED)       │
              └─────────┬──────────┘
                        ▼
         staged apply → production + audit trail
                        │
                        ▼
              ┌───────────────────┐
              │  Live dashboard     │  (HTML + Chart.js)
              │  Streamlit app      │  (browser front end)
              └───────────────────┘
```

---

## Project structure

### 1. Data generation and loading
| File | Purpose |
|---|---|
| `generate_realistic_bank_data.py` | Generates the synthetic dataset (9 CSVs) with realistic formats and injected errors |
| `load_to_oracle.py` | Loads all CSVs into Oracle |
| `01_create_tables_oracle.sql` | Table DDL |

### 2. Rule-based SQL pipeline (run in order)
| File | Action steps | Purpose |
|---|---|---|
| `dq_step1_profile.sql` / `.py` | 27 | Statistical profiling of every column |
| `dq_step2_rules.sql` | 28 | AI-suggested rules, pending owner approval |
| `dq_step3_scores.sql` | 29 | Calculates DQ scores per rule/table/dimension |
| `dq_step5_anomalies.sql` | 36 | Relationship checks (orphan records) and broken date logic |
| `dq_step6_spike_detection.py` | 35 | Robust Z-score statistical spike detection |
| `dq_step7_root_cause.sql` | 37 | Classifies anomalies: isolated error vs. upstream feed failure |
| `dq_step8_dashboard_v2.py` | 30, 38–40 | Live dashboard: scores, anomalies, root cause, accuracy |
| `dq_step9_accuracy.sql` | 40 | Precision/recall vs. hidden ground truth |
| `dq_step10_corrections.sql` | 41–44 | Rule-based correction suggestions with history-based confidence |
| `dq_step11_review_apply_audit.sql` | 45–47 | Review queue, staged apply, audit trail |
| `dq_step12_monthly_retrain.sql` | 48 | Recalculates confidence scores from the latest decisions |

### 3. Machine learning pipeline
| File | Purpose |
|---|---|
| `dq_ml_pipeline1.py` | Core `DQPipeline` class: learns rules from data, trains/validates a Random Forest error detector, generates correction suggestions, saves/loads the trained model |
| `dq_step13_ml_confidence.py` | Logistic Regression model for correction-approval confidence |
| `dq_step14_isolation_forest.py` | Isolation Forest for multivariate loan anomaly detection |
| `dq_step15_ml_to_corrections.py` | Pushes ML-generated suggestions into the shared review queue |
| `dq_check_new_data.py` | Applies an already-trained model to new data (CLI, no retraining) |
| `dq_frontend_app.py` | Streamlit browser app: upload/pull data, view dashboard, submit to review queue |

### 4. Documentation
| File | Purpose |
|---|---|
| `DQ_Project_Report_Suman_P.docx` | Full project report: methodology, results, traceability matrix |

---

## Setup

### Requirements
- Oracle Database (tested on Oracle XE 21c)
- Python 3.10+
- Oracle SQL Developer (recommended for running the `.sql` scripts)

### Install Python dependencies
```bash
pip install pandas numpy oracledb scikit-learn joblib streamlit
```

### Configure database credentials
Each Python script has these constants near the top — update them to match your environment:
```python
DB_USER = "system"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "XE"
```

---

## How to run it, end to end

```bash
# 1. Generate and load the dataset
python generate_realistic_bank_data.py .
python load_to_oracle.py

# 2. Run the SQL pipeline in SQL Developer, in order:
#    01_create_tables_oracle.sql (if not already run by the loader)
#    dq_step1_profile.sql → dq_step2_rules.sql → dq_step3_scores.sql
#    dq_step5_anomalies.sql → dq_step7_root_cause.sql → dq_step9_accuracy.sql
#    dq_step10_corrections.sql → dq_step11_review_apply_audit.sql

# 3. Run the spike detector and dashboard
python dq_step6_spike_detection.py
python dq_step8_dashboard_v2.py        # opens dq_dashboard.html

# 4. Train and validate the ML models
python dq_ml_pipeline.py               # trains, validates, saves dq_pipeline_trained.joblib
python dq_step13_ml_confidence.py
python dq_step14_isolation_forest.py
python dq_step15_ml_to_corrections.py  # pushes ML suggestions into the review queue
```

---

## Checking new data (day-to-day use)

Once a model is trained and saved, you don't need to retrain to check new data.

**Command line:**
```bash
python dq_check_new_data.py --csv new_customers.csv
python dq_check_new_data.py --since-days 1     # or pull recent Oracle rows instead
```

**Browser app:**
```bash
streamlit run dq_frontend_app.py
```
Opens at `http://localhost:8501` — upload a file or pull from Oracle, see the quality dashboard, and push findings to the review queue with one click.

> **Important:** a trained model's rules are tied to the exact column names of the table it was trained on. Train a separate model per table (e.g. one for `customer_master`, a separate one for `account_master`) — see *Limitations* below.

---

## Results (from the project's own test run)

| Metric | Result |
|---|---|
| Overall rule-based DQ score | 99.04% |
| Anomalies detected (relationships, dates, spikes) | 4,158+ |
| Root-cause classification | 4,081 isolated errors / 77 upstream feed-failure anomalies across 3 batches |
| Rule-based anomaly detection accuracy | 100% recall, 86.3% precision |
| ML error-detection classifier (Random Forest) | AUC 0.82–0.90 (validation ≈ test, no overfitting), 100% precision, 65–80% recall |
| ML correction-confidence classifier (Logistic Regression) | AUC 0.89, calibrated to within ≈1pp of true historical approval rates |
| Isolation Forest (loans, complementary layer) | ≈31–41% precision, ≈33–50% recall |
| Corrections resolved / rejected / pending (after review) | 7,615 / 1,201 / 794 |

Exact numbers vary slightly between runs because the underlying dataset is randomly generated each time.

---

## Limitations (known, and by design)

- **The ML pipeline checks one table's own columns, not relationships between tables.** Orphan foreign keys (e.g. an account pointing to a non-existent customer) and cross-column logic (e.g. a closed account with a nonzero balance) are **not** detectable by `dq_ml_pipeline.py` — they require `dq_step5_anomalies.sql`, which is built for exactly this.
- **A trained model only recognizes the columns it was trained on.** Applying a model trained on `customer_master` to a structurally different table (e.g. `account_master`) will silently skip almost every check, not error out. Train one model per table.
- **Recall is intentionally capped by design in several places.** Suggestions with no safe automatic fix (e.g. a 5-digit PIN code, a one-character-short PAN) are deliberately left for manual review rather than guessed — this is a safety choice, not a shortfall.
- **Isolation Forest underperforms the hand-written SQL rules on this project's specific injected errors**, because most of those errors are simple, single-column mistakes that a written rule catches more precisely than an unsupervised model can. It is included as a complementary layer, not a replacement.

---

## Tech stack

- **Database:** Oracle (SQL, PL/SQL)
- **Python:** pandas, NumPy, oracledb, scikit-learn, joblib
- **Dashboard:** HTML + Chart.js (generated by Python)
- **Front end:** Streamlit

---

## Author

Suman.P
