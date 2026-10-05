#!/usr/bin/env python3
"""
STEPS 38-39 - Automated Anomaly Report + Dashboard  (action steps 38, 39)

Extends the step-30 dashboard with an ANOMALY section:
  - anomaly counts by domain (table) and severity           (step 38)
  - root-cause split: isolated errors vs upstream failures   (step 37 result)
  - the suspect batches, with a recommended action           (step 38)
  - a live anomaly dashboard panel                           (step 39)

Run any time after re-running dq_step5/6/7 (Oracle) to refresh the page.

Setup:   pip install oracledb
Run:     python dq_step8_dashboard_v2.py
Output:  dq_dashboard.html
"""
import datetime as dt
import json
import webbrowser

import oracledb

DB_USER = "system"
DB_PASSWORD = "suman_123456"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "xe"

OUT_FILE = "dq_dashboard.html"
OPEN_IN_BROWSER = True

RECOMMENDED_ACTION = {
    "ORPHAN_RECORD": "Trace the missing parent record; likely a late-arriving or failed upstream load. Quarantine until resolved.",
    "BROKEN_DATE_LOGIC": "Verify against the source document (loan agreement). Likely a data entry swap of two date fields.",
    "RELATIONSHIP_MISMATCH": "Confirm the customer's current address; PIN may be outdated rather than wrong.",
    "AMOUNT_SPIKE": "Check whether a bulk/batch disbursement or a promotional campaign explains the volume before treating as an error.",
}


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or {})
    cols = [d[0].lower() for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def main():
    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
                            service_name=DB_SERVICE)
    cur = conn.cursor()

    # ---- DQ score section (from step 29) ----
    overall = fetch_all(cur, """
        SELECT ROUND(AVG(pass_rate_pct), 2) AS score FROM dq_rule_results
         WHERE run_id = (SELECT MAX(run_id) FROM dq_rule_results)
    """)[0]["score"]

    by_table = fetch_all(cur, """
        SELECT table_name, COUNT(*) AS rules_applied, ROUND(AVG(pass_rate_pct), 2) AS score,
               SUM(violation_count) AS violations
          FROM dq_rule_results WHERE run_id = (SELECT MAX(run_id) FROM dq_rule_results)
         GROUP BY table_name ORDER BY score ASC
    """)

    trend = fetch_all(cur, """
        SELECT run_id, ROUND(AVG(pass_rate_pct), 2) AS score
          FROM dq_rule_results GROUP BY run_id ORDER BY run_id
    """)

    # ---- Anomaly section (steps 36-37) ----
    anomaly_total = fetch_all(cur, "SELECT COUNT(*) AS c FROM dq_anomalies")[0]["c"]

    by_type_sev = fetch_all(cur, """
        SELECT anomaly_type, severity, COUNT(*) AS cnt
          FROM dq_anomalies GROUP BY anomaly_type, severity ORDER BY cnt DESC
    """)

    by_table_anom = fetch_all(cur, """
        SELECT table_name, COUNT(*) AS cnt
          FROM dq_anomalies GROUP BY table_name ORDER BY cnt DESC
    """)

    root_cause = fetch_all(cur, """
        SELECT NVL(root_cause, 'NOT_CLASSIFIED') AS root_cause, COUNT(*) AS cnt
          FROM dq_anomalies GROUP BY NVL(root_cause, 'NOT_CLASSIFIED') ORDER BY cnt DESC
    """)

    suspect_batches = fetch_all(cur, """
        SELECT table_name, source_system, batch_id, COUNT(*) AS cnt,
               MAX(anomaly_type) AS sample_type
          FROM dq_anomalies WHERE root_cause = 'UPSTREAM_FEED_FAILURE'
         GROUP BY table_name, source_system, batch_id ORDER BY cnt DESC
    """)

    sample_isolated = fetch_all(cur, """
        SELECT anomaly_id, table_name, record_id, anomaly_type, severity, description
          FROM dq_anomalies WHERE root_cause = 'ISOLATED_ENTRY_ERROR'
         ORDER BY anomaly_id FETCH FIRST 15 ROWS ONLY
    """)

    # ---- Accuracy vs ground truth (step 40) ----
    accuracy_checks = [
        ("Orphan accounts", "ACCOUNT_MASTER", "ORPHAN_RECORD", "ACCOUNT_MASTER", "ORPHAN_CUSTOMER_ID"),
        ("Orphan loans", "LOAN_ACCOUNTS", "ORPHAN_RECORD", "LOAN_ACCOUNTS", "ORPHAN_CUSTOMER_ID"),
        ("Orphan transactions", "TRANSACTIONS", "ORPHAN_RECORD", "TRANSACTIONS", "ORPHAN_ACCOUNT_NUMBER"),
    ]
    accuracy_rows = []
    for label, a_table, a_type, e_table, e_type in accuracy_checks:
        r = fetch_all(cur, """
            SELECT
              (SELECT COUNT(*) FROM dq_anomalies a WHERE a.table_name=:t1 AND a.anomaly_type=:at
                 AND EXISTS (SELECT 1 FROM error_log_ground_truth e WHERE UPPER(e.table_name)=:t2
                              AND e.record_id=a.record_id AND e.error_type=:et)) AS tp,
              (SELECT COUNT(*) FROM dq_anomalies a WHERE a.table_name=:t1 AND a.anomaly_type=:at
                 AND NOT EXISTS (SELECT 1 FROM error_log_ground_truth e WHERE UPPER(e.table_name)=:t2
                                 AND e.record_id=a.record_id AND e.error_type=:et)) AS fp,
              (SELECT COUNT(*) FROM error_log_ground_truth e WHERE UPPER(e.table_name)=:t2 AND e.error_type=:et
                 AND NOT EXISTS (SELECT 1 FROM dq_anomalies a WHERE a.table_name=:t1
                                 AND a.anomaly_type=:at AND a.record_id=e.record_id)) AS fn
            FROM dual
        """, {"t1": a_table, "at": a_type, "t2": e_table, "et": e_type})[0]
        accuracy_rows.append({"label": label, "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})

    r = fetch_all(cur, """
        SELECT
          (SELECT COUNT(*) FROM dq_anomalies a WHERE a.anomaly_type='BROKEN_DATE_LOGIC'
             AND EXISTS (SELECT 1 FROM error_log_ground_truth e WHERE UPPER(e.table_name)='LOAN_ACCOUNTS'
                          AND e.record_id=a.record_id
                          AND e.error_type IN ('MATURITY_BEFORE_DISBURSEMENT','MATURITY_OVER_30_YEARS'))) AS tp,
          (SELECT COUNT(*) FROM dq_anomalies a WHERE a.anomaly_type='BROKEN_DATE_LOGIC'
             AND NOT EXISTS (SELECT 1 FROM error_log_ground_truth e WHERE UPPER(e.table_name)='LOAN_ACCOUNTS'
                             AND e.record_id=a.record_id
                             AND e.error_type IN ('MATURITY_BEFORE_DISBURSEMENT','MATURITY_OVER_30_YEARS'))) AS fp,
          (SELECT COUNT(*) FROM error_log_ground_truth e WHERE UPPER(e.table_name)='LOAN_ACCOUNTS'
             AND e.error_type IN ('MATURITY_BEFORE_DISBURSEMENT','MATURITY_OVER_30_YEARS')
             AND NOT EXISTS (SELECT 1 FROM dq_anomalies a WHERE a.anomaly_type='BROKEN_DATE_LOGIC'
                             AND a.record_id=e.record_id)) AS fn
        FROM dual
    """)[0]
    accuracy_rows.append({"label": "Broken date logic", "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})

    tp_sum = sum(x["tp"] for x in accuracy_rows)
    fp_sum = sum(x["fp"] for x in accuracy_rows)
    fn_sum = sum(x["fn"] for x in accuracy_rows)
    precision = round(100 * tp_sum / max(tp_sum + fp_sum, 1), 1)
    recall = round(100 * tp_sum / max(tp_sum + fn_sum, 1), 1)

    conn.close()

    data = dict(overall=overall, by_table=by_table, trend=trend, anomaly_total=anomaly_total,
               by_type_sev=by_type_sev, by_table_anom=by_table_anom, root_cause=root_cause,
               suspect_batches=suspect_batches, sample_isolated=sample_isolated,
               accuracy_rows=accuracy_rows, precision=precision, recall=recall,
               generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))

    html = build_html(data)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Dashboard written to {OUT_FILE}  (DQ score {overall}%, anomalies {anomaly_total})")
    if OPEN_IN_BROWSER:
        webbrowser.open(OUT_FILE)


def color(score):
    if score is None:
        return "#999"
    return "#1e8e3e" if score >= 99 else "#e8a400" if score >= 95 else "#d93025"


def sev_color(sev):
    return {"CRITICAL": "#d93025", "HIGH": "#e8710a", "MEDIUM": "#e8a400"}.get(sev, "#9aa0a6")


def build_html(d):
    table_rows = "".join(
        f"<tr><td>{r['table_name']}</td><td>{r['rules_applied']}</td>"
        f"<td style='color:{color(r['score'])};font-weight:600'>{r['score']}%</td>"
        f"<td>{r['violations']:,}</td></tr>" for r in d["by_table"])

    type_sev_rows = "".join(
        f"<tr><td>{r['anomaly_type']}</td>"
        f"<td style='color:{sev_color(r['severity'])};font-weight:600'>{r['severity']}</td>"
        f"<td>{r['cnt']:,}</td></tr>" for r in d["by_type_sev"])

    table_anom_rows = "".join(f"<tr><td>{r['table_name']}</td><td>{r['cnt']:,}</td></tr>"
                              for r in d["by_table_anom"])

    rc_total = sum(r["cnt"] for r in d["root_cause"]) or 1
    rc_rows = "".join(
        f"<tr><td>{r['root_cause']}</td><td>{r['cnt']:,}</td>"
        f"<td>{round(100 * r['cnt'] / rc_total, 1)}%</td></tr>" for r in d["root_cause"])

    batch_rows = "".join(
        f"<tr><td>{r['table_name']}</td><td>{r['source_system']}</td><td>{r['batch_id']}</td>"
        f"<td style='color:#d93025;font-weight:700'>{r['cnt']}</td>"
        f"<td>{RECOMMENDED_ACTION.get(r['sample_type'], 'Investigate this batch as a group before reviewing individual records.')}</td></tr>"
        for r in d["suspect_batches"])
    if not batch_rows:
        batch_rows = "<tr><td colspan='5' style='color:#9aa0a6'>No batches currently flagged as feed failures.</td></tr>"

    iso_rows = "".join(
        f"<tr><td>{r['anomaly_id']}</td><td>{r['table_name']}</td><td>{r['record_id']}</td>"
        f"<td>{r['anomaly_type']}</td>"
        f"<td style='color:{sev_color(r['severity'])}'>{r['severity']}</td>"
        f"<td>{r['description']}</td></tr>" for r in d["sample_isolated"])

    trend_labels = json.dumps([f"Run {t['run_id']}" for t in d["trend"]])
    trend_scores = json.dumps([t["score"] for t in d["trend"]])
    type_labels = json.dumps(sorted({r["anomaly_type"] for r in d["by_type_sev"]}))
    type_totals = {}
    for r in d["by_type_sev"]:
        type_totals[r["anomaly_type"]] = type_totals.get(r["anomaly_type"], 0) + r["cnt"]
    type_values = json.dumps([type_totals[t] for t in json.loads(type_labels)])
    rc_labels = json.dumps([r["root_cause"] for r in d["root_cause"]])
    rc_values = json.dumps([r["cnt"] for r in d["root_cause"]])

    acc_labels = json.dumps([r["label"] for r in d["accuracy_rows"]])
    acc_tp = json.dumps([r["tp"] for r in d["accuracy_rows"]])
    acc_fp = json.dumps([r["fp"] for r in d["accuracy_rows"]])
    acc_rows_html = "".join(
        f"<tr><td>{r['label']}</td><td style='color:#1e8e3e'>{r['tp']}</td>"
        f"<td style='color:#d93025'>{r['fp']}</td><td style='color:#e8a400'>{r['fn']}</td></tr>"
        for r in d["accuracy_rows"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Bank Data Quality Governance Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{ --bg:#0f1115; --card:#171a21; --text:#e8eaed; --muted:#9aa0a6; --border:#2a2e37; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
         background: var(--bg); color: var(--text); padding: 24px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 17px; margin: 32px 0 12px; border-top: 1px solid var(--border); padding-top: 20px; }}
  .sub {{ color: var(--muted); margin-bottom: 20px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 16px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }}
  .tile {{ text-align:center; }}
  .tile .num {{ font-size: 42px; font-weight: 700; }}
  .tile .lbl {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }}
  .section-title {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; }}
  canvas {{ max-height: 240px; }}
</style>
</head>
<body>
  <h1>Bank Data Quality Governance Dashboard</h1>
  <div class="sub">Metric 04.01.05 &middot; Generated {d['generated']}</div>

  <div class="grid">
    <div class="card tile"><div class="num" style="color:{color(d['overall'])}">{d['overall']}%</div>
      <div class="lbl">Overall Data Quality Score</div></div>
    <div class="card tile"><div class="num" style="color:#e8a400">{d['anomaly_total']:,}</div>
      <div class="lbl">Total Open Anomalies</div></div>
    <div class="card tile"><div class="num" style="color:#d93025">{len(d['suspect_batches'])}</div>
      <div class="lbl">Suspect Batches (Upstream Feed Failures)</div></div>
  </div>

  <h2>Data Quality Scores (Steps 28-30)</h2>
  <div class="grid">
    <div class="card"><div class="section-title">Score by Table</div>
      <table><thead><tr><th>Table</th><th>Rules</th><th>Score</th><th>Violations</th></tr></thead>
      <tbody>{table_rows}</tbody></table></div>
    <div class="card"><div class="section-title">Score Trend Across Runs</div>
      <canvas id="trendChart"></canvas></div>
  </div>

  <h2>Anomaly Report (Steps 33-38)</h2>
  <div class="grid">
    <div class="card"><div class="section-title">Anomalies by Type</div>
      <canvas id="typeChart"></canvas></div>
    <div class="card"><div class="section-title">Root Cause Split</div>
      <canvas id="rcChart"></canvas></div>
    <div class="card"><div class="section-title">Anomalies by Table</div>
      <table><thead><tr><th>Table</th><th>Count</th></tr></thead><tbody>{table_anom_rows}</tbody></table></div>
  </div>

  <div class="grid" style="grid-template-columns: 1fr;">
    <div class="card">
      <div class="section-title">Anomaly Count by Type &amp; Severity</div>
      <table><thead><tr><th>Anomaly Type</th><th>Severity</th><th>Count</th></tr></thead>
      <tbody>{type_sev_rows}</tbody></table>
    </div>

    <div class="card">
      <div class="section-title">Root Cause Summary</div>
      <table><thead><tr><th>Root Cause</th><th>Count</th><th>% of Total</th></tr></thead>
      <tbody>{rc_rows}</tbody></table>
    </div>

    <div class="card">
      <div class="section-title">Suspect Batches - Investigate First (Step 37)</div>
      <table><thead><tr><th>Table</th><th>Source System</th><th>Batch</th><th>Anomalies</th><th>Recommended Action</th></tr></thead>
      <tbody>{batch_rows}</tbody></table>
    </div>

    <div class="card">
      <div class="section-title">Sample Isolated Errors (need individual review - Step 45 queue)</div>
      <table><thead><tr><th>ID</th><th>Table</th><th>Record</th><th>Type</th><th>Severity</th><th>Description</th></tr></thead>
      <tbody>{iso_rows}</tbody></table>
    </div>
  </div>

  <h2>Detection Accuracy vs Ground Truth (Step 40)</h2>
  <div class="grid">
    <div class="card tile"><div class="num" style="color:{color(d['precision'])}">{d['precision']}%</div>
      <div class="lbl">Precision - of what we flagged, how much was real</div></div>
    <div class="card tile"><div class="num" style="color:{color(d['recall'])}">{d['recall']}%</div>
      <div class="lbl">Recall - of everything real, how much we caught</div></div>
  </div>
  <div class="grid" style="grid-template-columns: 1fr;">
    <div class="card">
      <div class="section-title">True vs False Positives by Check</div>
      <canvas id="accChart"></canvas>
    </div>
    <div class="card">
      <div class="section-title">Accuracy Detail</div>
      <table><thead><tr><th>Check</th><th>True Positive</th><th>False Positive</th><th>False Negative</th></tr></thead>
      <tbody>{acc_rows_html}</tbody></table>
    </div>
  </div>

<script>
new Chart(document.getElementById('trendChart'), {{
  type: 'line',
  data: {{ labels: {trend_labels}, datasets: [{{ label: 'Score %', data: {trend_scores},
           borderColor: '#4f9cff', backgroundColor: '#4f9cff33', fill: true, tension: 0.3 }}] }},
  options: {{ scales: {{ y: {{ min: 0, max: 100, ticks: {{ color: '#9aa0a6' }} }}, x: {{ ticks: {{ color: '#9aa0a6' }} }} }},
             plugins: {{ legend: {{ display: false }} }} }}
}});
new Chart(document.getElementById('typeChart'), {{
  type: 'bar',
  data: {{ labels: {type_labels}, datasets: [{{ label: 'Count', data: {type_values}, backgroundColor: '#4f9cff' }}] }},
  options: {{ indexAxis: 'y', scales: {{ x: {{ ticks: {{ color: '#9aa0a6' }} }}, y: {{ ticks: {{ color: '#9aa0a6' }} }} }},
             plugins: {{ legend: {{ display: false }} }} }}
}});
new Chart(document.getElementById('rcChart'), {{
  type: 'doughnut',
  data: {{ labels: {rc_labels}, datasets: [{{ data: {rc_values}, backgroundColor: ['#e8a400','#d93025','#4f9cff'] }}] }},
  options: {{ plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#e8eaed' }} }} }} }}
}});
new Chart(document.getElementById('accChart'), {{
  type: 'bar',
  data: {{ labels: {acc_labels}, datasets: [
    {{ label: 'True Positive', data: {acc_tp}, backgroundColor: '#1e8e3e' }},
    {{ label: 'False Positive', data: {acc_fp}, backgroundColor: '#d93025' }} ] }},
  options: {{ scales: {{ y: {{ ticks: {{ color: '#9aa0a6' }} }}, x: {{ ticks: {{ color: '#9aa0a6' }} }} }},
             plugins: {{ legend: {{ labels: {{ color: '#e8eaed' }} }} }} }}
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
