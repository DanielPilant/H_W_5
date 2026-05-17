"""
generate_report.py - Step 6 (Reporting) of the Real-Time Spam Filter project.

Reads `alerts.txt`, `console.consumer.txt`, and `emails_master.csv`,
computes Min, Median, p95, and Max latencies, merges true labels into the 
per-message consumer log, and emits a standalone HTML report (`full_report.html`).
"""

import html
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ALERTS_FILE = "alerts.txt"
CONSUMER_LOG = "console.consumer.txt"
MASTER_CSV = "emails_master.csv"
OUTPUT_HTML = "full_report.html"

LATENCY_COLS = ["kafka_transit_ms", "inference_ms", "e2e_ms"]
LATENCY_LABELS = {
    "kafka_transit_ms": "Kafka Transit Latency",
    "inference_ms":     "Inference Latency",
    "e2e_ms":           "End-to-End Latency",
}

# ---------------------------------------------------------------------------
def compute_latency_stats(alerts: pd.DataFrame) -> dict:
    out = {}
    for col in LATENCY_COLS:
        values = pd.to_numeric(alerts[col], errors="coerce").dropna()
        out[col] = {
            "min":    float(values.min()) if len(values) else float("nan"),
            "median": float(values.median()) if len(values) else float("nan"),
            "p95":    float(np.percentile(values, 95)) if len(values) else float("nan"),
            "max":    float(values.max()) if len(values) else float("nan"),
            "mean":   float(values.mean()) if len(values) else float("nan"),
            "n":      int(len(values)),
        }
    return out

def classify_row(true_label, final_vote) -> str:
    try:
        t = int(true_label)
        p = int(final_vote)
    except (TypeError, ValueError):
        return "unknown"
    if t == 1 and p == 1: return "spam-tp"
    if t == 0 and p == 1: return "fp"
    if t == 0 and p == 0: return "ham-tn"
    if t == 1 and p == 0: return "fn"
    return "unknown"

# ---------------------------------------------------------------------------
def build_html(merged: pd.DataFrame, stats: dict) -> str:
    counts = merged["row_class"].value_counts().to_dict()
    total = len(merged)

    # latency cards (top section)
    latency_cards = ""
    for col in LATENCY_COLS:
        s = stats[col]
        latency_cards += f"""
        <div class="card">
          <h3>{LATENCY_LABELS[col]}</h3>
          <p class="metric"><span class="lbl">Median</span> {s['median']:.3f} ms</p>
          <p class="metric"><span class="lbl">p95</span> {s['p95']:.3f} ms</p>
          <div style="margin-top: 8px; font-size: 13px; color: var(--muted);">
            <span class="lbl">Min:</span> {s['min']:.3f} ms | 
            <span class="lbl">Max:</span> {s['max']:.3f} ms
          </div>
          <p class="sub" style="margin-top:8px;">mean {s['mean']:.3f} ms &middot; n = {s['n']:,}</p>
        </div>
        """

    # outcome summary cards
    summary_cards = f"""
      <div class="card sm-card"><h4>Total processed</h4><p class="metric">{total:,}</p></div>
      <div class="card sm-card spam-tp"><h4>Spam correctly caught</h4><p class="metric">{counts.get('spam-tp', 0):,}</p></div>
      <div class="card sm-card fp"><h4>False positives</h4><p class="metric">{counts.get('fp', 0):,}</p></div>
      <div class="card sm-card ham-tn"><h4>Ham correctly passed</h4><p class="metric">{counts.get('ham-tn', 0):,}</p></div>
      <div class="card sm-card fn"><h4>Spam missed</h4><p class="metric">{counts.get('fn', 0):,}</p></div>
    """

    # per-message table rows - mapping actual logs to assignment visual layout
    rows_html = []
    for _, r in merged.iterrows():
        cls = r["row_class"]
        true_str = str(int(r["true_label"])) if pd.notna(r["true_label"]) else "?"
        
        rows_html.append(
            f'<tr class="{cls}">'
            f'<td>{html.escape(str(r["record_id"]))}</td>'
            f'<td>{html.escape(str(r.get("source", "")))}</td>'
            f'<td>{true_str}</td>'
            f'<td>{int(r["logistic_pred"])}</td>'
            f'<td>{int(r["svm_pred"])}</td>'
            f'<td>{int(r["rf_pred"])}</td>'
            f'<td>{int(r["final_vote"])}</td>'
            f'<td>{float(r["confidence_score"]):.4f}</td>'
            f'<td>{html.escape(str(r["timestamp_received"]))}</td>'
            f'</tr>'
        )
    table_html = "\n".join(rows_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Real-Time Spam Filter - Full Report</title>
<style>
  :root {{
    --bg: #f5f7fb;
    --card: #ffffff;
    --ink: #1f2937;
    --muted: #6b7280;
    --red: #ef4444;
    --orange: #f59e0b;
    --green: #10b981;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--ink);
  }}
  h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
  h2 {{ font-size: 20px; margin-top: 32px; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }}
  p.sub {{ color: var(--muted); margin: 0; font-size: 12px; }}
  .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 16px; }}
  .card {{ background: var(--card); border-radius: 10px; padding: 18px 20px; box-shadow: 0 1px 2px rgba(0,0,0,.06); }}
  .card h3 {{ margin: 0 0 12px 0; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
  .card h4 {{ margin: 0 0 6px 0; font-size: 13px; color: var(--muted); }}
  .metric {{ margin: 4px 0; font-size: 22px; font-weight: 600; }}
  .lbl {{ font-size: 12px; color: var(--muted); font-weight: 500; margin-right: 6px; display: inline-block; min-width: 56px; }}
  .sm-card.spam-tp {{ border-left: 4px solid var(--red); }}
  .sm-card.fp      {{ border-left: 4px solid var(--orange); }}
  .sm-card.ham-tn  {{ border-left: 4px solid var(--green); }}
  .sm-card.fn      {{ border-left: 4px solid #d97706; }}
  .table-wrap {{ max-height: 75vh; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }}
  th {{ background: #f3f4f6; position: sticky; top: 0; z-index: 1; }}
  tr.spam-tp {{ background: #fee2e2; }}
  tr.fp      {{ background: #fed7aa; }}
  tr.ham-tn  {{ background: #ffffff; }}
  tr.fn      {{ background: #fef3c7; }}
  tr.unknown {{ background: #f9fafb; color: var(--muted); }}
</style>
</head>
<body>
  <h1>Real-Time Spam Filter &mdash; Full Report</h1>
  <p class="sub">Generated from <code>alerts.txt</code> + <code>console.consumer.txt</code> + <code>emails_master.csv</code></p>
  <h2>Latency Metrics</h2>
  <div class="grid">{latency_cards}</div>
  <h2>Outcome Summary</h2>
  <div class="grid">{summary_cards}</div>
  <h2>Per-message Details</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>record_id</th><th>source</th><th>true_label</th>
          <th>logistic</th><th>svm</th><th>rf</th>
          <th>vote</th><th>confidence</th><th>timestamp_received</th>
        </tr>
      </thead>
      <tbody>
{table_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""

# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading logs...")
    alerts = pd.read_csv(ALERTS_FILE)
    consumer = pd.read_csv(CONSUMER_LOG)
    master = pd.read_csv(MASTER_CSV, usecols=["record_id", "label", "source"])

    print("Computing latency stats...")
    stats = compute_latency_stats(alerts)
    for col in LATENCY_COLS:
        s = stats[col]
        print(f"  {LATENCY_LABELS[col]:<24} min={s['min']:.3f}ms | median={s['median']:.3f}ms | p95={s['p95']:.3f}ms | max={s['max']:.3f}ms")

    print("Merging consumer log with true labels...")
    merged = consumer.merge(master, on="record_id", how="left")
    merged["true_label"] = pd.to_numeric(merged["label"], errors="coerce")
    
    # Matches the actual columns from your live consumer output
    merged["row_class"] = merged.apply(
        lambda r: classify_row(r["true_label"], r["final_vote"]), axis=1
    )

    print(f"Writing {OUTPUT_HTML} ({len(merged):,} rows)...")
    html_doc = build_html(merged, stats)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Done. Open '{OUTPUT_HTML}' in any browser.")

if __name__ == "__main__":
    main()