"""
consumer.py - Steps 4 & 5 (Real-Time Inference + Majority Voting) of the
Real-Time Spam Filter project.

For every message on Kafka topic `email-stream`:
  * Record t2 (receipt time) the instant the message arrives.
  * Parse t1 (timestamp_sent) injected by the producer.
  * Scale the 57 UCI features with the saved StandardScaler.
  * Predict with LogReg / SVM / RandomForest.
  * Majority vote -> final 0/1 prediction.
  * Average probability of the voted class across models -> confidence.
  * Record t3 (after inference); log results + latency metrics to disk.

Ctrl+C shuts down cleanly (logs flushed, consumer closed).
"""

import csv
import json
import pickle
import datetime

import numpy as np
from kafka import KafkaConsumer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "email-stream"
GROUP_ID = "spam-group-recovery"

SCALER_FILE = "scaler.pkl"
MODEL_FILES = {
    "logistic": "model_logistic.pkl",
    "svm":      "model_svm.pkl",
    "rf":       "model_rf.pkl",
}

FEATURE_PREFIXES = ("word_freq_", "char_freq_", "capital_")
FEATURE_HEADER_SOURCE = "emails_master.csv"   # we only read its header line

CONSUMER_LOG = "console.consumer.txt"
ALERTS_LOG = "alerts.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(csv_path: str) -> list:
    """Return the ordered list of 57 feature column names used during training."""
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
    cols = [c for c in header if c.startswith(FEATURE_PREFIXES)]
    if len(cols) != 57:
        print(f"WARNING: expected 57 feature columns, found {len(cols)} in {csv_path}")
    return cols


def parse_iso(ts: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(ts)


def ms_between(a: datetime.datetime, b: datetime.datetime) -> float:
    return (b - a).total_seconds() * 1000.0


def to_float(value) -> float:
    """Robust string-or-number -> float (CSV values arrive as strings)."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading scaler and models...")
    scaler = load_pickle(SCALER_FILE)
    models = {name: load_pickle(path) for name, path in MODEL_FILES.items()}
    feature_cols = load_feature_columns(FEATURE_HEADER_SOURCE)
    print(f"Loaded {len(models)} models, {len(feature_cols)} feature columns.")

    print(f"Connecting to Kafka {BOOTSTRAP_SERVERS}  topic={TOPIC}")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id=GROUP_ID,
        enable_auto_commit=True,
    )

    consumer_log = open(CONSUMER_LOG, "w", newline="", encoding="utf-8")
    alerts_log = open(ALERTS_LOG, "w", newline="", encoding="utf-8")
    cw = csv.writer(consumer_log)
    aw = csv.writer(alerts_log)
    cw.writerow([
        "record_id", "timestamp_received",
        "logistic_pred", "svm_pred", "rf_pred",
        "final_vote", "confidence_score",
    ])
    aw.writerow(["record_id", "kafka_transit_ms", "inference_ms", "e2e_ms"])

    processed = 0
    print("Listening for messages... (Ctrl+C to stop)")
    try:
        for message in consumer:
            # t2: the instant we receive the message
            t2 = datetime.datetime.now()
            payload = message.value
            record_id = payload.get("record_id", "")

            # t1 was injected by the producer at send time
            t1_raw = payload.get("timestamp_sent")
            try:
                t1 = parse_iso(t1_raw) if t1_raw else t2
            except Exception:
                t1 = t2

            # ---- inference window ----
            features = np.array(
                [[to_float(payload.get(c)) for c in feature_cols]],
                dtype=float,
            )
            scaled = scaler.transform(features)

            preds = {}
            probs = {}
            for name, model in models.items():
                preds[name] = int(model.predict(scaled)[0])
                probs[name] = model.predict_proba(scaled)[0]

            spam_votes = sum(preds.values())
            final_vote = 1 if spam_votes >= 2 else 0
            confidence = float(np.mean([probs[name][final_vote] for name in models]))

            t3 = datetime.datetime.now()
            # ---- end inference window ----

            kafka_ms = ms_between(t1, t2)
            inference_ms = ms_between(t2, t3)
            e2e_ms = ms_between(t1, t3)

            cw.writerow([
                record_id,
                t2.isoformat(),
                preds["logistic"], preds["svm"], preds["rf"],
                final_vote, f"{confidence:.4f}",
            ])
            aw.writerow([
                record_id,
                f"{kafka_ms:.3f}", f"{inference_ms:.3f}", f"{e2e_ms:.3f}",
            ])

            processed += 1
            if processed % 1000 == 0:
                consumer_log.flush()
                alerts_log.flush()
                print(f"  processed {processed} messages...")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt - shutting down...")
    finally:
        print(f"Total processed: {processed}")
        try:
            consumer.close()
            print("Kafka consumer closed.")
        except Exception as exc:
            print(f"Error closing consumer: {exc}")
        consumer_log.close()
        alerts_log.close()
        print(f"Logs written to '{CONSUMER_LOG}' and '{ALERTS_LOG}'.")


if __name__ == "__main__":
    main()
