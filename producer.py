"""
producer.py - Step 2 of the Real-Time Spam Filter project.

Streams every row of `emails_master.csv` into the Kafka topic `email-stream`
on a local broker (`localhost:9092`), routing across 3 partitions by source:

    source == 'uci'      -> partition 0
    source == 'berkeley' -> partition 1
    otherwise            -> partition 2

Each message is the original CSV row + an injected ISO `timestamp_sent` (t1).
A flat log of every send is written to `console.producer.txt`.
"""

import csv
import json
import time
import datetime

from kafka import KafkaProducer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "email-stream"
INPUT_CSV = "emails_master.csv"
LOG_FILE = "console.producer.txt"
SEND_DELAY_SEC = 0.05            # 50ms pacing
PARTITION_MAP = {"uci": 0, "berkeley": 1}
DEFAULT_PARTITION = 2


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Starting Kafka producer -> {BOOTSTRAP_SERVERS}  topic={TOPIC}")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    sent = 0
    try:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as log, \
             open(INPUT_CSV, "r", encoding="utf-8") as f:

            log.write("record_id,timestamp_sent,true_label,source\n")
            reader = csv.DictReader(f)

            for row in reader:
                # t1 = the exact moment we hand the message to Kafka
                t1 = datetime.datetime.now().isoformat()
                message = {**row, "timestamp_sent": t1}

                # Route to one of the 3 partitions based on `source`
                source = row.get("source", "")
                partition = PARTITION_MAP.get(source, DEFAULT_PARTITION)

                producer.send(TOPIC, value=message, partition=partition)

                # Per-row log line
                label = row.get("label", "")
                if label == "" or label is None:
                    label = "?"
                log.write(f"{row['record_id']},{t1},{label},{source}\n")

                sent += 1
                if sent % 1000 == 0:
                    print(f"  sent {sent} messages...")

                time.sleep(SEND_DELAY_SEC)

        producer.flush()
        print(f"Producer finished. Total messages sent: {sent}")

    finally:
        producer.close()


if __name__ == "__main__":
    main()
