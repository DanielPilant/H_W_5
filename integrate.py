"""
integrate.py - Step 1 of the Real-Time Spam Filter project.

Builds a unified `emails_master.csv` (60 columns, ~12,949 rows) by merging:
  * UCI Spambase (id=94)                              -> structured features
  * SpamAssassin public corpus (Berkeley DS100 set)   -> raw emails, parsed
    into the SAME 57-feature UCI schema via dynamic extraction.

Run:
    python integrate.py
"""

import os
import re
import email
import tarfile
import logging
import urllib.request
from collections import Counter

import pandas as pd
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPAMASSASSIN_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
BERKELEY_DIR = "berkeley_data"
OUTPUT_CSV = "emails_master.csv"

# (archive filename, label (1=spam, 0=ham), extracted sub-folder name)
ARCHIVES = [
    ("20021010_spam.tar.bz2",       1, "spam"),
    ("20021010_easy_ham.tar.bz2",   0, "easy_ham"),
    ("20021010_hard_ham.tar.bz2",   0, "hard_ham"),
    ("20030228_easy_ham.tar.bz2",   0, "easy_ham"),
    ("20030228_easy_ham_2.tar.bz2", 0, "easy_ham_2"),
    ("20030228_hard_ham.tar.bz2",   0, "hard_ham"),
    ("20030228_spam.tar.bz2",       1, "spam"),
    ("20030228_spam_2.tar.bz2",     1, "spam_2"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("integrate")


# ---------------------------------------------------------------------------
# 1. Download + extract SpamAssassin archives
# ---------------------------------------------------------------------------
def download_archive(filename: str, dest_dir: str) -> str:
    """Download a single .tar.bz2 archive (skipped if already on disk)."""
    url = SPAMASSASSIN_BASE + filename
    local_path = os.path.join(dest_dir, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        log.info(f"  cached  : {filename}")
        return local_path
    log.info(f"  download: {url}")
    tmp_path = local_path + ".part"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, local_path)
    return local_path


def extract_archive(tar_path: str, dest_dir: str) -> None:
    """Extract a .tar.bz2 archive into dest_dir (overwrites existing files)."""
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=dest_dir)


def prepare_berkeley_corpus() -> None:
    """Download every required SpamAssassin archive and unpack it locally."""
    os.makedirs(BERKELEY_DIR, exist_ok=True)
    log.info("=== Preparing SpamAssassin corpus ===")
    for filename, _label, _subdir in ARCHIVES:
        tar_path = download_archive(filename, BERKELEY_DIR)
        extract_archive(tar_path, BERKELEY_DIR)
    log.info("Corpus download + extraction complete.")


# ---------------------------------------------------------------------------
# 2. Feature extraction (the core logic)
# ---------------------------------------------------------------------------
WORD_PREFIX = "word_freq_"
CHAR_PREFIX = "char_freq_"
CAP_AVG = "capital_run_length_average"
CAP_LONG = "capital_run_length_longest"
CAP_TOTAL = "capital_run_length_total"

# A "word" per UCI Spambase docs: a run of alphanumeric characters
# bounded by non-alphanumerics or string edges.
WORD_RE = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)
CAP_RE = re.compile(r"[A-Z]+")


def extract_uci_features(raw_text: str, uci_column_names) -> dict:
    """Parse a raw email and produce the exact 57-feature UCI vector.

    * word_freq_X : 100 * count(X) / total_words  (case-insensitive, \\b...\\b)
    * char_freq_X : 100 * count(X) / total_chars
    * capital_run_length_{average,longest,total} : over runs matching [A-Z]+
    """
    features = {col: 0.0 for col in uci_column_names}
    if not raw_text:
        return features

    lowered = raw_text.lower()
    words = WORD_RE.findall(lowered)
    n_words = len(words)
    n_chars = len(raw_text)
    word_counts: Counter = Counter(words) if n_words else Counter()

    for col in uci_column_names:
        if col.startswith(WORD_PREFIX):
            token = col[len(WORD_PREFIX):].lower()
            if n_words and token:
                features[col] = 100.0 * word_counts.get(token, 0) / n_words
        elif col.startswith(CHAR_PREFIX):
            ch = col[len(CHAR_PREFIX):]
            if n_chars and ch:
                features[col] = 100.0 * raw_text.count(ch) / n_chars

    caps = CAP_RE.findall(raw_text)
    if caps:
        lengths = [len(r) for r in caps]
        total = sum(lengths)
        if CAP_AVG in features:
            features[CAP_AVG] = total / len(lengths)
        if CAP_LONG in features:
            features[CAP_LONG] = float(max(lengths))
        if CAP_TOTAL in features:
            features[CAP_TOTAL] = float(total)

    return features


# ---------------------------------------------------------------------------
# 3. Berkeley email parsing
# ---------------------------------------------------------------------------
def read_email_text(file_path: str) -> str:
    """Read a raw email file and return its textual body (headers stripped)."""
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    raw_text = raw_bytes.decode("latin-1", errors="ignore")
    try:
        msg = email.message_from_string(raw_text)
        body_chunks = []
        for part in msg.walk():
            ctype = part.get_content_type()
            if not ctype.startswith("text/"):
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "latin-1"
            try:
                body_chunks.append(payload.decode(charset, errors="ignore"))
            except (LookupError, TypeError):
                body_chunks.append(payload.decode("latin-1", errors="ignore"))
        body = "\n".join(body_chunks).strip()
        return body if body else raw_text
    except Exception:
        return raw_text


def collect_berkeley_emails(uci_column_names) -> pd.DataFrame:
    """Walk every extracted SpamAssassin folder and build a feature DataFrame."""
    log.info("=== Parsing SpamAssassin emails ===")
    rows = []
    parsed = 0
    skipped = 0
    for _filename, label, subdir in ARCHIVES:
        folder = os.path.join(BERKELEY_DIR, subdir)
        if not os.path.isdir(folder):
            log.warning(f"  missing directory: {folder}")
            continue

        folder_parsed = 0
        for entry in os.listdir(folder):
            fp = os.path.join(folder, entry)
            # Skip directories, the SpamAssassin `cmds` index file, and dot-files
            if not os.path.isfile(fp) or entry == "cmds" or entry.startswith("."):
                skipped += 1
                continue
            try:
                text = read_email_text(fp)
                features = extract_uci_features(text, uci_column_names)
                features["label"] = label
                features["source"] = "berkeley"
                rows.append(features)
                parsed += 1
                folder_parsed += 1
            except Exception as exc:
                skipped += 1
                log.debug(f"  skip {fp}: {exc}")
        log.info(f"  {subdir:<14} label={label}  parsed={folder_parsed}")

    log.info(f"Berkeley totals -> parsed={parsed}  skipped={skipped}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. UCI loader
# ---------------------------------------------------------------------------
def load_uci():
    """Fetch UCI Spambase and return (dataframe, list_of_57_feature_columns)."""
    log.info("=== Fetching UCI Spambase (id=94) ===")
    spambase = fetch_ucirepo(id=94)
    feature_cols = list(spambase.data.features.columns)
    target_col = list(spambase.data.targets.columns)[0]

    uci_df = pd.concat([spambase.data.features, spambase.data.targets], axis=1)
    uci_df = uci_df.rename(columns={target_col: "label"})
    uci_df["source"] = "uci"
    log.info(f"UCI rows={len(uci_df)}  features={len(feature_cols)}  target='{target_col}'")
    return uci_df, feature_cols


# ---------------------------------------------------------------------------
# 5. Merge + write
# ---------------------------------------------------------------------------
def main() -> None:
    uci_df, feature_cols = load_uci()
    prepare_berkeley_corpus()
    berkeley_df = collect_berkeley_emails(feature_cols)

    log.info("=== Merging UCI + Berkeley ===")
    column_order = feature_cols + ["label", "source"]
    uci_aligned = uci_df.reindex(columns=column_order)
    berkeley_aligned = berkeley_df.reindex(columns=column_order)

    master_df = pd.concat([uci_aligned, berkeley_aligned], ignore_index=True)

    # Numeric coercion + NaN -> 0 for the 57 feature columns
    for col in feature_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors="coerce").fillna(0.0)
    master_df["label"] = master_df["label"].fillna(0).astype(int)
    master_df["source"] = master_df["source"].fillna("unknown").astype(str)

    # record_id must be the very first column
    master_df.insert(0, "record_id", [f"REC_{i}" for i in range(len(master_df))])

    master_df.to_csv(OUTPUT_CSV, index=False)

    log.info("=== Done ===")
    log.info(f"Output         : {OUTPUT_CSV}")
    log.info(f"Final shape    : rows={master_df.shape[0]}  cols={master_df.shape[1]}")
    log.info(f"Source counts  :\n{master_df['source'].value_counts().to_string()}")
    log.info(f"Label counts   :\n{master_df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
