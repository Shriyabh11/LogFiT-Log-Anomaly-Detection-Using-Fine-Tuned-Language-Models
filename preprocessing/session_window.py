"""
Preprocessing script for HDFS dataset.
"""

import re
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


data_dir   = r"data/HDFS"
log_file   = "HDFS.log"
label_file = "anomaly_label.csv"


log_path   = os.path.join(data_dir, log_file)
label_path = os.path.join(data_dir, label_file)
train_path = os.path.join(data_dir, "train.csv")
test_path  = os.path.join(data_dir, "test.csv")


def extract_block_ids(line: str):
    """
    Extract all HDFS block IDs from a log line.
    Block IDs follow pattern: blk_<number> or blk_-<number>

    Example:
        "Receiving block blk_-1608999687919862906 src: /10.250.19.102"
        -> ["blk_-1608999687919862906"]
    """
    return re.findall(r"blk_-?\d+", line)


def extract_message(line: str):
    """
    Extract just the message part from a raw HDFS log line.
    
    Raw format: Date Time PID Level Component: Message
    Example:
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_123"
        -> "dfs.DataNode$DataXceiver: Receiving block blk_123"
    
    We keep Component + Message (skip date, time, pid, level)
    This is the semantic content LogFiT needs.
    """
    parts = line.strip().split(None, 5)  # split on whitespace, max 6 parts
    if len(parts) >= 6:
        return parts[4] + " " + parts[5]   # Component: Message
    elif len(parts) == 5:
        return parts[4]
    return line.strip()


def parse_raw_log(log_path: str):
    """
    Parse raw HDFS.log file.
    Groups log messages by Block ID into sessions.

    Returns:
        sessions : dict { block_id -> [message1, message2, ...] }
        stats    : dict with parsing statistics
    """
    print(f"[1/4] Parsing raw log file: {log_path}")
    print(f"      (This may take a while for full dataset...)")

    sessions      = defaultdict(list)
    total_lines   = 0
    matched_lines = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Parsing lines"):
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            block_ids    = extract_block_ids(line)

            if block_ids:
                matched_lines += 1
                message = extract_message(line)
                for block_id in block_ids:
                    sessions[block_id].append(message)

    stats = {
        "total_lines"   : total_lines,
        "matched_lines" : matched_lines,
        "total_sessions": len(sessions)
    }

    print(f"    Total lines parsed  : {total_lines:,}")
    print(f"    Lines with block ID : {matched_lines:,}")
    print(f"    Unique sessions     : {len(sessions):,}")

    return sessions, stats


def load_labels(label_path: str):
    """
    Load anomaly labels from anomaly_label.csv
    Columns: BlockId, Label (Normal/Anomaly)

    Returns dict: { block_id -> 0 or 1 }
    0 = Normal, 1 = Anomaly
    """
    print(f"[2/4] Loading labels: {label_path}")
    df = pd.read_csv(label_path)
    df.columns = [c.strip() for c in df.columns]

    label_map = {}
    for _, row in df.iterrows():
        block_id          = str(row["BlockId"]).strip()
        label             = 0 if str(row["Label"]).strip() == "Normal" else 1
        label_map[block_id] = label

    normal_count  = sum(1 for v in label_map.values() if v == 0)
    anomaly_count = sum(1 for v in label_map.values() if v == 1)

    print(f"    Total labeled blocks : {len(label_map):,}")
    print(f"    Normal sessions      : {normal_count:,}")
    print(f"    Anomaly sessions     : {anomaly_count:,}")
    print(f"    Anomaly ratio        : {anomaly_count/len(label_map)*100:.2f}%")

    return label_map


def build_dataframe(sessions: dict, label_map: dict):
    """
    Build DataFrame where each row = one log paragraph (session).

    Columns:
        block_id      : HDFS block ID
        log_paragraph : all log messages joined as one paragraph
        num_lines     : number of log lines in this session
        label         : 0=Normal, 1=Anomaly
    """
    print("[3/4] Building dataframe...")
    rows    = []
    skipped = 0

    for block_id, messages in tqdm(sessions.items(), desc="Building rows"):
        if block_id not in label_map:
            skipped += 1
            continue

        log_paragraph = " ".join(messages)
        label         = label_map[block_id]

        rows.append({
            "block_id"      : block_id,
            "log_paragraph" : log_paragraph,
            "num_lines"     : len(messages),
            "label"         : label
        })

    if skipped > 0:
        print(f"    Skipped {skipped:,} sessions (block ID not in label file)")

    df = pd.DataFrame(rows)

    normal_count  = len(df[df["label"] == 0])
    anomaly_count = len(df[df["label"] == 1])

    print(f"    Total sessions  : {len(df):,}")
    print(f"    Normal          : {normal_count:,}")
    print(f"    Anomaly         : {anomaly_count:,}")
    print(f"    Anomaly ratio   : {anomaly_count/len(df)*100:.2f}%")
    print(f"    Avg lines/session: {df['num_lines'].mean():.1f}")

    return df


def split_and_save(df: pd.DataFrame, train_path: str, test_path: str,
                   train_ratio: float = 0.8):
    """
    Split into train and test sets.

    IMPORTANT - Paper says:
        Train = NORMAL logs ONLY (self supervised learning)
        Test  = normal + anomaly logs (for evaluation)

    Split:
        Train : 80% of normal logs
        Test  : 20% of normal logs + ALL anomaly logs
    """
    print("[4/4] Splitting and saving...")

    normal_df  = df[df["label"] == 0].reset_index(drop=True)
    anomaly_df = df[df["label"] == 1].reset_index(drop=True)

    train_size     = int(len(normal_df) * train_ratio)
    train_df       = normal_df.iloc[:train_size]
    test_normal_df = normal_df.iloc[train_size:]

    test_df = pd.concat([test_normal_df, anomaly_df]).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"\n✓ train.csv saved -> {train_path}")
    print(f"  Rows         : {len(train_df):,}")
    print(f"  All normal   : yes (for self supervised training)")

    print(f"\n✓ test.csv saved  -> {test_path}")
    print(f"  Total rows   : {len(test_df):,}")
    print(f"  Normal       : {len(test_normal_df):,}")
    print(f"  Anomaly      : {len(anomaly_df):,}")
    print(f"  Anomaly ratio: {len(anomaly_df)/len(test_df)*100:.2f}%")


def print_samples(df: pd.DataFrame):
    """Print sample paragraphs for sanity check."""
    print("\n── Sample Normal Log Paragraph ──")
    sample = df[df["label"] == 0].iloc[0]
    print(f"Block ID  : {sample['block_id']}")
    print(f"Num lines : {sample['num_lines']}")
    print(f"Paragraph : {sample['log_paragraph'][:300]}...")

    print("\n── Sample Anomaly Log Paragraph ──")
    sample = df[df["label"] == 1].iloc[0]
    print(f"Block ID  : {sample['block_id']}")
    print(f"Num lines : {sample['num_lines']}")
    print(f"Paragraph : {sample['log_paragraph'][:300]}...")


def main():
    print("=" * 55)
    print("  HDFS Preprocessing - Session Window (LogFiT)")
    print("=" * 55)

    # Check files exist
    assert os.path.exists(log_path),   f"Log file not found   : {log_path}"
    assert os.path.exists(label_path), f"Label file not found : {label_path}"

    # Run pipeline
    sessions,  stats = parse_raw_log(log_path)
    label_map        = load_labels(label_path)
    df               = build_dataframe(sessions, label_map)

    # Sanity check
    print_samples(df)

    # Save
    split_and_save(df, train_path, test_path)

    print("\n" + "=" * 55)
    print("  Preprocessing Complete!")
    print("=" * 55)
    print(f"  Output folder : {data_dir}")
    print(f"  train.csv     : {train_path}")
    print(f"  test.csv      : {test_path}")


if __name__ == "__main__":
    main()