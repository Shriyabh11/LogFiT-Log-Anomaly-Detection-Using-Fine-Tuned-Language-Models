"""
Preprocessing script for BGL and Thunderbird datasets.
"""

import os
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm


dataset_name = "Thunderbird"           # "BGL" or "Thunderbird"
data_dir     = r"data/Thunderbird"    # change to data/Thunderbird for Thunderbird
log_file     = "Thunderbird.log"      # change to Thunderbird.log for Thunderbird
window_size  = 60              # seconds — paper uses 10, 30, 60


log_path   = os.path.join(data_dir, log_file)
train_path = os.path.join(data_dir, f"train_{window_size}s.csv")
test_path  = os.path.join(data_dir, f"test_{window_size}s.csv")


def parse_bgl_line(line: str):
    """
    Parse a single BGL log line.

    BGL format:
    - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-... RAS KERNEL INFO msg

    Field 0: label ("-" or error type)
    Field 1: unix timestamp (integer)
    Field 4: detailed timestamp with microseconds
    Field 9+: message

    Returns (unix_timestamp, message, label) or None
    """
    parts = line.strip().split()
    if len(parts) < 10:
        return None

    label_str = parts[0]
    label     = 0 if label_str == "-" else 1

    try:
        # Field 1 is unix timestamp in seconds
        ts_seconds = float(parts[1])
    except ValueError:
        return None

    # Message starts from field 9 onwards
    message = " ".join(parts[9:])

    return ts_seconds, message, label


def parse_thunderbird_line(line: str):
    """
    Parse a single Thunderbird log line.

    Thunderbird format similar to BGL:
    - 1131567595 2005.11.09 tbird-admin1 2005-11-09-15.19.55 ... msg

    Field 0: label
    Field 1: unix timestamp
    Field 5+: message

    Returns (unix_timestamp, message, label) or None
    """
    parts = line.strip().split()
    if len(parts) < 6:
        return None

    label_str = parts[0]
    label     = 0 if label_str == "-" else 1

    try:
        ts_seconds = float(parts[1])
    except ValueError:
        return None

    # Message starts from field 5 onwards
    message = " ".join(parts[5:])

    return ts_seconds, message, label


def parse_log_file(log_path: str, dataset_name: str):
    """
    Parse raw log file.
    Returns list of (timestamp, message, label) sorted by time.
    """
    print(f"[1/4] Parsing {dataset_name} log: {log_path}")
    print(f"      (This may take a few minutes for full dataset...)")

    records      = []
    total        = 0
    parse_errors = 0
    normal_count = 0
    anomaly_count= 0

    parse_fn = parse_bgl_line if dataset_name == "BGL" else parse_thunderbird_line

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Parsing lines"):
            line = line.strip()
            if not line:
                continue

            total += 1
            result = parse_fn(line)

            if result is None:
                parse_errors += 1
                continue

            ts, msg, label = result
            records.append((ts, msg, label))

            if label == 0:
                normal_count  += 1
            else:
                anomaly_count += 1

    # Sort by timestamp
    records.sort(key=lambda x: x[0])

    print(f"\n    Total lines     : {total:,}")
    print(f"    Parse errors    : {parse_errors:,}")
    print(f"    Valid records   : {len(records):,}")
    print(f"    Normal lines    : {normal_count:,}")
    print(f"    Anomaly lines   : {anomaly_count:,}")
    print(f"    Anomaly ratio   : {anomaly_count/max(len(records),1)*100:.2f}%")

    return records


def apply_sliding_window(records: list, window_size: int):
    """
    Group log lines into non-overlapping time windows.

    Each window:
        - Spans [t_start, t_start + window_size) seconds
        - Label = 1 (anomaly) if ANY line in window is anomalous
        - Label = 0 (normal) if ALL lines are normal

    Paper uses 10s, 30s, 60s windows for BGL and Thunderbird.
    """
    print(f"\n[2/4] Applying sliding window (size={window_size}s)...")

    if not records:
        return pd.DataFrame()

    windows = []

    current_start    = records[0][0]
    current_messages = []
    current_label    = 0

    for ts, message, label in tqdm(records, desc="Windowing"):

        # Check if this record falls outside current window
        if ts >= current_start + window_size:

            # Save current window if it has messages
            if current_messages:
                windows.append({
                    "log_paragraph" : " ".join(current_messages),
                    "label"         : current_label,
                    "num_lines"     : len(current_messages),
                    "window_start"  : current_start
                })

            # Start new window at this timestamp
            current_start    = ts
            current_messages = []
            current_label    = 0

        current_messages.append(message)
        if label == 1:
            current_label = 1  # window is anomaly if ANY line is anomaly

    # Don't forget the last window
    if current_messages:
        windows.append({
            "log_paragraph" : " ".join(current_messages),
            "label"         : current_label,
            "num_lines"     : len(current_messages),
            "window_start"  : current_start
        })

    df = pd.DataFrame(windows)

    normal_windows  = len(df[df["label"] == 0])
    anomaly_windows = len(df[df["label"] == 1])

    print(f"\n    Total windows   : {len(df):,}")
    print(f"    Normal windows  : {normal_windows:,}")
    print(f"    Anomaly windows : {anomaly_windows:,}")
    print(f"    Anomaly ratio   : {anomaly_windows/max(len(df),1)*100:.2f}%")
    print(f"    Avg lines/window: {df['num_lines'].mean():.1f}")

    return df


def split_and_save(df: pd.DataFrame, train_path: str, test_path: str,
                   train_ratio: float = 0.8):
    """
    Split into train and test.

    Train: NORMAL windows ONLY (self supervised)
    Test : remaining normal + ALL anomaly windows
    """
    print(f"\n[3/4] Splitting and saving...")

    normal_df  = df[df["label"] == 0].reset_index(drop=True)
    anomaly_df = df[df["label"] == 1].reset_index(drop=True)

    train_size     = int(len(normal_df) * train_ratio)
    train_df       = normal_df.iloc[:train_size]
    test_normal_df = normal_df.iloc[train_size:]

    test_df = pd.concat([test_normal_df, anomaly_df]).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\n✓ {os.path.basename(train_path)} saved -> {train_path}")
    print(f"  Rows         : {len(train_df):,} (all normal)")
    print(f"\n✓ {os.path.basename(test_path)} saved  -> {test_path}")
    print(f"  Total        : {len(test_df):,}")
    print(f"  Normal       : {len(test_normal_df):,}")
    print(f"  Anomaly      : {len(anomaly_df):,}")
    print(f"  Anomaly ratio: {len(anomaly_df)/max(len(test_df),1)*100:.2f}%")


def print_samples(df: pd.DataFrame):
    """Print sample windows for sanity check."""
    print("\n── Sample Normal Window ──")
    row = df[df["label"] == 0].iloc[0]
    print(f"  Lines     : {row['num_lines']}")
    print(f"  Paragraph : {row['log_paragraph'][:250]}...")

    if len(df[df["label"] == 1]) > 0:
        print("\n── Sample Anomaly Window ──")
        row = df[df["label"] == 1].iloc[0]
        print(f"  Lines     : {row['num_lines']}")
        print(f"  Paragraph : {row['log_paragraph'][:250]}...")
    else:
        print("\n  No anomaly windows found!")
        print("  Check that log file has anomaly lines (label != '-')")


def main():
    print("=" * 55)
    print(f"  {dataset_name} Preprocessing - Sliding Window (LogFiT)")
    print(f"  Window size : {window_size}s")
    print("=" * 55)

    assert os.path.exists(log_path), \
        f"Log file not found: {log_path}\nSet data_dir and log_file correctly!"

    # Pipeline
    records = parse_log_file(log_path, dataset_name)

    if not records:
        print("ERROR: No valid records parsed! Check log file format.")
        return

    df = apply_sliding_window(records, window_size)

    # Sanity check
    print_samples(df)

    # Save
    split_and_save(df, train_path, test_path)

    print("\n" + "=" * 55)
    print(f"  {dataset_name} Preprocessing Complete!")
    print("=" * 55)
    print(f"  Output: {data_dir}/")
    print(f"\n  To run for different window sizes, change:")
    print(f"  window_size = 10  # or 30 or 60")


if __name__ == "__main__":
    main()