# Log Anomaly Detection using Transformer Models

## Overview

This project implements a **log anomaly detection system using transformer-based language models**. The goal is to detect abnormal system behavior by learning patterns of **normal log sequences** and identifying deviations from them.

Large distributed systems generate massive volumes of logs describing system operations, failures, warnings, and events. Manually analyzing these logs is impractical, so automated anomaly detection methods are required.

This project preprocesses system logs and uses a **pretrained transformer model (RoBERTa / Longformer)** to learn contextual patterns in log sequences and detect anomalies.

---

# Methodology

The pipeline consists of three main stages:

1. **Data Preparation**
2. **Model Training**
3. **Anomaly Detection**

### 1. Data Preparation

Raw logs are converted into structured sequences suitable for transformer models.

Steps include:

* Parsing raw log files
* Extracting useful information (timestamps, messages, identifiers)
* Grouping logs into sequences
* Creating structured log paragraphs
* Preparing train and test datasets

Two preprocessing strategies are used depending on the dataset structure.

#### Session Window (HDFS)

HDFS logs contain **block identifiers representing file operations**.
Logs with the same block ID are grouped into a **session** representing one operation.

Example:

Raw logs:

```
Receiving block blk_123
PacketResponder started
PacketResponder finished
```

Session paragraph:

```
Receiving block blk_123 PacketResponder started PacketResponder finished
```

#### Sliding Time Window (BGL, Thunderbird)

BGL and Thunderbird logs are **continuous streams without session identifiers**.
Logs are grouped into **fixed time windows** (e.g., 60 seconds).

Example:

Logs within a time window:

```
CPU check
Disk read
Memory warning
```

Window paragraph:

```
CPU check Disk read Memory warning
```

Each paragraph becomes a **single input sample for the model**.

---

### 2. Model Training

A pretrained transformer model (RoBERTa or Longformer) is fine-tuned on log sequences.

Training uses **Masked Sentence Prediction (MSP)**:

* Random tokens are masked in the log sequence.
* The model learns to predict the masked tokens based on context.
* Training is performed using **normal logs only**.

This allows the model to learn patterns of **normal system behavior**.

Loss function:

* Cross-entropy loss is used to optimize token prediction.

---

### 3. Anomaly Detection

During inference:

1. Tokens in a log sequence are masked.
2. The model predicts the masked tokens.
3. Prediction accuracy is used to compute an **anomaly score**.

If the model fails to predict tokens correctly, it indicates that the log sequence deviates from normal patterns.

Logs with low prediction accuracy are flagged as **anomalies**.

---

# Datasets

The project uses three widely used system log datasets.

## HDFS

* Source: Hadoop Distributed File System
* Logs describe file block operations in distributed storage
* Sessions are identified by **block IDs**

## BGL

* Source: Blue Gene/L supercomputer logs
* Contains hardware events, system warnings, and failures
* Logs are continuous event streams

## Thunderbird

* Source: large-scale cluster system logs
* Very large dataset with millions of log messages
* Logs are processed using sliding time windows

Dataset source:
[https://zenodo.org/records/8196385](https://zenodo.org/records/8196385)

---

# Labeling

Each log sequence has a label representing system behavior.

```
0 → Normal
1 → Anomaly
```

For sliding windows:

* If **any log inside the window is anomalous**, the entire window is labeled as anomaly.

---

# Repository Structure

```
project/
│
├── data/
│   ├── HDFS/
│   ├── BGL/
│   └── Thunderbird/
│
├── preprocessing/
│   ├── session_window.py
│   └── sliding_window.py
│
├── notebooks/
│   └── eda.ipynb
│
├── models/
│   └── transformer_model.py
│
└── README.md
```

---

# Exploratory Data Analysis (EDA)

EDA was performed to understand dataset characteristics such as:

* class imbalance (normal vs anomaly)
* log length distribution
* dataset size and structure

These insights guided preprocessing decisions.

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/log-anomaly-detection.git
cd log-anomaly-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running Preprocessing

### HDFS Session Window

```
python preprocessing/session_window.py
```

### BGL / Thunderbird Sliding Window

```
python preprocessing/sliding_window.py
```

These scripts generate processed datasets for training.

---

# Technologies Used

* Python
* Pandas
* NumPy
* HuggingFace Transformers
* RoBERTa / Longformer
* Matplotlib
* tqdm

---

# Contributions

Main contributions of this project include:

* Implemented preprocessing pipelines for log datasets
* Designed session window and sliding window grouping methods
* Performed exploratory data analysis on log datasets
* Generated structured log sequences for transformer-based training
* Implemented anomaly detection using transformer models

---

# Future Work

Possible improvements include:

* real-time anomaly detection
* log template extraction
* larger transformer models for long sequences
* integrating system monitoring dashboards

---

# License

This project is for academic and research purposes.
