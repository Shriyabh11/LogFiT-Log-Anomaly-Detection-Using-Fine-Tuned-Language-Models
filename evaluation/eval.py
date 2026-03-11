import os
import re
import random
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

CONFIG = {
    "model_path"  : "/kaggle/input/datasets/akshits999/highpresmodel",
    "test_csv" : "/kaggle/input/datasets/akshits999/test-data/test.csv",
    "paragraph_col"  : "log_paragraph",
    "label_col"  : "label", "max_seq_len"  : 512,"sentence_mask_ratio" : 0.5,
    "token_mask_ratio" : 0.8,"batch_size"  : 32, "seed" : 42,
    "k" : 5,"threshold"  : 0.903,
    "sample_size"  : 5000,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedSentencePredictionCollator:
    def __init__(self, tokenizer, cfg):
        self.tokenizer  = tokenizer
        self.max_seq_len  = cfg["max_seq_len"]
        self.sentence_mask_ratio = cfg["sentence_mask_ratio"]
        self.token_mask_ratio  = cfg["token_mask_ratio"]
        self.mask_token_id  = tokenizer.mask_token_id
        self.special_token_ids = set(tokenizer.all_special_ids)

    def _split_sentences(self, paragraph):
        sentences = re.split(r'(?=dfs\.|hadoop\.|INFO|WARN|ERROR)', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [paragraph]

    def __call__(self, batch):
        encoding = self.tokenizer(
            batch,
            padding="max_length",  truncation=True,
            max_length=self.max_seq_len,  return_tensors="pt",
            return_offsets_mapping=False,
        )

        input_ids = encoding["input_ids"].clone()
        attention_mask = encoding["attention_mask"]
        labels   = torch.full_like(input_ids, -100)

        for b in range(input_ids.size(0)):
            seq_len = attention_mask[b].sum().item()
            valid_pos = [
                i for i in range(1, int(seq_len) - 1)
                if input_ids[b, i].item() not in self.special_token_ids
            ]
            if not valid_pos:
                continue

            sentences  = self._split_sentences(batch[b])
            n_sentences  = len(sentences)
            tokens_per_sent = max(1, len(valid_pos) // n_sentences)
            sent_spans  = []
            for s in range(n_sentences):
                start = s * tokens_per_sent
                end   = (s + 1) * tokens_per_sent if s < n_sentences - 1 else len(valid_pos)
                sent_spans.append(valid_pos[start:end])
            n_sents_to_mask = max(1, int(n_sentences * self.sentence_mask_ratio))
            sent_indices    = random.sample(range(n_sentences), n_sents_to_mask)
            for si in sent_indices:
                span = sent_spans[si]
                if not span:
                    continue
                n_tok_to_mask = max(1, int(len(span) * self.token_mask_ratio))
                tok_indices   = random.sample(span, n_tok_to_mask)
                for ti in tok_indices:
                    labels[b, ti]    = input_ids[b, ti]
                    input_ids[b, ti] = self.mask_token_id

        return {
            "input_ids"     : input_ids, "attention_mask": attention_mask,
            "labels"        : labels,
        }


def get_scores(model, tokenizer, paragraphs, cfg):
    collator = MaskedSentencePredictionCollator(tokenizer, cfg)
    scores = []
    total_batches = (len(paragraphs) + cfg["batch_size"] - 1) // cfg["batch_size"]

    for i in tqdm(range(0, len(paragraphs), cfg["batch_size"]),
                  total=total_batches, desc="Scoring", leave=True):
        batch  = paragraphs[i:i+cfg["batch_size"]]
        enc = collator(batch)
        input_ids  = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        labels  = enc["labels"].to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        mask_positions = (labels != -100)
        topk_preds = torch.topk(logits, k=cfg["k"], dim=-1).indices
        labels_exp = labels.unsqueeze(-1).expand_as(topk_preds)
        correct_mask  = (topk_preds == labels_exp).any(dim=-1) & mask_positions

        for b in range(input_ids.size(0)):
            n_masked = mask_positions[b].sum().item()
            if n_masked == 0:
                scores.append(1.0)
                continue
            scores.append(correct_mask[b].sum().item() / n_masked)

    return scores


def main():
    cfg = CONFIG

    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model_path"], local_files_only=True)
    model = RobertaForMaskedLM.from_pretrained(cfg["model_path"], local_files_only=True)
    model.eval()
    model.to(DEVICE)
    test_df = pd.read_csv(cfg["test_csv"])
    test_df = test_df.sample(n=cfg["sample_size"], random_state=cfg["seed"])
    paragraphs = test_df[cfg["paragraph_col"]].dropna().astype(str).tolist()
    true_labels = test_df[cfg["label_col"]].tolist()
    scores = get_scores(model, tokenizer, paragraphs, cfg)
    preds = [1 if s < cfg["threshold"] else 0 for s in scores]
    p = precision_score(true_labels, preds, zero_division=0)
    r = recall_score(true_labels, preds, zero_division=0)
    f = f1_score(true_labels, preds, zero_division=0)

    print(f"Precision : {p:.4f}")
    print(f"Recall : {r:.4f}")
    print(f"F1 Score : {f:.4f}")


if __name__ == "__main__":
    main()
