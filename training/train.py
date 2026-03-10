import os
import re
import random
import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (RobertaTokenizerFast,RobertaForMaskedLM,
get_linear_schedule_with_warmup,)
from torch.optim import AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

CONFIG = {"train_csv": "/kaggle/input/datasets/kirabitesthedust/ds-project/HDFSTrain.csv",
    "test_csv": "/kaggle/input/datasets/kirabitesthedust/testforhdfs/test.csv",
    "paragraph_col": "log_paragraph","label_col" : "label",
    "output_dir" : "/kaggle/working/logfit_roberta",
    "base_model": "roberta-base",
    "max_seq_len" : 512,"sentence_mask_ratio": 0.5,
    "token_mask_ratio": 0.8,"epochs" : 4,
    "batch_size" : 8,"grad_accum_steps" : 4,"learning_rate" : 1e-5, "weight_decay" : 0.01,
    "warmup_ratio" : 0.1,"max_grad_norm": 1.0,
    "k_folds" : 3,"n_train_samples" : 50000,"n_train_per_fold" : 20000,
    "seed": 42, "topk_candidates" : [5, 9, 12],"threshold_steps": 20,
    "precision_target" : 0.90,"recall_target" : 0.90,"min_f1_floor": 0.60,}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs(CONFIG["output_dir"], exist_ok=True)

class HDFSLogDataset(Dataset):
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
    def __len__(self):
        return len(self.paragraphs)
    def __getitem__(self, idx):
        return self.paragraphs[idx]

class MaskedSentencePredictionCollator:
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.max_seq_len = cfg["max_seq_len"]
        self.sentence_mask_ratio = cfg["sentence_mask_ratio"]
        self.token_mask_ratio = cfg["token_mask_ratio"]
        self.mask_token_id = tokenizer.mask_token_id
        self.special_token_ids = set(tokenizer.all_special_ids)
    def _split_sentences(self, paragraph):
        sentences = re.split(r'(?=dfs\.|hadoop\.|INFO|WARN|ERROR)', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [paragraph]

    def __call__(self, batch):
        encoding = self.tokenizer(batch,padding="max_length",truncation=True,
            max_length=self.max_seq_len,return_tensors="pt",
            return_offsets_mapping=False,)
        input_ids = encoding["input_ids"].clone()
        attention_mask = encoding["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        for b in range(input_ids.size(0)):
            seq_len = attention_mask[b].sum().item()
            valid_pos = [
                i for i in range(1, int(seq_len) - 1)
                if input_ids[b, i].item() not in self.special_token_ids
            ]

            if not valid_pos:
                continue
            sentences  = self._split_sentences(batch[b])
            n_sentences = len(sentences)
            tokens_per_sent = max(1, len(valid_pos) // n_sentences)
            sent_spans = []
            for s in range(n_sentences):
                start = s*tokens_per_sent
                end   = (s+1)*tokens_per_sent if s < n_sentences - 1 else len(valid_pos)
                sent_spans.append(valid_pos[start:end])
            n_sents_to_mask = max(1, int(n_sentences * self.sentence_mask_ratio))
            sent_indices = random.sample(range(n_sentences), n_sents_to_mask)

            for si in sent_indices:
                span = sent_spans[si]
                if not span:
                    continue
                n_tok_to_mask = max(1, int(len(span) * self.token_mask_ratio))
                tok_indices   = random.sample(span, n_tok_to_mask)
                for ti in tok_indices:
                    labels[b, ti]    = input_ids[b, ti]
                    input_ids[b, ti] = self.mask_token_id

        return{"input_ids" : input_ids,
            "attention_mask": attention_mask,
            "labels" : labels,}
def get_topk_accuracy(model, tokenizer, paragraphs, cfg, k, batch_size=16):
    model.eval()
    collator = MaskedSentencePredictionCollator(tokenizer, cfg)
    scores = []

    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i+batch_size]
        enc = collator(batch)
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        labels = enc["labels"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        for b in range(input_ids.size(0)):
            mask_positions = (input_ids[b] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                scores.append(1.0)
                continue
            correct = 0
            n_masked = 0
            for pos in mask_positions:
                true_id = labels[b, pos].item()
                if true_id == -100:
                    continue
                topk_ids = torch.topk(logits[b, pos], k=k).indices.tolist()
                if true_id in topk_ids:
                    correct += 1
                n_masked += 1

            scores.append(correct / n_masked if n_masked > 0 else 1.0)

    model.train()
    return scores

def find_best_threshold(model, tokenizer, normal_paras, anomaly_paras, cfg):
    print("\nSearching for best threshold (P > 0.90 and R > 0.90) ->")

    n_eval       = min(1000, len(normal_paras), len(anomaly_paras))
    eval_normal  = random.sample(normal_paras, n_eval)
    eval_anomaly = random.sample(anomaly_paras, n_eval)
    eval_paras   = eval_normal + eval_anomaly
    true_labels  = [0] * n_eval + [1] * n_eval

    best_balanced = {"precision": 0,"recall": 0, "f1": 0, "threshold": 0.5, "k": 9}
    best_precision = {"precision": 0,"recall": 0, "f1": 0, "threshold": 0.5, "k": 9}
    for k in cfg["topk_candidates"]:
        print(f"    Trying k={k}->")
        scores     = get_topk_accuracy(model, tokenizer, eval_paras, cfg, k=k)
        train_acc  = float(np.mean(scores[:n_eval]))
        low        = max(0.0, train_acc - 0.30)
        thresholds = np.linspace(train_acc, low, cfg["threshold_steps"])

        for thresh in thresholds:
            preds = [1 if s < thresh else 0 for s in scores]
            p = precision_score(true_labels, preds, zero_division=0)
            r = recall_score(true_labels, preds, zero_division=0)
            f = f1_score(true_labels, preds, zero_division=0)
            if (p >= cfg["precision_target"] and
                r >= cfg["recall_target"] and
                f > best_balanced["f1"]):
                best_balanced = {
                    "precision": round(p, 4),
                    "recall" : round(r, 4),
                    "f1" : round(f, 4),
                    "threshold" : round(float(thresh), 4),
                    "k" : k,
                    "target_hit": True,}
            if (p >= cfg["precision_target"] and
                f >= cfg["min_f1_floor"] and
                f > best_precision["f1"]):
                best_precision = {
                    "precision": round(p, 4),
                    "recall" : round(r, 4),
                    "f1" : round(f, 4),
                    "threshold" : round(float(thresh), 4),
                    "k": k,
                    "target_hit": False,}

    if best_balanced["f1"] > 0:
        print(f"Both targets hit → k={best_balanced['k']}  "
              f"threshold={best_balanced['threshold']}  "
              f"P={best_balanced['precision']}  R={best_balanced['recall']}  "
              f"F1={best_balanced['f1']}")
        return best_balanced

    if best_precision["f1"] > 0:
        print(f"Both targets not simultaneously achievable.")
        print(f"Best with P >= 0.90 -> k={best_precision['k']}  "
              f"threshold={best_precision['threshold']}  "
              f"P={best_precision['precision']}  R={best_precision['recall']}  "
              f"F1={best_precision['f1']}")
        return best_precision

    
    print(f"Precision target not reachable, using best F1.")
    best_f1 = {"precision": 0, "recall": 0, "f1": 0, "threshold": 0.5, "k": 9, "target_hit": False}
    for k in cfg["topk_candidates"]:
        scores     = get_topk_accuracy(model, tokenizer, eval_paras, cfg, k=k)
        train_acc  = float(np.mean(scores[:n_eval]))
        low        = max(0.0, train_acc - 0.30)
        thresholds = np.linspace(train_acc, low, cfg["threshold_steps"])
        for thresh in thresholds:
            preds = [1 if s < thresh else 0 for s in scores]
            p = precision_score(true_labels, preds, zero_division=0)
            r = recall_score(true_labels, preds, zero_division=0)
            f = f1_score(true_labels, preds, zero_division=0)
            if f > best_f1["f1"]:
                best_f1 = {
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f, 4),
                    "threshold" : round(float(thresh), 4),
                    "k" : k,
                    "target_hit": False,
                }
    print(f"    Best F1 -> k={best_f1['k']}  threshold={best_f1['threshold']}  "
          f"P={best_f1['precision']}  R={best_f1['recall']}  F1={best_f1['f1']}")
    return best_f1


def train_one_fold(fold_idx, train_paragraphs, normal_val, anomaly_val, cfg, tokenizer, model):
    print(f"\n{'='*55}")
    print(f"Fold {fold_idx+1}/{cfg['k_folds']}")
    print(f"{'='*55}")
    dataset = HDFSLogDataset(train_paragraphs)
    collator = MaskedSentencePredictionCollator(tokenizer, cfg)
    loader = DataLoader(dataset,batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collator,num_workers=2,pin_memory=True,)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg["weight_decay"],},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,},
    ]
    optimizer = AdamW(param_groups, lr=cfg["learning_rate"], eps=1e-5)
    total_steps = (len(loader)//cfg["grad_accum_steps"]) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model.train()
    model.to(DEVICE)
    total_batches = len(loader) * cfg["epochs"]
    save_interval  = total_batches//4
    saves_done = 0
    global_batch = 0
    best_loss  = float("inf")
    fold_loss_history = []

    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss/cfg["grad_accum_steps"]
            loss.backward()
            epoch_loss += outputs.loss.item()
            n_batches += 1
            global_batch += 1
            if (step+1) % cfg["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            avg_loss = epoch_loss/n_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            if saves_done < 4 and global_batch >= (saves_done+1)*save_interval:
                ckpt_path = os.path.join(cfg["output_dir"], f"fold{fold_idx+1}_ckpt{saves_done+1}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                saves_done += 1
                print(f"\nCheckpoint {saves_done}/4 saved -> {ckpt_path}(loss={avg_loss:.4f})")
        avg_epoch_loss = epoch_loss/n_batches
        fold_loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} avg loss:{avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

    threshold_result = find_best_threshold(model, tokenizer, normal_val, anomaly_val, cfg)
    return best_loss, fold_loss_history, threshold_result

def main():
    cfg = CONFIG

    print(f"Loading train: {cfg['train_csv']}")
    train_df   = pd.read_csv(cfg["train_csv"])
    paragraphs = train_df[cfg["paragraph_col"]].dropna().astype(str).tolist()
    print(f"Total normal paragraphs: {len(paragraphs)}")
    if len(paragraphs) > cfg["n_train_samples"]:
        paragraphs = random.sample(paragraphs, cfg["n_train_samples"])
        print(f"Sampled to: {len(paragraphs)}")
    print(f"\nLoading test: {cfg['test_csv']}")
    test_df     = pd.read_csv(cfg["test_csv"])
    normal_val  = test_df[test_df[cfg["label_col"]] == 0][cfg["paragraph_col"]].dropna().astype(str).tolist()
    anomaly_val = test_df[test_df[cfg["label_col"]] == 1][cfg["paragraph_col"]].dropna().astype(str).tolist()
    print(f"Val normal: {len(normal_val)}  |  Val anomaly: {len(anomaly_val)}")

    print(f"\nLoading model: {cfg['base_model']}")
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["base_model"])
    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=cfg["seed"])
    fold_losses = []
    all_loss_history = {}
    all_thresholds = []

    for fold_idx, (train_idx, _) in enumerate(kf.split(paragraphs)):
        fold_model      = RobertaForMaskedLM.from_pretrained(cfg["base_model"])
        fold_paragraphs = [paragraphs[i] for i in train_idx][:cfg["n_train_per_fold"]]

        best_loss, loss_history, threshold_result = train_one_fold(
            fold_idx, fold_paragraphs, normal_val, anomaly_val, cfg, tokenizer, fold_model
        )
        fold_losses.append(best_loss)
        all_loss_history[f"fold_{fold_idx+1}"] = loss_history
        all_thresholds.append(threshold_result)

        final_path = os.path.join(cfg["output_dir"], f"fold{fold_idx+1}_final")
        fold_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"\nFinal model fold {fold_idx+1} saved -> {final_path}")

        del fold_model
        torch.cuda.empty_cache()

    best_fold_idx = max(
        range(len(all_thresholds)),
        key=lambda i: (
            all_thresholds[i].get("target_hit", False),
            all_thresholds[i]["f1"],
        )
    )

    print(f"\n{'='*55}")
    print("  Training is over->")
    print(f" Fold losses     : {[f'{l:.4f}' for l in fold_losses]}")
    print(f"  Mean loss       : {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
    print(f"\n  Threshold results per fold:")
    for i, t in enumerate(all_thresholds):
        hit = "YES" if t.get("target_hit") else "X"
        print(f"    Fold {i+1} [{hit}] → k={t['k']}  thresh={t['threshold']}  "
              f"P={t['precision']}  R={t['recall']}  F1={t['f1']}")
    print(f"\n  ★ Best fold: {best_fold_idx+1}  "
          f"(P={all_thresholds[best_fold_idx]['precision']}  "
          f"R={all_thresholds[best_fold_idx]['recall']}  "
          f"F1={all_thresholds[best_fold_idx]['f1']})")
    print(f"  → Use fold{best_fold_idx+1}_final for inference")
    print(f"{'='*55}")

    results = {
        "fold_best_losses": fold_losses,
        "mean_loss": float(np.mean(fold_losses)),
        "std_loss": float(np.std(fold_losses)),
        "loss_history": all_loss_history,
        "threshold_results" : all_thresholds,
        "best_fold" : best_fold_idx + 1,
        "best_threshold_cfg" : all_thresholds[best_fold_idx],
        "config" : cfg,
    }
    results_path = os.path.join(cfg["output_dir"], "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {results_path}")


if __name__ == "__main__":
    main()


