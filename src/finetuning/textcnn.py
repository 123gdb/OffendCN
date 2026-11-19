# -*- coding: utf-8 -*-
"""TextCNN 对照实验。用法：python textcnn.py --task offense"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA_DIR as DATA_ROOT, MODELS_DIR, RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(DATA_ROOT)

SEED = 42
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 150
MAX_LEN = 160
FILTER_SIZES = [2, 3, 4]
FILTER_NUM = 128
DROPOUT = 0.3

MODEL_PATH = str(MODELS_DIR / "chinese-roberta-wwm-ext-base")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== 数据加载 ========================

def load_offense():
    with open(os.path.join(DATA_DIR, "OffendCN.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"content": x["content"].strip(), "label": 1 if x["offensive"] else 0}
            for x in data]


def load_target():
    with open(os.path.join(DATA_DIR, "subsets", "targeted_offensive_data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    res = []
    for x in data:
        cat = x.get("four_tuple", {}).get("Target_Category")
        if cat is None:
            continue
        res.append({"content": x["content"].strip(), "label": cat})
    return res


def load_expression():
    with open(os.path.join(DATA_DIR, "subsets", "targeted_offensive_data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    res = []
    for x in data:
        expr = x.get("expression_type")
        if expr is None or expr == "":
            continue
        res.append({"content": x["content"].strip(), "label": expr})
    return res


def load_dialect():
    with open(os.path.join(DATA_DIR, "subsets", "dialect_offensive_data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    res = []
    for x in data:
        info = x.get("dialect_info") or {}
        if not info.get("is_dialect_offensive", False):
            continue
        group = info.get("dialect_group")
        if group is None or group == "":
            continue
        res.append({"content": x["content"].strip(), "label": group})
    return res


def load_dialect_binary():
    with open(os.path.join(DATA_DIR, "subsets", "offensive_data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    res = []
    for x in data:
        info = x.get("dialect_info") or {}
        res.append({"content": x["content"].strip(),
                    "label": 1 if info.get("is_dialect_offensive", False) else 0})
    return res


LOADERS = {
    "offense": load_offense,
    "target": load_target,
    "expression": load_expression,
    "dialect": load_dialect,
    "dialect_binary": load_dialect_binary,
}


# ======================== 模型组件 ========================

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(item["content"], max_length=self.max_len,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        return enc["input_ids"].flatten(), torch.tensor(item["label"], dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, filter_num, dropout, class_num):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, filter_num, kernel_size=k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        outs = []
        for conv in self.convs:
            c = conv(x)
            c = torch.max_pool1d(c, c.shape[-1]).squeeze(-1)
            outs.append(c)
        out = self.dropout(torch.cat(outs, dim=1))
        return self.fc(out)


def evaluate(model, loader, classes=None):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ids, lab in loader:
            logits = model(ids.to(DEVICE))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(lab.numpy())

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "weighted_recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    if classes:
        names = [str(c) for c in classes]
        report = classification_report(labels, preds, labels=list(range(len(names))),
                                       target_names=names, output_dict=True, zero_division=0)
        for name in names:
            r = report.get(name, {})
            metrics[f"f1_{name}"] = r.get("f1-score", 0)
            metrics[f"support_{name}"] = r.get("support", 0)
    return metrics


def print_metrics(title, m):
    print("=" * 45)
    print(f"  {title}")
    print("=" * 45)
    print(f"  Accuracy           : {m['accuracy']:.4f}")
    print(f"  Macro Precision    : {m['macro_precision']:.4f}")
    print(f"  Macro Recall       : {m['macro_recall']:.4f}")
    print(f"  Macro F1           : {m['macro_f1']:.4f}")
    print(f"  Weighted Precision : {m['weighted_precision']:.4f}")
    print(f"  Weighted Recall    : {m['weighted_recall']:.4f}")
    print(f"  Weighted F1        : {m['weighted_f1']:.4f}")


# ======================== 主流程 ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="offense", choices=list(LOADERS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"task={args.task}  epochs={args.epochs}  device={DEVICE}")

    all_data = LOADERS[args.task]()

    encoder = LabelEncoder()
    labels = encoder.fit_transform([x["label"] for x in all_data])
    for item, lid in zip(all_data, labels):
        item["label"] = int(lid)
    classes = list(encoder.classes_)
    num_labels = len(classes)

    print(f"总样本数: {len(all_data)}")
    print("类别映射:", dict(enumerate(classes)))
    for name, cnt in zip(classes, np.bincount(labels)):
        print(f"  {name}: {cnt}")

    train_raw, temp_raw = train_test_split(
        all_data, test_size=0.2, random_state=SEED,
        stratify=[x["label"] for x in all_data])
    val_raw, test_raw = train_test_split(
        temp_raw, test_size=0.5, random_state=SEED,
        stratify=[x["label"] for x in temp_raw])
    print(f"训练集:{len(train_raw)} 验证集:{len(val_raw)} 测试集:{len(test_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    train_loader = DataLoader(TextDataset(train_raw, tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_raw, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TextDataset(test_raw, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

    model = TextCNN(tokenizer.vocab_size, 768, FILTER_SIZES, FILTER_NUM,
                    DROPOUT, num_labels).to(DEVICE)
    bert = AutoModel.from_pretrained(MODEL_PATH)
    model.embedding.weight.data.copy_(bert.embeddings.word_embeddings.weight.data)
    model.embedding.weight.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    pre = evaluate(model, test_loader)
    print_metrics("训练前测试集基线（未微调）", pre)

    best_f1, best_state, history = 0.0, None, []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for ids, lab in train_loader:
            ids, lab = ids.to(DEVICE), lab.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(ids), lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Macro-F1: {val['macro_f1']:.4f}")
        history.append({"epoch": epoch + 1, **{f"val_{k}": v for k, v in val.items()
                                               if isinstance(v, float)}})

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, test_loader, classes)
    print_metrics("训练后测试结果", test)

    out_dir = os.path.join(str(RESULTS_DIR), "textcnn", f"{args.task}_s{SEED}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task": args.task,
            "model": "TextCNN",
            "seed": SEED,
            "epochs": args.epochs,
            "best_val_macro_f1": best_f1,
            "classes": classes,
            "split": {"train": len(train_raw), "val": len(val_raw), "test": len(test_raw)},
            "pretrain_test": pre,
            "test": test,
            "log_history": history,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
