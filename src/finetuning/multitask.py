# -*- coding: utf-8 -*-
"""多任务联合训练（硬共享编码器 + 5 个任务头）。

用法：python multitask.py
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR

# ===================== 配置区（与单任务基线保持一致） =====================
MODEL_NAME = str(MODELS_DIR / "chinese-roberta-wwm-ext-base")
DATA_PATH = str(DATA_DIR / "OffendCN.json")
SEED = 42
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 消融开关 =====================
# 取值: "full" | "only_off" | "no_dialect" | "no_target" | "no_expression"
# 可直接改这里；也可用环境变量 MTL_ABLATION 覆盖（供 ablation.py 批量运行）
# 输出目录会按配置自动分开，不会互相覆盖
ABLATION = os.environ.get("MTL_ABLATION", "full").strip()

_ABLATION_SWITCH = {
    "full":          (True,  True,  True),   # (USE_TARGET, USE_EXPR, USE_DIALECT)
    "only_off":      (False, False, False),
    "no_dialect":    (True,  True,  False),
    "no_target":     (False, True,  True),
    "no_expression": (True,  False, True),
}
_ABLATION_TAG = {
    "full": "mtl-full",
    "only_off": "mtl-only-off",
    "no_dialect": "mtl-no-dialect",
    "no_target": "mtl-no-target",
    "no_expression": "mtl-no-expression",
}
if ABLATION not in _ABLATION_SWITCH:
    raise ValueError(f"未知 ABLATION: {ABLATION}，可选 {list(_ABLATION_SWITCH)}")

USE_TARGET, USE_EXPR, USE_DIALECT = _ABLATION_SWITCH[ABLATION]
# SUPERVISED_HEADS: 参与 loss 的头（消融只改这里）
SUPERVISED_HEADS = (
    ["off"]
    + (["target"] if USE_TARGET else [])
    + (["expr"] if USE_EXPR else [])
    + (["dialect_bin", "dialect"] if USE_DIALECT else [])
)
# ALL_HEADS: 每个配置都创建、都评估的头（保证 5 个任务在所有配置下都有指标）
ALL_HEADS = ["off", "target", "expr", "dialect_bin", "dialect"]
HEAD_NAMES = {
    "off": "offensive",
    "target": "target",
    "expr": "expr",
    "dialect_bin": "dialect_binary",
    "dialect": "dialect_group",
}
# 本配置的输出目录（训练结果与详细分析都写这里）
MTL_DST = str(RESULTS_DIR / f"{os.path.basename(MODEL_NAME)}-{_ABLATION_TAG[ABLATION]}")
# ========================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_data(file_path):
    """读取主数据集，抽取五个任务的标签（缺失记为 None）。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = []
    for item in data:
        ft = item.get("four_tuple")
        target = ft.get("Target_Category") if isinstance(ft, dict) else None
        di = item.get("dialect_info", {})
        is_dialect = bool(di.get("is_dialect_offensive"))
        # 方言分组标签只在「方言冒犯」样本上有效
        dialect = di.get("dialect_group") if is_dialect else None
        dialect = dialect if dialect else None
        res.append({
            "content": item["content"].strip(),
            "off": 1 if item["offensive"] else 0,
            "target": target if target else None,
            "expr": item.get("expression_type") or None,
            # 方言冒犯二分类口径与 dialect_binary.py 一致：只在冒犯样本上定义（780 正 / 5591 负）
            "dialect_bin": (1 if is_dialect else 0) if item["offensive"] else -1,
            "dialect": dialect,
        })
    return res


def encode_column(items, key):
    """把类别名编码为整数 id，缺失样本记为 -1。"""
    present = sorted({x[key] for x in items if x[key] is not None})
    le = LabelEncoder().fit(present)
    ids = [int(le.transform([x[key]])[0]) if x[key] is not None else -1 for x in items]
    return ids, le


all_items = load_data(DATA_PATH)

# 类别编码（在整体上一次性 fit，保证三个划分的 id 一致、无类别泄露到未知 id）
target_ids, target_le = encode_column(all_items, "target")
expr_ids, expr_le = encode_column(all_items, "expr")
dialect_ids, dialect_le = encode_column(all_items, "dialect")
for i, x in enumerate(all_items):
    x["target_id"] = target_ids[i]
    x["expr_id"] = expr_ids[i]
    x["dialect_id"] = dialect_ids[i]

print("目标类别数:", len(target_le.classes_), list(target_le.classes_))
print("表达类别数:", len(expr_le.classes_), list(expr_le.classes_))
print("方言类别数:", len(dialect_le.classes_), list(dialect_le.classes_))

# ===== 与单任务保持一致的划分 =====
# 每个监督头在自己的子集上独立切 8:1:1（和 finetune.py 完全相同的 SPLIT_SEED 与 stratify），
# 这样各头的测试集与单任务实验逐条相同，指标可直接比较。
# 为避免跨头泄漏：某条样本只要在任意一个头里落在 test/val，就从所有头的训练集中剔除。
SPLIT_FIELD = {
    "off": "off",
    "target": "target_id",
    "expr": "expr_id",
    "dialect_bin": "dialect_bin",
    "dialect": "dialect_id",
}

per_head = {}
for _head, _field in SPLIT_FIELD.items():
    _sub = [x for x in all_items if x[_field] >= 0]
    _tr, _temp = train_test_split(
        _sub, test_size=0.2, random_state=SEED,
        stratify=[x[_field] for x in _sub]
    )
    _va, _te = train_test_split(
        _temp, test_size=0.5, random_state=SEED,
        stratify=[x[_field] for x in _temp]
    )
    per_head[_head] = {
        "train": {x["content"] for x in _tr},
        "val": {x["content"] for x in _va},
        "test": {x["content"] for x in _te},
    }
    print(f"  {_head:<12} 子集 {len(_sub):6d}  train/val/test = "
          f"{len(_tr)}/{len(_va)}/{len(_te)}")

test_keys = set().union(*(per_head[h]["test"] for h in per_head))
val_keys = set().union(*(per_head[h]["val"] for h in per_head))
train_keys = {x["content"] for x in all_items} - test_keys - val_keys


def _mask(item, split):
    """只保留该头在当前划分下的标签，其余置 -1（损失里忽略）"""
    y = dict(item)
    for _head, _field in SPLIT_FIELD.items():
        if item["content"] not in per_head[_head][split]:
            y[_field] = -1
    return y


train_items = [_mask(x, "train") for x in all_items if x["content"] in train_keys]
val_items = [_mask(x, "val") for x in all_items if x["content"] in val_keys]
test_items = [_mask(x, "test") for x in all_items if x["content"] in test_keys]
print(f"训练集:{len(train_items)} 验证集:{len(val_items)} 测试集:{len(test_items)}")
for _head, _field in SPLIT_FIELD.items():
    _n = sum(1 for x in test_items if x[_field] >= 0)
    print(f"  {_head:<12} 测试集有效样本 {_n}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class MTLDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        enc = tokenizer(
            d["content"], max_length=MAX_LEN, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "off": torch.tensor(d["off"], dtype=torch.long),
            "target": torch.tensor(d["target_id"], dtype=torch.long),
            "expr": torch.tensor(d["expr_id"], dtype=torch.long),
            "dialect_bin": torch.tensor(d["dialect_bin"], dtype=torch.long),
            "dialect": torch.tensor(d["dialect_id"], dtype=torch.long),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "off": torch.stack([b["off"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "expr": torch.stack([b["expr"] for b in batch]),
        "dialect_bin": torch.stack([b["dialect_bin"] for b in batch]),
        "dialect": torch.stack([b["dialect"] for b in batch]),
    }


train_loader = DataLoader(MTLDataset(train_items), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(MTLDataset(val_items), batch_size=BATCH_SIZE, collate_fn=collate)
test_loader = DataLoader(MTLDataset(test_items), batch_size=BATCH_SIZE, collate_fn=collate)


class MTLRoBERTa(nn.Module):
    def __init__(self, n_target, n_expr, n_dialect):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        # 所有头都创建：未监督的头也保留，用于评估「该任务完全不给监督」的表现
        self.head_off = nn.Linear(dim, 2)
        self.head_target = nn.Linear(dim, n_target)
        self.head_expr = nn.Linear(dim, n_expr)
        self.head_dialect_bin = nn.Linear(dim, 2)
        self.head_dialect = nn.Linear(dim, n_dialect)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state[:, 0, :])
        return {
            "off": self.head_off(h),
            "target": self.head_target(h),
            "expr": self.head_expr(h),
            "dialect_bin": self.head_dialect_bin(h),
            "dialect": self.head_dialect(h),
        }


model = MTLRoBERTa(len(target_le.classes_), len(expr_le.classes_), len(dialect_le.classes_)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
# 方言冒犯二分类与 dialect_binary.py 同口径：只在冒犯样本上训练（780 正 / 5591 负），
# 用 N/(K*n_c) 加权，避免全判为负
_bin_labels = [x["dialect_bin"] for x in train_items if x["dialect_bin"] != -1]
_bin_counts = np.bincount(_bin_labels, minlength=2)
_bin_weight = len(_bin_labels) / (2 * np.maximum(_bin_counts, 1))
criterion_bin = nn.CrossEntropyLoss(
    weight=torch.tensor(_bin_weight, dtype=torch.float, device=DEVICE)
)
print("方言冒犯二分类类别权重:", dict(enumerate(np.round(_bin_weight, 3))),
      "| 训练样本", _bin_counts.tolist())


def masked_ce(logits, labels, crit=None):
    """对缺失标签（-1）做掩码后的交叉熵；本 batch 无有效标签时返回 None（不计入平均）。"""
    crit = crit or criterion
    mask = labels != -1
    if mask.sum() == 0:
        return None
    return crit(logits[mask], labels[mask])


def base_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "weighted_recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def collect_preds(model, loader):
    """收集 5 个头在给定数据集上的标签与预测（缺失标签 -1 跳过）。"""
    model.eval()
    buf = {h: {"p": [], "l": []} for h in ALL_HEADS}
    with torch.no_grad():
        for b in loader:
            outs = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            for h in ALL_HEADS:
                lab = b[h].to(DEVICE)
                # 缺失标签（-1）一律跳过；改用各头独立划分后，off 也可能出现 -1
                mask = lab != -1
                if mask.any():
                    buf[h]["p"] += outs[h][mask].argmax(-1).cpu().tolist()
                    buf[h]["l"] += lab[mask].cpu().tolist()
    return buf


def evaluate(model, loader):
    """5 个头全部评估，与消融开关无关：去掉监督 ≠ 去掉指标。"""
    buf = collect_preds(model, loader)
    return {
        HEAD_NAMES[h]: base_metrics(buf[h]["l"], buf[h]["p"])
        for h in ALL_HEADS
        if buf[h]["l"]
    }


def detail_report(model, loader):
    """每个头的混淆矩阵、逐类指标、预测分布（用于分析数据）。"""
    buf = collect_preds(model, loader)
    out = {}
    for h in ALL_HEADS:
        lab, pred = buf[h]["l"], buf[h]["p"]
        if not lab:
            continue
        cls = sorted(set(lab) | set(pred))
        rep = classification_report(lab, pred, labels=cls, output_dict=True, zero_division=0)
        out[HEAD_NAMES[h]] = {
            "n": len(lab),
            "confusion_matrix": {
                "labels": cls,
                "matrix": confusion_matrix(lab, pred, labels=cls).tolist(),
            },
            "per_class": {
                str(c): {k: round(rep[str(c)][k], 4)
                         for k in ("precision", "recall", "f1-score", "support")}
                for c in cls
            },
            "pred_dist": {str(c): pred.count(c) for c in cls},
        }
    return out


# 只做详细评估（加载已训练权重），不训练：用于分析每类的预测分布
if os.environ.get("MTL_EVAL_ONLY") == "1":
    _ckpt = os.path.join(MTL_DST, "best_mtl_roberta.pt")
    if not os.path.exists(_ckpt):
        raise FileNotFoundError(f"找不到已训练权重: {_ckpt}")
    model.load_state_dict(torch.load(_ckpt, map_location=DEVICE))
    _detail = {
        "ablation": ABLATION,
        "supervised_heads": SUPERVISED_HEADS,
        "test": detail_report(model, test_loader),
    }
    _detail_path = os.path.join(MTL_DST, "detail_test.json")
    with open(_detail_path, "w", encoding="utf-8") as f:
        json.dump(_detail, f, ensure_ascii=False, indent=2)
    print(f"已保存详细分析 -> {_detail_path}")
    for _h, _v in _detail["test"].items():
        print(f"{_h}: n={_v['n']} 预测分布={_v['pred_dist']} "
              f"逐类={ {k: v['f1-score'] for k, v in _v['per_class'].items()} }")
    raise SystemExit(0)

# 训练前基线（未微调）
print(f"===== 消融配置: {ABLATION} | 监督头: {'+'.join(SUPERVISED_HEADS)} | loss 按监督头取平均 =====")
pre = evaluate(model, test_loader)
print("===== 训练前测试集基线（未微调，5 个头全评） =====")
print(pre)

# 只算零微调基线后退出：不训练、不覆盖已训练模型，供 ablation.py 汇总
if os.environ.get("MTL_BASELINE_ONLY") == "1":
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    _pre_path = os.path.join(str(RESULTS_DIR), "pretrain_baseline.json")
    with open(_pre_path, "w", encoding="utf-8") as f:
        json.dump({"note": "训练前测试集基线（未微调，随机初始化分类头）",
                   "test": pre},
                  f, ensure_ascii=False, indent=2)
    print(f"已保存零微调基线 -> {_pre_path}")
    raise SystemExit(0)

best_f1 = -1.0
best_state = None
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    off_loss_sum = 0.0
    for b in train_loader:
        optimizer.zero_grad()
        outs = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        loss_off = criterion(outs["off"], b["off"].to(DEVICE))
        # 只对「本配置有监督」且 batch 内有有效标签的头计入平均
        losses = [loss_off]
        if USE_TARGET:
            l = masked_ce(outs["target"], b["target"].to(DEVICE))
            if l is not None:
                losses.append(l)
        if USE_EXPR:
            l = masked_ce(outs["expr"], b["expr"].to(DEVICE))
            if l is not None:
                losses.append(l)
        if USE_DIALECT:
            l = masked_ce(outs["dialect_bin"], b["dialect_bin"].to(DEVICE), criterion_bin)
            if l is not None:
                losses.append(l)
            l = masked_ce(outs["dialect"], b["dialect"].to(DEVICE))
            if l is not None:
                losses.append(l)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        off_loss_sum += loss_off.item()

    val = evaluate(model, val_loader)
    val_desc = " | ".join(
        f"val_{HEAD_NAMES[h]}_macro_f1 {val[HEAD_NAMES[h]]['macro_f1']:.4f}"
        for h in ALL_HEADS if HEAD_NAMES[h] in val
    )
    print(f"Epoch {epoch + 1}/{EPOCHS} | loss {total_loss / len(train_loader):.4f} "
          f"(off {off_loss_sum / len(train_loader):.4f}) | {val_desc}")
    # 模型选择统一以冒犯主任务的验证 macro_f1 为准，与消融配置无关
    if val["offensive"]["macro_f1"] > best_f1:
        best_f1 = val["offensive"]["macro_f1"]
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
test = evaluate(model, test_loader)
print("===== 训练后全部测试集指标（5 个头） =====")
print(test)

os.makedirs(MTL_DST, exist_ok=True)
torch.save(best_state, os.path.join(MTL_DST, "best_mtl_roberta.pt"))
print(f"已保存最优权重 -> {MTL_DST}/best_mtl_roberta.pt")

# 供 ablation.py 汇总的指标文件
with open(os.path.join(MTL_DST, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "ablation": ABLATION,
            "supervised_heads": SUPERVISED_HEADS,
            "evaluated_heads": ALL_HEADS,
            "best_val_offensive_macro_f1": best_f1,
            "test": test,
        },
        f, ensure_ascii=False, indent=2,
    )
print(f"已保存指标 -> {MTL_DST}/metrics.json")