# -*- coding: utf-8 -*-
"""单任务微调 + 知识增强。

用法：python finetune.py --task target --mode baseline
task: offense|target|expression|dialect；mode: baseline|target|dialect|pinyin|full
"""

import argparse
import glob
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.nn import CrossEntropyLoss

import knowledge_aug

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 42
SPLIT_SEED = 42          # 数据划分固定，所有 seed 跑同一批数据
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
MAX_LEN = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_CONFIG = {
    "offense": {
        "model": str(MODELS_DIR / "chinese-roberta-wwm-ext-base"),
        "data": str(DATA_DIR / "OffendCN.json"),
        "weighted": False,
    },
    "target": {
        "model": str(MODELS_DIR / "chinese-roberta-wwm-ext-base"),
        "data": str(DATA_DIR / "subsets" / "targeted_offensive_data.json"),
        "weighted": False,
    },
    "expression": {
        "model": str(MODELS_DIR / "chinese-roberta-wwm-ext-base"),
        "data": str(DATA_DIR / "subsets" / "targeted_offensive_data.json"),
        "weighted": False,
    },
    "dialect": {
        "model": str(MODELS_DIR / "chinese-roberta-wwm-ext-base"),
        "data": str(DATA_DIR / "subsets" / "dialect_offensive_data.json"),
        "weighted": True,
    },
    # 方言冒犯二分类：在全部冒犯样本上判断是否方言冒犯（780 正 / 5591 负）
    "dialect_binary": {
        "model": str(MODELS_DIR / "chinese-roberta-wwm-ext-base"),
        "data": str(DATA_DIR / "subsets" / "offensive_data.json"),
        "weighted": True,
    },
}

MODE_FEATURES = {
    "baseline": set(),
    "target": {"target"},
    "dialect": {"dialect"},
    "pinyin": {"pinyin"},
    "full": {"target", "dialect", "pinyin"},
}

# ============ 直接运行（IDE 点运行 / 不传参数）时改这几行 ============
DEFAULT_TASK = "expression"     # offense | target | expression | dialect
DEFAULT_MODE = "full"      # baseline | target | dialect | pinyin | full
DEFAULT_SEED = SEED
# ==================================================================


# ======================== 数据加载 ========================

def load_offense_data(file_path):
    """冒犯检测（二分类）：读取全量 OffendCN.json 的 offensive 字段"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = []
    for item in data:
        content = item["content"].strip()
        res.append({
            "content": content,
            "label": 1 if item["offensive"] else 0,
        })
    return res


def load_target_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = []
    for item in data:
        content = item["content"].strip()
        four_tuple = item.get("four_tuple")
        if not isinstance(four_tuple, dict):
            continue
        target_category = four_tuple.get("Target_Category")
        if target_category is None:
            continue
        res.append({"content": content, "label": target_category})
    return res


def load_expression_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = []
    for item in data:
        content = item["content"].strip()
        expression = item.get("expression_type")
        if expression is None or expression == "":
            continue
        res.append({"content": content, "label": expression})
    return res


def load_dialect_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = []
    for item in data:
        content = item["content"].strip()
        dialect_info = item.get("dialect_info", {})
        if not dialect_info.get("is_dialect_offensive", False):
            continue
        dialect_group = dialect_info.get("dialect_group")
        if dialect_group is None or dialect_group == "":
            continue
        res.append({"content": content, "label": dialect_group})
    return res


def load_dialect_binary_data(file_path):
    """方言冒犯二分类：在冒犯样本上判断是否属于方言冒犯"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = []
    for item in data:
        dialect_info = item.get("dialect_info") or {}
        res.append({
            "content": item["content"].strip(),
            "label": 1 if dialect_info.get("is_dialect_offensive", False) else 0,
        })
    return res


LOADERS = {
    "offense": load_offense_data,
    "target": load_target_data,
    "expression": load_expression_data,
    "dialect": load_dialect_data,
    "dialect_binary": load_dialect_binary_data,
}


# ======================== 训练组件 ========================

CLASSES = []   # 由 main 赋值，供 compute_metrics 输出每类指标


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "weighted_recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    if CLASSES:
        names = [str(c) for c in CLASSES]
        report = classification_report(
            labels, preds, labels=list(range(len(names))),
            target_names=names, output_dict=True, zero_division=0,
        )
        for name in names:
            r = report.get(name, {})
            metrics[f"f1_{name}"] = r.get("f1-score", 0)
            metrics[f"support_{name}"] = r.get("support", 0)
    return metrics


class WeightedTrainer(Trainer):
    """方言与方言二分类任务使用的类别加权损失"""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels.to(logits.device))
        return (loss, outputs) if return_outputs else loss


def kg_collate(features):
    """所有序列已 pad 到 MAX_LEN，直接堆叠即可"""
    batch = {}
    for key in features[0]:
        if key == "content":
            continue
        batch[key] = torch.tensor([f[key] for f in features])
    if "label" in batch:
        batch["labels"] = batch.pop("label")
    return batch


# ======================== 主流程 ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, choices=list(TASK_CONFIG))
    parser.add_argument("--mode", default=None, choices=list(MODE_FEATURES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--weighted", action="store_true",
                        help="强制开启类别加权（默认只有 dialect 任务开启）")
    parser.add_argument("--keep-ckpt", action="store_true",
                        help="保留 checkpoint（默认跑完删除，只留 metrics.json）")
    parser.add_argument("--oversample", type=float, default=0.0,
                        help="少数类过采样强度：0=关闭；0.5=把少数类补到最多类的 50%%")
    parser.add_argument("--model", default=None,
                        help="换底座：模型目录名（如 bert-base-chinese）或绝对路径；"
                             "不填则用 TASK_CONFIG 里该任务的默认底座")
    args = parser.parse_args()

    if args.task is None:
        args.task = DEFAULT_TASK
    if args.mode is None:
        args.mode = DEFAULT_MODE

    cfg = TASK_CONFIG[args.task]
    need = MODE_FEATURES[args.mode]

    # 换底座：--model 给相对名时按 models/ 解析；输出目录带上模型名避免互相覆盖
    if args.model:
        model_name = args.model if os.path.isabs(args.model) else str(MODELS_DIR / args.model)
        model_tag = "_" + os.path.basename(os.path.normpath(model_name))
    else:
        model_name = cfg["model"]
        model_tag = ""

    # 固定随机种子：必须在建模之前，保证模型初始化可复现
    set_seed(args.seed)

    print(f"task={args.task}  mode={args.mode}  seed={args.seed}  "
          f"features={sorted(need) or '无'}  "
          f"model={os.path.basename(model_name)}")

    # ---------- 数据 ----------
    rows = LOADERS[args.task](cfg["data"])
    df_all = pd.DataFrame(rows)
    print("总样本数:", len(df_all))
    print(df_all["label"].value_counts())

    encoder = LabelEncoder()
    df_all["label"] = encoder.fit_transform(df_all["label"])
    print("类别映射:")
    for i, c in enumerate(encoder.classes_):
        print(i, "->", c)
    num_labels = len(encoder.classes_)
    CLASSES[:] = list(encoder.classes_)

    # 数据划分固定用 SPLIT_SEED，--seed 只影响模型初始化与训练过程，
    # 保证不同种子之间用的是同一批 train/val/test
    train_df, temp_df = train_test_split(
        df_all, test_size=0.2, random_state=SPLIT_SEED, stratify=df_all["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SPLIT_SEED, stratify=temp_df["label"]
    )
    print(f"训练集:{len(train_df)} 验证集:{len(val_df)} 测试集:{len(test_df)}")

    # 少数类过采样（只动训练集，验证/测试不变）
    if args.oversample > 0:
        counts = train_df["label"].value_counts()
        target_n = int(counts.max() * args.oversample)
        parts = [train_df]
        for lab, cnt in counts.items():
            if cnt < target_n:
                extra = train_df[train_df["label"] == lab].sample(
                    n=target_n - cnt, replace=True, random_state=args.seed
                )
                parts.append(extra)
        train_df = pd.concat(parts).sample(
            frac=1, random_state=args.seed
        ).reset_index(drop=True)
        print(f"过采样后训练集: {len(train_df)}  目标每类 {target_n}")

    # ---------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(examples):
        encoded = tokenizer(
            examples["content"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_offsets_mapping=bool(need),
        )
        if need:
            offsets = encoded.pop("offset_mapping")
            knowledge_aug.add_knowledge_features(
                encoded, examples["content"], offsets, tokenizer, need, MAX_LEN
            )
        return encoded

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["content", "label"]], preserve_index=False)
        return ds.map(tokenize_func, batched=True)

    train_ds, val_ds, test_ds = to_dataset(train_df), to_dataset(val_df), to_dataset(test_df)

    # ---------- 模型 ----------
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if need:
        model = knowledge_aug.KGClassifier(
            base, num_labels, knowledge_aug.feature_dims(need),
            hidden=256,
        ).to(DEVICE)
        print(f"知识分支: {knowledge_aug.feature_dims(need)}")
    else:
        model = base.to(DEVICE)

    out_dir = os.path.join(
        str(RESULTS_DIR), "kg",
        f"{args.task}_{args.mode}_s{args.seed}"
        + ("_w" if args.weighted else "")
        + (f"_o{args.oversample}" if args.oversample > 0 else "")
        + model_tag,
    )

    ta_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,          # 只保留最优 checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=args.seed,
        logging_strategy="no",
        log_level="warning",
        remove_unused_columns=not need,
    )
    # 只存权重，不存 optimizer/scheduler/rng（1.1GB → 约 0.4GB）
    if "save_only_model" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["save_only_model"] = True
    train_args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = dict(
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if need:
        trainer_kwargs["data_collator"] = kg_collate
    if cfg["weighted"] or args.weighted:
        counts = train_df["label"].value_counts().sort_index()
        weights = len(train_df) / (len(counts) * counts.values)
        trainer_kwargs["class_weights"] = torch.tensor(weights, dtype=torch.float).to(DEVICE)
        trainer = WeightedTrainer(model=model, **trainer_kwargs)
    else:
        trainer = Trainer(model=model, **trainer_kwargs)

    trainer.train()
    test_metrics = trainer.evaluate(test_ds)

    print("=" * 45)
    print(f"  {args.task} / {args.mode} 测试集结果")
    print("=" * 45)
    for name, key in [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted Precision", "weighted_precision"),
        ("Weighted Recall", "weighted_recall"),
        ("Weighted F1", "weighted_f1"),
    ]:
        print(f"  {name:<20}: {test_metrics['eval_' + key]:.4f}")
    print("  ---- 各类别 ----")
    for c in encoder.classes_:
        f1_c = test_metrics.get(f"eval_f1_{c}", 0)
        sup_c = test_metrics.get(f"eval_support_{c}", 0)
        print(f"  {str(c):<20}: F1={f1_c:.4f}  Support={int(sup_c)}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task": args.task,
            "mode": args.mode,
            "test": test_metrics,
            "config": {
                "model": os.path.basename(model_name),
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "max_len": MAX_LEN,
                "seed": args.seed,
                "split_seed": SPLIT_SEED,
                "oversample": args.oversample,
                "classes": list(encoder.classes_),
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"指标已保存: {os.path.join(out_dir, 'metrics.json')}")

    # 默认清理 checkpoint：训练完权重就没用了，只留 metrics.json
    if not args.keep_ckpt:
        for ck in glob.glob(os.path.join(out_dir, "checkpoint-*")):
            shutil.rmtree(ck, ignore_errors=True)
        print("已清理 checkpoint，仅保留 metrics.json（需要权重时加 --keep-ckpt）")


if __name__ == "__main__":
    main()
