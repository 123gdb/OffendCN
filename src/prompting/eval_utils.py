# -*- coding: utf-8 -*-
"""提示评测公共模块：数据加载、真值提取、指标计算、结果保存。"""

import json
import os
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RESULTS_DIR, SPLITS_DIR

# 数据划分从 data/splits/ 读取，结果写到 results/，
# 保证无论从哪个目录启动，读写的都是同一批 json。

# 方言冒犯二分类的两个类别（顺序与原脚本一致）
LABELS = ["非方言冒犯", "方言冒犯"]

METRIC_NAMES = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]


# ===================== 数据加载 / 结果落盘 =====================

def load_json(filename):
    with open(os.path.join(str(SPLITS_DIR), filename), encoding="utf-8") as f:
        return json.load(f)


def save_results(results, filename):
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    path = os.path.join(str(RESULTS_DIR), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


# ===================== 真值提取（各任务保持原脚本写法） =====================

def gold_target(item):
    """target 任务：four_tuple.Target_Category。"""
    return item.get("four_tuple", {}).get("Target_Category", "")


def gold_target_optional(item):
    """qwen-0shot-target.py 的写法：four_tuple 不是 dict 时返回空串。"""
    ft = item.get("four_tuple")
    return ft.get("Target_Category", "") if isinstance(ft, dict) else ""


def gold_expression(item):
    return item.get("expression_type", "")


def gold_dialect(item):
    return item.get("dialect_info", {}).get("dialect_group", "")


def gold_dialect_binary(item):
    dialect_info = item.get("dialect_info") or {}
    return "方言冒犯" if dialect_info.get("is_dialect_offensive") else "非方言冒犯"


# ===================== 指标计算 =====================

def compute_metrics(y_true, y_pred):
    """accuracy / macro P,R,F1 / weighted P,R,F1（zero_division=0，与原脚本一致）。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def binary_confusion(y_true, y_pred, safe=False):
    """返回 TN, FP, FN, TP。

    safe=True 对应 chatglm 两个 offense 脚本的兜底逻辑；
    safe=False 对应 qwen 两个 offense 脚本的直接 ravel()。
    """
    cm = confusion_matrix(y_true, y_pred)
    if not safe:
        return cm.ravel()

    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        labels = sorted(set(y_true) | set(y_pred))
        if labels == [0]:
            TN, FP, FN, TP = cm[0, 0], 0, 0, 0
        elif labels == [1]:
            TN, FP, FN, TP = 0, 0, 0, cm[0, 0]
        else:
            TN = FP = FN = TP = 0
    return TN, FP, FN, TP


def print_metrics(
    y_true,
    y_pred,
    title="",
    round_digits=4,
    fpr_fnr=False,
    fpr_label="FPR",
    fnr_label="FNR",
    binary_safe=False,
):
    """按原脚本格式打印指标。

    round_digits=None 时不做 round（qwen-zero-shot.py 的写法）。
    """
    m = compute_metrics(y_true, y_pred)
    if title:
        print(title)
    for name in METRIC_NAMES:
        value = m[name]
        if round_digits is not None:
            value = round(value, round_digits)
        print(f"{name}:", value)

    if fpr_fnr:
        TN, FP, FN, TP = binary_confusion(y_true, y_pred, safe=binary_safe)
        fpr = FP / (FP + TN)
        fnr = FN / (FN + TP)
        if round_digits is not None:
            fpr = round(fpr, round_digits)
            fnr = round(fnr, round_digits)
        print(f"{fpr_label}:", fpr)
        print(f"{fnr_label}:", fnr)


def print_binary_label_report(y_true, y_pred):
    """方言冒犯二分类的各类别详细指标（四个原脚本一致）。"""
    print("\n" + "=" * 60)
    print("【各类别详细指标】")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=LABELS,
        zero_division=0, output_dict=True
    )
    print(f"{'类别':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 56)
    for name in LABELS:
        r = report.get(name, {})
        print(f"{name:<12} {r.get('precision', 0):>10.4f} {r.get('recall', 0):>10.4f} "
              f"{r.get('f1-score', 0):>10.4f} {r.get('support', 0):>10.0f}")
