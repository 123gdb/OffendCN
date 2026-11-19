# -*- coding: utf-8 -*-
"""MTL 消融：依次跑 4 组配置。

用法：python ablation.py [--skip-existing]
产物：results/ablation_logs/、results/MTL消融汇总.xlsx
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "multitask.py")
OUT_DIR = str(RESULTS_DIR)
LOG_DIR = os.path.join(OUT_DIR, "ablation_logs")
SUMMARY_XLSX = os.path.join(OUT_DIR, "MTL消融汇总.xlsx")
BASELINE_JSON = os.path.join(OUT_DIR, "pretrain_baseline.json")

# (配置名, 说明, 输出目录后缀)
CONFIGS = [
    ("only_off", "仅冒犯信号", "mtl-only-off"),
    ("no_dialect", "去掉方言", "mtl-no-dialect"),
    ("no_target", "去掉目标", "mtl-no-target"),
    ("no_expression", "去掉表达方式", "mtl-no-expression"),
]

# metrics.json 里 test 的键名（multitask.py 的 HEAD_NAMES 值）
HEAD_NAMES = {
    "offensive": "冒犯",
    "target": "目标",
    "expr": "表达",
    "dialect_binary": "方言二分类",
    "dialect_group": "方言分组",
}

# metrics.json 里 supervised_heads 的键名（multitask.py 的内部头名），与上表不同
SUP_HEAD_CN = {
    "off": "冒犯",
    "target": "目标",
    "expr": "表达",
    "dialect_bin": "方言二分类",
    "dialect": "方言分组",
}


def find_metrics(tag):
    """在 results 下查找该配置的 metrics.json。"""
    suffix = f"-{tag}"
    for root, _dirs, files in os.walk(OUT_DIR):
        if "metrics.json" in files and root.endswith(suffix):
            with open(os.path.join(root, "metrics.json"), "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def run_one(cfg, tag, skip_existing=False, extra_env=None, log_name=None):
    log_path = os.path.join(LOG_DIR, f"{log_name or cfg}.log")

    if skip_existing and find_metrics(tag) is not None:
        print(f"[跳过] {cfg} 已有 metrics.json")
        return 0

    env = dict(os.environ)
    env["MTL_ABLATION"] = cfg
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [sys.executable, SCRIPT],
            cwd=HERE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        proc.wait()
    return proc.returncode


def fill_metrics(row, test):
    """把某个配置的 test 指标展开成一行。"""
    for key, cn in HEAD_NAMES.items():
        if key in test:
            t = test[key]
            row[f"{cn}_accuracy"] = round(t["accuracy"], 4)
            row[f"{cn}_macro_precision"] = round(t["macro_precision"], 4)
            row[f"{cn}_macro_recall"] = round(t["macro_recall"], 4)
            row[f"{cn}_macro_f1"] = round(t["macro_f1"], 4)
            row[f"{cn}_weighted_precision"] = round(t["weighted_precision"], 4)
            row[f"{cn}_weighted_recall"] = round(t["weighted_recall"], 4)
            row[f"{cn}_weighted_f1"] = round(t["weighted_f1"], 4)
    return row


def build_summary():
    rows = []
    for cfg, desc, tag in CONFIGS:
        m = find_metrics(tag)
        row = {"配置": cfg, "说明": desc}
        if m is None:
            row["状态"] = "未运行"
            rows.append(row)
            continue

        row["状态"] = "完成"
        row["监督头"] = "+".join(
            SUP_HEAD_CN.get(h, h) for h in m.get("supervised_heads", [])
        )
        row["最优验证_macro_f1"] = round(
            m.get("best_val_offensive_macro_f1", m.get("best_val_macro_f1", 0)), 4
        )
        rows.append(fill_metrics(row, m.get("test") or {}))

    # 零微调基线：5 个配置的随机初始化完全相同，只需一组
    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON, "r", encoding="utf-8") as f:
            b = json.load(f)
        row = {"配置": "零微调基线", "说明": "未微调（随机初始化头）", "状态": "完成",
               "监督头": "无", "最优验证_macro_f1": None}
        rows.append(fill_metrics(row, b.get("test") or {}))
    else:
        rows.append({"配置": "零微调基线", "说明": "未微调（随机初始化头）",
                     "状态": "未生成"})

    return rows


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    skip_existing = "--skip-existing" in sys.argv
    summary_only = "--summary-only" in sys.argv

    failed = []
    if "--detail" in sys.argv:
        # 只加载已训练权重做详细分析（混淆矩阵/逐类指标/预测分布），不训练
        targets = [c for c in CONFIGS if c[0] in sys.argv[1:]] or CONFIGS
        for cfg, desc, tag in targets:
            print(f"\n{'=' * 60}\n[详细分析] {cfg} —— {desc}\n{'=' * 60}")
            code = run_one(cfg, tag, extra_env={"MTL_EVAL_ONLY": "1"},
                           log_name=f"detail_{cfg}")
            if code != 0:
                failed.append(cfg)
                print(f"[失败] {cfg} 详细分析退出码 {code}")
        if failed:
            print(f"\n以下配置详细分析失败: {failed}")
            sys.exit(1)
        return

    if "--baseline-only" in sys.argv:
        print(f"\n{'=' * 60}\n[0] 零微调基线（未微调，只评估不训练）\n{'=' * 60}")
        code = run_one("full", "mtl-full", extra_env={"MTL_BASELINE_ONLY": "1"},
                       log_name="baseline")
        if code != 0:
            print(f"[失败] 零微调基线退出码 {code}，日志见 {LOG_DIR}/baseline.log")
            sys.exit(1)
        print(f"已生成 {BASELINE_JSON}")
        return

    if summary_only:
        print("[--summary-only] 跳过训练，直接用已有 metrics.json 重建汇总表")
    else:
        print(f"\n{'=' * 60}\n[0] 零微调基线（未微调，只评估不训练）\n{'=' * 60}")
        run_one("full", "mtl-full", extra_env={"MTL_BASELINE_ONLY": "1"}, log_name="baseline")

        for i, (cfg, desc, tag) in enumerate(CONFIGS, 1):
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(CONFIGS)}] {cfg} —— {desc}")
            print(f"{'=' * 60}")
            code = run_one(cfg, tag, skip_existing)
            if code != 0:
                failed.append(cfg)
                print(f"[失败] {cfg} 退出码 {code}，日志见 {LOG_DIR}")

    rows = build_summary()
    df = pd.DataFrame(rows)
    df.to_excel(SUMMARY_XLSX, index=False)

    print(f"\n{'=' * 60}\n汇总（已写入 {SUMMARY_XLSX}）\n{'=' * 60}")
    cols = ["配置", "说明", "状态", "监督头", "最优验证_macro_f1"]
    for cn in HEAD_NAMES.values():
        cols += [f"{cn}_macro_f1", f"{cn}_weighted_f1"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\n完整指标（含各任务 accuracy / macro_precision / macro_recall / "
          f"weighted_precision / weighted_recall）见 {SUMMARY_XLSX}")

    if failed:
        print(f"\n以下配置运行失败: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
