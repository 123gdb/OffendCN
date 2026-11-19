# -*- coding: utf-8 -*-
"""Qwen3-14B（Ollama）评测：5 个任务 × 0/5-shot。

用法：python eval_qwen.py --task offense --shot 0
"""

import argparse
import os
import random

import requests
from tqdm import tqdm

from eval_utils import (
    gold_dialect,
    gold_dialect_binary,
    gold_expression,
    gold_target,
    gold_target_optional,
    load_json,
    print_binary_label_report,
    print_metrics,
    save_results,
)

from prompts import binary_index, build_prompt, parse_answer, select_examples

MODEL_NAME = "qwen3:14b-Q4_K_M"
MODEL_TAG = os.path.basename(MODEL_NAME).replace(":", "-").replace(" ", "-")
URL = "http://localhost:11434/api/generate"

RESULT_FILES = {
    ("offense", 0): f"{MODEL_TAG}_zero_result.json",
    ("offense", 5): f"{MODEL_TAG}_5shot_result.json",
    ("target", 0): f"{MODEL_TAG}_0shot_target_result.json",
    ("target", 5): f"{MODEL_TAG}_5shot_target_result.json",
    ("expression", 0): f"{MODEL_TAG}_0shot_expression_result.json",
    ("expression", 5): f"{MODEL_TAG}_5shot_expression_result.json",
    ("dialect", 0): f"{MODEL_TAG}_0shot_dialect_result.json",
    ("dialect", 5): f"{MODEL_TAG}_5shot_dialect_result.json",
    ("dialect_binary", 0): f"{MODEL_TAG}_0shot_dialect_binary_result.json",
    ("dialect_binary", 5): f"{MODEL_TAG}_5shot_dialect_binary_result.json",
}

# 统一采样参数：与 eval_chatglm.py 完全一致
BASE_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": 8192,
}


# ===================== 推理 =====================

def ask_qwen(prompt, options=None, strip=False, verbose=False):
    """Ollama /api/generate 调用，采样参数与 eval_chatglm.py 保持一致。"""
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {**BASE_OPTIONS, **(options or {})},
    }

    r = requests.post(URL, json=data, timeout=600)
    result = r.json()

    if strip:
        raw_resp = result.get("response", "").strip()
        if verbose:
            print("RAW输出:", repr(raw_resp))
        return raw_resp

    return result["response"]


# ===================== offense：prompt / 解析 =====================

def offense_prompt_0shot(text):
    return build_prompt("offense", text, shot=0)


def offense_prompt_5shot(examples, text):
    return build_prompt("offense", text, shot=5)


def parse_offense_0shot(ans):
    return binary_index(parse_answer("offense", ans))


def parse_offense_5shot(ans):
    return binary_index(parse_answer("offense", ans))


# ===================== target：prompt / 解析 =====================

def target_prompt_0shot(text):
    return build_prompt("target", text, shot=0)


def target_prompt_5shot(examples, text):
    return build_prompt("target", text, shot=5)


def parse_target_0shot(ans):
    return parse_answer("target", ans)


def parse_target_5shot(ans):
    return parse_answer("target", ans)


# ===================== expression：prompt / 解析 =====================

def expression_prompt_0shot(text):
    return build_prompt("expression", text, shot=0)


def expression_prompt_5shot(examples, text):
    return build_prompt("expression", text, shot=5)


def parse_expression(ans):
    return parse_answer("expression", ans)


# ===================== dialect：prompt / 解析 =====================

def dialect_prompt_0shot(text):
    return build_prompt("dialect", text, shot=0)


def dialect_prompt_5shot(examples, text):
    return build_prompt("dialect", text, shot=5)


def parse_dialect_0shot(ans):
    return parse_answer("dialect", ans)


def parse_dialect_5shot(ans):
    return parse_answer("dialect", ans)


# ===================== dialect_binary：prompt / 解析 =====================

def dialect_binary_prompt_0shot(text):
    return build_prompt("dialect_binary", text, shot=0)


def dialect_binary_prompt_5shot(examples, text):
    return build_prompt("dialect_binary", text, shot=5)


def parse_binary(ans):
    return parse_answer("dialect_binary", ans)


# ===================== 各任务主流程 =====================

def run_offense(shot):
    if shot == 5:
        test = load_json("test.json")

        examples = select_examples("offense", 5)

        print("\n=====5-shot示例=====")
        for e in examples:
            print(
                e["content"],
                e["offensive"]
            )
    else:
        test = load_json("test.json")

    y_true = []
    y_pred = []
    results = []

    for item in tqdm(test):
        text = item["content"]

        if shot == 0:
            prompt = offense_prompt_0shot(text)
        else:
            prompt = offense_prompt_5shot(examples, text)

        answer = ask_qwen(prompt)

        if shot == 0:
            pred = parse_offense_0shot(answer)

            if pred != -1:
                y_true.append(
                    1 if item["offensive"] else 0
                )
                y_pred.append(pred)

            results.append(
                {
                    "content": text,
                    "gold": item["offensive"],
                    "answer": answer,
                    "pred": pred
                }
            )
        else:
            pred = parse_offense_5shot(answer)

            if pred != -1:
                gold = (
                    1
                    if item["offensive"]
                    else 0
                )
                y_true.append(gold)
                y_pred.append(pred)

            results.append(
                {
                    "content": item["content"],
                    "gold": gold if pred != -1 else None,
                    "response": answer,
                    "prediction": pred
                }
            )

    if shot == 0:
        mode = "zero"
        print("\n==========", mode, "==========")
        print_metrics(
            y_true, y_pred,
            round_digits=None,
            fpr_fnr=True,
            fpr_label="FPR误判率",
            fnr_label="FNR漏判率",
            binary_safe=False,
        )
    else:
        print_metrics(
            y_true, y_pred,
            title="\n==========整体测试集==========",
            round_digits=4,
            fpr_fnr=True,
            fpr_label="FPR",
            fnr_label="FNR",
            binary_safe=False,
        )

    save_results(results, RESULT_FILES[("offense", shot)])


def run_target(shot):
    if shot == 5:
        test = load_json("target_test.json")

        examples = select_examples("target", 5)

        print("\n=====5-shot示例=====")
        for e in examples:
            print(
                e["content"],
                "=>",
                gold_target(e)
            )
    else:
        test = load_json("target_test.json")

    y_true = []
    y_pred = []
    results = []

    for item in tqdm(test):
        text = item["content"]

        if shot == 0:
            prompt = target_prompt_0shot(text)
            answer = ask_qwen(prompt, strip=True)
            pred = parse_target_0shot(answer)
            gold = gold_target_optional(item)
            # 只统计五个有效类别
            if pred != "Unknown" and gold in ["Region", "Gender", "Race", "LGBTQ", "Other"]:
                y_true.append(gold)
                y_pred.append(pred)
        else:
            prompt = target_prompt_5shot(examples, text)
            answer = ask_qwen(
                prompt, strip=True, verbose=True
            )
            pred = parse_target_5shot(answer)
            gold = gold_target(item)
            if pred != "Unknown" and gold:
                y_true.append(gold)
                y_pred.append(pred)

        results.append({
            "content": text,
            "gold": gold,
            "answer": answer,
            "pred": pred
        })

    if shot == 0:
        title = "\n========== Target任务B =========="
    else:
        title = "\n========== Task B Target =========="
    print_metrics(y_true, y_pred, title=title)

    save_results(results, RESULT_FILES[("target", shot)])


def run_expression(shot):
    if shot == 5:
        examples = select_examples("expression", 5)
        print("Selected examples:", examples)

    test = load_json("target_test.json")

    y_true = []
    y_pred = []
    results = []

    for item in tqdm(test):
        text = item["content"]

        if shot == 0:
            prompt = expression_prompt_0shot(text)
        else:
            prompt = expression_prompt_5shot(examples, text)

        answer = ask_qwen(prompt)
        pred = parse_expression(answer)
        gold = gold_expression(item)

        if pred != "Unknown" and gold:
            y_true.append(gold)
            y_pred.append(pred)

        results.append({
            "content": text,
            "gold": gold,
            "answer": answer,
            "pred": pred
        })

    print_metrics(y_true, y_pred, title="\n==========Task C Expression==========")

    save_results(results, RESULT_FILES[("expression", shot)])


def run_dialect(shot):
    if shot == 5:
        examples = select_examples("dialect", 5)

    test = load_json("dialect_test.json")

    y_true = []
    y_pred = []
    results = []

    for item in tqdm(test):
        text = item["content"]

        if shot == 0:
            prompt = dialect_prompt_0shot(text)
            answer = ask_qwen(prompt)
            pred = parse_dialect_0shot(answer)
            gold = gold_dialect(item)
            # 0-shot 原脚本只判断 gold 非空，Unknown 预测也计入
            if gold:
                y_true.append(gold)
                y_pred.append(pred)
        else:
            prompt = dialect_prompt_5shot(examples, text)
            answer = ask_qwen(prompt)
            pred = parse_dialect_5shot(answer)
            gold = gold_dialect(item)
            if pred != "Unknown" and gold:
                y_true.append(gold)
                y_pred.append(pred)

        results.append({
            "content": text,
            "gold": gold,
            "answer": answer,
            "pred": pred
        })

    if shot == 0:
        title = "\n==========Task D Dialect Group=========="
    else:
        title = "\n==========Task D Dialect Group (5-shot)=========="
    print_metrics(y_true, y_pred, title=title)

    save_results(results, RESULT_FILES[("dialect", shot)])


def run_dialect_binary(shot):
    if shot == 5:
        examples = select_examples("dialect_binary", 5)
        print(f"示例条数: {len(examples)} | 正负分布: "
              f"{[gold_dialect_binary(e) for e in examples].count('方言冒犯')} 正 / "
              f"{[gold_dialect_binary(e) for e in examples].count('非方言冒犯')} 负")

    test = load_json("dialect_bin_test.json")

    y_true = []
    y_pred = []
    results = []
    unknown = 0

    for idx, item in enumerate(tqdm(test)):
        text = item["content"]

        if shot == 0:
            prompt = dialect_binary_prompt_0shot(text)
        else:
            prompt = dialect_binary_prompt_5shot(examples, text)

        answer = ask_qwen(prompt)
        pred = parse_binary(answer)
        gold = gold_dialect_binary(item)

        print(f"\n===== 第 {idx + 1} 条 =====")
        print("文本:", text)
        print("模型原始输出:", repr(answer))
        print("解析后预测:", pred)
        print("真实标签:", gold)

        if pred != "Unknown":
            y_true.append(gold)
            y_pred.append(pred)
        else:
            unknown += 1

        results.append({
            "content": text,
            "gold": gold,
            "answer": answer,
            "pred": pred,
        })

    print(f"\n解析失败（Unknown）: {unknown} 条")

    shot_name = "0-shot" if shot == 0 else "5-shot"
    print("\n" + "=" * 60)
    print(f"【方言冒犯二分类（{shot_name}）评估指标】")
    print("=" * 60)
    print_metrics(y_true, y_pred)
    print_binary_label_report(y_true, y_pred)

    path = save_results(results, RESULT_FILES[("dialect_binary", shot)])
    print(f"结果已保存: {path}")


# ===================== 入口 =====================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-14B（Ollama）冒犯语言评测")
    parser.add_argument(
        "--task",
        required=True,
        choices=["offense", "target", "expression", "dialect", "dialect_binary"],
    )
    parser.add_argument("--shot", required=True, type=int, choices=[0, 5])
    args = parser.parse_args()

    random.seed(42)

    dispatch = {
        "offense": run_offense,
        "target": run_target,
        "expression": run_expression,
        "dialect": run_dialect,
        "dialect_binary": run_dialect_binary,
    }
    dispatch[args.task](args.shot)


if __name__ == "__main__":
    main()
