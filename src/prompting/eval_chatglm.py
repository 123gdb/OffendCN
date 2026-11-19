# -*- coding: utf-8 -*-
"""ChatGLM3-6B（Ollama）评测：5 个任务 × 0/5-shot。

用法：python eval_chatglm.py --task offense --shot 0
前置：ollama 服务运行在 localhost:11434。
"""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import requests
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_utils import (
    compute_metrics,
    gold_dialect,
    gold_dialect_binary,
    gold_expression,
    gold_target,
    load_json,
    print_binary_label_report,
    print_metrics,
    save_results,
)

from prompts import binary_index, build_prompt, parse_answer, select_examples

MODEL_NAME = "chatglm3:6b"
MODEL_TAG = "chatglm3-6b"
URL = "http://localhost:11434/api/chat"
NUM_CTX = 8192

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

# ===================== 推理 =====================

def ask_chatglm(prompt, max_new_tokens=20):
    """Ollama /api/chat 调用，采样参数与 eval_qwen.py 保持一致。"""
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": NUM_CTX,
            "num_predict": max_new_tokens,
        },
    }
    r = requests.post(URL, json=data, timeout=600)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()


def ask_chatglm_generate(prompt):
    """offense 任务：最多 5 个 token，与 Qwen 侧参数一致。"""
    return ask_chatglm(prompt, max_new_tokens=5)


# ===================== offense：prompt / 解析 =====================

def offense_prompt_0shot(text):
    return build_prompt("offense", text, shot=0)


def offense_prompt_5shot(examples, text):
    return build_prompt("offense", text, shot=5)


def parse_answer_offense(ans):
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


def parse_expression_0shot(ans):
    return parse_answer("expression", ans)


def parse_expression_5shot(ans):
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
    result = []

    for item in tqdm(test):
        text = item["content"]

        if shot == 0:
            prompt = offense_prompt_0shot(text)
        else:
            prompt = offense_prompt_5shot(examples, text)

        response = ask_chatglm_generate(prompt)
        pred = parse_answer_offense(response)

        if shot == 0:
            if pred != -1:
                y_true.append(
                    1 if item["offensive"] else 0
                )
                y_pred.append(pred)
            result.append(
                {
                    "content": text,
                    "gold": item["offensive"],
                    "answer": response,
                    "pred": pred
                }
            )
        else:
            if pred != -1:
                gold = (
                    1
                    if item["offensive"]
                    else 0
                )
                y_true.append(gold)
                y_pred.append(pred)
            else:
                gold = None
            result.append(
                {
                    "content": item["content"],
                    "gold": gold,
                    "response": response,
                    "prediction": pred
                }
            )

    if shot == 0:
        mode = "zero"
        print("\n==========", mode, "==========")
        print_metrics(
            y_true, y_pred,
            round_digits=4,
            fpr_fnr=True,
            fpr_label="FPR误判率",
            fnr_label="FNR漏判率",
            binary_safe=True,
        )
    else:
        print_metrics(
            y_true, y_pred,
            title="\n==========整体测试集==========",
            round_digits=4,
            fpr_fnr=True,
            fpr_label="FPR",
            fnr_label="FNR",
            binary_safe=True,
        )

    save_results(result, RESULT_FILES[("offense", shot)])


def run_target(shot):
    if shot == 5:
        test = load_json("target_test.json")

        examples = select_examples("target", 5)

        print("\n=====5-shot示例=====")
        for e in examples:
            print(e["content"], "=>", gold_target(e))
    else:
        test = load_json("target_test.json")

    y_true = []
    y_pred = []
    results = []

    for item in tqdm(test):
        text = item["content"]

        if shot == 0:
            prompt = target_prompt_0shot(text)
        else:
            prompt = target_prompt_5shot(examples, text)

        # 两个 target 脚本均为 max_new_tokens=20
        raw_answer = ask_chatglm(prompt, max_new_tokens=20)

        if shot == 0:
            pred = parse_target_0shot(raw_answer)
            gold = gold_target(item)
            # 仅当真值有效、预测不为Unknown时计入评测
            if pred != "Unknown" and gold in ["Region", "Gender", "Race", "LGBTQ", "Other"]:
                y_true.append(gold)
                y_pred.append(pred)
        else:
            pred = parse_target_5shot(raw_answer)
            gold = gold_target(item)
            if pred != "Unknown" and gold:
                y_true.append(gold)
                y_pred.append(pred)

        results.append({
            "content": text,
            "gold_label": gold,
            "model_raw_output": raw_answer,
            "final_pred": pred
        })

    print_metrics(y_true, y_pred, title="\n==========Target任务B==========")

    save_results(results, RESULT_FILES[("target", shot)])


def run_expression(shot):
    if shot == 5:
        examples = select_examples("expression", 5)
        print("Selected examples:", [e.get("expression_type", "") for e in examples])

    test = load_json("target_test.json")

    y_true = []
    y_pred = []
    results = []

    for idx, item in enumerate(tqdm(test)):
        text = item["content"]

        if shot == 0:
            prompt = expression_prompt_0shot(text)
        else:
            prompt = expression_prompt_5shot(examples, text)

        # expression 两个脚本均为 max_new_tokens=20
        response = ask_chatglm(prompt, max_new_tokens=20)

        if shot == 0:
            pred = parse_expression_0shot(response)
        else:
            pred = parse_expression_5shot(response)

        gold = gold_expression(item)

        print(f"\n===== 第 {idx + 1} 条 =====")
        print("文本:", text)
        print("模型原始输出:", repr(response))
        print("解析后预测:", pred)
        print("真实标签:", gold)

        if pred != "Unknown" and gold:
            y_true.append(gold)
            y_pred.append(pred)

        results.append({
            "content": text,
            "gold": gold,
            "answer": response,
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

    for idx, item in enumerate(tqdm(test)):
        text = item["content"]

        if shot == 0:
            prompt = dialect_prompt_0shot(text)
            response = ask_chatglm(prompt, max_new_tokens=20)
            pred = parse_dialect_0shot(response)
        else:
            prompt = dialect_prompt_5shot(examples, text)
            response = ask_chatglm(prompt, max_new_tokens=20)
            pred = parse_dialect_5shot(response)

        gold = gold_dialect(item)

        print(f"\n===== 第 {idx + 1} 条 =====")
        print("文本:", text)
        print("模型原始输出:", repr(response))
        print("解析后预测:", pred)
        print("真实标签:", gold)

        if pred != "Unknown" and gold:
            y_true.append(gold)
            y_pred.append(pred)

        results.append({
            "content": text,
            "gold": gold,
            "answer": response,
            "pred": pred
        })

    if shot == 0:
        print_class_distribution(y_true, y_pred)
        report_dialect_0shot(y_true, y_pred)
    else:
        print_metrics(
            y_true, y_pred,
            title="\n==========Task D Dialect Group (5-shot)=========="
        )

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
            response = ask_chatglm(prompt, max_new_tokens=20)
        else:
            prompt = dialect_binary_prompt_5shot(examples, text)
            response = ask_chatglm(prompt, max_new_tokens=20)

        pred = parse_binary(response)
        gold = gold_dialect_binary(item)

        print(f"\n===== 第 {idx + 1} 条 =====")
        print("文本:", text)
        print("模型原始输出:", repr(response))
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
            "answer": response,
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
    parser = argparse.ArgumentParser(description="ChatGLM3-6B 冒犯语言评测")
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
