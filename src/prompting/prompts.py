# -*- coding: utf-8 -*-
"""Shared prompt templates and answer parsing for LLM evaluation."""

import json
import os
import re

from common.paths import SPLITS_DIR

TASK_SPECS = {
    "offense": {
        "train_file": "train.json",
        "instruction": "判断下面这段中文网络文本是否包含冒犯性内容。",
        "definitions": [
            ("offensive", "文本包含对他人的侮辱、攻击、贬低、嘲讽，或针对某群体、身份、地域的歧视性表达。"),
            ("non-offensive", "文本不含上述冒犯性内容，包括普通陈述、观点表达和无攻击性的情绪表达。"),
        ],
        "labels": ["non-offensive", "offensive"],
    },
    "target": {
        "train_file": "target_train.json",
        "instruction": "下面文本已经确认属于针对性冒犯文本，请判断其攻击目标的类别。",
        "definitions": [
            ("Region", "攻击对象是地域、地区或来自某地的人群。"),
            ("Gender", "攻击对象是某一性别或性别相关群体。"),
            ("Race", "攻击对象是民族、种族或相关群体。"),
            ("LGBTQ", "攻击对象是性少数群体。"),
            ("Other", "攻击对象是其他个人、群体或身份类别。"),
        ],
        "labels": ["Region", "Gender", "Race", "LGBTQ", "Other"],
    },
    "expression": {
        "train_file": "target_train.json",
        "instruction": "下面文本已经确认属于针对性冒犯文本，请判断其表达形式。",
        "definitions": [
            ("Explicit", "显性表达：直接使用辱骂词、贬损词或明确的攻击性语句。"),
            ("Implicit", "隐性表达：通过隐喻、反讽、刻板印象、暗示等方式实施冒犯，字面不一定含脏话。"),
            ("Reported", "转述表达：复述、引用或陈述他人的冒犯性言论。"),
        ],
        "labels": ["Explicit", "Implicit", "Reported"],
    },
    "dialect": {
        "train_file": "dialect_train.json",
        "instruction": "下面文本已经确认属于方言冒犯文本，请判断它所属的方言组别。",
        "definitions": [
            ("N-Mandarin", "北方官话（含普通话及其北方变体）。"),
            ("SW-Mandarin", "西南官话（四川话、重庆话、云南话等）。"),
            ("Wu", "吴语（上海话、苏州话、温州话等）。"),
            ("Yue", "粤语（广东话、广府话等）。"),
            ("Min", "闽语（闽南语、闽东话、潮汕话等）。"),
            ("Other", "其他方言或混合方言。"),
        ],
        "labels": ["N-Mandarin", "SW-Mandarin", "Wu", "Yue", "Min", "Other"],
    },
    "dialect_binary": {
        "train_file": "dialect_bin_train.json",
        "instruction": "判断下面这段冒犯文本是否依托方言词汇或方言表达实现冒犯。",
        "definitions": [
            ("方言冒犯", "冒犯通过方言词汇、方言表达或方言谐音实现。"),
            ("非方言冒犯", "使用普通话或网络流行语表达攻击，不含方言成分。"),
        ],
        "labels": ["非方言冒犯", "方言冒犯"],
    },
}


def _load_split(filename):
    path = os.path.join(str(SPLITS_DIR), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 {path}，请先在仓库根目录运行：python src/prompting/make_splits.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label_of(task, item):
    if task == "offense":
        return "offensive" if item.get("offensive") else "non-offensive"
    if task == "target":
        return (item.get("four_tuple") or {}).get("Target_Category", "Other")
    if task == "expression":
        return item.get("expression_type") or "Explicit"
    if task == "dialect":
        return (item.get("dialect_info") or {}).get("dialect_group") or "Other"
    if task == "dialect_binary":
        return "方言冒犯" if (item.get("dialect_info") or {}).get("is_dialect_offensive") else "非方言冒犯"
    raise ValueError(task)


_EXAMPLE_CACHE = {}


def select_examples(task, k=5):
    """Return the same k training examples for every model."""
    if task in _EXAMPLE_CACHE:
        return _EXAMPLE_CACHE[task]

    data = _load_split(TASK_SPECS[task]["train_file"])
    groups = {}
    for item in data:
        groups.setdefault(_label_of(task, item), []).append(item)

    # 按类别样本数从多到少轮转取，保证小样本类别也能出现在示例中
    order = sorted(groups, key=lambda x: (-len(groups[x]), x))
    picked = []
    cursor = {label: 0 for label in order}
    while len(picked) < k:
        added = False
        for label in order:
            rows = groups[label]
            if cursor[label] < len(rows):
                picked.append(rows[cursor[label]])
                cursor[label] += 1
                added = True
                if len(picked) == k:
                    break
        if not added:
            break

    _EXAMPLE_CACHE[task] = picked
    return picked


def build_prompt(task, text, shot=0):
    spec = TASK_SPECS[task]
    labels = spec["labels"]

    prompt = "你是中文网络冒犯语言分析专家。\n\n"
    prompt += f"任务：\n{spec['instruction']}\n\n"
    prompt += "标签定义：\n"
    for name, desc in spec["definitions"]:
        prompt += f"- {name}：{desc}\n"
    prompt += "\n输出要求：\n"
    prompt += f"只输出一个标签，必须是以下之一：{'、'.join(labels)}。\n"
    prompt += "不要解释，不要输出标点、引号或其他文字。\n"

    if shot:
        prompt += f"\n下面给出 {shot} 个示例：\n"
        for i, item in enumerate(select_examples(task, k=shot)):
            prompt += f"\n示例{i + 1}\n文本：\n{item['content']}\n标签：\n{_label_of(task, item)}\n"

    prompt += f"\n待判断文本：\n{text}\n\n标签：\n"
    return prompt


_ASCII_PATTERNS = {
    "target": [
        (r"\blgbtq\b", "LGBTQ"),
        (r"\bgender\b", "Gender"),
        (r"\brace\b", "Race"),
        (r"\bregion\b", "Region"),
        (r"\bother\b", "Other"),
    ],
    "expression": [
        (r"\bexplicit\b", "Explicit"),
        (r"\bimplicit\b", "Implicit"),
        (r"\breported\b", "Reported"),
    ],
    "dialect": [
        (r"\bsw-mandarin\b", "SW-Mandarin"),
        (r"\bn-mandarin\b", "N-Mandarin"),
        (r"\byue\b", "Yue"),
        (r"\bwu\b", "Wu"),
        (r"\bmin\b", "Min"),
        (r"\bother\b", "Other"),
    ],
}

_CHINESE_PATTERNS = {
    "target": [
        ("性少数", "LGBTQ"),
        ("性别", "Gender"),
        ("种族", "Race"),
        ("地域", "Region"),
        ("其他", "Other"),
    ],
    "expression": [
        ("显性", "Explicit"),
        ("隐性", "Implicit"),
        ("转述", "Reported"),
    ],
    "dialect": [
        ("西南", "SW-Mandarin"),
        ("北方", "N-Mandarin"),
        ("粤", "Yue"),
        ("吴", "Wu"),
        ("闽", "Min"),
        ("其他", "Other"),
    ],
}


def _match_line(task, line):
    line = line.strip().lower()

    if task == "offense":
        if any(k in line for k in ["non-offensive", "non offensive", "not offensive", "非冒犯", "不冒犯", "没有冒犯"]):
            return "non-offensive"
        if "offensive" in line or "冒犯" in line:
            return "offensive"
        return None

    if task == "dialect_binary":
        if any(k in line for k in ["非方言", "不是方言", "不含方言", "non-dialect", "non dialect"]):
            return "非方言冒犯"
        if any(k in line for k in ["方言冒犯", "是方言", "含方言", "dialect"]):
            return "方言冒犯"
        return None

    for pattern, label in _ASCII_PATTERNS.get(task, []):
        if re.search(pattern, line):
            return label
    for keyword, label in _CHINESE_PATTERNS.get(task, []):
        if keyword in line:
            return label
    return None


def parse_answer(task, ans):
    """Map a raw model answer to a canonical label, or "Unknown"."""
    if not ans or not ans.strip():
        return "Unknown"

    lines = [line for line in ans.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        label = _match_line(task, line)
        if label:
            return label

    label = _match_line(task, ans)
    return label or "Unknown"


def binary_index(label):
    """offense 任务的历史返回约定：1=offensive, 0=non-offensive, -1=解析失败。"""
    return {"offensive": 1, "non-offensive": 0}.get(label, -1)

