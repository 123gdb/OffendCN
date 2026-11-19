# -*- coding: utf-8 -*-
"""生成 data/splits/（12 个划分文件）与 data/subsets/（3 个子集）。"""
import json
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA_DIR, SPLITS_DIR, SUBSETS_DIR

V2 = os.path.join(str(DATA_DIR), "OffendCN.json")
SEED = 42

FIELD_ORDER = [
    "content", "topic", "platform", "offensive",
    "four_tuple", "dialect_info", "expression_type", "offensive_base_type",
]


def reformat(item):
    """v2 样本 → 原 schema（字段顺序、字段名都与旧文件一致）"""
    out = {k: item.get(k) for k in FIELD_ORDER}
    out["offensive_base_type"] = item.get("offensive_type") or ""
    return out


def label_of(item, key):
    if key == "offensive":
        return item["offensive"]
    if key == "target":
        return (item.get("four_tuple") or {}).get("Target_Category") or "Other"
    if key == "dialect":
        return (item.get("dialect_info") or {}).get("dialect_group") or "Other"
    if key == "dialect_binary":
        return "方言冒犯" if (item.get("dialect_info") or {}).get("is_dialect_offensive") else "非方言冒犯"
    raise ValueError(key)


def make_split(items, key):
    """0.2 → 0.5 的 8:1:1 分层划分，返回 (train, dev, test)"""
    labels = [label_of(x, key) for x in items]
    idx = list(range(len(items)))
    tr, temp = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=labels)
    dv, te = train_test_split(temp, test_size=0.5, random_state=SEED,
                              stratify=[labels[i] for i in temp])
    return ([items[i] for i in tr], [items[i] for i in dv], [items[i] for i in te])


def dump(name, data):
    os.makedirs(str(SPLITS_DIR), exist_ok=True)
    with open(os.path.join(str(SPLITS_DIR), name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_subset(name, data):
    os.makedirs(str(SUBSETS_DIR), exist_ok=True)
    with open(os.path.join(str(SUBSETS_DIR), name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    with open(V2, "r", encoding="utf-8") as f:
        v2 = json.load(f)

    tasks = [
        ("冒犯识别", [reformat(x) for x in v2], "offensive", ["train.json", "dev.json", "test.json"]),
        ("目标/表达", [reformat(x) for x in v2 if x.get("offensive_type") == "Tar-off"],
         "target", ["target_train.json", "target_dev.json", "target_test.json"]),
        ("方言", [reformat(x) for x in v2 if (x.get("dialect_info") or {}).get("is_dialect_offensive")],
         "dialect", ["dialect_train.json", "dialect_dev.json", "dialect_test.json"]),
        ("方言冒犯二分类", [reformat(x) for x in v2 if x.get("offensive")],
         "dialect_binary", ["dialect_bin_train.json", "dialect_bin_dev.json", "dialect_bin_test.json"]),
    ]

    print(f"数据源：{V2}（{len(v2)} 条）")
    for task_name, items, key, names in tasks:
        tr, dv, te = make_split(items, key)
        print(f"\n=== {task_name}：{len(items)} 条 → 训练 {len(tr)} / 验证 {len(dv)} / 测试 {len(te)} ===")
        for name, data in zip(names, (tr, dv, te)):
            dump(name, data)
            print(f"  {name}: {len(data)} 条")

    # 子集：fine-tuning 与四元组抽取直接读取
    subsets = {
        "offensive_data.json": [x for x in v2 if x.get("offensive") is True],
        "targeted_offensive_data.json": [
            x for x in v2
            if isinstance(x.get("four_tuple"), dict) and x["four_tuple"].get("Target_Category")
        ],
        "dialect_offensive_data.json": [
            x for x in v2 if (x.get("dialect_info") or {}).get("is_dialect_offensive")
        ],
    }
    print("\n=== 子集 ===")
    for name, rows in subsets.items():
        dump_subset(name, rows)
        print(f"  {name}: {len(rows)} 条")

    # 校验
    print("\n" + "=" * 55)
    for task_name, items, key, names in tasks:
        got = []
        for name in names:
            with open(os.path.join(str(SPLITS_DIR), name), "r", encoding="utf-8") as f:
                got.append(json.load(f))
        total = sum(len(g) for g in got)
        merged = [x for g in got for x in g]
        src_contents = {x["content"] for x in items}
        print(f"{task_name}: {len(items)} → 划分合计 {total} | 条数一致: {total == len(items)} | "
              f"内容与源子集一致: {all(x['content'] in src_contents for x in merged)}")
    with open(os.path.join(str(SPLITS_DIR), "train.json"), "r", encoding="utf-8") as f:
        sample = json.load(f)[0]
    print("字段:", list(sample.keys())[:8])
    print("offensive=True 占比（train）:",
          round(sum(1 for x in json.load(open(os.path.join(str(SPLITS_DIR), "train.json"), encoding="utf-8"))
                    if x["offensive"]) / len(json.load(open(os.path.join(str(SPLITS_DIR), "train.json"), encoding="utf-8"))), 4))


if __name__ == "__main__":
    main()
