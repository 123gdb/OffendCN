"""四元组抽取：用任意 Ollama 模型批量预测。

用法：python generate_ollama.py --model llama3:8b
"""

import argparse
import json
import os
import re
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SUBSETS_DIR

# ============================================================
# 1. Config 配置
# ============================================================
DEFAULT_MODEL = "llama3:8b"   # 可通过 --model 覆盖
MODEL_NAME = DEFAULT_MODEL    # 运行时由 main() 赋值

OLLAMA_HOST = "http://localhost:11434"

# 测试模式
TEST_MODE = False        # True: 只测试少量数据; False: 全量运行
TEST_NUM = 10            # 测试模式下运行的样本数

# 重跑失败样本模式
RERUN_FAILED = False     # True: 只重跑之前预测失败的样本; False: 正常模式

# 数据路径：Target / Argument / Offensive_Span 的评测输入
DATA_PATH = str(SUBSETS_DIR / "targeted_offensive_data.json")

# 输出文件名从模型名自动生成，替换 :/\ 为 -
def output_path_for(model_name: str) -> str:
    safe_name = model_name.replace(":", "-").replace("/", "-").replace("\\", "-")
    return f"{safe_name}_quadruple_predictions.json"


# 每 N 条保存一次中间结果（防止中断丢失）
SAVE_INTERVAL = 50

# 请求间隔（秒），避免压垮 Ollama
REQUEST_INTERVAL = 0.0

# 单条请求超时（秒）
REQUEST_TIMEOUT = 120

# 统一采样参数（与 prompting 实验一致）
NUM_CTX = 8192
MAX_NEW_TOKENS = 512


# ============================================================
# 2. Prompt 模板
# ============================================================
PROMPT_TEMPLATE = """# 任务说明

你是一名中文网络攻击性语言语义分析专家。

你的任务是根据输入文本和人工提供的 Target Category，
恢复文本中的 Target-aware 攻击语义结构四元组：

Target Category - Target - Argument - Offensive Span


本任务不是简单判断文本是否攻击，
而是分析攻击行为中的：

1. 攻击对象是谁；
2. 哪些文本片段体现攻击；
3. 攻击者针对该对象表达了什么负面观点或攻击逻辑。


# 输入信息

输入包括：

Text：
待分析的中文文本。

Target Category：
人工标注的攻击对象类别。

注意：

Target Category 是人工提供的信息，
代表攻击对象所属类别，
不要修改该类别。


# 字段定义


## 1. Target

表示文本中实际受到攻击、贬低、侮辱或负面评价的对象。

要求：

- 可以是个人、群体、地域、身份类别等。
- 如果多个对象共同受到攻击，可以使用顿号分隔。
- 不要提取仅被提及但未受到攻击的对象。

示例：

文本：
河南人都是骗子。

Target：
河南人


## 2. Argument

表示攻击者针对目标对象表达的负面观点、攻击原因或攻击逻辑。

注意：

Argument 是对攻击语义的解释，
不是文本中的攻击词。

需要说明：

- 攻击者利用了什么身份特征；
- 表达了什么负面评价；
- 使用了什么刻板印象、贬低逻辑或攻击意图。


重要：

Argument 与 Offensive Span 不同：

Offensive Span：
必须来自原始文本。

Argument：
使用概括性的自然语言描述，
不要求出现在原文中。

禁止：

- 直接复制 Offensive Span；
- 只输出辱骂词；
- 将攻击片段作为解释。


错误示例：

文本：
河南人都是骗子。

错误 Argument：
都是骗子

正确 Argument：
基于地域身份对河南群体进行负面刻板化描述，将个体行为泛化到整个群体。


## 3. Offensive Span

表示原始文本中能够支撑攻击判断的完整语义片段。

要求：

- 必须严格来自原始文本；
- 不允许改写或总结；
- 不选择过短导致语义缺失的片段；
- 不无条件复制整句话；
- 选择能够独立体现攻击性的最小充分语义范围。


示例：

文本：
你们这些外地人都是没素质的垃圾。

Offensive Span：
外地人都是没素质的垃圾


文本：
你这个哈批，什么都不懂。

Offensive Span：
哈批


文本：
上海排外，是第一天么？

Offensive Span：
上海排外


# 方言和网络表达处理

中文网络攻击表达可能包含：

- 方言词；
- 地方俚语；
- 网络黑话；
- 谐音变体。


分析时：

- 不仅依据普通话字面含义；
- 根据上下文判断其攻击含义；
- 如果方言词具有明确侮辱意义，应作为 Offensive Span；
- 如果只是普通方言表达，不应强行判断为攻击。


# 输出要求

只输出 JSON。

不要输出解释、分析过程或 Markdown。

输出格式：

{
  "Target": "",
  "Argument": "",
  "Offensive_Span": ""
}


# 重要提示

当前文本已经过人工确认，属于攻击性文本。

你必须根据文本内容，输出你对攻击语义的分析结果。

即使表达方式隐晦、方言化或网络化，也必须给出你的判断。

严禁输出空 JSON 或空字段。每个字段都必须填写你的分析结果。


# 当前任务

Text:
{{CONTENT}}


Target Category:
{{TARGET_CATEGORY}}


请输出攻击语义四元组："""


# ============================================================
# 3. 模型调用
# ============================================================
def build_prompt(content: str, target_category: str) -> str:
    """将 content 和 target_category 填入 Prompt 模板。"""
    return PROMPT_TEMPLATE.replace("{{CONTENT}}", content).replace("{{TARGET_CATEGORY}}", target_category)


def call_ollama(prompt: str) -> str:
    """调用本地 Ollama chat API，返回模型回复文本。

    options 对所有模型统一：temperature=0、num_ctx=8192、num_predict=512。
    enable_thinking 仅对 Qwen3 系列开启关闭，避免思考模式拖慢生成。
    """
    url = f"{OLLAMA_HOST}/api/chat"
    options = {
        "temperature": 0.0,
        "num_ctx": NUM_CTX,
        "num_predict": MAX_NEW_TOKENS,
    }
    if "qwen3" in MODEL_NAME.lower():
        options["enable_thinking"] = False

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": options,
    }
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def extract_json(text: str) -> dict | None:
    """从模型回复文本中提取 JSON 对象。"""
    # DeepSeek-R1 等推理模型会先输出思考过程，先剔除
    text = re.sub(r" thinking>.*? response", "", text, flags=re.DOTALL).strip()

    # 尝试从 markdown 代码块中提取
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        candidate = code_block_match.group(1).strip()
    else:
        # 尝试匹配第一个 JSON 对象
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
        else:
            candidate = text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = candidate.replace("'", '"')
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


def predict_one(content: str, target_category: str) -> dict:
    """
    对单条样本进行预测，返回 {"Target": ..., "Argument": ..., "Offensive_Span": ...}。
    任何异常都会返回空 dict，不会抛出。
    """
    prompt = build_prompt(content, target_category)

    try:
        raw_response = call_ollama(prompt)
    except requests.exceptions.ConnectionError:
        print(f"    [ERROR] Ollama 连接失败")
        return {}
    except requests.exceptions.Timeout:
        print(f"    [ERROR] 请求超时 ({REQUEST_TIMEOUT}s)")
        return {}
    except Exception as e:
        print(f"    [ERROR] 调用异常: {e}")
        return {}

    result = extract_json(raw_response)
    if result is None:
        print(f"    [WARN] JSON 解析失败，原始回复: {raw_response[:200]}")
        return {}

    return result


# ============================================================
# 4. 主流程
# ============================================================

def is_prediction_empty(pred: dict) -> bool:
    """判断预测结果是否为空（全部字段空白）。"""
    return not (pred.get("Target", "").strip() or pred.get("Argument", "").strip() or pred.get("Offensive_Span", "").strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="OffendCN 四元组抽取：用任意 Ollama 模型批量预测"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama 模型名（默认 {DEFAULT_MODEL}），如 llama3:8b / chatglm3:6b / qwen3:14b-Q4_K_M",
    )
    return parser.parse_args()


def main():
    global MODEL_NAME

    args = parse_args()
    MODEL_NAME = args.model
    output_path = output_path_for(MODEL_NAME)

    print("=" * 60)
    print("OffendCN 四元组理解实验 — 批量测试")
    print(f"模型: {MODEL_NAME}")
    print(f"测试模式: {TEST_MODE}  (TEST_NUM={TEST_NUM})")
    print(f"重跑失败: {RERUN_FAILED}")
    print(f"数据: {DATA_PATH}")
    print(f"输出: {output_path}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ============================================================
    # 模式：重跑失败样本
    # ============================================================
    if RERUN_FAILED:
        if not os.path.exists(output_path):
            print("错误: 找不到预测结果文件，无法重跑失败样本")
            return

        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 找出失败的条目
        failed_indices = [i for i, r in enumerate(results) if is_prediction_empty(r["prediction"])]
        print(f"\n已有结果: {len(results)} 条")
        print(f"失败样本: {len(failed_indices)} 条")
        if not failed_indices:
            print("没有失败样本，无需重跑。")
            return

        print(f"\n开始重跑 {len(failed_indices)} 条失败样本...")
        start_time = time.time()
        rerun_success = 0
        rerun_fail = 0

        for idx, fi in enumerate(failed_indices):
            record = results[fi]
            content = record["content"]
            target_category = record["target_category"]
            print(f"\n  [{idx+1}/{len(failed_indices)}] content: {content[:60]}...")

            prediction = predict_one(content, target_category)
            if prediction and not is_prediction_empty(prediction):
                results[fi]["prediction"] = {
                    "Target": prediction.get("Target", ""),
                    "Argument": prediction.get("Argument", ""),
                    "Offensive_Span": prediction.get("Offensive_Span", ""),
                }
                rerun_success += 1
            else:
                rerun_fail += 1

            # 定期保存
            if (idx + 1) % SAVE_INTERVAL == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                elapsed = time.time() - start_time
                print(f"  [CHECKPOINT] 已重跑 {idx+1} 条 | 成功 {rerun_success} / 仍失败 {rerun_fail} | 耗时 {elapsed:.1f}s")

        # 最终保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("重跑失败样本完成!")
        print(f"  重跑总数:   {len(failed_indices)}")
        print(f"  修复成功:   {rerun_success}")
        print(f"  仍然失败:   {rerun_fail}")
        print(f"  总耗时:     {total_time:.1f}s")
        print(f"  结果文件:   {output_path}")
        print("=" * 60)
        return

    # ============================================================
    # 模式：正常批量预测
    # ============================================================

    # ---- 读取数据 ----
    print("\n[1/4] 读取数据...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    total_all = len(all_data)
    print(f"  数据集总样本数: {total_all}")

    # 过滤：只保留 four_tuple 为对象（含 Target_Category）的针对型冒犯样本
    data = [item for item in all_data if isinstance(item.get("four_tuple"), dict)]
    offensive_count = len(data)
    non_offensive_count = total_all - offensive_count
    print(f"  含四元组的针对型冒犯样本: {offensive_count}")
    print(f"  其余样本（four_tuple 为占位字符串，已跳过）: {non_offensive_count}")

    if TEST_MODE:
        data = data[:TEST_NUM]
        print(f"  TEST_MODE: 只取前 {len(data)} 条")
    else:
        print(f"  全量运行: {len(data)} 条")

    # ---- 初始化 / 断点续跑 ----
    results = []
    success_count = 0
    fail_count = 0
    start_idx = 0

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        success_count = sum(1 for r in results if r["prediction"].get("Target") or r["prediction"].get("Argument") or r["prediction"].get("Offensive_Span"))
        fail_count = start_idx - success_count
        print(f"\n  断点续跑: 已有 {start_idx} 条结果，从第 {start_idx + 1} 条继续")
    else:
        print(f"\n  从头开始")

    # ---- 批量预测 ----
    print(f"\n[2/4] 开始批量预测...")
    start_time = time.time()

    for i in range(start_idx, len(data)):
        item = data[i]
        content = item.get("content", "")
        target_category = item.get("four_tuple", {}).get("Target_Category", "")
        gold = item.get("four_tuple", {})
        gold_tuple = {
            "Target": gold.get("Target", ""),
            "Argument": gold.get("Argument", ""),
            "Offensive_Span": gold.get("Offensive_Span", ""),
        }

        print(f"\n  [{i+1}/{len(data)}] content: {content[:60]}...")

        prediction = predict_one(content, target_category)

        if prediction:
            success_count += 1
        else:
            fail_count += 1

        record = {
            "model_name": MODEL_NAME,
            "content": content,
            "target_category": target_category,
            "gold": gold_tuple,
            "prediction": {
                "Target": prediction.get("Target", ""),
                "Argument": prediction.get("Argument", ""),
                "Offensive_Span": prediction.get("Offensive_Span", ""),
            },
        }
        results.append(record)

        # 定期保存中间结果
        if (i + 1) % SAVE_INTERVAL == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            elapsed = time.time() - start_time
            print(f"  [CHECKPOINT] 已保存 {len(results)} 条 | 成功 {success_count} / 失败 {fail_count} | 耗时 {elapsed:.1f}s")

        # 请求间隔
        if REQUEST_INTERVAL > 0:
            time.sleep(REQUEST_INTERVAL)

    # ---- 最终保存 ----
    print(f"\n[3/4] 保存最终结果到 {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 统计 ----
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("[4/4] 完成!")
    print(f"  总样本数:   {len(data)}")
    print(f"  成功:       {success_count}")
    print(f"  失败:       {fail_count}")
    print(f"  总耗时:     {total_time:.1f}s")
    print(f"  平均耗时:   {total_time/len(data):.1f}s/条")
    print(f"  结果文件:   {output_path}")
    print(f"  结束时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()