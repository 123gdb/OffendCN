"""四元组抽取评价（BERTScore / Soft-F1）。

用法：python evaluate.py --input <predictions.json>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MODELS_DIR


# ============================================================
# 1. Config 配置
# ============================================================
DEFAULT_INPUT = "llama3-8b_quadruple_predictions.json"

# 输出文件名由模型名自动生成，无需手动修改
# 例如 qwen3:14b 预测 → qwen3-14b_evaluation_results.json
OUTPUT_SUMMARY_TEMPLATE = "{model_name}_evaluation_results.json"
OUTPUT_PER_SAMPLE_TEMPLATE = "{model_name}_per_sample_scores.json"

# 本地中文 RoBERTa 模型路径
BERT_MODEL_PATH = str(MODELS_DIR / "chinese-roberta-wwm-ext-base")

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. BERTScore 计算模块
# ============================================================

def load_bertscore_model(model_path: str, device: str):
    """手动加载 BERTScore 所需的 tokenizer 和 model

    绕过 AutoModel.from_pretrained 在 CPU 环境可能崩溃的问题，
    使用手动构建模型 + 权重映射的方式。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BERT 模型路径不存在: {model_path}")

    print(f"  加载 tokenizer: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)

    print(f"  加载 config & 构建模型...")
    config = BertConfig.from_pretrained(model_path)
    model = BertModel(config)

    print(f"  加载权重文件...")
    state_dict = torch.load(
        os.path.join(model_path, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    # 去除 bert. 前缀（chinese-roberta-wwm-ext 权重键名带 bert. 前缀）
    state_dict = {k[5:] if k.startswith("bert.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"  BERT 模型已加载: {model_path}")
    print(f"  设备: {device}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return tokenizer, model


def compute_single_bertscore(ref: str, cand: str, tokenizer, model, device: str) -> float:
    """计算单对文本的 BERTScore F1"""
    if not ref.strip() or not cand.strip():
        return 0.0

    # Tokenize
    ref_inputs = tokenizer(ref, return_tensors="pt", truncation=True, max_length=512).to(device)
    cand_inputs = tokenizer(cand, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        ref_emb = model(**ref_inputs).last_hidden_state[0]  # (ref_len, hidden)
        cand_emb = model(**cand_inputs).last_hidden_state[0]  # (cand_len, hidden)

    # 归一化（余弦相似度 = 归一化后的点积）
    ref_emb = F.normalize(ref_emb, dim=-1)
    cand_emb = F.normalize(cand_emb, dim=-1)

    # 余弦相似度矩阵
    sim = torch.mm(ref_emb, cand_emb.T)  # (ref_len, cand_len)

    # Recall: 每个 ref token 找最相似的 cand token
    recall = sim.max(dim=1).values.mean().item()
    # Precision: 每个 cand token 找最相似的 ref token
    precision = sim.max(dim=0).values.mean().item()

    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return f1


# ============================================================
# 3. Span Soft-F1 计算模块
# ============================================================

def compute_span_soft_f1(gold_span: str, pred_span: str) -> float:
    """字符级 Span Soft-F1

    Args:
        gold_span: 人工标注的攻击性文本片段
        pred_span: 模型预测的攻击性文本片段

    Returns:
        Soft-F1 分数 (0.0 ~ 1.0)
    """
    if not pred_span.strip():
        return 0.0

    if not gold_span.strip():
        return None  # gold 为空，该字段不参与计算

    g_chars = set(gold_span)
    p_chars = set(pred_span)

    if len(p_chars) == 0:
        return 0.0

    intersection = g_chars & p_chars

    precision = len(intersection) / len(p_chars)
    recall = len(intersection) / len(g_chars)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_span_length_ratio(gold_span: str, pred_span: str) -> float:
    """Span 长度比例

    Length_Ratio = len(Gold) / len(Prediction)
    如果 prediction 比 gold 长，自动产生惩罚（ratio < 1）。

    Args:
        gold_span: 人工标注的攻击性文本片段
        pred_span: 模型预测的攻击性文本片段

    Returns:
        Length_Ratio (0.0 ~ 1.0)，prediction 为空时返回 0.0，gold 为空时返回 None
    """
    if not pred_span.strip():
        return 0.0

    if not gold_span.strip():
        return None

    gold_len = len(gold_span)
    pred_len = len(pred_span)

    if pred_len == 0:
        return 0.0

    ratio = gold_len / pred_len
    # 上限 1.0：只惩罚预测过长，不奖励预测过短
    return min(ratio, 1.0)


def compute_span_adjusted_f1(gold_span: str, pred_span: str) -> float:
    """带长度惩罚的 Span Adjusted F1

    Span_Adjusted_F1 = Span_Soft_F1 × Length_Ratio

    Args:
        gold_span: 人工标注的攻击性文本片段
        pred_span: 模型预测的攻击性文本片段

    Returns:
        Adjusted F1 分数 (0.0 ~ 1.0)，prediction 为空时返回 0.0，gold 为空时返回 None
    """
    if not pred_span.strip():
        return 0.0

    if not gold_span.strip():
        return None

    soft_f1 = compute_span_soft_f1(gold_span, pred_span)
    if soft_f1 is None:
        return None

    length_ratio = compute_span_length_ratio(gold_span, pred_span)
    if length_ratio is None:
        return None

    return soft_f1 * length_ratio


# ============================================================
# 4. 单条样本评分
# ============================================================

def score_sample(gold: dict, pred: dict, tokenizer, model, device: str) -> dict:
    """对单条样本计算所有指标

    Returns:
        {
            "Target_BERTScore": float | None,
            "Argument_BERTScore": float | None,
            "Span_Soft_F1": float | None,
            "Span_Length_Ratio": float | None,
            "Span_Adjusted_F1": float | None,
            "Quadruple_Score": float | None,
        }
        None 表示 gold 为空，该字段不参与统计
    """
    scores = {}

    # --- Target BERTScore ---
    gold_target = gold.get("Target", "").strip()
    pred_target = pred.get("Target", "").strip()

    if gold_target:
        if pred_target:
            scores["Target_BERTScore"] = compute_single_bertscore(
                gold_target, pred_target, tokenizer, model, device
            )
        else:
            scores["Target_BERTScore"] = 0.0  # prediction 为空
    else:
        scores["Target_BERTScore"] = None  # gold 为空，跳过

    # --- Argument BERTScore ---
    gold_argument = gold.get("Argument", "").strip()
    pred_argument = pred.get("Argument", "").strip()

    if gold_argument:
        if pred_argument:
            scores["Argument_BERTScore"] = compute_single_bertscore(
                gold_argument, pred_argument, tokenizer, model, device
            )
        else:
            scores["Argument_BERTScore"] = 0.0
    else:
        scores["Argument_BERTScore"] = None

    # --- Span Soft-F1 ---
    gold_span = gold.get("Offensive_Span", "").strip()
    pred_span = pred.get("Offensive_Span", "").strip()

    if gold_span:
        scores["Span_Soft_F1"] = compute_span_soft_f1(gold_span, pred_span)
    else:
        scores["Span_Soft_F1"] = None

    # --- Span Length Ratio ---
    if gold_span:
        scores["Span_Length_Ratio"] = compute_span_length_ratio(gold_span, pred_span)
    else:
        scores["Span_Length_Ratio"] = None

    # --- Span Adjusted F1 ---
    if gold_span:
        scores["Span_Adjusted_F1"] = compute_span_adjusted_f1(gold_span, pred_span)
    else:
        scores["Span_Adjusted_F1"] = None

    # --- Quadruple Score（使用 Span_Adjusted_F1） ---
    valid = [v for v in [scores["Target_BERTScore"], scores["Argument_BERTScore"], scores["Span_Adjusted_F1"]] if v is not None]
    if valid:
        scores["Quadruple_Score"] = sum(valid) / len(valid)
    else:
        scores["Quadruple_Score"] = None

    return scores


# ============================================================
# 5. 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="OffendCN 四元组抽取结果评价（BERTScore / Soft-F1）"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"预测结果文件（默认 {DEFAULT_INPUT}），与 generate_ollama.py 的输出对应",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input

    print("=" * 60)
    print("OffendCN 四元组生成实验 — 评价模块")
    print(f"输入: {input_path}")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # ---- 加载数据 ----
    print("\n[1/5] 加载预测结果...")
    with open(input_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    total = len(predictions)
    print(f"  样本总数: {total}")

    model_name = predictions[0].get("model_name", "unknown") if predictions else "unknown"
    print(f"  模型名称: {model_name}")

    # 根据模型名生成输出文件名（替换特殊字符防止路径问题）
    safe_name = model_name.replace(":", "-").replace("/", "-").replace("\\", "-")
    output_summary = OUTPUT_SUMMARY_TEMPLATE.format(model_name=safe_name)
    output_per_sample = OUTPUT_PER_SAMPLE_TEMPLATE.format(model_name=safe_name)
    print(f"  输出汇总: {output_summary}")
    print(f"  逐条分数: {output_per_sample}")

    # ---- 加载 BERTScore 模型 ----
    print("\n[2/5] 加载 BERTScore 模型...")
    tokenizer, model = load_bertscore_model(BERT_MODEL_PATH, DEVICE)

    # ---- 逐条评分 ----
    print(f"\n[3/5] 计算评价指标 ({total} 条)...")
    per_sample_scores = []
    generation_success = 0

    # 收集各指标有效值，用于计算平均分
    target_scores = []
    argument_scores = []
    span_scores = []
    span_length_ratios = []
    span_adjusted_scores = []
    quadruple_scores = []

    error_indices = []

    for i, item in enumerate(tqdm(predictions, desc="  评分进度", unit="条")):
        gold = item.get("gold", {})
        pred = item.get("prediction", {})

        try:
            scores = score_sample(gold, pred, tokenizer, model, DEVICE)
        except Exception as e:
            print(f"\n  [ERROR] 样本 {i+1} 评分失败: {e}")
            error_indices.append(i + 1)
            # 失败样本：全部设为 0 或 None
            scores = {
                "Target_BERTScore": 0.0 if gold.get("Target", "").strip() else None,
                "Argument_BERTScore": 0.0 if gold.get("Argument", "").strip() else None,
                "Span_Soft_F1": 0.0 if gold.get("Offensive_Span", "").strip() else None,
                "Span_Length_Ratio": 0.0 if gold.get("Offensive_Span", "").strip() else None,
                "Span_Adjusted_F1": 0.0 if gold.get("Offensive_Span", "").strip() else None,
                "Quadruple_Score": None,
            }

        # 记录 per-sample
        per_sample_scores.append({
            "content": item.get("content", ""),
            "Target_BERTScore": scores["Target_BERTScore"],
            "Argument_BERTScore": scores["Argument_BERTScore"],
            "Span_Soft_F1": scores["Span_Soft_F1"],
            "Span_Length_Ratio": scores["Span_Length_Ratio"],
            "Span_Adjusted_F1": scores["Span_Adjusted_F1"],
            "Quadruple_Score": scores["Quadruple_Score"],
        })

        # 收集有效分数用于平均
        if scores["Target_BERTScore"] is not None:
            target_scores.append(scores["Target_BERTScore"])
        if scores["Argument_BERTScore"] is not None:
            argument_scores.append(scores["Argument_BERTScore"])
        if scores["Span_Soft_F1"] is not None:
            span_scores.append(scores["Span_Soft_F1"])
        if scores["Span_Length_Ratio"] is not None:
            span_length_ratios.append(scores["Span_Length_Ratio"])
        if scores["Span_Adjusted_F1"] is not None:
            span_adjusted_scores.append(scores["Span_Adjusted_F1"])
        if scores["Quadruple_Score"] is not None:
            quadruple_scores.append(scores["Quadruple_Score"])

        # 统计生成成功率：prediction 三个字段至少一个非空
        pred_target = pred.get("Target", "").strip()
        pred_argument = pred.get("Argument", "").strip()
        pred_span = pred.get("Offensive_Span", "").strip()
        if pred_target or pred_argument or pred_span:
            generation_success += 1

    if error_indices:
        print(f"\n  评分异常样本编号: {error_indices}")

    # ---- 计算平均指标 ----
    print("\n[4/5] 汇总统计...")

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 6) if lst else 0.0

    average = {
        "Target_BERTScore": safe_mean(target_scores),
        "Argument_BERTScore": safe_mean(argument_scores),
        "Span_Soft_F1": safe_mean(span_scores),
        "Span_Length_Ratio": safe_mean(span_length_ratios),
        "Span_Adjusted_F1": safe_mean(span_adjusted_scores),
        "Quadruple_Score": safe_mean(quadruple_scores),
    }

    generation_rate = round(generation_success / total, 6) if total > 0 else 0.0

    # ---- 输出汇总 ----
    summary = {
        "model_name": model_name,
        "sample_num": total,
        "generation_rate": generation_rate,
        "average": average,
    }

    print(f"\n  {'='*40}")
    print(f"  模型:                {model_name}")
    print(f"  样本总数:            {total}")
    print(f"  生成成功率:          {generation_rate:.4f} ({generation_success}/{total})")
    print(f"  Target_BERTScore:    {average['Target_BERTScore']:.4f}")
    print(f"  Argument_BERTScore:  {average['Argument_BERTScore']:.4f}")
    print(f"  Span_Soft_F1:        {average['Span_Soft_F1']:.4f}")
    print(f"  Span_Length_Ratio:   {average['Span_Length_Ratio']:.4f}")
    print(f"  Span_Adjusted_F1:    {average['Span_Adjusted_F1']:.4f}")
    print(f"  Quadruple_Score:     {average['Quadruple_Score']:.4f}")
    print(f"  {'='*40}")

    # ---- 保存结果 ----
    print(f"\n[5/5] 保存结果...")

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  汇总 -> {output_summary}")

    with open(output_per_sample, "w", encoding="utf-8") as f:
        json.dump(per_sample_scores, f, ensure_ascii=False, indent=2)
    print(f"  逐条 -> {output_per_sample}")

    print("\n" + "=" * 60)
    print("评价完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()