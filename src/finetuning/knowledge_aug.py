# -*- coding: utf-8 -*-
"""知识增强模块：target / dialect / pinyin 三路知识向量与 CLS 拼接后共用分类头。"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import VOCAB_DIR as VOCAB_DIR_PATH

# ======================== 词表路径 ========================

VOCAB_DIR = str(VOCAB_DIR_PATH)
FINAL_VOCAB_PATH = os.path.join(VOCAB_DIR, "final_vocab.json")
FILTERED_VOCAB_PATH = os.path.join(VOCAB_DIR, "filtered_vocab.json")

TYPE_TO_TARGET = {"Gender": 1, "Region": 2, "Race": 3, "LGBTQ": 4, "Others": 0}
DIALECT_TO_ID = {
    "mandarin": 0, "北方官话": 1, "粤语": 2, "西南官话": 3,
    "吴语": 4, "闽语": 5, "其他": 6,
}

NUM_TARGET_IDS = 5      # 0 = 未命中
NUM_DIALECT_IDS = 7     # 0 = 未命中

_cache = {}


# ======================== 词表 → word→id 映射 ========================

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_vocab_maps():
    """构建 target / dialect / pinyin 三套 word→id 映射（带缓存）

    pinyin 的 id 从 1 开始（0 留给「未命中」），因此 embedding 大小是 num_pinyin + 1。
    """
    if _cache:
        return _cache

    final_vocab = _read_json(FINAL_VOCAB_PATH)
    filtered_vocab = _read_json(FILTERED_VOCAB_PATH)

    w2t = {}
    for w, info in filtered_vocab.items():
        tid = TYPE_TO_TARGET.get(info.get("type"), 0)
        if tid != 0:          # Others 不参与 target
            w2t[w] = tid

    w2d = {}
    for w, info in final_vocab.items():
        did = DIALECT_TO_ID.get(info.get("dialect"), 0)
        if did != 0:          # 普通话不参与 dialect
            w2d[w] = did

    pinyins = sorted({info["pinyin"] for info in final_vocab.values() if info.get("pinyin")})
    p2i = {p: i + 1 for i, p in enumerate(pinyins)}
    w2p = {w: p2i[info["pinyin"]] for w, info in final_vocab.items() if info.get("pinyin")}

    _cache.update(
        w2t=w2t, w2d=w2d, w2p=w2p,
        num_pinyin=len(pinyins) + 1,
    )
    return _cache


# ======================== char span 匹配 ========================

def _occurrences(text, word):
    res, start = [], 0
    while True:
        i = text.find(word, start)
        if i == -1:
            break
        res.append((i, i + len(word)))
        start = i + 1
    return res


def _token_span(offsets, cs, ce):
    """word 的字符区间 → token 区间；token 边界必须与 word 边界完全对齐"""
    ts = te = None
    for i, (a, b) in enumerate(offsets):
        if a == b:            # special token
            continue
        if a < ce and b > cs:
            if ts is None:
                ts = i
            te = i + 1
    if ts is None:
        return None
    if offsets[ts][0] == cs and offsets[te - 1][1] == ce:
        return ts, te
    return None


def _token_ids_for(text, offsets, word_to_id, length):
    ids = [0] * length
    for word, vid in word_to_id.items():
        if word not in text:
            continue
        for cs, ce in _occurrences(text, word):
            span = _token_span(offsets, cs, ce)
            if span is None:
                continue
            for j in range(span[0], span[1]):
                if ids[j] == 0:
                    ids[j] = vid
    return ids


def add_knowledge_features(batch, texts, offsets, tokenizer, need, max_len):
    """给 tokenizer 输出补齐 target_ids / dial_ids / pinyin_ids

    batch: tokenizer(...) 的返回 dict（会被原地修改）
    need:  {"target", "dialect", "pinyin"} 的子集
    """
    maps = get_vocab_maps()
    n = len(texts)

    if "target" in need:
        batch["target_ids"] = [
            _token_ids_for(texts[i], offsets[i], maps["w2t"], max_len) for i in range(n)
        ]
    if "dialect" in need:
        batch["dial_ids"] = [
            _token_ids_for(texts[i], offsets[i], maps["w2d"], max_len) for i in range(n)
        ]
    if "pinyin" in need:
        batch["pinyin_ids"] = [
            _token_ids_for(texts[i], offsets[i], maps["w2p"], max_len) for i in range(n)
        ]
    return batch


def feature_dims(need):
    """每种知识特征的 embedding 词表大小"""
    maps = get_vocab_maps()
    dims = {}
    if "target" in need:
        dims["target_ids"] = NUM_TARGET_IDS
    if "dialect" in need:
        dims["dial_ids"] = NUM_DIALECT_IDS
    if "pinyin" in need:
        dims["pinyin_ids"] = maps["num_pinyin"]
    return dims


# ======================== 模型 ========================

class KGClassifier(nn.Module):
    """知识增强分类器：知识特征与文本 CLS 拼接后共用一个分类头

    logits = Linear( cat([LayerNorm(CLS), LayerNorm(知识特征)]) )
    CLS 用 last_hidden_state[:,0,:]，与知识特征拼接后共用一个分类头
    """

    def __init__(self, base_model, num_labels, dims, hidden=256):
        super().__init__()
        self.base = base_model
        self.num_labels = num_labels
        h = base_model.config.hidden_size
        emb_dim = h

        self.embeds = nn.ModuleDict({
            name: nn.Embedding(size, emb_dim) for name, size in dims.items()
        })

        # 每路特征各自 LayerNorm（768 维），与 LayerNorm(CLS) 拼接后共用一个 Linear
        self.feat_norms = nn.ModuleDict({
            name: nn.LayerNorm(emb_dim) for name in dims
        })
        self.cls_norm = nn.LayerNorm(h)
        self.cls_drop = nn.Dropout(base_model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(h + emb_dim * len(dims), num_labels)

    def _pool(self, name, ids):
        emb = self.embeds[name](ids.clamp(min=0))
        w = (ids > 0).float().unsqueeze(-1)
        pooled = (emb * w).sum(1) / w.sum(1).clamp(min=1.0)
        hit = (w.sum(1) > 0).float()
        return pooled, hit

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):
        base_out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        pooled = []
        for name in self.embeds:
            ids = kwargs.get(name)
            if ids is None:
                ids = torch.zeros_like(input_ids)
            p, _ = self._pool(name, ids)
            p = self.feat_norms[name](p)   # 每路各自 LayerNorm
            pooled.append(p)

        kg_feat = torch.cat(pooled, dim=-1)
        cls = base_out.hidden_states[-1][:, 0, :]
        text = self.cls_drop(self.cls_norm(cls))
        logits = self.classifier(torch.cat([text, kg_feat], dim=-1))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
