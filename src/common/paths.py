# -*- coding: utf-8 -*-
"""项目路径常量。ROOT 由本文件位置推算。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # → OffendCN/

DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"     # 预训练底座放这里，由用户在 README 里自行下载
RESULTS_DIR = ROOT / "results"
VOCAB_DIR = ROOT / "data" / "vocab"
SUBSETS_DIR = DATA_DIR / "subsets"   # 未随仓库发布的子集，生成规则见 README
