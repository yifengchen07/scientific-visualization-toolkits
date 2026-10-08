"""loaders —— 读实验结果文件的小工具"""
import glob
from pathlib import Path

import pandas as pd

__all__ = ["read_jsonl"]


def read_jsonl(path, str_cols=("sample_id",), add_source=True, **kwargs):
    """
    读 .jsonl（每行一个 JSON）成 DataFrame。path 可以是：
        单个文件   "results/Qwen_full.jsonl"
        通配符     "results/*_full.jsonl"        —— 多个模型的结果一次读入并拼接
        文件夹     "results"                      —— 读其中所有 .jsonl

    str_cols   : 强制读成字符串的列（避免 sample_id 之类的编号被当成数字、丢掉前导 0）
    add_source : 增加 _source 列，记录每行来自哪个文件
    其余参数传给 pd.read_json。
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    elif any(ch in str(path) for ch in "*?["):
        files = sorted(Path(f) for f in glob.glob(str(path)))
    else:
        files = [p]
    if not files:
        raise FileNotFoundError(f"没找到 jsonl 文件: {path}")
    frames = []
    for f in files:
        d = pd.read_json(f, lines=True, dtype={c: str for c in str_cols}, **kwargs)
        if add_source:
            d["_source"] = f.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
