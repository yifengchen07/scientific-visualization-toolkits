"""各绘图函数共用的小工具（不对外导出，pl.save 除外）"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

from .utils import pubstyle as ps


def get_ax(ax, figsize):
    """ax 为 None 时新建画布，否则直接画在传入的 ax 上。"""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


_MATH = re.compile(r"\$[^$]*\$")


def _is_mixed(text):
    """是否同时含 $公式$ 和非 ASCII（中文）字符：这种标签需要拆段渲染。"""
    return "$" in text and any(ord(ch) > 127 for ch in _MATH.sub("", text))


def set_labels(ax, title=None, xlabel=None, ylabel=None):
    """
    统一设置标题和坐标轴标签。None = 不设置。
    含 “中文 + $公式$” 的标签会自动改用 ps.mixed_label 拆段渲染（否则中文会变方块）。
    """
    items = ((title, "title", ax.set_title), (xlabel, "x", ax.set_xlabel), (ylabel, "y", ax.set_ylabel))
    offsets = {"title": 0.03, "x": 0.13, "y": 0.14}
    for text, where, setter in items:
        if text is None:
            continue
        if _is_mixed(text):
            ps.mixed_label(ax, text, where=where, offset=offsets[where])
        else:
            setter(text)


def darken(color, factor=0.15):
    r, g, b = to_rgb(color)
    return to_hex((r * (1 - factor), g * (1 - factor), b * (1 - factor)))


def groups_from(data, labels=None):
    """
    把“分组数据”统一成 (名称列表, 数组列表)。data 可以是：
      dict {名称: 数组} / DataFrame（每列一组）/ 数组的列表
    """
    if isinstance(data, pd.DataFrame):
        names = [str(c) for c in data.columns]
        arrs = [data[c].dropna().to_numpy(float) for c in data.columns]
    elif isinstance(data, dict):
        names = [str(k) for k in data.keys()]
        arrs = [np.asarray(v, float) for v in data.values()]
    else:
        arrs = [np.asarray(v, float) for v in data]
        names = [f"Group {i + 1}" for i in range(len(arrs))]
    if labels is not None:
        if len(labels) != len(arrs):
            raise ValueError(f"labels 有 {len(labels)} 个，但数据有 {len(arrs)} 组")
        names = list(labels)
    return names, arrs


def save(obj, path, dpi=300, formats=("png", "pdf"), **kwargs):
    """
    保存图。obj 可以是 Figure / Axes / SubFigure / 返回的 (ax1, ax2)。
    path 带扩展名（如 "fig1.pdf"）则只存这一种；不带扩展名则同时存 formats 里的各种格式。
    返回保存的文件路径列表。
    """
    if isinstance(obj, (tuple, list)):
        obj = obj[0]
    fig = obj if hasattr(obj, "savefig") else obj.figure
    p = Path(path)
    if p.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".tif", ".tiff"}:
        paths = [str(p)]
    else:
        paths = [f"{p}.{ext}" for ext in formats]
    for p in paths:
        fig.savefig(p, dpi=dpi, **kwargs)
    return paths
