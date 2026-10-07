"""
plots —— 科研绘图工具箱。用法：

    import plots as pl
    pl.bar({"Control": a, "Treat A": b, "Treat B": c}, ylabel="Value")

导入时会自动启用 pubstyle 样式（Times New Roman + SimSun 回退 + cm 公式）。
需要换字体 / 公式字体时：pl.use_style(cn_font="SimHei", math_fontset="stix")
"""
from utils import pubstyle as ps
from utils import palettes
from utils.palettes import register_palette, register_cmap

from ._common import save
from .relation import regression, joint, pair, ellipse, hexbin, dual_axis
from .matrix import corr_heatmap, q_heatmap
from .compare import box, bar, grouped_bar, radar
from .composition import donut, stacked_bar, smooth_area
from .distribution import ridgeline

__all__ = [
    # 关系
    "regression", "joint", "pair", "ellipse", "hexbin", "dual_axis",
    # 矩阵
    "corr_heatmap", "q_heatmap",
    # 比较
    "box", "bar", "grouped_bar", "radar",
    # 构成
    "donut", "stacked_bar", "smooth_area",
    # 分布
    "ridgeline",
    # 工具
    "save", "use_style", "ps", "palettes", "register_palette", "register_cmap",
]


def use_style(**kwargs):
    """重新应用样式，参数同 pubstyle.use（cn_font / en_font / math_fontset / seaborn_context / extra_rc）。"""
    ps.use(**kwargs)


use_style()   # 导入即启用默认样式
