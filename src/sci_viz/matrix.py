"""矩阵类：相关性热力图、Q 版圆角热力图"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .utils import palettes
from ._common import get_ax


# ======================================================================
def corr_heatmap(df, ax=None, *, title="Correlation Matrix", cmap=None, annot=True, fmt=".2f",
                 method="pearson", triangle=None, cbar_label="Pearson Correlation (r)",
                 figsize=(6.5, 5.2)):
    """
    相关系数热力图（对 df 的数值列自动算相关矩阵）。         [axes 级，返回 ax]

    cmap     : 色表，默认 "blue_red"（10 个锚点、20 级）；可传名称 / 颜色列表 / Colormap
    method   : "pearson" / "spearman" / "kendall"
    triangle : None = 完整矩阵；"lower" / "upper" = 只画下 / 上三角
    """
    ax = get_ax(ax, figsize)
    corr = df.corr(method=method, numeric_only=True)

    mask = None
    if triangle == "lower":
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif triangle == "upper":
        mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    sns.heatmap(corr, ax=ax, cmap=palettes.get_cmap(cmap, "corr_heatmap"), vmax=1, vmin=-1, center=0,
                annot=annot, fmt=fmt, linewidths=0.5, mask=mask,
                cbar_kws={"shrink": 0.8, "label": cbar_label}, square=True, alpha=0.85)
    if title:
        ax.set_title(title, pad=10)
    return ax


# ======================================================================
def q_heatmap(df, ax=None, *, cmap=None, high_color=None, corner_radius=0.3, cell_gap=0.1,
              label_size=7, cbar_label="Z-score", figsize=None):
    """
    Q 版圆角热力图（每个格子是圆角矩形）+ 色条。              [axes 级，返回 ax]

    df         : DataFrame，行 = 行标签，列 = 列标签
    cmap       : 色表，默认 "q_heat"（低=蓝，中=灰白，高=红）
    high_color : 只想换“高值”颜色时用（保持低/中不变），如 high_color="#E63946"
    figsize    : None = 按行列数自动估算
    """
    rows, cols = df.shape
    if ax is None:
        if figsize is None:
            figsize = ((cols * 0.8 + 2) * 0.65, (rows * 0.5 + 1) * 0.65)
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True
    else:
        fig, own_fig = ax.figure, False

    if cmap is None and high_color is not None:
        base = palettes.get_cmap(None, "q_heatmap")
        cmap = [base(0.0), base(0.5), high_color]      # 低/中沿用当前主题，只换高值色
    my_cmap = palettes.get_cmap(cmap, "q_heatmap")

    vmin, vmax = df.min().min(), df.max().max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    for r in range(rows):
        for c in range(cols):
            rect = patches.FancyBboxPatch(
                (c + cell_gap / 2, rows - r - 1 + cell_gap / 2), 1 - cell_gap, 1 - cell_gap,
                boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                facecolor=my_cmap(norm(df.iloc[r, c])), edgecolor="none",
                mutation_scale=1, mutation_aspect=1)
            ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(df.columns, fontweight="bold", size=label_size)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(df.index[::-1], fontstyle="italic", size=label_size)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    sm = ScalarMappable(cmap=my_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=12, pad=0.08)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=label_size - 1)
    cbar.set_label(cbar_label, size=label_size, fontweight="bold")
    cbar.set_ticks(np.linspace(vmin, vmax, 5))                 # 先固定刻度再设标签，避免错位
    cbar.set_ticklabels(["Low", "", "", "", "High"])

    if own_fig:
        fig.tight_layout()
    return ax
