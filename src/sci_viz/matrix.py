"""矩阵类：相关性热力图、Q 版圆角热力图"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, to_rgb
from matplotlib.cm import ScalarMappable

from .utils import palettes
from ._common import get_ax, set_labels


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


# ======================================================================
def confusion_heatmap(cm, ax=None, *, labels=None, pred_labels=None, normalize="true", cmap=None,
                      annot="both", fontsize=7, title=None, xlabel="Predicted", ylabel="True",
                      cbar=True, cbar_label=None, xtick_rotation=0, figsize=None):
    """
    混淆矩阵热力图：格子里写“数量 + 百分比”，颜色按百分比深浅。     [axes 级，返回 ax]

    cm          : DataFrame（行 = 真实类别，列 = 预测类别，值 = 数量；可用 stats.confusion_table 生成），
                  或二维数组（此时用 labels / pred_labels 给行列命名）。行列数可以不同
    normalize   : "true" 按行归一化（每行 100% = 该真实类别的召回分布，推荐）
                  "pred" 按列归一化 / "all" 按总数 / None 不归一化（颜色按数量）
    annot       : "both" 数量 + 百分比 / "count" 只写数量 / "pct" 只写百分比 / None 不写
    cmap        : 色表，默认跟随主题的连续色（浅 -> 深）
    cbar_label  : 色条标签，默认按 normalize 自动生成
    """
    if not isinstance(cm, pd.DataFrame):
        cm = pd.DataFrame(np.asarray(cm), index=labels, columns=pred_labels)
    elif labels is not None or pred_labels is not None:
        cm = cm.copy()
        if labels is not None:
            cm.index = list(labels)
        if pred_labels is not None:
            cm.columns = list(pred_labels)
    counts = cm.to_numpy(float)
    nr, nc = counts.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        if normalize == "true":
            frac = counts / counts.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            frac = counts / counts.sum(axis=0, keepdims=True)
        elif normalize == "all":
            frac = counts / counts.sum()
        elif normalize is None:
            frac = counts / counts.max()
        else:
            raise ValueError("normalize 应为 'true' / 'pred' / 'all' / None")
    frac = np.nan_to_num(frac)
    pctv = frac * 100

    if figsize is None:
        figsize = (0.95 * nc + 1.5, 0.8 * nr + 1.0)
    ax = get_ax(ax, figsize)
    cmap_ = palettes.get_cmap(cmap, "confusion_heatmap")
    vmax = 1.0 if normalize in ("true", "pred") else float(frac.max() or 1.0)
    im = ax.imshow(frac, cmap=cmap_, vmin=0, vmax=vmax, aspect="equal")

    if annot:
        for i in range(nr):
            for j in range(nc):
                r, g, b = to_rgb(cmap_(frac[i, j] / vmax if vmax else 0))
                txt = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "#222222"
                if annot == "both":
                    s = f"{counts[i, j]:,.0f}\n({pctv[i, j]:.1f}%)" if normalize else f"{counts[i, j]:,.0f}"
                elif annot == "count":
                    s = f"{counts[i, j]:,.0f}"
                else:
                    s = f"{pctv[i, j]:.1f}%"
                ax.text(j, i, s, ha="center", va="center", fontsize=fontsize, color=txt)

    ax.set_xticks(range(nc))
    ax.set_xticklabels([str(c) for c in cm.columns], rotation=xtick_rotation,
                       ha="right" if xtick_rotation else "center")
    ax.set_yticks(range(nr))
    ax.set_yticklabels([str(i) for i in cm.index])
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    set_labels(ax, title, xlabel, ylabel)

    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.03, fraction=0.05)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
        if cbar_label is None:
            cbar_label = {"true": "Row-normalized (%)", "pred": "Column-normalized (%)",
                          "all": "Share of all samples (%)", None: "Relative count"}[normalize]
        cb.set_label(cbar_label, size=fontsize + 0.5)
        if normalize in ("true", "pred"):
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cb.set_ticklabels(["0", "25", "50", "75", "100"])
    return ax
