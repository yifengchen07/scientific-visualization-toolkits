"""组间比较：箱线图、均值柱状图（带散点与误差棒）、并列柱状图、环形雷达图"""
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from utils import pubstyle as ps
from utils import palettes
from ._common import get_ax, set_labels, groups_from


# ======================================================================
def box(data, ax=None, *, labels=None, palette=None, show_mean=True, widths=0.5, title=None,
        xlabel=None, ylabel=None, figsize=(4.8, 3.6)):
    """
    分组箱线图（箱体填色，+ 号标均值）。                    [axes 级，返回 ax]

    data    : dict {组名: 数组} / DataFrame（每列一组）/ 数组的列表
    labels  : 组名（不给则用 dict 的键或 DataFrame 列名）
    palette : 各组颜色
    """
    ax = get_ax(ax, figsize)
    names, arrs = groups_from(data, labels)
    colors = palettes.get_palette(palette, "box", n=len(arrs))

    bp = ax.boxplot(arrs, patch_artist=True, showmeans=show_mean, widths=widths,
                    medianprops={"color": "black", "linewidth": 1.5},
                    meanprops={"marker": "+", "markeredgecolor": "black", "markersize": 5},
                    whiskerprops={"color": "black", "linewidth": 1.5},
                    capprops={"color": "black", "linewidth": 1.5},
                    boxprops={"color": "red", "linewidth": 1.5})
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax)
    return ax


# ======================================================================
def bar(data, ax=None, *, labels=None, palette=None, show_points=True, jitter=0.04, seed=None,
        title=None, xlabel=None, ylabel=None, figsize=(3.9, 3.3)):
    """
    均值柱状图：柱高 = 均值，误差棒 = 标准误（SEM），叠加抖动散点。   [axes 级，返回 ax]

    data    : dict {组名: 数组} / DataFrame（每列一组）/ 数组的列表
    palette : 各组颜色
    seed    : 抖动随机种子（想让散点位置可复现时设置）
    """
    ax = get_ax(ax, figsize)
    names, arrs = groups_from(data, labels)
    colors = palettes.get_palette(palette, "bar", n=len(arrs))
    rng = np.random.default_rng(seed)

    for i, (a, c) in enumerate(zip(arrs, colors)):
        mean = a.mean()
        sem = a.std(ddof=1) / np.sqrt(len(a))
        ax.bar(i, mean, color=c, alpha=0.5, width=0.6, edgecolor=c, linewidth=2, zorder=1)
        ax.errorbar(i, mean, yerr=sem, fmt="none", ecolor="black", capsize=5, elinewidth=1.2, zorder=4)
        if show_points:
            ax.scatter(i + rng.normal(0, jitter, size=len(a)), a, color=c, alpha=0.9, s=30,
                       edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax)
    return ax


# ======================================================================
def grouped_bar(df, category, series=None, ax=None, *, palette=None, bar_width=0.35, title=None,
                xlabel=None, ylabel=None, legend=True, figsize=(4.0, 2.9)):
    """
    并列柱状图：每个类别下并排若干个系列。                   [axes 级，返回 ax]

    df       : DataFrame
    category : 作为横轴类别的列名（如 "Time"）
    series   : 要并排比较的列名列表；None = 除 category 外的所有列
    palette  : 各系列颜色
    """
    ax = get_ax(ax, figsize)
    series = list(series) if series is not None else [c for c in df.columns if c != category]
    colors = palettes.get_palette(palette, "grouped_bar", n=len(series))

    index = np.arange(len(df))
    n_bars = len(series)
    for i, col in enumerate(series):
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(index + offset, df[col], bar_width, label=col, color=colors[i], alpha=0.6,
               edgecolor=colors[i], linewidth=2)

    ax.set_xticks(index)
    ax.set_xticklabels(df[category])
    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax)
    if legend:
        ax.legend(frameon=False)
    return ax


# ======================================================================
def radar(data, criteria, ax=None, *, palette=None, ring_palette=None, ring_height=0.15, rmax=1.0,
          ncol=4, figsize=(5, 5)):
    """
    环形雷达图：外圈为各指标的彩色色块。                     [axes 级（极坐标），返回 ax]

    data         : dict {模型名: 各指标的值}，值的个数须等于 len(criteria)
    criteria     : 指标名列表
    palette      : 各模型颜色
    ring_palette : 外圈各指标色块的颜色
    rmax         : 数据的最大刻度（数据在 0~1 之间就用 1；在 0~100 就设 100）
    ax           : 可传入极坐标 ax（plt.subplots(subplot_kw=dict(polar=True))）
    """
    n = len(criteria)
    for name, v in data.items():
        if len(v) != n:
            raise ValueError(f"{name!r} 有 {len(v)} 个值，但指标有 {n} 个")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    cols = palettes.get_palette(palette, "radar", n=len(data))
    rings = palettes.get_palette(ring_palette, "radar_ring", n=n)

    angles = np.linspace(0, 2 * pi, n, endpoint=False).tolist()
    angles += angles[:1]

    for i, (name, values) in enumerate(data.items()):
        vals = list(values) + [values[0]]                   # 闭合，不修改传入的数据
        ax.plot(angles, vals, color=cols[i], linewidth=2, label=name, marker="o", markersize=4)
        ax.fill(angles, vals, color=cols[i], alpha=0.1)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(180 / n)                       # 半径刻度放在两个指标之间，不压住指标名
    ax.set_yticks(np.linspace(0, rmax, 6)[1:])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    for i, (label, color) in enumerate(zip(criteria, rings)):
        ax.bar(angles[i], ring_height * rmax, width=2 * pi / n, bottom=1.02 * rmax, color=color,
               alpha=0.6, edgecolor="none")
        ax.text(angles[i], 1.1 * rmax, label, ha="center", va="center", weight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=ncol, frameon=False)
    ax.set_ylim(0, 1.2 * rmax)
    return ax
