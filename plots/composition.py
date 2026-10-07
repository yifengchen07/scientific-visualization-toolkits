"""构成 / 占比：甜甜圈饼图、堆积柱状图、平滑堆叠面积图"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

from utils import pubstyle as ps
from utils import palettes
from ._common import get_ax, set_labels


# ======================================================================
def donut(values, labels, ax=None, *, center_text="Composition", palette=None, width=0.35,
          legend_title="Categories", pct_fontsize=7, figsize=(5.2, 4.0)):
    """
    甜甜圈饼图（扇区显示百分比，右侧图例）。                  [axes 级，返回 ax]

    values      : 各扇区数值（自动换算成百分比）
    labels      : 各扇区名称
    center_text : 圆环中心文字，可含换行 "Cell\\nViability"；None = 不显示
    palette     : 各扇区颜色
    width       : 圆环厚度（0~1）
    """
    ax = get_ax(ax, figsize)
    colors = palettes.get_palette(palette, "donut", n=len(values))

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, pctdistance=0.82,
        textprops={"fontweight": "bold"},
        wedgeprops={"width": width, "edgecolor": "w", "linewidth": 2})
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(pct_fontsize)

    if center_text:
        ax.text(0, 0, center_text, ha="center", va="center", fontweight="bold", color="#333333")
    ax.legend(wedges, labels, title=legend_title, loc="center left",
              bbox_to_anchor=(1.3, 0, 0.5, 1), frameon=False)
    ax.axis("equal")
    return ax


# ======================================================================
def stacked_bar(df, category, stack=None, ax=None, *, palette=None, bar_width=0.6, title=None,
                xlabel=None, ylabel=None, legend=True, figsize=(3.6, 3.6)):
    """
    堆积柱状图。                                            [axes 级，返回 ax]

    df       : DataFrame
    category : 作为横轴类别的列名
    stack    : 需要堆叠的列名列表（自下而上）；None = 除 category 外的所有列
    palette  : 各层颜色（自下而上）
    """
    ax = get_ax(ax, figsize)
    stack = list(stack) if stack is not None else [c for c in df.columns if c != category]
    colors = palettes.get_palette(palette, "stacked_bar", n=len(stack))

    x_pos = np.arange(len(df))
    bottom = np.zeros(len(df))
    for i, col in enumerate(stack):
        values = df[col].to_numpy(float)
        ax.bar(x_pos, values, bar_width, bottom=bottom, label=col, color=colors[i],
               edgecolor="white", linewidth=1, alpha=0.9)
        bottom += values

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df[category])
    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax)
    if legend:
        ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    return ax


# ======================================================================
def smooth_area(df, ax=None, *, palette=None, transition_ratio=0.3, xlabel=None, ylabel=None,
                title=None, legend=True, figsize=(7.2, 3.9)):
    """
    平滑阶梯过渡的百分比堆叠面积图。                          [axes 级，返回 ax]

    df               : DataFrame，第 1 列为类别（时间/月份），其余列为各层数值（每行总和应为 100）
    palette          : 各层颜色（自下而上）
    transition_ratio : 过渡带占比（0~1），越小越接近阶梯图，越大越圆润
    """
    ax = get_ax(ax, figsize)
    categories = df.iloc[:, 0].values
    data_values = df.iloc[:, 1:].to_numpy(float).T
    n_layers, n_steps = data_values.shape
    colors = palettes.get_palette(palette, "smooth_area", n=n_layers)

    pts = 100
    trans_w = int(pts * transition_ratio)
    flat_w = pts - trans_w

    y_fine = [[] for _ in range(n_layers)]
    for i in range(n_steps - 1):
        for j in range(n_layers):
            y0, y1 = data_values[j, i], data_values[j, i + 1]
            y_fine[j].extend(np.full(flat_w, y0))
            y_fine[j].extend(y0 + (y1 - y0) * expit(np.linspace(-10, 10, trans_w)))
    for j in range(n_layers):
        y_fine[j].extend(np.full(pts, data_values[j, -1]))

    x_fine = np.arange(len(y_fine[0]))
    ax.stackplot(x_fine, np.array(y_fine), labels=df.columns[1:], colors=colors, alpha=0.9,
                 edgecolor="white", linewidth=0.3)

    ax.set_xticks(np.arange(0, n_steps * pts, pts) + pts / 2)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, x_fine[-1])
    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax, "both")
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    return ax
