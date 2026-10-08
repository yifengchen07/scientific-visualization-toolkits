"""组间比较：箱线图、均值柱状图（带散点与误差棒）、并列柱状图、环形雷达图"""
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import pubstyle as ps
from .utils import palettes
from ._common import get_ax, set_labels, groups_from
from . import stats as _stats


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
        ax.bar(i, mean, color=c, alpha=0.85, width=0.6, linewidth=2, zorder=1)
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
        ax.bar(index + offset, df[col], bar_width, label=col, color=colors[i], alpha=0.85,
                linewidth=2)

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


# ======================================================================
def _rate_long(data, by, value, hue, order, hue_order, ci):
    """把各种输入统一成长表：group, [hue,] k, n, rate, lo, hi（0~1 小数）。"""
    if isinstance(data, pd.DataFrame):
        if by is None or value is None:
            raise ValueError("data 是 DataFrame 时需要指定 by=（分组列）和 value=（布尔 / 0-1 列）")
        return _stats.rate_table(data, by, value, hue=hue, order=order, hue_order=hue_order, ci=ci)
    if isinstance(data, dict):
        rows = []
        for name, v in data.items():
            if isinstance(v, (tuple, list)) and len(v) == 2 and np.isscalar(v[0]) and np.isscalar(v[1]):
                k, n = int(v[0]), int(v[1])
            else:                                         # 布尔 / 0-1 数组
                arr = np.asarray(v).astype(float)
                k, n = int(np.nansum(arr)), int(np.sum(~np.isnan(arr)))
            rows.append(dict(group=str(name), k=k, n=n))
        g = pd.DataFrame(rows)
        g["rate"] = g["k"] / g["n"]
        g["lo"], g["hi"] = _stats.wilson_ci(g["k"].to_numpy(), g["n"].to_numpy(), ci)
        if order is not None:
            g = g.set_index("group").loc[list(order)].reset_index()
        return g
    raise TypeError("data 应为 DataFrame（配合 by/value）或 dict {组名: (成功数, 总数) 或 布尔数组}")


def rate_bar(data, ax=None, *, by=None, value=None, hue=None, order=None, hue_order=None, labels=None,
             ci=0.95, show_values=True, show_n=True, baseline=None, baseline_label=None,
             horizontal=False, palette=None, rate_label="Rate (%)", title=None, xlabel=None, ylabel=None,
             xtick_rotation=None, figsize=(3.9, 3.3)):
    """
    比例柱状图：柱高 = 某事件发生的比例（如准确率、召回率、True 的占比），误差棒 = Wilson 置信区间。 [axes 级，返回 ax]

    data    : DataFrame（配合 by= 分组列、value= 布尔/0-1 列，可选 hue= 再分一层）
              或 dict {组名: (成功数, 总数)} / {组名: 布尔数组}
    order / hue_order : 组 / hue 的显示顺序
    labels  : 重命名各组的显示名（与 order 同长度）
    ci      : 置信水平（默认 0.95）
    show_values : 柱顶标注百分数；show_n：横轴标签下方标注样本量 (n=…)（有 hue 时不标）
    baseline    : 画一条虚线基线（0~1 小数，如 stats.majority_baseline 的返回值 0.691），baseline_label 为其说明文字
    xtick_rotation : 横轴标签旋转角度；None = 标签较长时自动旋转 30°
    horizontal  : True = 横向柱状图（组名很长时用）
    rate_label  : 比例所在坐标轴的默认标签；xlabel / ylabel 可单独覆盖
    palette     : 各组（有 hue 时为各 hue）颜色，默认跟随主题
    """
    ax = get_ax(ax, figsize)
    tab = _rate_long(data, by, value, hue, order, hue_order, ci)
    groups = list(dict.fromkeys(tab["group"]))
    hues = list(dict.fromkeys(tab["hue"])) if "hue" in tab else [None]
    names = list(labels) if labels is not None else [str(g) for g in groups]
    if len(names) != len(groups):
        raise ValueError(f"labels 有 {len(names)} 个，但分组有 {len(groups)} 个")
    colors = palettes.get_palette(palette, "rate_bar", n=len(hues) if "hue" in tab else len(groups))
    nh = len(hues)
    w = 0.6 if nh == 1 else 0.8 / nh
    pct = lambda x: np.asarray(x, float) * 100
    ymax = float(np.nanmax(pct(tab["hi"])))
    pad = ymax * 0.02 + 1.0

    for gi, g in enumerate(groups):
        for hi_, h in enumerate(hues):
            row = tab[(tab["group"] == g) & ((tab["hue"] == h) if "hue" in tab else True)]
            if row.empty:
                continue
            r = row.iloc[0]
            c = colors[hi_] if "hue" in tab else colors[gi]
            pos = gi + ((hi_ - (nh - 1) / 2) * w if nh > 1 else 0)
            val, lo, hi = pct(r["rate"]), pct(r["lo"]), pct(r["hi"])
            err = [[val - lo], [hi - val]]
            lab = (str(h) if gi == 0 else None) if "hue" in tab else None
            if horizontal:
                ax.barh(pos, val, height=w * 0.92, color=c, alpha=0.5, edgecolor=c, linewidth=2, zorder=2, label=lab)
                ax.errorbar(val, pos, xerr=err, fmt="none", ecolor="black", capsize=3, elinewidth=1.1, zorder=4)
                if show_values:
                    ax.text(hi + pad, pos, f"{val:.1f}", va="center", ha="left", fontsize=7)
            else:
                ax.bar(pos, val, width=w * 0.92, color=c, alpha=0.85, linewidth=2, zorder=2, label=lab)
                ax.errorbar(pos, val, yerr=err, fmt="none", ecolor="black", capsize=3, elinewidth=1.1, zorder=4)
                if show_values:
                    ax.text(pos, hi + pad, f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ticklabels = names
    if show_n and "hue" not in tab:
        ns = tab.set_index("group")["n"].reindex(groups).to_numpy()
        ticklabels = [f"{nm}\n(n={int(n):,})" for nm, n in zip(names, ns)]
    top = min(max(ymax + 14, 20), 118)
    ticks = [t for t in np.arange(0, 101, 20) if t <= top]
    if horizontal:
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(ticklabels)
        ax.invert_yaxis()
        ax.set_xlim(0, top)
        ax.set_xticks(ticks)
        if baseline is not None:
            ax.axvline(baseline * 100, ls="--", lw=1, color="#555555", zorder=1)
            if baseline_label:
                ax.text(baseline * 100, 1.0, baseline_label, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6.5, color="#555555")
    else:
        ax.set_xticks(range(len(groups)))
        rot = xtick_rotation
        if rot is None:
            rot = 30 if max(len(t.split("\n")[0]) for t in ticklabels) * len(ticklabels) > 40 else 0
        ax.set_xticklabels(ticklabels, rotation=rot, ha="right" if rot else "center",
                           rotation_mode="anchor" if rot else "default")
        ax.set_ylim(0, top)
        ax.set_yticks(ticks)
        if baseline is not None:
            ax.axhline(baseline * 100, ls="--", lw=1, color="#555555", zorder=1)
            if baseline_label:                              # 在最右侧留出空位写说明，避免压住柱子
                ax.set_xlim(-0.6, len(groups) - 0.4 + 0.75)
                ax.text(len(groups) - 0.4 + 0.72, baseline * 100 + 1, baseline_label.replace(" ", "\n", 1),
                        ha="right", va="bottom", fontsize=6.5, color="#555555", linespacing=0.95)
    if "hue" in tab:
        ax.legend(frameon=False, loc="best")
    if horizontal:
        set_labels(ax, title, xlabel if xlabel is not None else rate_label, ylabel)
    else:
        set_labels(ax, title, xlabel, ylabel if ylabel is not None else rate_label)
    ps.clean_axis(ax)
    return ax
