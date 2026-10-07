"""变量之间的关系：回归、联合分布、散点矩阵、置信椭圆、六边形分箱、双 Y 轴"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from scipy import stats

from .utils import pubstyle as ps
from .utils import palettes
from ._common import get_ax, set_labels, darken


def _emptiest_corner(x, y, ax, avoid_upper_left):
    """在几个角落里挑数据点最少的作为图例位置（统计框占着左上角时不选左上）。"""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
    corners = {"lower right": (fx > 0.6) & (fy < 0.4), "upper right": (fx > 0.6) & (fy > 0.6),
               "lower left": (fx < 0.4) & (fy < 0.4)}
    if not avoid_upper_left:
        corners["upper left"] = (fx < 0.4) & (fy > 0.6)
    return min(corners, key=lambda k: corners[k].sum())


# ======================================================================
def regression(x, y, ax=None, *, xlabel="Independent Variable", ylabel="Dependent Variable",
               title="Linear Regression", palette=None, ci=0.95, show_stats=True,
               legend=True, legend_loc=None, figsize=(3.6, 3.0)):
    """
    散点 + 线性回归 + 置信带 + R² / p / N 标注。          [axes 级，返回 ax]

    x, y     : 一维数据（自动忽略 NaN）
    palette  : 最多 3 色 [散点, 拟合线, 置信带]；只给 2 色时置信带沿用拟合线颜色
    ci       : 置信水平（回归均值线的双侧置信带），默认 0.95
    show_stats : 是否在左上角标注 R²、p、N
    legend_loc : 图例位置，如 "upper right"；None = 自动挑数据最少的角落
    """
    ax = get_ax(ax, figsize)
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    pal = palettes.get_palette(palette, "regression")
    c_pt, c_ln = pal[0], pal[1]
    c_ci = pal[2] if len(pal) > 2 else c_ln

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    idx = np.argsort(x)
    x_s = x[idx]
    line = slope * x_s + intercept

    ax.scatter(x, y, color=c_pt, s=25, label="Data Points")
    ax.plot(x_s, line, color=c_ln, lw=2, label="Fitted Line")

    n = len(x)
    dof = n - 2
    t_crit = stats.t.ppf(1 - (1 - ci) / 2, dof)
    resid = y - (slope * x + intercept)
    s_err = np.sqrt(np.sum(resid ** 2) / dof)
    band = t_crit * s_err * np.sqrt(1 / n + (x_s - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(x_s, line - band, line + band, color=c_ci, alpha=0.15, label=f"{ci * 100:g}% CI")

    if show_stats:
        txt = f"$R^2 = {r_value ** 2:.3f}$\n$p = {p_value:.4g}$\n$N = {n}$"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#bdc3c7"))

    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax)
    if legend:
        ax.legend(frameon=False, loc=legend_loc or _emptiest_corner(x, y, ax, show_stats))
    return ax


# ======================================================================
def joint(x, y, fig=None, *, xlabel="Independent Variable", ylabel="Outcome Y", palette=None,
          bins=20, figsize=(3.6, 3.1)):
    """
    主图散点 + 回归线，上方和右侧为边缘直方图。          [figure 级，返回 fig]

    fig     : None = 新建画布；也可传入 SubFigure 把多个 joint 排在一张图里：
                  fig = plt.figure(figsize=(7.2, 3.1)); a, b = fig.subfigures(1, 2)
                  pl.joint(x1, y1, a); pl.joint(x2, y2, b)
    palette : 最多 3 色 [散点与直方图, 拟合线, 散点边缘]；只给 2 色时边缘取散点色的加深色
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    x, y = np.asarray(x, float), np.asarray(y, float)

    pal = palettes.get_palette(palette, "joint")
    c_pt, c_ln = pal[0], pal[1]
    c_edge = pal[2] if len(pal) > 2 else darken(c_pt)

    gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.08, wspace=0.08)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_hx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_hy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(x, y, color=c_pt, edgecolor=c_edge, s=30)
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    xr = np.linspace(x.min(), x.max(), 100)
    ax_main.plot(xr, slope * xr + intercept, color=c_ln, lw=2)
    ax_main.text(0.05, 0.95, f"$R^2 = {r_value ** 2:.2f}$\n$p = {p_value:.2g}$",
                 transform=ax_main.transAxes, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#bdc3c7"))

    ax_hx.hist(x, bins=bins, color=c_pt, edgecolor="white")
    ax_hx.axis("off")
    ax_hy.hist(y, bins=bins, color=c_pt, edgecolor="white", orientation="horizontal")
    ax_hy.axis("off")

    set_labels(ax_main, None, xlabel, ylabel)
    ps.clean_axis(ax_main, "both")
    return fig


# ======================================================================
def pair(df, vars=None, hue=None, *, palette=None, title="auto", height=2.5):
    """
    多变量散点矩阵：下/上三角为回归散点，对角线为核密度。   [figure 级，返回 fig]

    df      : DataFrame
    vars    : 要画的列名列表；None = 所有数值列
    hue     : 分组列名（可选）
    palette : 各分组颜色（按分组顺序）
    title   : "auto" = 自动标题；None = 不要标题；或自定义字符串
    """
    vars = list(vars) if vars is not None else list(df.select_dtypes("number").columns)
    plot_kws = {"scatter_kws": {"alpha": 0.7, "s": 20, "rasterized": True}, "line_kws": {"lw": 1.5}}
    diag_kws = {"fill": True, "alpha": 0.7}

    if hue is not None:
        pal = palettes.get_palette(palette, "pair", n=df[hue].nunique())
        g = sns.pairplot(df, vars=vars, hue=hue, kind="reg", diag_kind="kde", palette=pal,
                         corner=False, height=height, plot_kws=plot_kws, diag_kws=diag_kws)
    else:
        c = palettes.get_palette(palette, "pair")[0]
        plot_kws["color"] = c
        diag_kws["color"] = c
        g = sns.pairplot(df, vars=vars, kind="reg", diag_kind="kde", corner=False, height=height,
                         plot_kws=plot_kws, diag_kws=diag_kws)

    sns.despine(fig=g.fig, trim=False)
    if title == "auto":
        title = f"Multivariate Correlation Analysis: {', '.join(vars)}"
    if title:
        g.fig.suptitle(title, y=1.02, fontsize=11, fontweight="bold")
    return g.fig


# ======================================================================
def ellipse(groups, ax=None, *, n_std=2.0, palette=None, xlabel="Feature X (units)",
            ylabel="Feature Y (units)", title=None, legend=True, figsize=(4.8, 3.6)):
    """
    各组散点 + 带填充的置信椭圆。                         [axes 级，返回 ax]

    groups  : dict {组名: (x, y)}，例如 {"Group A": (x1, y1), "Group B": (x2, y2)}
    n_std   : 椭圆半径 = n_std 倍标准差（2.0 约对应 95%）
    palette : 各组颜色
    """
    ax = get_ax(ax, figsize)
    colors = palettes.get_palette(palette, "ellipse", n=len(groups))

    for (name, (x, y)), color in zip(groups.items(), colors):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ax.scatter(x, y, s=30, color=color, edgecolors="white", linewidth=0.5)

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell = Ellipse((0, 0), width=np.sqrt(1 + pearson) * 2, height=np.sqrt(1 - pearson) * 2,
                      facecolor=color, edgecolor=color, alpha=0.5, linestyle="--", linewidth=1.5,
                      label=name)
        tf = (transforms.Affine2D().rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
              .translate(np.mean(x), np.mean(y)))
        ell.set_transform(tf + ax.transData)
        ax.add_patch(ell)

    ax.relim()
    ax.autoscale_view()
    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax, "both")
    if legend:
        ax.legend(frameon=False)
    return ax


# ======================================================================
def hexbin(x, y, *, xlabel="X", ylabel="Y", title="Hexbin Chart", cmap=None, palette=None,
           gridsize=20, bins=20, figsize=(5, 5)):
    """
    六边形分箱图 + 上方/右侧边缘直方图 + 颜色条。          [figure 级，返回 fig]

    cmap    : 色表（默认 "Blues"）。名称 / 颜色列表 / Colormap 均可
    palette : 边缘直方图颜色（取第 1 个）
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    c_hist = palettes.get_palette(palette, "hexbin")[0]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_hx = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_hy = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    hb = ax_main.hexbin(x, y, gridsize=gridsize, cmap=palettes.get_cmap(cmap, "hexbin"),
                        mincnt=1, edgecolors="none")
    ax_hx.hist(x, bins=bins, color=c_hist, edgecolor="white", linewidth=0.8, alpha=0.8)
    ax_hy.hist(y, bins=bins, color=c_hist, edgecolor="white", linewidth=0.8, alpha=0.8,
               orientation="horizontal")

    for a in (ax_hx, ax_hy):
        a.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False,
                      labelbottom=False, labelleft=False)
        for sp in a.spines.values():
            sp.set_visible(False)

    set_labels(ax_main, None, xlabel, ylabel)
    ps.clean_axis(ax_main, "both")

    cax = fig.add_axes([0.95, 0.25, 0.02, 0.4])
    fig.colorbar(hb, cax=cax, label="Count")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    return fig


# ======================================================================
def dual_axis(x, y1, y2, ax=None, *, xlabel="X", ylabel1="Y1", ylabel2="Y2", title=None,
              palette=None, xtick_step=None, xtick_rotation=None, figsize=(7.2, 3.1)):
    """
    双 Y 轴折线图（两条曲线量纲不同，对比变化趋势）。       [axes 级，返回 (ax_左, ax_右)]

    x        : 横轴数据。字符串（如日期文本）按顺序等距排列，数值/日期则按实际值
    palette  : 2 色 [左轴, 右轴]；轴标签和刻度也会染成对应颜色
    xtick_step : 字符串横轴时每隔多少个点标一个刻度；None = 自动（约 10 个）
    xtick_rotation : 横轴刻度旋转角度；None = 字符串横轴转 90°，其余不转
    """
    ax1 = get_ax(ax, figsize)
    c1, c2 = palettes.get_palette(palette, "dual_axis", n=2)

    x = pd.Series(x).reset_index(drop=True)
    categorical = not (pd.api.types.is_numeric_dtype(x) or pd.api.types.is_datetime64_any_dtype(x))
    xs = np.arange(len(x)) if categorical else x.to_numpy()

    ax1.plot(xs, np.asarray(y1), color=c1)
    ax2 = ax1.twinx()
    ax2.plot(xs, np.asarray(y2), color=c2)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2.set_ylabel(ylabel2, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    if categorical:
        step = xtick_step or max(1, len(x) // 10)
        ax1.set_xticks(np.arange(0, len(x), step))
        ax1.set_xticklabels(x.iloc[::step])
    rot = xtick_rotation if xtick_rotation is not None else (90 if categorical else 0)
    ax1.tick_params(axis="x", rotation=rot)

    ps.clean_axis(ax1, "both")
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    if title:
        ax2.set_title(title)
    return ax1, ax2
