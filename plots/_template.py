"""
新增一种图的模板（复制本文件内容到对应类别的模块里改，不要直接 import 本文件）。

四步走：
  1. 在 plots/ 下合适的模块（relation / matrix / compare / composition / distribution）里写函数，
     或者新建一个模块，如 plots/timeseries.py
  2. 配色走 palettes.get_palette / get_cmap 即可自动跟随主题（离散色用主题 cat，连续色用 seq）；默认角色分配见 palettes._theme_palette_spec
  3. 在 plots/__init__.py 里 import 并写进 __all__
  4. 在 examples/gallery.py 里加一个 demo_xxx 并登记到 GALLERY（它同时就是冒烟测试）

函数约定：
  - 第 1 个参数是数据；第 2 个位置参数永远是 ax（figure 级函数是 fig）；其余全部关键字参数（* 之后）
  - axes 级：ax=None 时自己新建画布，返回 ax；figure 级：返回 fig
  - 颜色一律走 palettes.get_palette(palette, "函数名", n=组数) / palettes.get_cmap(cmap, "函数名")
  - 标签用 set_labels（自动处理 “中文 + $公式$” 混排），坐标轴整理用 ps.clean_axis
  - 不要在函数里调用 plt.show() / sns.set_theme()，也不要硬编码字号（交给 pubstyle）
"""
import numpy as np

from utils import pubstyle as ps
from utils import palettes
from ._common import get_ax, set_labels, groups_from


def violin(data, ax=None, *, labels=None, palette=None, title=None, xlabel=None, ylabel=None,
           figsize=(4.8, 3.6)):
    """
    一句话说明这是什么图。                                   [axes 级，返回 ax]

    data    : dict {组名: 数组} / DataFrame（每列一组）/ 数组的列表
    labels  : 组名（可选）
    palette : 各组颜色
    """
    ax = get_ax(ax, figsize)
    names, arrs = groups_from(data, labels)
    colors = palettes.get_palette(palette, "violin", n=len(arrs))      # 自动跟随主题 cat；classic 主题才需要在 DEFAULTS 补 "violin"

    parts = ax.violinplot(arrs, showmeans=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.8)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    set_labels(ax, title, xlabel, ylabel)
    ps.clean_axis(ax)
    return ax
