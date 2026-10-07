"""分布类：山脊图"""
import matplotlib.pyplot as plt

from .utils import palettes


def ridgeline(df, by, column, *, cmap=None, fade=True, alpha=0.75, overlap=2, title=None,
              xlabel=None, background="#ffffff", figsize=(5.5, 4.0), **joyplot_kwargs):
    """
    山脊图：用多条错开的密度曲线比较不同组的分布。            [figure 级，返回 fig]

    df      : DataFrame（长表）
    by      : 分组列名（每个取值一条山脊）
    column  : 要画分布的数值列名
    cmap    : 渐变色表，默认 "autumn"。名称 / 颜色列表 / Colormap 均可
    overlap : 山脉重叠程度（越大越紧凑）
    其余参数原样传给 joypy.joyplot（如 linewidth、ylabelsize 等）

    依赖 joypy：pip install joypy。注意 joypy 目前与 pandas>=3 不兼容，需 pandas<3。
    """
    try:
        import joypy
    except ImportError as e:
        raise ImportError("ridgeline 需要 joypy：pip install joypy") from e

    try:
        fig, axes = joypy.joyplot(df, by=by, column=column, colormap=palettes.get_cmap(cmap, "ridgeline"),
                                  fade=fade, alpha=alpha, background=background, overlap=overlap,
                                  title=title, figsize=figsize, **joyplot_kwargs)
    except TypeError as e:
        if "generator" in str(e):
            raise RuntimeError("joypy 与当前 pandas 版本不兼容（pandas>=3 会触发此错误），"
                               "请降级：pip install 'pandas<3'") from e
        raise
    if xlabel:
        axes[-1].set_xlabel(xlabel)
    return fig
