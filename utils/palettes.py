"""
palettes —— 配色注册表（离散配色 + 连续/发散色表 + 各图的默认配色）

三种使用方式：
    1. 单次调用指定：   pl.bar(data, palette="sci6")  /  palette=["#1f77b4", "#ff7f0e"]
    2. 注册自己的配色： palettes.register_palette("mine", ["#264653", "#2a9d8f", "#e9c46a"])
                        之后 pl.bar(data, palette="mine")
    3. 全局覆盖：       palettes.use("mine")     # 所有图的离散配色都改用它
                        palettes.use()            # 恢复各图原来的默认配色

连续色表（热力图、hexbin、山脊图）同理：cmap="blue_red" / cmap=["#79C7FF", "#F0F0F0", "#FF5A5F"] / cmap="viridis"
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap, to_rgb

# ----------------------------------------------------------------------
# 1. 离散配色：名称 -> 颜色列表（前 N 个颜色依次给第 1..N 组）
#    以下全部来自原 notebook，名称里的数字 = 颜色个数
# ----------------------------------------------------------------------
PALETTES = {
    "soft4":     ["#8EC8ED", "#AED594", "#D693BE", "#F5B3A5"],
    "teal6":     ["#257D8B", "#68BED9", "#BFDFD2", "#EAA558", "#ED8D5A", "#EFCE87"],
    "vivid3":    ["#23BAC5", "#EECA40", "#FD763F"],
    "red_blue4": ["#DB3124", "#FFDF92", "#90BEE0", "#4B74B2"],
    "sci6":      ["#2878B5", "#9AC9E0", "#C82423", "#F8AC8C", "#72BA68", "#BA8FBB"],
    "sunset5":   ["#326195", "#A1CADF", "#FEF9C3", "#F2A561", "#C52B28"],
    "ring6":     ["#ADD8E6", "#B0C4DE", "#F08080", "#FFCC99", "#FFE4B5", "#D3EDD3"],
    "sky_rose":  ["#87C3E4", "#F48892"],
    # pubstyle 里的 Okabe-Ito 色盲友好色板
    "okabe_ito": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"],
}

# ----------------------------------------------------------------------
# 2. 连续 / 发散色表：名称 -> (颜色锚点, 离散级数 N；None = 连续 256 级)
# ----------------------------------------------------------------------
_CMAPS = {
    # 相关性热力图：10 个锚点，按 20 级离散化（与原图一致）
    "blue_red": (["#104680", "#317CB7", "#6DADE5", "#B6D7E8", "#E9F0F4",
                  "#F1E3CE", "#F6B293", "#DC6D57", "#B72230", "#6D011F"], 20),
    # Q 版圆角热力图：低(蓝) - 中(灰白) - 高(红)
    "q_heat": (["#79C7FF", "#F0F0F0", "#FF5A5F"], None),
}

# ----------------------------------------------------------------------
# 3. 每种图的默认配色（保持与原 notebook 一致）。key = 绘图函数名
#    注：多角色的图用位置区分角色，见各函数 docstring
# ----------------------------------------------------------------------
DEFAULTS = {
    "regression":  ["#87C3E4", "#F48892", "#e74c3c"],   # 散点, 拟合线, 置信带
    "joint":       ["#87C3E4", "#F48892", "#6EB8E0"],   # 散点&直方图, 拟合线, 散点边缘
    "pair":        ["#91CAE8", "#F48892"],              # 各分组
    "ellipse":     PALETTES["soft4"],                   # 各分组
    "hexbin":      ["#8ED1F2"],                         # 边缘直方图
    "dual_axis":   ["tab:red", "tab:blue"],             # 左轴, 右轴
    "box":         PALETTES["teal6"],                   # 各分组
    "bar":         PALETTES["vivid3"],                  # 各分组
    "grouped_bar": PALETTES["vivid3"],                  # 各系列
    "radar":       PALETTES["teal6"],                   # 各模型
    "radar_ring":  PALETTES["ring6"],                   # 外圈色块（各指标）
    "stacked_bar": PALETTES["red_blue4"],               # 各层（自下而上）
    "donut":       PALETTES["sci6"],                    # 各扇区
    "smooth_area": PALETTES["sunset5"],                 # 各层（自下而上）
}
DEFAULT_CMAPS = {
    "corr_heatmap": "blue_red",
    "q_heatmap": "q_heat",
    "hexbin": "Blues",
    "ridgeline": "autumn",
}

_override = {"palette": None, "cmap": None}


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _check(colors, what="颜色"):
    colors = list(colors)
    if not colors:
        raise ValueError(f"{what}列表不能为空")
    for c in colors:
        try:
            to_rgb(c)
        except ValueError:
            raise ValueError(f"无法识别的{what}: {c!r}（请用 '#RRGGBB' 或 matplotlib 颜色名）") from None
    return colors


def _to_list(spec):
    if isinstance(spec, str):
        if spec in PALETTES:
            return list(PALETTES[spec])
        try:                                   # 单个颜色，如 "#87C3E4" / "tab:red"
            to_rgb(spec)
            return [spec]
        except ValueError:
            raise ValueError(f"未知配色名 {spec!r}。可用: {sorted(PALETTES)}；"
                             f"也可以直接传颜色列表，或用 register_palette 注册。") from None
    return _check(spec)


def register_palette(name, colors):
    """注册一套离散配色，之后可用 palette="name"。同名会覆盖。"""
    PALETTES[name] = _check(colors)
    return PALETTES[name]


def register_cmap(name, colors, n=None):
    """注册一个连续/发散色表（colors 为颜色锚点；n 为离散级数，None = 连续）。"""
    _CMAPS[name] = (_check(colors), n)


def get_palette(spec=None, key=None, n=None):
    """
    解析配色。spec 可以是：None（用全局覆盖或 key 对应的默认）、配色名、颜色列表、单个颜色。
    n 不为 None 时，返回长度恰好为 n 的列表（不够则循环使用）。
    """
    if spec is None:
        spec = _override["palette"]
    if spec is None:
        spec = DEFAULTS[key]
    colors = _to_list(spec)
    if n is not None:
        colors = [colors[i % len(colors)] for i in range(n)]
    return colors


def get_cmap(spec=None, key=None, n=None):
    """解析色表。spec 可以是：None、已注册名、matplotlib 色表名、颜色列表、Colormap 对象。"""
    if spec is None:
        spec = _override["cmap"]
    if spec is None:
        spec = DEFAULT_CMAPS[key]
    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, str):
        if spec in _CMAPS:
            colors, nb = _CMAPS[spec]
            return LinearSegmentedColormap.from_list(spec, colors, N=n or nb or 256)
        return plt.get_cmap(spec)              # matplotlib 自带名称，如 "viridis" / "Blues_r"
    return LinearSegmentedColormap.from_list("custom", _check(spec), N=n or 256)


def use(palette=None, cmap=None):
    """全局覆盖所有图的离散配色 / 色表。不带参数 = 恢复各图默认。"""
    if palette is not None:
        _to_list(palette)
    if cmap is not None:
        get_cmap(cmap, key="corr_heatmap")
    _override["palette"] = palette
    _override["cmap"] = cmap


def list_palettes():
    return {"palettes": sorted(PALETTES), "cmaps": sorted(_CMAPS)}


def show():
    """画出所有已注册的配色与色表，方便挑选。返回 Figure。"""
    names, cnames = list(PALETTES), list(_CMAPS)
    nrows = len(names) + len(cnames)
    fig, axes = plt.subplots(nrows, 1, figsize=(6.0, 0.3 * nrows + 0.4))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.02, hspace=0.25)
    for ax, name in zip(axes, names):
        cols = PALETTES[name]
        for i, c in enumerate(cols):
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=c))
        ax.set_xlim(0, max(len(cols), 8)); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(-0.02, 0.5, name, transform=ax.transAxes, ha="right", va="center")
    grad = np.linspace(0, 1, 256)[None, :]
    for ax, name in zip(axes[len(names):], cnames):
        ax.imshow(grad, aspect="auto", cmap=get_cmap(name)); ax.axis("off")
        ax.text(-0.02, 0.5, name, transform=ax.transAxes, ha="right", va="center")
    return fig
