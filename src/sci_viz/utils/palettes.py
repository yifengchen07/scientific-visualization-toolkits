"""
palettes —— 配色注册表（离散配色 + 连续/发散色表 + 各图的默认配色）

三种使用方式：
    1. 单次调用指定：   pl.bar(data, palette="sci6")  /  palette=["#1f77b4", "#ff7f0e"]
    2. 注册自己的配色： palettes.register_palette("mine", ["#264653", "#2a9d8f", "#e9c46a"])
                        之后 pl.bar(data, palette="mine")
    3. 全局覆盖：       palettes.use("mine")     # 所有图的离散配色都改用它
                        palettes.use()            # 恢复各图原来的默认配色

4. 主题（整篇论文配色统一）：palettes.set_theme("ocean")  —— 所有图的默认配色都从该主题派生
                        palettes.register_theme("mine", cat=[...], accent="#...")   # 自己定义主题
                        优先级：单次调用 palette= > palettes.use() 全局覆盖 > 主题 overrides > 主题默认

连续色表（热力图、hexbin、山脊图）同理：cmap="blue_red" / cmap=["#79C7FF", "#F0F0F0", "#FF5A5F"] / cmap="viridis"
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap, to_rgb, to_hex

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
    "rate_bar":    PALETTES["vivid3"],                  # 各组（有 hue 时为各 hue）
    "radar":       PALETTES["teal6"],                   # 各模型
    "radar_ring":  PALETTES["ring6"],                   # 外圈色块（各指标）
    "stacked_bar": PALETTES["red_blue4"],               # 各层（自下而上）
    "donut":       PALETTES["sci6"],                    # 各扇区
    "smooth_area": PALETTES["sunset5"],                 # 各层（自下而上）
}
DEFAULT_CMAPS = {
    "corr_heatmap": "blue_red",
    "q_heatmap": "q_heat",
    "confusion_heatmap": "Blues",
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
    解析配色。spec 可以是：None（依次用 全局覆盖 -> 主题 -> classic 默认）、配色名、颜色列表、单个颜色。
    n 不为 None 时，返回长度恰好为 n 的列表（不够则循环使用）。
    """
    if spec is None:
        spec = _override["palette"]
    if spec is None:
        spec = _theme_palette_spec(key, n)
    colors = _to_list(spec)
    if n is not None:
        colors = [colors[i % len(colors)] for i in range(n)]
    return colors


def get_cmap(spec=None, key=None, n=None):
    """解析色表。spec 可以是：None、已注册名、matplotlib 色表名、颜色列表、Colormap 对象。"""
    if spec is None:
        spec = _override["cmap"]
    if spec is None:
        spec = _theme_cmap_spec(key)
    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, str):
        if spec in _CMAPS:
            colors, nb = _CMAPS[spec]
            return LinearSegmentedColormap.from_list(spec, colors, N=n or nb or 256)
        return plt.get_cmap(spec)              # matplotlib 自带名称，如 "viridis" / "Blues_r"
    return LinearSegmentedColormap.from_list("custom", _check(spec), N=n or 256)


def use(palette=None, cmap=None):
    """全局覆盖所有图的离散配色 / 色表（优先于主题）。不带参数 = 取消覆盖，回到主题。"""
    if palette is not None:
        _to_list(palette)
    if cmap is not None:
        get_cmap(cmap, key="corr_heatmap")
    _override["palette"] = palette
    _override["cmap"] = cmap


# ----------------------------------------------------------------------
# 4. 主题：一组相互匹配的颜色，各种图的默认配色都由它派生
#    cat     离散主色序列（柱、箱、饼、雷达……按顺序取）
#    accent  强调色（拟合线、双轴的另一侧、对比色）
#    seq     连续色表锚点：浅 -> 深（hexbin、山脊图）
#    div     发散色表 5 个锚点：负 -> 中性 -> 正（相关性热力图、Q 版热力图、堆叠面积）
#    overrides  {函数名: 配色名/颜色列表/色表} 个别图用特殊配色，其余仍跟主题
# ----------------------------------------------------------------------
def _mix(c, other, t):
    """把颜色 c 向 other 混合比例 t（0~1），返回 '#RRGGBB'。"""
    a, b = np.array(to_rgb(c)), np.array(to_rgb(other))
    return to_hex(a * (1 - t) + b * t)


def _tint(c, t=0.55):
    return _mix(c, "#ffffff", t)


THEMES = {
    "classic": None,   # 特殊：各图使用 DEFAULTS / DEFAULT_CMAPS（原 notebook 的配色，彼此不统一）
    # 海洋：蓝 + 青绿为主，珊瑚红点缀（默认）
    "ocean": dict(
        cat=["#2B6A99", "#7DB9DE", "#7FC4B8", "#F2B880", "#E8836B", "#B3A4D0"],
        accent="#D9534F",
        seq=["#EAF3FA", "#7DB9DE", "#1B4F7A"],
        div=["#2B6A99", "#A9CFE6", "#F7F4EA", "#F2B880", "#D9534F"]),
    # 日落：暖橙红为主，深蓝点缀
    "sunset": dict(
        cat=["#C8553D", "#F28F3B", "#F6C667", "#8FB8D8", "#3D5A80", "#9CC5A1"],
        accent="#3D5A80",
        seq=["#FFF4E0", "#F6A04D", "#A83A2A"],
        div=["#3D5A80", "#9DBBD6", "#FBF3E4", "#F6A04D", "#B8412E"]),
    # 森林：绿 + 土黄，砖红点缀
    "forest": dict(
        cat=["#2F6B4F", "#7FB685", "#C9DFA4", "#E9C46A", "#D98B5F", "#6C8EAD"],
        accent="#B5483A",
        seq=["#EEF5E6", "#8CC08F", "#255C42"],
        div=["#2F6B4F", "#A8D2AA", "#F6F3E7", "#E9C46A", "#B5483A"]),
    # Nature 风：高饱和，适合投稿期刊
    "nature": dict(
        cat=["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"],
        accent="#E64B35",
        seq=["#EEF3F8", "#8491B4", "#2E3F6B"],
        div=["#3C5488", "#9FB1D4", "#F6F3EE", "#F39B7F", "#C9341F"]),

    # ---- 用户提供的 11 色渐变（蓝 -> 青 -> 绿 -> 黄 -> 橙 -> 红）----
    # cat：从 11 色里挑出相邻差异大的 8 个，保证分类图好区分；div：完整 11 色原序，热力图/面积图用
    "dawn": dict(
        cat=["#7b95c6", "#f59c7c", "#67a583", "#49c2d9", "#c85e62", "#a2c986", "#fded95", "#a1d8e8"],
        accent="#f47254",
        seq=["#fded95", "#f59c7c", "#c85e62"],
        div=["#7b95c6", "#49c2d9", "#a1d8e8", "#67a583", "#a2c986", "#d0e2c0",
             "#fded95", "#ffc1a6", "#f59c7c", "#f47254", "#c85e62"]),
    # ---- 用户提供的 8 色马卡龙 ----
    "macaron": dict(
        cat=["#5b99c5", "#faa256", "#badaba", "#f0afaf", "#c2bfd7", "#cde0cc", "#f7bfbf", "#b1ddf0"],
        accent="#faa256",
        seq=["#e6f1f8", "#8fbcdc", "#3f7aa8"],
        div=["#5b99c5", "#b1ddf0", "#f6f3ee", "#f0afaf", "#f08a3a"]),
    # ---- Nature 2025 论文配色“日出印象”（色号取自小红书笔记截图：#385a75 #6192b5 #98b4ce #eec0a3 #d2d0dc #f2d9be #f0e4d2 #f4f6f1）----
    # cat：蓝系 + 蜜桃/奶油，按相邻差异大重排；accent 与发散色表暖端的 #d98f62 是在 #eec0a3 基础上加深的（原色太浅，线条看不清）
    "sunrise_impression": dict(
        cat=["#6192b5", "#eec0a3", "#385a75", "#98b4ce", "#d2d0dc", "#f2d9be"],
        accent="#d98f62",
        seq=["#f4f6f1", "#98b4ce", "#385a75"],
        div=["#385a75", "#6192b5", "#98b4ce", "#f4f6f1", "#f2d9be", "#eec0a3", "#d98f62"]),
    # ---- 梵高《奥维尔附近的平原》蓝-黄-绿（色号取自小红书笔记截图：83a0bf baced3 ebd976 c8b015 abc08d 879e48）----
    # cat 保持截图原顺序；accent 直接用画中芥黄 #c8b015；发散色表两端 #5b7ca3 / #9c8a0e 是加深的（原色太浅，热力图对比弱）；seq 末端 #5f7a2e 也是加深的橄榄绿
    "auvers": dict(
        cat=["#83a0bf", "#baced3", "#ebd976", "#c8b015", "#abc08d", "#879e48"],
        accent="#c8b015",
        seq=["#f3f2e0", "#abc08d", "#5f7a2e"],
        div=["#5b7ca3", "#83a0bf", "#baced3", "#f6f4e6", "#ebd976", "#c8b015", "#9c8a0e"]),
    # ---- 莫奈《睡莲》黄绿 + 灰蓝（色号取自小红书笔记截图：7c9559 90ac7c bdbb55 deb956 9dbdd2 779ebd）----
    # 尽量还原：cat / accent / 发散色表全部用原色，顺序与截图柱状图 A~F 一致；只有中性色 #f4f4ec 和 seq 的浅端 #eef4f8 是我补的
    "waterlily": dict(
        cat=["#7c9559", "#90ac7c", "#bdbb55", "#deb956", "#9dbdd2", "#779ebd"],
        accent="#deb956",
        seq=["#eef4f8", "#9dbdd2", "#779ebd"],
        div=["#779ebd", "#9dbdd2", "#f4f4ec", "#bdbb55", "#deb956"]),
    # ---- 小红书笔记截图的三套蓝绿配色（色号均取自截图）----
    # umap：UMAP 细胞类型四色。发散色表的中性色 #f4f8f2、seq 浅端 #e6f1f8 是补的
    "umap": dict(
        cat=["#367DB0", "#3D9F3C", "#9DC7DD", "#9ED17B"],
        accent="#3D9F3C",
        seq=["#e6f1f8", "#9DC7DD", "#367DB0"],
        div=["#367DB0", "#9DC7DD", "#f4f8f2", "#9ED17B", "#3D9F3C"]),
    # blue_ramp：蓝色渐变（深 -> 浅，适合剂量/时间等有序分组），强调色用截图里的绿 #519D78；发散色表 蓝 -> 近白 -> 绿
    "blue_ramp": dict(
        cat=["#04579B", "#3492B2", "#58B8D1", "#96C2D4", "#BAD2E1", "#D8E5F7"],
        accent="#519D78",
        seq=["#DBF1FA", "#58B8D1", "#04579B"],
        div=["#04579B", "#3492B2", "#ACEEFE", "#F7FEF0", "#BFE8C1", "#8BCF8B", "#519D78"]),
    # green_ramp：绿色渐变（深 -> 浅），强调色用截图里的蓝 #5385BD；发散色表 绿 -> 近白 -> 蓝
    "green_ramp": dict(
        cat=["#519D78", "#8BCF8B", "#92C2A6", "#AADCA9", "#C4E9CA", "#CEEFCC"],
        accent="#5385BD",
        seq=["#F3FBF2", "#8BCF8B", "#519D78"],
        div=["#519D78", "#8BCF8B", "#DDF3DE", "#D6F6FF", "#58B8D1", "#5385BD"]),
    # ---- monet：通用科研主题。从莫奈名作的主色里取色相（睡莲的蓝、日出的橙、吉维尼的青绿、罂粟的玫红、干草堆的金黄、薰衣草紫），
    # 再手工调整明度/饱和度，使 6 个主色在正常视觉与三种色盲模拟下都分得开（相邻最小色差 ≈ 12~30，与 Okabe-Ito 同量级）。
    # 非原画取样。cat 前 6 个为主色，第 7、8 个（深海蓝、浅天蓝）用于类别 > 6 时。
    "monet": dict(
        cat=["#3D78AE", "#E8703A", "#4F9F8A", "#D4607A", "#EDCB62", "#B4A6D8", "#2C4A6E", "#8FB9D6"],
        accent="#E8703A",
        seq=["#EAF2F8", "#8FB6D6", "#2C4A6E"],
        div=["#2F6495", "#8FB6D6", "#F6F1E7", "#F2B07F", "#C85A25"]),
    # ---- 以下为 Paul Tol 色板（色号来自 tueplots 文档），seq/div 为配套推导 ----
    "tol_bright": dict(
        cat=["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"],
        accent="#EE6677", seq=["#EEF3F9", "#7FA3CC", "#2A4D7A"],
        div=["#2A5C9A", "#99BBDD", "#F7F4EE", "#EE9AA6", "#C4364F"]),
    "tol_vibrant": dict(
        cat=["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"],
        accent="#CC3311", seq=["#E6F3FA", "#33BBEE", "#00507F"],
        div=["#0077BB", "#99D5F0", "#F7F4EE", "#F5B087", "#CC3311"]),
    "tol_muted": dict(
        cat=["#332288", "#88CCEE", "#44AA99", "#117733", "#DDCC77", "#CC6677", "#AA4499", "#882255"],
        accent="#CC6677", seq=["#ECEAF5", "#8E86C4", "#332288"],
        div=["#332288", "#9F9AD0", "#F7F4EE", "#E0A0AA", "#882255"]),
    # ---- 以下为医学/综合期刊风格（ggsci 色号，按记忆录入，未逐一对照官方来源，用前请自行核对）----
    "aaas": dict(      # Science 系
        cat=["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#5F559B"],
        accent="#EE0000", seq=["#ECEEF6", "#8A92C4", "#3B4992"],
        div=["#3B4992", "#A5ABD2", "#F6F3EE", "#F08C8C", "#BB0000"]),
    "nejm": dict(
        cat=["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD"],
        accent="#BC3C29", seq=["#EAF3F9", "#6FA9D2", "#0A5B8E"],
        div=["#0072B5", "#9CC8E2", "#F7F3EA", "#E9A383", "#BC3C29"]),
    "lancet": dict(
        cat=["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91"],
        accent="#ED0000", seq=["#E8EFF7", "#6C97C6", "#00468B"],
        div=["#00468B", "#9DB8DA", "#F6F3EE", "#F59A8E", "#AD002A"]),
    "jama": dict(
        cat=["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599"],
        accent="#B24745", seq=["#E9EEF0", "#7C98A1", "#374E55"],
        div=["#374E55", "#9DB2B8", "#F6F2EA", "#E3AE7A", "#B24745"]),
    # ---- 自配的低饱和主题 ----
    "slate": dict(     # 石板灰蓝 + 琥珀，稳重
        cat=["#2F3E4E", "#5B7A99", "#9DB4C8", "#D9A441", "#C0504D", "#6E9F8D"],
        accent="#C0504D", seq=["#EDF1F5", "#7F9BB5", "#2F3E4E"],
        div=["#2F3E4E", "#9DB4C8", "#F6F3EC", "#E8C27A", "#C0504D"]),
    "lavender": dict(  # 薰衣草紫 + 玫粉
        cat=["#6C5B9E", "#A99CCB", "#E8A0BF", "#7FB8C9", "#F3C98B", "#8DAA91"],
        accent="#D9577F", seq=["#F1EEF8", "#A99CCB", "#4B3D7A"],
        div=["#4B3D7A", "#B5ABD6", "#F7F3F0", "#F0B3C8", "#D9577F"]),
    "earth": dict(     # 大地色：赭、麦、橄榄、靛
        cat=["#8C5E3C", "#C4975A", "#D9C79C", "#6B8F71", "#4A6B7C", "#B5543C"],
        accent="#B5543C", seq=["#F6F0E4", "#C4975A", "#6B4226"],
        div=["#4A6B7C", "#A5BAC3", "#F6F1E6", "#D9B27A", "#B5543C"]),
    # 注意：先 palettes.set_theme(...)，各函数的 cat 取前 N 个；类别 > 6~8 个时颜色会循环，建议换 Tol Muted
    # 色盲友好（Okabe-Ito）
    "okabe": dict(
        cat=["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
        accent="#D55E00",
        seq=["#EAF3FA", "#56B4E9", "#004C7A"],
        div=["#0072B2", "#9CCBE6", "#F5F2E8", "#F0C060", "#D55E00"]),
}
_THEME_OVERRIDES = {}                      # 当前主题生效的 overrides（register 时的 + set_theme 传入的）
_current = {"name": "ocean", "overrides": {}}

_NEUTRAL_MID = "#F0F0F0"


def _fill_theme(cat, accent=None, seq=None, div=None):
    """补全缺省项：只给 cat 就能得到一个完整主题。"""
    cat = _check(cat, "主题主色")
    accent = accent or cat[min(4, len(cat) - 1)]
    to_rgb(accent)
    seq = _check(seq, "seq") if seq else [_tint(cat[0], 0.9), _tint(cat[0], 0.4), _mix(cat[0], "#000000", 0.35)]
    div = _check(div, "div") if div else [cat[0], _tint(cat[0], 0.6), "#F7F4EA", _tint(accent, 0.5), accent]
    return dict(cat=cat, accent=accent, seq=seq, div=div)


def register_theme(name, cat, accent=None, seq=None, div=None, overrides=None):
    """
    注册一个主题。只需要 cat（主色序列），其余自动推导；想精细控制再给 accent / seq / div。
        palettes.register_theme("mine", cat=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
    overrides: {函数名: 配色} —— 个别图用特殊配色，如 {"bar": "vivid3", "corr_heatmap": "viridis"}
    """
    t = _fill_theme(cat, accent, seq, div)
    t["overrides"] = dict(overrides or {})
    THEMES[name] = t
    return t


def _active():
    t = THEMES[_current["name"]]
    return t


def _theme_palette_spec(key, n):
    name = _current["name"]
    ov = {**(THEMES[name] or {}).get("overrides", {}), **_current["overrides"]}
    if key in ov:
        return ov[key]
    t = THEMES[name]
    if t is None:                                   # classic
        return DEFAULTS.get(key, PALETTES["sci6"])        # 新图未登记时的兜底
    cat, acc = t["cat"], t["accent"]
    contrast = [cat[0], acc] + cat[1:]
    roles = {
        "regression":  [cat[0], acc, acc],          # 散点, 拟合线, 置信带
        "joint":       [cat[0], acc, _mix(cat[0], "#000000", 0.15)],
        "pair":        contrast,
        "hexbin":      [cat[0]],
        "dual_axis":   [acc, cat[0]],               # 左轴, 右轴
        "radar_ring":  [_tint(c, 0.5) for c in cat],
    }
    if key == "smooth_area":                        # 自下而上沿发散色表取色
        m = n or 5
        cm = LinearSegmentedColormap.from_list("layers", t["div"])
        return [to_hex(cm(v)) for v in (np.linspace(0, 1, m) if m > 1 else [0.0])]
    return roles.get(key, cat)                      # 其余（ellipse/box/bar/grouped_bar/radar/stacked_bar/donut/新图）都用 cat


def _theme_cmap_spec(key):
    name = _current["name"]
    ov = {**(THEMES[name] or {}).get("overrides", {}), **_current["overrides"]}
    if key in ov:
        return ov[key]
    t = THEMES[name]
    if t is None:
        return DEFAULT_CMAPS.get(key, "viridis")
    if key == "corr_heatmap":
        return LinearSegmentedColormap.from_list("theme_div", t["div"], N=20)
    if key == "q_heatmap":
        return [t["div"][1], _NEUTRAL_MID, t["div"][-1]]
    return LinearSegmentedColormap.from_list("theme_seq", t["seq"])      # hexbin / ridgeline / 新增连续图


def set_theme(name="ocean", overrides=None, sync_matplotlib=True):
    """
    切换主题。之后所有图的默认配色都从该主题派生（单次 palette= / palettes.use() 仍然优先）。
        palettes.set_theme("sunset")
        palettes.set_theme("ocean", overrides={"bar": "vivid3"})       # 主题不变，只有 bar 用特殊配色
        palettes.set_theme("classic")                                  # 回到原 notebook 的配色
    sync_matplotlib=True 时同步 matplotlib 默认色循环 / 默认色表，用 plt.plot 等直接画的图也跟主题一致。
    """
    if name not in THEMES:
        raise ValueError(f"未知主题 {name!r}。可用: {sorted(THEMES)}；或用 register_theme 注册。")
    _current["name"] = name
    _current["overrides"] = dict(overrides or {})
    if sync_matplotlib:
        import matplotlib as mpl
        from cycler import cycler
        t = THEMES[name]
        cols = t["cat"] if t else ["#87C3E4", "#F48892", "#AED594", "#D693BE", "#F5B3A5", "#68BED9"]
        mpl.rcParams["axes.prop_cycle"] = cycler(color=cols)
    return name


def get_theme():
    """当前主题名。"""
    return _current["name"]


def list_themes():
    return sorted(THEMES)


def show_themes(names=None):
    """画出主题总览：每行 = 主色序列 + 强调色 + 连续色表 + 发散色表。返回 Figure。"""
    names = names or list(THEMES)
    fig, axes = plt.subplots(len(names), 1, figsize=(6.4, 0.42 * len(names) + 0.3))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.98, bottom=0.02, hspace=0.35)
    keep = dict(_current)
    for ax, nm in zip(axes, names):
        ax.axis("off"); ax.set_xlim(0, 24); ax.set_ylim(0, 1)
        _current["name"], _current["overrides"] = nm, {}
        cols = get_palette(None, "bar", n=6) if THEMES[nm] else get_palette(None, "box", n=6)
        for i, c in enumerate(cols):
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=c))
        acc = get_palette(None, "regression")[1]
        ax.add_patch(plt.Rectangle((6.2, 0), 0.9, 1, color=acc))
        for x0, cm in ((7.6, get_cmap(None, "hexbin")), (15.8, get_cmap(None, "corr_heatmap", n=256))):
            ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cm,
                      extent=(x0, x0 + 7.6, 0, 1))
        ax.set_xlim(0, 24)
        ax.text(-0.01, 0.5, nm + (" *" if nm == keep["name"] else ""), transform=ax.transAxes,
                ha="right", va="center")
    _current.update(keep)
    return fig


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
