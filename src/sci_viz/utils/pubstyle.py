"""
pubstyle —— 论文级 matplotlib 排版（英文/数字 Times New Roman，中文 SimSun，公式 Computer Modern）

用法：
    import pubstyle as ps
    ps.use()                                   # 一行启用全部样式
    fig, ax = plt.subplots()
    ax.plot(x, y, color=ps.COLORS["blue"])
    ps.clean_axis(ax)
    ps.panel_label(ax, "(a)")
    ps.mixed_label(ax, r"样本量 $n$ (Sample size)", where="x")
"""
import glob
import os
import re
import warnings

import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import (TextArea, DrawingArea, HPacker, VPacker,
                                  AnchoredOffsetbox)

__all__ = ["COLORS", "use", "clean_axis", "panel_label",
           "mixed_label", "mixed_legend"]

# ----------------------------------------------------------------------
# 配色（Okabe-Ito 色盲友好色板 + 中性色）
# ----------------------------------------------------------------------
COLORS = {
    "blue": "#0072B2", "sky": "#56B4E9", "green": "#009E73",
    "orange": "#E69F00", "vermillion": "#D55E00", "purple": "#CC79A7",
    "yellow": "#F0E442", "black": "#222222", "gray": "#7A7A7A",
    "light_gray": "#D9D9D9", "panel": "#F7F7F7",
}

# ----------------------------------------------------------------------
# 字体注册：字体装在用户目录但 matplotlib 没识别时，自动找文件并注册
# ----------------------------------------------------------------------
_FONT_DIRS = [
    "~/Library/Fonts", "/Library/Fonts", "/System/Library/Fonts/Supplemental",
    "C:/Windows/Fonts", "/usr/share/fonts", "~/.fonts", "~/.local/share/fonts",
]
_FONT_FILES = {                       # 字体名 -> 可能的文件名（小写，不含扩展名）
    "SimSun": ["simsun", "simsunb"],
    "Times New Roman": ["times new roman", "times"],
}


def _ensure_font(name):
    """名字已被识别就直接返回 True；否则在常见目录里找文件并注册。"""
    if name in {f.name for f in fm.fontManager.ttflist}:
        return True
    stems = _FONT_FILES.get(name, [name.lower()])
    for d in _FONT_DIRS:
        d = os.path.expanduser(d)
        for path in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            base, ext = os.path.splitext(os.path.basename(path).lower())
            if ext in (".ttf", ".ttc", ".otf") and base in stems:
                try:
                    fm.fontManager.addfont(path)
                except Exception:
                    continue
                if name in {f.name for f in fm.fontManager.ttflist}:
                    return True
    return False


# ----------------------------------------------------------------------
# 一键启用样式
# ----------------------------------------------------------------------
def use(cn_font="SimSun", en_font="Times New Roman", math_fontset="cm",
        seaborn_context="paper", extra_rc=None):
    """
    启用论文样式。
    cn_font / en_font : 中文字体 / 英文数字字体（en 在前，缺字形时回退到 cn）
    math_fontset      : 公式字体，"cm" = LaTeX 默认；想要接近 Times 用 "stix"
    seaborn_context   : "paper" / "notebook" / ...；None 则不调用 seaborn。
                        注意：seaborn 的 context 会改字号，放在最后以保持与原版一致
    extra_rc          : dict，额外覆盖的 rcParams
    """
    for name in (en_font, cn_font):
        if not _ensure_font(name):
            warnings.warn(f"未找到字体 {name!r}，相关文字可能显示为方块或被替换。",
                          stacklevel=2)

    mpl.rcParams.update({
        "font.family": [en_font, cn_font],     # 直接写成列表才会触发回退
        "axes.unicode_minus": False,
        "mathtext.fontset": math_fontset,

        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.25,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    if seaborn_context:
        try:
            import seaborn as sns
            sns.set_context(seaborn_context)
        except ImportError:
            pass

    if extra_rc:
        mpl.rcParams.update(extra_rc)


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def clean_axis(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    return ax


def panel_label(ax, label, x=-0.14, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


_MATH_SPLIT = re.compile(r"(\$[^$]+\$)")


def _split_math(text):
    return [p.strip() for p in _MATH_SPLIT.split(text) if p.strip()]


# ----------------------------------------------------------------------
# 中文 + 公式混排：标题 / 坐标轴标签
# ----------------------------------------------------------------------
def mixed_label(ax, text, where="x", offset=0.13, **textprops):
    """
    where : "x" | "y" | "title"
    offset: 离坐标轴的距离（轴比例单位），按刻度标签占的空间微调
    """
    parts = _split_math(text)

    if where == "y":
        props = dict(size=mpl.rcParams["axes.labelsize"], rotation=90,
                     ha="left", va="bottom", **textprops)
        boxes = [TextArea(p, textprops=props) for p in parts][::-1]
        box = VPacker(children=boxes, align="bottom", pad=0, sep=3)
        loc, anchor = "center right", (-offset, 0.5)
    else:
        if where == "title":
            props = dict(size=mpl.rcParams["axes.titlesize"],
                         weight=mpl.rcParams["axes.titleweight"], **textprops)
            loc, anchor = "lower center", (0.5, 1 + offset)
        else:
            props = dict(size=mpl.rcParams["axes.labelsize"], **textprops)
            loc, anchor = "upper center", (0.5, -offset)
        boxes = [TextArea(p, textprops=props) for p in parts]
        box = HPacker(children=boxes, align="baseline", pad=0, sep=3)

    ab = AnchoredOffsetbox(loc=loc, child=box, pad=0, borderpad=0, frameon=False,
                           bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
    ax.add_artist(ab)
    return ab


# ----------------------------------------------------------------------
# 中文 + 公式混排：图例
# ----------------------------------------------------------------------
def mixed_legend(ax, handles=None, labels=None, loc="upper right",
                 bbox_to_anchor=None, frameon=False, row_sep=4,
                 handle_len=22, **textprops):
    """
    handles / labels 不传则自动取 ax.get_legend_handles_labels()。
    loc 用字符串（不支持 "best"）；bbox_to_anchor 为轴比例坐标，可把图例挪到轴外。
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()

    fs = mpl.rcParams["legend.fontsize"]
    rows = []
    for h, lab in zip(handles, labels):
        da = DrawingArea(handle_len, fs * 1.2)
        if isinstance(h, Line2D):
            da.add_artist(Line2D([0, handle_len], [fs * 0.6] * 2,
                                 color=h.get_color(), ls=h.get_linestyle(),
                                 lw=h.get_linewidth(), alpha=h.get_alpha()))
            if h.get_marker() not in (None, "None", "", " "):
                da.add_artist(Line2D([handle_len / 2], [fs * 0.6],
                                     marker=h.get_marker(), ms=h.get_markersize(),
                                     color=h.get_color(), ls="none"))
        else:  # Patch / PolyCollection 等色块
            fc = h.get_facecolor()
            if hasattr(fc, "__len__") and len(fc) and hasattr(fc[0], "__len__"):
                fc = fc[0]
            ec = h.get_edgecolor()
            if hasattr(ec, "__len__") and len(ec) and hasattr(ec[0], "__len__"):
                ec = ec[0] if len(ec) else "none"
            da.add_artist(Rectangle((0, fs * 0.1), handle_len, fs, fc=fc, ec=ec,
                                    alpha=h.get_alpha()))

        txt = HPacker(children=[TextArea(p, textprops=dict(size=fs, **textprops))
                                for p in _split_math(lab)],
                      align="baseline", pad=0, sep=3)
        rows.append(HPacker(children=[da, txt], align="center", pad=0, sep=6))

    box = VPacker(children=rows, align="left", pad=0, sep=row_sep)
    ab = AnchoredOffsetbox(loc=loc, child=box, pad=0.3, borderpad=0.4,
                           frameon=frameon, bbox_to_anchor=bbox_to_anchor,
                           bbox_transform=ax.transAxes if bbox_to_anchor else None)
    if frameon:
        ab.patch.set_edgecolor("#CCCCCC")
        ab.patch.set_linewidth(0.6)
    ax.add_artist(ab)
    return ab
