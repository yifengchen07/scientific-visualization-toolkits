"""
画廊：用模拟数据把每个函数都画一遍，既是用法示例，也是冒烟测试。

    python examples/gallery.py                 # 画全部，保存到 examples/output/
    python examples/gallery.py bar box donut   # 只画指定的几个
    python examples/gallery.py --show          # 画完弹窗显示（默认只保存）
    python examples/gallery.py --list          # 列出所有图名
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sci_viz as pl

OUT = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------- 各图的演示
def demo_regression():
    np.random.seed(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x1 = np.random.rand(40) * 10
    pl.regression(x1, 2 * x1 + 5 + np.random.randn(40) * 2, a1, title="Group A: Strong Correlation")
    x2 = np.random.rand(40) * 10
    pl.regression(x2, 0.5 * x2 + 10 + np.random.randn(40) * 5, a2, title="Group B: Weak Correlation")
    fig.tight_layout()
    return fig


def demo_joint():
    np.random.seed(0)
    fig = plt.figure(figsize=(7.2, 3.1))
    a, b = fig.subfigures(1, 2, wspace=0.1)
    x1 = np.random.normal(50, 10, 200)
    pl.joint(x1, 0.8 * x1 + np.random.normal(0, 5, 200), a, xlabel="Control Group X", ylabel="Outcome Y")
    x2 = np.random.normal(40, 15, 200)
    pl.joint(x2, -0.4 * x2 + 100 + np.random.normal(0, 10, 200), b, xlabel="Treatment Group X",
             ylabel="Outcome Y")
    fig.suptitle("Comparative Joint Distribution Analysis", fontsize=11, fontweight="bold")
    return fig


def demo_pair():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({"Feature_A": np.random.normal(10, 2, n), "Feature_B": np.random.normal(20, 5, n),
                       "Feature_C": np.random.normal(30, 8, n),
                       "Group": np.random.choice(["Control", "Treatment"], n)})
    df["Feature_B"] += 0.6 * df["Feature_A"]
    df["Feature_C"] -= 0.4 * df["Feature_B"]
    return pl.pair(df, vars=["Feature_A", "Feature_B", "Feature_C"], hue="Group")


def _corr_data(n=100):
    np.random.seed(88)
    f1, f2, f3 = np.linspace(0, 10, n), np.linspace(10, 0, n), np.random.normal(5, 2, n)
    d = {}
    d["Var_A"] = f1 + np.random.normal(0, 0.5, n)
    d["Var_B"] = 0.9 * f1 + np.random.normal(0, 0.3, n)
    d["Var_C"] = 1.2 * f1 + np.random.normal(0, 0.8, n)
    d["Var_D"] = f2 + np.random.normal(0, 0.4, n)
    d["Var_E"] = 0.85 * f2 + np.random.normal(0, 0.2, n)
    d["Var_F"] = -0.95 * d["Var_A"] + np.random.normal(0, 0.5, n)
    d["Var_G"] = 0.7 * d["Var_D"] + 0.3 * f3 + np.random.normal(0, 0.5, n)
    d["Var_H"] = d["Var_B"] * 0.98 + np.random.normal(0, 0.05, n)
    d["Var_I"] = 0.5 * f1 + 0.5 * f3 + np.random.normal(0, 1.0, n)
    d["Var_J"] = np.random.normal(10, 2, n)
    return pd.DataFrame(d)


def demo_corr_heatmap():
    return pl.corr_heatmap(_corr_data(), title="heatmap").figure


def demo_ellipse():
    np.random.seed(42)
    x1 = np.random.rand(50) * 10
    y1 = 1.5 * x1 + 2 + np.random.randn(50) * 2
    x2 = np.random.rand(50) * 10
    y2 = -1.2 * x2 + 15 + np.random.randn(50) * 2
    ax = pl.ellipse({"Group A (High)": (x1, y1), "Group B (Low)": (x2, y2)},
                    palette=["#8EC8ED", "#D693BE"],
                    title="Scientific Analysis: Scatter with 95% Confidence Ellipses")
    return ax.figure


def demo_hexbin():
    np.random.seed(42)
    x, y = np.random.multivariate_normal([3.0, 6.0], [[0.2, 0.1], [0.1, 0.5]], 200).T
    return pl.hexbin(x, y, xlabel="sepal width (cm)", ylabel="sepal length (cm)")


def demo_dual_axis():
    np.random.seed(0)
    n = 574
    t = np.arange(n)
    dates = pd.date_range("1967-07-01", periods=n, freq="MS").strftime("%Y-%m-%d")
    ax1, _ = pl.dual_axis(dates, 10 + 3 * np.sin(t / 60) + np.random.randn(n) * 0.3,
                          6000 + 2500 * np.sin(t / 90) + t * 8, xlabel="Year",
                          ylabel1="Personal Savings Rate", ylabel2="# Unemployed (1000's)",
                          title="Personal Savings Rate vs Unemployed")
    ax1.figure.tight_layout()
    return ax1.figure


def demo_box():
    np.random.seed(10)
    data = {"Control": np.random.normal(100, 10, 200), "Group A": np.random.normal(90, 20, 200),
            "Group B": np.random.normal(110, 15, 200), "Group C": np.random.normal(105, 5, 200),
            "Group D": np.random.normal(115, 5, 200), "Group E": np.random.normal(95, 5, 200)}
    ax = pl.box(data, title="Experiment Results", ylabel="Value (Units)")
    ax.figure.tight_layout()
    return ax.figure


def demo_bar():
    np.random.seed(42)
    data = {"Control": np.random.normal(15, 3, 20), "Treat A": np.random.normal(20, 4, 20),
            "Treat B": np.random.normal(12, 2, 20)}
    ax = pl.bar(data, ylabel="Measured Value (Units)", seed=0)
    ax.figure.tight_layout()
    return ax.figure


def demo_grouped_bar():
    df = pd.DataFrame({"Time": ["Day 1", "Day 3", "Day 7"], "Control": [10, 15, 12],
                       "Treatment": [12, 22, 30]})
    ax = pl.grouped_bar(df, "Time", ylabel="Intensity (a.u.)")
    ax.figure.tight_layout()
    return ax.figure


def demo_radar():
    data = {"RF": [0.75, 0.82, 0.78, 0.80, 0.65, 0.72], "SVM": [0.65, 0.85, 0.60, 0.75, 0.55, 0.88],
            "XGB": [0.88, 0.70, 0.75, 0.72, 0.68, 0.75], "LGBM": [0.78, 0.75, 0.82, 0.78, 0.62, 0.70]}
    ax = pl.radar(data, ["Acc", "Prec", "Rec", "F1", "AUC", "MCC"],
                  palette=["#257D8B", "#ED8D5A", "#68BED9", "#EAA558"])
    ax.figure.tight_layout()
    return ax.figure


def demo_stacked_bar():
    df = pd.DataFrame({"Sample": ["Site A", "Site B", "Site C"], "Type_1": [40, 30, 20],
                       "Type_2": [35, 40, 30], "Type_3": [25, 30, 25]})
    ax = pl.stacked_bar(df, "Sample", ylabel="Relative Abundance (%)")
    ax.figure.tight_layout()
    return ax.figure


def demo_donut():
    ax = pl.donut([40, 30, 20, 10], ["Control", "Treatment A", "Treatment B", "Placebo"],
                  center_text="Cell\nViability")
    ax.figure.tight_layout()
    return ax.figure


def demo_smooth_area():
    df = pd.DataFrame({"Month": ["January", "February", "March", "April", "May", "June"],
                       "Shade 5": [39, 5, 31, 31, 32, 10], "Shade 4": [8, 35, 14, 18, 28, 18],
                       "Shade 3": [15, 12, 9, 7, 20, 32], "Shade 2": [23, 30, 13, 27, 12, 25],
                       "Shade 1": [15, 18, 33, 17, 8, 15]})
    ax = pl.smooth_area(df)
    ax.figure.tight_layout()
    return ax.figure


def demo_q_heatmap():
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(12, 4), index=[f"Gene_{i}" for i in range(12)],
                      columns=["Liver", "Muscle", "Brain", "Kidney"])
    return pl.q_heatmap(df).figure


def demo_ridgeline():
    np.random.seed(42)
    df = pd.DataFrame({"Year": np.repeat(range(2010, 2021), 100),
                       "Value": np.random.randn(1100).cumsum() + np.tile(np.linspace(0, 10, 11), 100)})
    return pl.ridgeline(df, by="Year", column="Value", title="this is title", xlabel="Value Axis")


# ---------------------------------------------------------------- 配色 / 公式的用法演示
def demo_custom_palette():
    """自定义配色：注册一套自己的颜色，之后按名字使用。"""
    pl.register_palette("mine", ["#264653", "#2a9d8f", "#e9c46a"])
    np.random.seed(1)
    data = {"A": np.random.normal(5, 1, 20), "B": np.random.normal(7, 1, 20), "C": np.random.normal(6, 1, 20)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    pl.bar(data, a1, palette="mine", title='palette="mine"')
    pl.box(data, ax=a2, palette=["#E63946", "#457B9D", "#1D3557"], title="palette=[...]")
    fig.tight_layout()
    return fig


def demo_mixed_label():
    """标签里“中文 + $公式$”混排：自动拆段渲染。"""
    np.random.seed(0)
    x = np.random.rand(40) * 10
    ax = pl.regression(x, 2 * x + 5 + np.random.randn(40) * 2,
                       xlabel=r"样本量 $n$ (Sample size)", ylabel=r"均方误差 $\mathrm{MSE}(\hat{\theta})$",
                       title=r"正态分布 $N(\mu,\ \sigma^2)$")
    return ax.figure


def demo_themes():
    """主题总览 + 同一批图换主题。"""
    return pl.palettes.show_themes()


def demo_palettes():
    return pl.palettes.show()


GALLERY = {
    "regression": demo_regression, "joint": demo_joint, "pair": demo_pair,
    "corr_heatmap": demo_corr_heatmap, "ellipse": demo_ellipse, "hexbin": demo_hexbin,
    "dual_axis": demo_dual_axis, "box": demo_box, "bar": demo_bar, "grouped_bar": demo_grouped_bar,
    "radar": demo_radar, "stacked_bar": demo_stacked_bar, "donut": demo_donut,
    "smooth_area": demo_smooth_area, "q_heatmap": demo_q_heatmap, "ridgeline": demo_ridgeline,
    "custom_palette": demo_custom_palette, "mixed_label": demo_mixed_label, "palettes": demo_palettes, "themes": demo_themes,
}


def main(argv):
    show = "--show" in argv
    names = [a for a in argv if not a.startswith("--")]
    if "--list" in argv:
        print("\n".join(GALLERY))
        return
    names = names or list(GALLERY)
    OUT.mkdir(exist_ok=True)
    for name in names:
        try:
            fig = GALLERY[name]()
        except KeyError:
            print(f"[skip] 没有 {name!r}，可用: {', '.join(GALLERY)}")
            continue
        except Exception as e:                         # 某张图失败不影响其他图
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            continue
        paths = pl.save(fig, OUT / name, formats=("png",), dpi=200)
        print(f"[ok]   {name:14s} -> {paths[0]}")
        if not show:
            plt.close(fig)
    if show:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
