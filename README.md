# sci_viz —— 科研绘图工具箱

一行调用画出论文级图表：英文/数字 Times New Roman，中文 SimSun，公式 LaTeX 字体；配色可自定义；以后加新图只需要写一个函数。

```python
import plots as pl

pl.bar({"Control": a, "Treat A": b, "Treat B": c}, ylabel="Measured Value (Units)")
pl.regression(x, y, title="Group A")
pl.corr_heatmap(df)
pl.save(fig_or_ax, "figures/fig1")        # 同时存 png + pdf
```

## 目录结构

```
sci_viz/
├── utils/
│   ├── pubstyle.py       # 样式：字体回退、rcParams、中文+公式混排（mixed_label / mixed_legend）
│   └── palettes.py       # 配色注册表：离散配色、色表、各图默认配色、自定义
├── plots/
│   ├── __init__.py       # 统一出口：import plots as pl
│   ├── relation.py       # regression  joint  pair  ellipse  hexbin  dual_axis
│   ├── matrix.py         # corr_heatmap  q_heatmap
│   ├── compare.py        # box  bar  grouped_bar  radar
│   ├── composition.py    # donut  stacked_bar  smooth_area
│   ├── distribution.py   # ridgeline
│   ├── _common.py        # 共用小工具（取 ax、标签、分组数据、保存）
│   └── _template.py      # 新增一种图的模板
├── test.ipynb            # 完整测试/教程：每种图的生成方式、并排布局、配色、保存
├── examples/gallery.py   # 把每个函数都画一遍（示例 + 冒烟测试）
└── requirements.txt
```

脚本放在仓库根目录就能直接 `import plots as pl`。放在子目录（如 `notebooks/`）时先加一行：

```python
import sys; sys.path.append("..")     # 按层级调整，指向仓库根目录
```

## 函数一览

所有函数的第 1 个参数是数据，**第 2 个位置参数永远是 `ax`**（`joint` 是 `fig`），其余参数都是关键字参数。
axes 级函数不传 `ax` 时自动新建画布，返回 `ax`；figure 级函数返回 `fig`。

| 函数 | 用途 | 数据输入 | 返回 | `palette` 的含义 |
|---|---|---|---|---|
| `regression(x, y)` | 散点 + 回归线 + 置信带 + R²/p/N（图例自动避开统计框，可用 `legend_loc` 指定）| 两个一维数组 | ax | [散点, 拟合线, 置信带] |
| `joint(x, y)` | 散点回归 + 边缘直方图 | 两个一维数组 | fig | [散点与直方图, 拟合线, 散点边缘] |
| `pair(df, vars, hue)` | 多变量散点矩阵 | DataFrame | fig | 各分组 |
| `ellipse({组名: (x, y)})` | 分组散点 + 置信椭圆 | dict | ax | 各分组 |
| `hexbin(x, y)` | 六边形分箱 + 边缘直方图 | 两个一维数组 | fig | [边缘直方图]；`cmap` 控制分箱色表 |
| `dual_axis(x, y1, y2)` | 双 Y 轴折线 | 三个一维数组 | (ax左, ax右) | [左轴, 右轴] |
| `corr_heatmap(df)` | 相关系数热力图 | DataFrame（数值列） | ax | 用 `cmap`（默认 `blue_red`） |
| `q_heatmap(df)` | Q 版圆角热力图 | DataFrame | ax | 用 `cmap` / `high_color` |
| `box(data)` | 分组箱线图 | dict / DataFrame / 列表 | ax | 各组 |
| `bar(data)` | 均值 ± SEM 柱状图 + 抖动散点 | dict / DataFrame / 列表 | ax | 各组 |
| `grouped_bar(df, category)` | 并列柱状图 | DataFrame | ax | 各系列 |
| `radar(data, criteria)` | 环形雷达图 | dict {模型: 各指标值} | ax | 各模型；`ring_palette` 为外圈 |
| `donut(values, labels)` | 甜甜圈饼图 | 两个列表 | ax | 各扇区 |
| `stacked_bar(df, category)` | 堆积柱状图 | DataFrame | ax | 各层（自下而上） |
| `smooth_area(df)` | 平滑百分比堆叠面积图 | DataFrame（首列为类别） | ax | 各层（自下而上） |
| `ridgeline(df, by, column)` | 山脊图 | 长表 DataFrame | fig | 用 `cmap`（默认 `autumn`） |

每个函数的详细参数看 docstring（`help(pl.bar)`）。想看全部效果：`python examples/gallery.py`，图保存在 `examples/output/`。

**分组数据**（`box`、`bar`）可以直接传：`{"A": arr1, "B": arr2}`、`df`（每列一组）或 `[arr1, arr2]`，组名用 `labels=[...]` 指定。

**组合子图**：

```python
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
pl.regression(x1, y1, a1, title="Group A")
pl.regression(x2, y2, a2, title="Group B")
fig.tight_layout()

# joint 是 figure 级，用 subfigures 排版
fig = plt.figure(figsize=(7.2, 3.1)); s1, s2 = fig.subfigures(1, 2)
pl.joint(x1, y1, s1); pl.joint(x2, y2, s2)
```

## 配色

**单次指定**：`palette` 可以是配色名、颜色列表、单个颜色。

```python
pl.bar(data, palette="sci6")
pl.bar(data, palette=["#E63946", "#457B9D", "#1D3557"])
pl.corr_heatmap(df, cmap="viridis")                       # matplotlib 色表名
pl.corr_heatmap(df, cmap=["#2a9d8f", "#ffffff", "#e76f51"])   # 自己的发散色
```

**注册自己的配色**，以后按名字用：

```python
pl.register_palette("mine", ["#264653", "#2a9d8f", "#e9c46a"])
pl.register_cmap("mine_div", ["#2a9d8f", "#ffffff", "#e76f51"])       # 连续/发散色表
pl.bar(data, palette="mine")
```

**全局覆盖**（所有图的离散配色 / 色表都改用它）：

```python
pl.palettes.use("mine")          # 全部用 mine
pl.palettes.use()                # 恢复各图原来的默认配色
```

**看有哪些配色**：`pl.palettes.show()` 会画出所有已注册配色的色块；`pl.palettes.list_palettes()` 返回名称。

内置配色名（全部来自原 notebook，数字 = 颜色个数）：`soft4` `teal6` `vivid3` `red_blue4` `sci6` `sunset5` `ring6` `sky_rose` `okabe_ito`；色表：`blue_red` `q_heat`。
各图的默认配色写在 `utils/palettes.py` 的 `DEFAULTS` 里，想永久改某张图的默认色，改那里即可。

> 颜色个数不够时按顺序循环使用；传了无法识别的颜色或配色名会直接报错并列出可用名称。

## 字体与公式

导入 `plots` 时自动启用 pubstyle：英文/数字 Times New Roman，中文自动回退 SimSun，`$...$` 里的公式用 Computer Modern。
需要调整：

```python
pl.use_style(cn_font="SimHei", math_fontset="stix")     # 参数见 utils/pubstyle.py 的 use()
```

**“中文 + 公式”混排的标签**（如 `r"样本量 $n$ (Sample size)"`）在 `title / xlabel / ylabel` 里会自动拆段渲染，不会出现方块。
`legend` 里的中文公式混排请用 `ps.mixed_legend`。

之后不要再调用 `sns.set_theme` / `sns.set_style`，它们会把字体设置重置掉。

## 保存

```python
pl.save(ax, "figures/fig1")                     # 不带扩展名 -> 同时存 png 和 pdf（dpi=300）
pl.save(fig, "figures/fig1.pdf")                # 带扩展名 -> 只存这一种
pl.save(ax, "figures/fig1", formats=("png", "svg"), dpi=600)
```

`pl.save` 接受 Figure / Axes / SubFigure，以及 `dual_axis` 返回的 `(ax1, ax2)`。

## 新增一种图

四步，详见 `plots/_template.py`：

1. 在 `plots/` 对应模块里写函数（遵守“数据第一、`ax` 第二、其余关键字”的约定）
2. 在 `utils/palettes.py` 的 `DEFAULTS` / `DEFAULT_CMAPS` 里加默认配色，key = 函数名
3. 在 `plots/__init__.py` 里 import 并写进 `__all__`
4. 在 `examples/gallery.py` 里加 `demo_xxx` 并登记到 `GALLERY`，跑一遍 `python examples/gallery.py xxx` 即是测试

## 依赖与已知问题

- `pip install -r requirements.txt`；matplotlib 需 ≥ 3.6（字体回退）
- 只有 `ridgeline` 依赖 joypy，且 **joypy 与 pandas ≥ 3 不兼容**（报 `'generator' object is not subscriptable`），需 `pip install "pandas<3"`
- 找不到 Times New Roman / SimSun 时 `ps.use()` 会给出警告；macOS 字体装在 `~/Library/Fonts` 即可被自动注册

## 与原 notebook 的差异

- `regression` 的置信带：原代码用 `t.ppf(0.95)`，实际是 90% 双侧区间却标成了 “95% CI”；现在按 `ci=0.95` 正确取 `t.ppf(0.975)`，带宽会略宽一点。想复现旧效果可传 `ci=0.90`
- 画布按论文尺寸等比缩小，字号、粗体统一交给 pubstyle，刻度方向为向外
- `radar` 不再修改传入的数据；`dual_axis` 的横轴刻度旋转已修复（原来设在了不显示的右轴上）
- `q_heatmap` 色条的 Low / High 现在固定落在两端
