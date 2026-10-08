"""
stats —— 分类 / 比较实验常用的统计小工具（纯 numpy / pandas / scipy，不画图）

    from sci_viz import stats
    stats.wilson_ci(k, n)                          # 比例的 Wilson 置信区间
    stats.binary_metrics(y_true, y_pred)           # 二分类指标表（Acc / Precision / Recall / Specificity / F1 / BalAcc / MCC + 置信区间）
    stats.rate_table(df, by="type", value="correct")   # 按组算比例 + 置信区间
    stats.confusion_table(y_true, y_pred)          # 混淆矩阵（DataFrame，行 = 真实，列 = 预测）
    stats.mcnemar_test(correct_a, correct_b)       # 两个模型在同一批样本上的配对比较
    stats.format_ci(0.656, 0.637, 0.675)           # -> "65.6 [63.7, 67.5]"

约定：除特别说明外，比例类结果用 0~1 的小数；画图函数（pl.rate_bar 等）内部自行换成百分数。
"""
import numpy as np
import pandas as pd
from scipy import stats as _st

__all__ = ["wilson_ci", "bootstrap_ci", "binary_metrics", "majority_baseline", "rate_table",
           "confusion_table", "mcnemar_test", "format_ci"]


# ----------------------------------------------------------------------
def wilson_ci(k, n, ci=0.95):
    """
    比例 k/n 的 Wilson 置信区间（比正态近似更稳，k 接近 0 或 n、n 较小时也可靠）。
    k, n 可以是标量或数组；n = 0 时返回 nan。返回 (lo, hi)，同形状，0~1 小数。
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = _st.norm.ppf(1 - (1 - ci) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    lo, hi = np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)
    lo, hi = np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)
    return (lo.item(), hi.item()) if lo.ndim == 0 else (lo, hi)


def bootstrap_ci(stat_fn, *arrays, n_boot=2000, ci=0.95, seed=0):
    """
    样本级 bootstrap 置信区间（百分位法）。
    stat_fn(*重采样后的 arrays) -> 标量；arrays 为等长的一维数组（如 y_true, y_pred）。
    返回 (点估计, lo, hi)。seed 固定则结果可复现。
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    est = stat_fn(*arrays)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat_fn(*[a[idx] for a in arrays])
    a = (1 - ci) / 2
    return est, float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))


# ----------------------------------------------------------------------
def _counts(yt, yp):
    tp = int(np.sum(yt & yp)); fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp)); tn = int(np.sum(~yt & ~yp))
    return tp, fp, fn, tn


def _f1(yt, yp):
    tp, fp, fn, _ = _counts(yt, yp)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan


def _bal_acc(yt, yp):
    tp, fp, fn, tn = _counts(yt, yp)
    r = tp / (tp + fn) if (tp + fn) else np.nan
    s = tn / (tn + fp) if (tn + fp) else np.nan
    return (r + s) / 2


def _mcc(yt, yp):
    tp, fp, fn, tn = _counts(yt, yp)
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else np.nan


def binary_metrics(y_true, y_pred, positive=1, ci=0.95, n_boot=2000, seed=0):
    """
    二分类指标表。positive 指定“正类”的取值（如有害 = 1，或字符串 "harmful"）。

    返回 DataFrame（index = 指标名），列：value / lo / hi / k / n
        Accuracy, Precision, Recall, Specificity —— 比例类，置信区间用 Wilson，k/n 是分子/分母
        F1, Balanced Acc., MCC                   —— 非比例类，置信区间用样本级 bootstrap（k/n 为空）
    数值均为 0~1 小数（MCC 为 -1~1）。附带 df.attrs["counts"] = dict(TP, FP, FN, TN)。
    """
    yt = np.asarray(y_true) == positive
    yp = np.asarray(y_pred) == positive
    if len(yt) != len(yp):
        raise ValueError(f"y_true 有 {len(yt)} 个，y_pred 有 {len(yp)} 个")
    tp, fp, fn, tn = _counts(yt, yp)
    N = len(yt)
    rows = {}
    for name, k, n in (("Accuracy", tp + tn, N), ("Precision", tp, tp + fp),
                       ("Recall", tp, tp + fn), ("Specificity", tn, tn + fp)):
        lo, hi = wilson_ci(k, n, ci)
        rows[name] = dict(value=k / n if n else np.nan, lo=lo, hi=hi, k=k, n=n)
    for name, fn_ in (("F1", _f1), ("Balanced Acc.", _bal_acc), ("MCC", _mcc)):
        est, lo, hi = bootstrap_ci(fn_, yt, yp, n_boot=n_boot, ci=ci, seed=seed)
        rows[name] = dict(value=est, lo=lo, hi=hi, k=np.nan, n=np.nan)
    out = pd.DataFrame(rows).T[["value", "lo", "hi", "k", "n"]]
    out.attrs["counts"] = dict(TP=tp, FP=fp, FN=fn, TN=tn)
    return out


def majority_baseline(y_true):
    """“全部预测成最多的那一类”能得到的准确率（0~1）。准确率低于它时说明模型还不如瞎猜多数类。"""
    s = pd.Series(np.asarray(y_true))
    return float(s.value_counts(normalize=True).iloc[0])


# ----------------------------------------------------------------------
def rate_table(df, by, value, hue=None, order=None, hue_order=None, ci=0.95):
    """
    按组统计“value 为真的比例”。value 列是布尔 / 0-1。
    返回长表：group, [hue,] k, n, rate, lo, hi（0~1 小数，Wilson 区间）。
    order / hue_order 指定组和 hue 的顺序（默认按出现顺序）。
    """
    keys = [by] + ([hue] if hue else [])
    g = df.groupby(keys, sort=False, observed=True)[value].agg(k="sum", n="count").reset_index()
    g = g.rename(columns={by: "group"})
    if hue:
        g = g.rename(columns={hue: "hue"})
    g["k"] = g["k"].astype(int)
    g["rate"] = g["k"] / g["n"]
    g["lo"], g["hi"] = wilson_ci(g["k"].to_numpy(), g["n"].to_numpy(), ci)
    if order is not None:
        g["group"] = pd.Categorical(g["group"], categories=list(order), ordered=True)
    if hue and hue_order is not None:
        g["hue"] = pd.Categorical(g["hue"], categories=list(hue_order), ordered=True)
    if order is None:                                    # 默认保持出现顺序，不按字母排
        g["group"] = pd.Categorical(g["group"], categories=list(dict.fromkeys(g["group"])), ordered=True)
    if hue and hue_order is None:
        g["hue"] = pd.Categorical(g["hue"], categories=list(dict.fromkeys(g["hue"])), ordered=True)
    g = g.sort_values(["group"] + (["hue"] if hue else []), kind="stable").reset_index(drop=True)
    g["group"] = g["group"].astype(object)
    if hue:
        g["hue"] = g["hue"].astype(object)
    return g


def confusion_table(y_true, y_pred, labels=None, pred_labels=None):
    """
    混淆矩阵（DataFrame，行 = 真实类别，列 = 预测类别）。
    labels / pred_labels 固定行 / 列的顺序，缺失的类别补 0（行列可以不同，如 真实 5 类 × 预测 5 类名称不同）。
    """
    yt, yp = pd.Series(np.asarray(y_true)), pd.Series(np.asarray(y_pred))
    ct = pd.crosstab(yt, yp)
    rows = list(labels) if labels is not None else list(ct.index)
    cols = list(pred_labels) if pred_labels is not None else list(ct.columns)
    out = ct.reindex(index=rows, columns=cols, fill_value=0).astype(int)
    out.index.name, out.columns.name = "True", "Predicted"
    return out


def mcnemar_test(correct_a, correct_b):
    """
    McNemar 精确检验：两个模型在**同一批样本**上的对错是否有显著差异（配对比较）。
    correct_a / correct_b：等长的布尔数组（每个样本是否预测正确，顺序要对齐同一批样本）。
    返回 dict：a_only（A 对 B 错）、b_only（B 对 A 错）、p_value（双侧精确二项检验）。
    """
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if len(a) != len(b):
        raise ValueError("两个模型的样本数不一致，请先按 sample_id 对齐")
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    m = a_only + b_only
    p = 1.0 if m == 0 else float(_st.binomtest(min(a_only, b_only), m, 0.5).pvalue)
    return dict(a_only=a_only, b_only=b_only, p_value=p)


def format_ci(value, lo, hi, digits=1, percent=True):
    """格式化成论文表格里的 “65.6 [63.7, 67.5]”。percent=True 时把 0~1 小数换成百分数。"""
    s = 100.0 if percent else 1.0
    return f"{value * s:.{digits}f} [{lo * s:.{digits}f}, {hi * s:.{digits}f}]"
