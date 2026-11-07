
import matplotlib

# 使用非交互式后端，避免 macOS GUI 线程冲突
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_curve(
    time,
    data,
    title="Curve",
    xlabel="Time (s)",
    ylabel="Value",
    save_path=None,
    figsize=(8, 4),
    dpi=200,
):
    """
    绘制一条曲线。

    Parameters
    ----------
    time : array-like
        时间数组。
    data : array-like
        数据数组。
    title : str
        图像标题。
    xlabel : str
        X 轴标签。
    ylabel : str
        Y 轴标签。
    save_path : str or Path, optional
        图片保存路径。
    figsize : tuple
        图片尺寸。
    dpi : int
        图片分辨率。

    Returns
    -------
    fig, ax
        Matplotlib Figure 和 Axes 对象。
    """

    time = np.asarray(time)
    data = np.asarray(data)

    if time.shape[0] != data.shape[0]:
        raise ValueError("time 和 data 的长度必须一致")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time, data, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_curves(
    time,
    data_list,
    titles,
    total_title="Simulation Results",
    rows=2,
    cols=2,
    xlabel="Time (s)",
    ylabels=None,
    save_path=None,
    figsize=None,
    dpi=200,
):
    """
    绘制多个子图。

    Parameters
    ----------
    time : array-like
        时间数组。
    data_list : list[array-like]
        多组数据，每组数据对应一个子图。
    titles : list[str]
        每个子图的标题。
    total_title : str
        整张图的总标题。
    rows : int
        子图行数。
    cols : int
        子图列数。
    xlabel : str
        X 轴标签。
    ylabels : list[str], optional
        每个子图的 Y 轴标签。
    save_path : str or Path, optional
        图片保存路径。
    figsize : tuple, optional
        图片尺寸。
    dpi : int
        图片分辨率。

    Returns
    -------
    fig, axes
        Matplotlib Figure 和 Axes 数组。
    """

    time = np.asarray(time)

    n = len(data_list)

    if n == 0:
        raise ValueError("data_list 不能为空")

    if len(titles) != n:
        raise ValueError("titles 的数量必须与 data_list 一致")

    if rows * cols < n:
        raise ValueError("rows * cols 必须大于等于数据组数")

    if ylabels is None:
        ylabels = ["Value"] * n

    if len(ylabels) != n:
        raise ValueError("ylabels 的数量必须与 data_list 一致")

    if figsize is None:
        figsize = (cols * 5, rows * 3.5)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for i, data in enumerate(data_list):
        data = np.asarray(data)

        if time.shape[0] != data.shape[0]:
            raise ValueError(
                f"第 {i} 组数据长度与 time 不一致"
            )

        ax = axes_flat[i]

        ax.plot(time, data, linewidth=1.5)

        ax.set_title(titles[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[i])

        ax.grid(True, linestyle="--", alpha=0.5)

    # 隐藏多余的子图
    for i in range(n, rows * cols):
        axes_flat[i].set_visible(False)

    fig.suptitle(total_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    # 测试数据
    t = np.linspace(0, 10, 1000)

    y1 = np.sin(t)
    y2 = np.cos(t)
    y3 = np.sin(2 * t)
    y4 = np.cos(2 * t)

    plot_curve(
        t,
        y1,
        title="Sine Wave",
        ylabel="Amplitude",
        save_path="results/sine.png",
    )

    plot_curves(
        time=t,
        data_list=[y1, y2, y3, y4],
        titles=[
            "sin(t)",
            "cos(t)",
            "sin(2t)",
            "cos(2t)",
        ],
        total_title="Trigonometric Functions",
        rows=2,
        cols=2,
        ylabels=[
            "Amplitude",
            "Amplitude",
            "Amplitude",
            "Amplitude",
        ],
        save_path="results/subplots.png",
    )

    plt.close("all")
