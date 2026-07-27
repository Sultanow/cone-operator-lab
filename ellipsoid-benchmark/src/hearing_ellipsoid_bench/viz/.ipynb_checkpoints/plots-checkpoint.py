from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_relative_error(df: pd.DataFrame, ax=None, label_col="method"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for label, part in df.groupby(label_col):
        ax.semilogy(part["k"], part["abs_rel_err"], ".", ms=2, label=label)
    ax.set_xlabel("Eigenvalue index k")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return ax


def plot_reverse_geometry(df: pd.DataFrame):
    if "C_rel_err_3term" in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        columns = ["V_rel_err_3term", "S_rel_err_3term", "C_rel_err_3term"]
        titles = ["Volume", "Surface area", "Integrated mean curvature"]
        for method, part in df.groupby("method"):
            for ax, col in zip(axes, columns):
                ax.semilogx(part["K"], 100 * part[col], "o-", label=method)
        for ax, title in zip(axes, titles):
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(title)
            ax.set_xlabel("K")
            ax.set_ylabel("relative error [%]")
            ax.grid(alpha=0.3, which="both")
            ax.legend()
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, part in df.groupby("method"):
        axes[0].semilogx(part["K"], 100 * part["V_rel_err_S_known"], "o-", label=method)
        axes[1].semilogx(part["K"], 100 * part["V_rel_err"], "o-", label=method)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Volume error with known surface")
    axes[1].set_title("2-term Weyl V/S fit")
    for ax in axes:
        ax.set_xlabel("K")
        ax.set_ylabel("relative error [%]")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    return fig
