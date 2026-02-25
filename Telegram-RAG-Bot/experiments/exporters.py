import os
import json
import csv
import asyncio
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")

async def export_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _write():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    await asyncio.to_thread(_write)


async def export_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _write():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            for r in rows:
                w.writerow(r)
    await asyncio.to_thread(_write)


async def save_plot_png(path, fig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _write():
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    await asyncio.to_thread(_write)


def quick_hist(values, title="Histogram", xlabel="value"):
    fig, ax = plt.subplots()
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return fig


def quick_line(x, y, xlabel="x", ylabel="y", title="Plot"):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig
