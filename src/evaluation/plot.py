"""
Entry point that generates every chart under evaluation_reports/charts,
plus the statistical JSON side-outputs (subsample stability, heatmap
patterns) that back them.

Run: python src/evaluation/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "evaluation_reports"
CHARTS = REPORTS / "charts"
STAT = REPORTS / "statistical"
COMP_SUMMARY = REPORTS / "computational" / "computational_summary.json"

SEED = 42
SUBSAMPLE_SIZES = [30, 60, 100]

BAR_COLOR = "#2a78d6"
CPU_COLOR = "#2a78d6"
GPU_COLOR = "#eb6834"

NAMES = {
    "baidu_ocr": "Unlimited OCR",
    "deepseekOCR": "DeepSeek-OCR",
    "deepseekOCR2": "DeepSeek-OCR-2",
    "dots_mocr": "dots.mocr",
    "glm_ocr": "GLM-OCR",
    "mineru": "MinerU2.5-Pro",
    "monkey_ocr": "MonkeyOCR-pro-3B",
    "paddle_vl_1.5": "PaddleOCR-VL-1.5",
    "paddle_vl_1.6": "PaddleOCR-VL-1.6",
    "tesseract": "Tesseract OCR",
}


def label(model: str) -> str:
    return NAMES.get(model, model)


# ---------------------------------------------------------------------------
# CER/WER/TEDS bar charts + TEDS-by-level heatmap
# ---------------------------------------------------------------------------

METRIC_COLORS = {
    "macro_cer": "#2a78d6",
    "macro_wer": "#eb6834",
    "average_TEDS": "#2a78d6",
    "average_TEDS-Struct": "#eb6834",
    "average_Cell-F1": "#1baf7a",
}

METRIC_LABELS = {
    "macro_cer": "CER",
    "macro_wer": "WER",
    "average_TEDS": "TEDS",
    "average_TEDS-Struct": "TEDS-Struct",
    "average_Cell-F1": "Cell-F1",
}

TASKS = [
    {
        "title": "Comparison of CER and WER on the English Handwritten Dataset",
        "report_dir": "handwritten_en",
        "metrics": ["macro_cer", "macro_wer"],
        "outfile": "handwritten_en_cer_wer.png",
        "higher_is_better": False,
    },
    {
        "title": "Comparison of CER on the Chinese Handwritten Dataset",
        "report_dir": "handwritten_zh",
        "metrics": ["macro_cer"],
        "outfile": "handwritten_zh_cer_wer.png",
        "higher_is_better": False,
    },
    {
        "title": "Comparison of TEDS, TEDS-Struct, and Cell-Level F1 on the "
                 "Table Dataset",
        "report_dir": "table",
        "metrics": ["average_TEDS", "average_TEDS-Struct", "average_Cell-F1"],
        "outfile": "table_teds_cellf1.png",
        "higher_is_better": True,
        "title_loc": "center",
    },
    {
        "type": "heatmap",
        "title": "Comparison of TEDS by Table Complexity Level Across Models",
        "report_dir": "table_by_level",
        "outfile": "table_teds_by_level.png",
        "title_loc": "center",
    },
]


def load_summary(report_dir: str, metrics: list[str]):
    rows = []

    for path in sorted((REPORTS / report_dir).glob("*_eval_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", path.stem.replace("_eval_report", ""))
        summary = data["summary"]
        rows.append((model, [summary[m] for m in metrics]))

    return rows


def plot_cer_wer_bars(task: dict, rows: list) -> Path:
    metrics = task["metrics"]
    higher_is_better = task["higher_is_better"]
    rows = sorted(rows, key=lambda r: r[1][0], reverse=higher_is_better)  # best first

    names = [label(r[0]) for r in rows]
    values = np.array([r[1] for r in rows])  # (n_engines, n_metrics)
    n = len(names)
    y = np.arange(n)[::-1]  # best at top

    n_metrics = len(metrics)
    bar_h = 0.7 / n_metrics
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * (bar_h + 0.04)

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * n + 1.5))

    xmax = values.max()

    for j, metric in enumerate(metrics):
        vals = values[:, j]
        yy = y + offsets[j]
        ax.barh(yy, vals, height=bar_h, color=METRIC_COLORS[metric],
                edgecolor="black", linewidth=0.5,
                label=METRIC_LABELS[metric], zorder=3)

        for yi, v in zip(yy, vals):
            ax.text(v + xmax * 0.012, yi, f"{v * 100:.1f}", va="center",
                    ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score (%)" if higher_is_better else "Error Rate (%)")
    ax.set_xlim(0, xmax * 1.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_title(task["title"], fontsize=12, loc=task.get("title_loc", "center"))

    if n_metrics > 1:
        ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    out = CHARTS / task["outfile"]
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def load_teds_by_level(report_dir: str):
    levels = None
    rows = []

    for path in sorted((REPORTS / report_dir).glob("*_eval_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", path.stem.replace("_eval_report", ""))
        if levels is None:
            levels = sorted(data["levels"])
        per_level = [data["levels"][lvl]["summary"]["average_TEDS"] for lvl in levels]
        rows.append((model, per_level, data["summary"]["average_TEDS"]))

    return levels, rows


def plot_teds_by_level_heatmap(task: dict) -> Path:
    levels, rows = load_teds_by_level(task["report_dir"])
    rows = sorted(rows, key=lambda r: r[2], reverse=True)  # best overall first

    names = [label(r[0]) for r in rows]
    matrix = np.array([r[1] for r in rows])  # (n_engines, n_levels)
    n = len(names)

    fig, ax = plt.subplots(figsize=(1.7 * len(levels) + 3, 0.5 * n + 2), dpi=160)
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([lvl.replace("_", " ").title() for lvl in levels], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9.5)

    for i in range(n):
        for j in range(len(levels)):
            v = matrix[i, j]
            ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.6 else "#0b0b0b")

    ax.set_title(task["title"], fontsize=12, loc=task.get("title_loc", "left"))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("TEDS (%)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    out = CHARTS / task["outfile"]
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Computational charts (execution time, CPU/GPU utilization)
# ---------------------------------------------------------------------------

def load_execution_times():
    data = json.loads(COMP_SUMMARY.read_text(encoding="utf-8"))
    engines = list(data.keys())
    means = np.array([data[e]["execution_time"]["mean"] for e in engines])
    return engines, means


def load_cpu_vs_gpu_utilization():
    data = json.loads(COMP_SUMMARY.read_text(encoding="utf-8"))
    engines = list(data.keys())
    cpu = np.array([data[e]["cpu_usage"]["mean"] for e in engines])
    gpu = np.array([data[e]["gpu_utilization"]["mean"] for e in engines])
    return engines, cpu, gpu


def plot_execution_time_comparison(engines, means) -> Path:
    order = np.argsort(means)[::-1]  # slowest at top -> fastest at bottom
    eng = [engines[i] for i in order]
    means_o = means[order]
    labels = [label(e) for e in eng]
    y = np.arange(len(eng))

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(eng) + 1.5))
    ax.barh(y, means_o, color=BAR_COLOR, edgecolor="black", linewidth=0.5)

    x_max = float(means_o.max())
    label_pad = x_max * 0.02
    for yi, m in zip(y, means_o):
        ax.text(m + label_pad, yi, f"{m:.1f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Execution Time (s)")
    ax.set_title("Execution Time Comparison")
    ax.set_xlim(0, x_max * 1.12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = CHARTS / "execution_time_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cpu_vs_gpu_utilization(engines, cpu, gpu) -> Path:
    order = np.argsort(gpu)[::-1]  # highest GPU utilization first
    eng = [engines[i] for i in order]
    cpu_o = cpu[order]
    gpu_o = gpu[order]
    labels = [label(e) for e in eng]
    x = np.arange(len(eng))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(1.1 * len(eng) + 2, 6))
    ax.bar(x - bar_w / 2, cpu_o, width=bar_w, color=CPU_COLOR,
           edgecolor="black", linewidth=0.5, label="CPU Usage")
    ax.bar(x + bar_w / 2, gpu_o, width=bar_w, color=GPU_COLOR,
           edgecolor="black", linewidth=0.5, label="GPU Utilization")

    y_max = float(max(cpu_o.max(), gpu_o.max()))
    label_pad = y_max * 0.015
    for xi, v in zip(x - bar_w / 2, cpu_o):
        ax.text(xi, v + label_pad, f"{v:.1f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + bar_w / 2, gpu_o):
        ax.text(xi, v + label_pad, f"{v:.1f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
    ax.set_ylabel("Utilization (%)")
    ax.set_title("CPU vs. GPU Utilization")
    ax.set_ylim(0, y_max * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    out = CHARTS / "cpu_vs_gpu_utilization.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Per-sample scores: shared loader, engine ordering/grouping
# ---------------------------------------------------------------------------

# Left -> right column order by model type (heatmaps / failure maps)
ENGINE_ORDER = [
    "tesseract",
    "baidu_ocr",
    "deepseekOCR",
    "deepseekOCR2",
    "paddle_vl_1.5",
    "paddle_vl_1.6",
    "dots_mocr",
    "glm_ocr",
    "mineru",
    "monkey_ocr",
]

GROUP_BOUNDARIES = [1, 2, 4, 6]  # after Classic, Baidu, DeepSeek, Paddle


def load_scores(report_dir: str, field: str):
    scores = {}
    for path in sorted((REPORTS / report_dir).glob("*_eval_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", path.stem.replace("_eval_report", ""))
        scores[model] = {row["id"]: float(row[field]) for row in data["details"]}

    doc_ids = sorted(set.intersection(*(set(s) for s in scores.values())))
    engines = [e for e in ENGINE_ORDER if e in scores]
    matrix = np.array([[scores[e][d] for e in engines] for d in doc_ids], dtype=float)
    return doc_ids, engines, matrix


# ---------------------------------------------------------------------------
# Per-sample summary bars + pattern analysis + subsample stability
# ---------------------------------------------------------------------------

PER_SAMPLE_TASKS = [
    {
        "name": "English CER",
        "report_dir": "handwritten_en",
        "field": "cer",
        "higher_is_better": False,
        "bad_threshold": 0.2,
        "metric_label": "CER",
    },
    {
        "name": "Chinese CER",
        "report_dir": "handwritten_zh",
        "field": "cer",
        "higher_is_better": False,
        "bad_threshold": 0.2,
        "metric_label": "CER",
    },
    {
        "name": "Table TEDS",
        "report_dir": "table",
        "field": "TEDS",
        "higher_is_better": True,
        "bad_threshold": 0.5,
        "metric_label": "TEDS",
    },
]


def pattern_summary(task: dict, doc_ids, engines, matrix: np.ndarray, top_k: int = 8):
    higher = task["higher_is_better"]
    bad_thr = task["bad_threshold"]

    if higher:
        error = 1.0 - matrix
        bad = matrix < bad_thr
    else:
        error = matrix
        bad = matrix > bad_thr

    mean_err = error.mean(axis=1)
    n_bad = bad.sum(axis=1).astype(int)
    hardest = np.argsort(-mean_err)[:top_k]
    shared = np.where(n_bad >= max(3, len(engines) // 2))[0]

    print(f"\n=== Patterns ({task['name']}) ===")
    print(f"{'rank':>4} {'idx':>4} {'id':<22} {'mean_err':>9} {'n_bad':>5}")
    for rank, i in enumerate(hardest, 1):
        print(
            f"{rank:4d} {i:4d} {doc_ids[i]:<22} {mean_err[i]:9.4f} {n_bad[i]:5d}"
        )
    print(
        f"Shared hard samples (>={max(3, len(engines) // 2)} engines bad): "
        f"{len(shared)} - indices {shared[:15].tolist()}"
        + (" ..." if len(shared) > 15 else "")
    )

    return {
        "hardest_samples": [
            {
                "index": int(i),
                "id": doc_ids[i],
                "mean_error": float(mean_err[i]),
                "n_engines_bad": int(n_bad[i]),
                "per_engine": {
                    label(e): float(matrix[i, j]) for j, e in enumerate(engines)
                },
            }
            for i in hardest
        ],
        "shared_hard_count": int(len(shared)),
        "shared_hard_indices": [int(i) for i in shared.tolist()],
        "bad_threshold": bad_thr,
        "higher_is_better": higher,
    }


def nested_subsample_indices(n: int, sizes: list[int], rng: np.random.Generator) -> dict[int, np.ndarray]:
    """One random permutation (seeded); first k indices for each size (30 subset of 60 subset of 100)."""
    perm = rng.permutation(n)
    return {size: np.sort(perm[: min(size, n)]) for size in sizes}


def subsample_sizes_vs_100(
    task: dict,
    doc_ids,
    engines,
    matrix: np.ndarray,
    rng: np.random.Generator,
    sizes: list[int] | None = None,
):
    sizes = sizes or SUBSAMPLE_SIZES
    n = len(doc_ids)
    subsets = nested_subsample_indices(n, sizes, rng)
    mean_by_size = {size: matrix[idx].mean(axis=0) for size, idx in subsets.items()}

    def ranks_for(means: np.ndarray) -> np.ndarray:
        if task["higher_is_better"]:
            order = np.argsort(-means)
        else:
            order = np.argsort(means)
        pos = np.empty(len(engines), dtype=int)
        pos[order] = np.arange(len(engines))
        return pos

    rank_100 = ranks_for(mean_by_size[100])
    top1_100 = engines[int(
        np.argsort(-mean_by_size[100] if task["higher_is_better"] else mean_by_size[100])[0]
    )]

    correlations = {}
    top1_match = {}
    for size in sizes:
        if size == 100:
            continue
        rank_size = ranks_for(mean_by_size[size])
        rho, p = spearmanr(rank_100, rank_size)
        correlations[str(size)] = {"rho": float(rho), "p": float(p)}
        top1_size = engines[int(
            np.argsort(-mean_by_size[size] if task["higher_is_better"] else mean_by_size[size])[0]
        )]
        top1_match[str(size)] = bool(top1_100 == top1_size)

    rows = []
    for j, e in enumerate(engines):
        row = {
            "engine": e,
            "display": label(e),
            "rank_100": int(rank_100[j] + 1),
        }
        for size in sizes:
            row[f"mean_{size}"] = float(mean_by_size[size][j])
            row[f"rank_{size}"] = int(ranks_for(mean_by_size[size])[j] + 1)
            if size != 100:
                row[f"abs_delta_{size}"] = float(abs(mean_by_size[100][j] - mean_by_size[size][j]))
        rows.append(row)

    print(f"\n=== Subsample {sizes} vs 100 ({task['name']}, seed={SEED}, nested) ===")
    for size in sizes:
        if size == 100:
            continue
        c = correlations[str(size)]
        top1_size = engines[int(np.argsort(-mean_by_size[size] if task["higher_is_better"] else mean_by_size[size])[0])]
        print(
            f"  N={size}: Spearman rho={c['rho']:.4f} (p={c['p']:.4g})  "
            f"top-1 {size}={label(top1_size)}  100={label(top1_100)}  "
            f"match={top1_match[str(size)]}"
        )
    header = f"{'Engine':<16}"
    for size in sizes:
        header += f" {'mean'+str(size):>9}"
    for size in sizes:
        if size != 100:
            header += f" {'rk'+str(size):>5}"
    header += f" {'rk100':>5}"
    print(header)
    for r in sorted(rows, key=lambda x: x["rank_100"]):
        line = f"{r['display']:<16}"
        for size in sizes:
            line += f" {r[f'mean_{size}']:9.4f}"
        for size in sizes:
            if size != 100:
                line += f" {r[f'rank_{size}']:5d}"
        line += f" {r['rank_100']:5d}"
        print(line)

    return {
        "metric": task["name"],
        "seed": SEED,
        "method": "nested_random_permutation",
        "n_full": n,
        "sample_sizes": sizes,
        "subset_indices": {str(k): [int(i) for i in v.tolist()] for k, v in subsets.items()},
        "spearman_vs_100": correlations,
        "top1_100": top1_100,
        "top1_match_vs_100": top1_match,
        "engines": rows,
    }


# ---------------------------------------------------------------------------
# Failure maps (standalone thresholds: CER/TEDS > 0.4)
# ---------------------------------------------------------------------------

FAILURE_MAP_TASKS = [
    {
        "name": "English CER",
        "report_dir": "handwritten_en",
        "field": "cer",
        "higher_is_better": False,
        "bad_threshold": 0.4,
        "failure_outfile": "handwritten_en_failure_map.png",
        "metric_label": "CER",
    },
    {
        "name": "Chinese CER",
        "report_dir": "handwritten_zh",
        "field": "cer",
        "higher_is_better": False,
        "bad_threshold": 0.4,
        "failure_outfile": "handwritten_zh_failure_map.png",
        "metric_label": "CER",
    },
    {
        "name": "Table TEDS",
        "report_dir": "table",
        "field": "TEDS",
        "higher_is_better": True,
        "bad_threshold": 0.4,
        "failure_outfile": "table_teds_failure_map.png",
        "metric_label": "TEDS",
    },
]


def failure_mask(matrix: np.ndarray, higher_is_better: bool, thr: float) -> np.ndarray:
    return matrix < thr if higher_is_better else matrix > thr


def plot_failure_map(task: dict, doc_ids, engines, matrix: np.ndarray) -> Path:
    """
    Top    - binary fail/pass grid (rows=engines, cols=sample index)
    Bottom - how many engines fail on each sample (common failures)
    """
    thr = task["bad_threshold"]
    bad = failure_mask(matrix, task["higher_is_better"], thr)
    n_docs, n_eng = bad.shape
    n_fail_per_sample = bad.sum(axis=1).astype(int)
    shared_cut = max(3, n_eng // 2)

    fail_grid = bad.T.astype(float)  # (engines, samples)

    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.8, 1.4], hspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    cmap = plt.cm.colors.ListedColormap(["#f0f0f0", "#c0392b"])
    ax0.imshow(fail_grid, aspect="auto", interpolation="nearest", cmap=cmap,
               vmin=0, vmax=1)
    ax0.set_yticks(range(n_eng))
    ax0.set_yticklabels([label(e) for e in engines], fontsize=9)
    for bound in GROUP_BOUNDARIES:
        if bound < n_eng:
            ax0.axhline(bound - 0.5, color="black", linewidth=1.0, alpha=0.75)
    thr_txt = (
        f"{task['metric_label']} < {thr:g}"
        if task["higher_is_better"]
        else f"{task['metric_label']} > {thr:g}"
    )
    fig.suptitle(
        f"{task['name']} - where each model fails "
        f"(red = {thr_txt}; grey = pass; n={n_docs} documents)",
        fontsize=12,
    )
    ax0.set_ylabel("Model")

    colors = np.where(
        n_fail_per_sample >= shared_cut,
        "#c0392b",
        np.where(n_fail_per_sample >= 2, "#e67e22", "#95a5a6"),
    )
    ax1.bar(np.arange(n_docs), n_fail_per_sample, color=colors, width=1.0,
            edgecolor="none")
    ax1.axhline(shared_cut, color="black", linestyle="--", linewidth=1)
    ax1.set_ylim(0, n_eng + 0.5)
    ax1.set_ylabel("# models failing")
    ax1.set_xlabel(f"Sample index (document id order, n={n_docs})")
    ax1.set_title("Common failures across models (taller = more models fail on that sample)")
    legend = [
        Patch(facecolor="#c0392b", edgecolor="none", label=f"Shared fail (>= {shared_cut} models)"),
        Patch(facecolor="#e67e22", edgecolor="none", label="Some fail (2 - " + str(shared_cut - 1) + " models)"),
        Patch(facecolor="#95a5a6", edgecolor="none", label="Isolated / no failure (0 - 1 models)"),
    ]
    ax1.legend(handles=legend, loc="upper right", fontsize=7.5, framealpha=0.9)
    ax1.set_xticks(list(range(0, n_docs, 5)))
    ax1.grid(axis="y", alpha=0.3)

    shared_idx = np.where(n_fail_per_sample >= shared_cut)[0]
    for i in shared_idx[:8]:
        ax1.annotate(
            str(i),
            xy=(i, n_fail_per_sample[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#922b21",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = CHARTS / task["failure_outfile"]
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI forest plot
# ---------------------------------------------------------------------------

IS_PCT = {"English CER", "English WER", "Chinese CER"}


def plot_bootstrap_panel(ax, metric: dict):
    engines = metric["engines"]  # already ranked 1..N in the JSON
    rows = sorted(engines, key=lambda r: r["rank"], reverse=True)  # rank 1 at top

    y = range(len(rows))
    scale = 100.0 if metric["metric"] in IS_PCT else 1.0
    means = [r["mean"] * scale for r in rows]
    lo = [r["ci_lower_2.5"] * scale for r in rows]
    hi = [r["ci_upper_97.5"] * scale for r in rows]
    err_lo = [m - l for m, l in zip(means, lo)]
    err_hi = [h - m for m, h in zip(means, hi)]
    labels = [label(r["engine"]) for r in rows]

    ax.errorbar(
        means, y, xerr=[err_lo, err_hi], fmt="none",
        ecolor="black", elinewidth=1.2, capsize=3, zorder=2,
    )
    ax.scatter(means, y, c=BAR_COLOR, s=45, edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    direction = "higher better" if metric["higher_is_better"] else "lower better"
    ax.set_title(f"{metric['metric']} ({direction})", fontsize=10.5)
    unit = "%" if metric["metric"] in IS_PCT else "score"
    ax.set_xlabel(f"Mean & 95% CI ({unit})", fontsize=9)


def plot_bootstrap_ci_forest() -> Path:
    data = json.loads((STAT / "bootstrap_ci.json").read_text(encoding="utf-8"))
    metrics = data["metrics"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric in zip(axes.flat, metrics):
        plot_bootstrap_panel(ax, metric)

    fig.suptitle(
        "Bootstrap 95% Confidence Intervals of Mean Performance",
        fontsize=18, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = CHARTS / "bootstrap_ci_forest.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    STAT.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        if task.get("type") == "heatmap":
            out = plot_teds_by_level_heatmap(task)
        else:
            rows = load_summary(task["report_dir"], task["metrics"])
            out = plot_cer_wer_bars(task, rows)
        print(f"Saved {out}")

    engines, means = load_execution_times()
    print(f"Saved {plot_execution_time_comparison(engines, means)}")

    engines, cpu, gpu_util = load_cpu_vs_gpu_utilization()
    print(f"Saved {plot_cpu_vs_gpu_utilization(engines, cpu, gpu_util)}")

    rng = np.random.default_rng(SEED)
    patterns = {}
    subsample = []
    for task in PER_SAMPLE_TASKS:
        doc_ids, engines, matrix = load_scores(task["report_dir"], task["field"])
        patterns[task["name"]] = pattern_summary(task, doc_ids, engines, matrix)
        subsample.append(subsample_sizes_vs_100(task, doc_ids, engines, matrix, rng))

    chart_paths = {"failure_maps": {}}
    for task in FAILURE_MAP_TASKS:
        doc_ids, engines, matrix = load_scores(task["report_dir"], task["field"])
        out = plot_failure_map(task, doc_ids, engines, matrix)
        chart_paths["failure_maps"][task["name"]] = str(out.relative_to(ROOT)).replace("\\", "/")
        print(f"Saved {out}")

    payload = {"seed": SEED, "sample_sizes": SUBSAMPLE_SIZES, "tasks": subsample}
    (STAT / "subsample_30_60_vs_100.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (STAT / "heatmap_patterns.json").write_text(
        json.dumps({"charts": chart_paths, "patterns": patterns}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved statistical JSON to {STAT}")

    print(f"Saved {plot_bootstrap_ci_forest()}")


if __name__ == "__main__":
    main()
