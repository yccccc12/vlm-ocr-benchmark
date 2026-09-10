"""
Sample-size stability analysis.

For N in {20, 40, 60, 80, 100}, repeat 1000 random subsampling experiments
(without replacement, drawn from the full N=100 corpus). For each repetition:
  1. Draw N document ids.
  2. Compute the mean metric per engine on that subset.
  3. Rank engines.
  4. Compare the ranking to the reference ranking (full N=100 dataset) via
     Spearman rho (and Kendall tau).

Aggregated over the 1000 repetitions this shows whether benchmark
conclusions (mean scores + rankings) are stable as sample size grows,
justifying the use of N=100.

Run: python src/evaluation/sample_size_stability.py
Outputs:
  evaluation_reports/statistical/sample_size_stability.json
  evaluation_reports/charts/sample_size_ranking_stability.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "evaluation_reports"
CHARTS = REPORTS / "charts"
OUT = REPORTS / "statistical"

SEED = 42
N_REPS = 1000
SAMPLE_SIZES = [20, 40, 60, 80, 100]

NAMES = {
    "baidu_ocr": "Baidu Unlimited OCR",
    "deepseekOCR": "DeepSeek-OCR",
    "deepseekOCR2": "DeepSeek-OCR2",
    "dots_mocr": "dots.mocr",
    "glm_ocr": "GLM-OCR",
    "mineru": "MinerU",
    "monkey_ocr": "MonkeyOCR",
    "paddle_vl_1.5": "PaddleOCR-VL-1.5",
    "paddle_vl_1.6": "PaddleOCR-VL-1.6",
    "tesseract": "Tesseract",
}

METRICS = [
    ("English CER", "handwritten_en", "cer", False),
    ("English WER", "handwritten_en", "wer", False),
    ("Chinese CER", "handwritten_zh", "cer", False),
    ("Table TEDS", "table", "TEDS", True),
    ("Table TEDS-Struct", "table", "TEDS-Struct", True),
    ("Table Cell-F1", "table", "Cell-F1", True),
]


def label(model: str) -> str:
    return NAMES.get(model, model)


def load_matrix(report_dir: str, field: str):
    scores = {}
    for path in sorted((REPORTS / report_dir).glob("*_eval_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", path.stem.replace("_eval_report", ""))
        scores[model] = {row["id"]: float(row[field]) for row in data["details"]}
    doc_ids = sorted(set.intersection(*(set(s) for s in scores.values())))
    engines = sorted(scores)
    matrix = np.array([[scores[e][d] for e in engines] for d in doc_ids], dtype=float)
    return doc_ids, engines, matrix


def ranks_from_means(means: np.ndarray, higher: bool) -> np.ndarray:
    order = np.argsort(-means) if higher else np.argsort(means)
    ranks = np.empty(len(means), dtype=int)
    ranks[order] = np.arange(1, len(means) + 1)
    return ranks


def ranks_from_means_batch(means: np.ndarray, higher: bool) -> np.ndarray:
    """Vectorized rank-from-mean for a (reps, n_engines) array -> ranks with the same shape."""
    order = np.argsort(-means, axis=1) if higher else np.argsort(means, axis=1)
    reps, n = means.shape
    ranks = np.empty_like(order)
    rep_idx = np.arange(reps)[:, None]
    ranks[rep_idx, order] = np.arange(1, n + 1)[None, :]
    return ranks


def spearman_rho_batch(ranks: np.ndarray, full_ranks: np.ndarray) -> np.ndarray:
    """Closed-form Spearman rho (no ties, since ranks are strict permutations). ranks: (reps, n)."""
    n = ranks.shape[1]
    d2 = ((ranks - full_ranks[None, :]) ** 2).sum(axis=1)
    return 1.0 - (6.0 * d2) / (n * (n**2 - 1))


def kendall_tau_batch(ranks: np.ndarray, full_ranks: np.ndarray) -> np.ndarray:
    """Vectorized Kendall tau (no ties) via pairwise concordance counting."""
    n = ranks.shape[1]
    diff_a = ranks[:, :, None] - ranks[:, None, :]
    diff_b = full_ranks[None, :, None] - full_ranks[None, None, :]
    sign_prod = np.sign(diff_a) * np.sign(diff_b)
    iu = np.triu_indices(n, k=1)
    concordance = sign_prod[:, iu[0], iu[1]].sum(axis=1)
    return concordance / (n * (n - 1) / 2)


def run_metric(metric_name: str, report_dir: str, field: str, higher: bool, rng: np.random.Generator):
    doc_ids, engines, matrix = load_matrix(report_dir, field)
    n_full = len(doc_ids)
    n_engines = len(engines)

    full_means = matrix.mean(axis=0)
    full_ranks = ranks_from_means(full_means, higher)
    top1_full = engines[int(np.argmin(full_ranks))]

    by_size = {}
    for size in SAMPLE_SIZES:
        size = min(size, n_full)

        if size >= n_full:
            idx = np.tile(np.arange(n_full), (N_REPS, 1))
        else:
            # Vectorized "N_REPS independent random samples without replacement":
            # argsort random keys per row, keep the first `size` columns.
            idx = np.argsort(rng.random((N_REPS, n_full)), axis=1)[:, :size]

        rep_means = matrix[idx].mean(axis=1)  # (N_REPS, n_engines)
        ranks = ranks_from_means_batch(rep_means, higher)  # (N_REPS, n_engines)
        rhos = spearman_rho_batch(ranks, full_ranks)
        taus = kendall_tau_batch(ranks, full_ranks)
        top1_idx = np.argmin(ranks, axis=1)
        top1_matches = np.array([engines[i] for i in top1_idx]) == top1_full

        mean_of_means = rep_means.mean(axis=0)
        std_across_reps = rep_means.std(axis=0, ddof=0)
        ci_lo, ci_hi = np.percentile(rep_means, [2.5, 97.5], axis=0)

        by_size[str(size)] = {
            "n_reps": N_REPS,
            "spearman_rho_mean": float(rhos.mean()),
            "spearman_rho_std": float(rhos.std(ddof=0)),
            "kendall_tau_mean": float(taus.mean()),
            "top1_match_rate": float(top1_matches.mean()),
            "engines": [
                {
                    "engine": e,
                    "display": label(e),
                    "mean_of_subsample_means": float(mean_of_means[j]),
                    "std_across_reps": float(std_across_reps[j]),
                    "ci_lower_2.5": float(ci_lo[j]),
                    "ci_upper_97.5": float(ci_hi[j]),
                    "ci_width": float(ci_hi[j] - ci_lo[j]),
                    "full_dataset_mean": float(full_means[j]),
                    "full_dataset_rank": int(full_ranks[j]),
                }
                for j, e in enumerate(engines)
            ],
        }

    return {
        "metric": metric_name,
        "report_dir": report_dir,
        "field": field,
        "higher_is_better": higher,
        "n_full": n_full,
        "engines": engines,
        "top1_full_dataset": label(top1_full),
        "by_sample_size": by_size,
    }


def print_ranking_table(result: dict):
    print(f"\n=== Ranking stability - {result['metric']} (reference: N={result['n_full']}, top-1={result['top1_full_dataset']}) ===")
    print(f"{'N':>5} {'Spearman rho':>14} {'Kendall tau':>13} {'Top-1 match rate':>18}")
    for size in SAMPLE_SIZES:
        s = result["by_sample_size"][str(min(size, result['n_full']))]
        print(f"{size:>5} {s['spearman_rho_mean']:>14.4f} {s['kendall_tau_mean']:>13.4f} {s['top1_match_rate']:>17.1%}")


METRIC_COLORS = {
    "English CER": "#2a78d6",
    "English WER": "#eb6834",
    "Chinese CER": "#1baf7a",
    "Table TEDS": "#eda100",
    "Table TEDS-Struct": "#e87ba4",
    "Table Cell-F1": "#008300",
}


def plot_figure3_ranking_stability(results: list[dict]):
    """Two-panel figure: Spearman rho (left) and Kendall tau (right) vs sample size,
    one line per metric, colored by METRIC_COLORS."""
    fig, (ax_rho, ax_tau) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

    for ax, key, title in [
        (ax_rho, "spearman_rho_mean", "Spearman's ρ vs. Full-Dataset Ranking"),
        (ax_tau, "kendall_tau_mean", "Kendall's τ vs. Full-Dataset Ranking"),
    ]:
        for r in results:
            sizes = [min(s, r["n_full"]) for s in SAMPLE_SIZES]
            ys = [r["by_sample_size"][str(s)][key] for s in sizes]
            ax.plot(sizes, ys, marker="o", markersize=6, linewidth=2,
                     color=METRIC_COLORS[r["metric"]], markeredgecolor="black",
                     markeredgewidth=0.5, label=r["metric"])
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_xlabel("Subsample Size (N)")
        ax.set_xticks(SAMPLE_SIZES)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    ax_rho.set_ylabel("Rank Correlation with Full Dataset (N=100)")
    ax_rho.set_ylim(0.6, 1.03)

    handles, labels = ax_rho.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Ranking Stability Across Subsample Sizes", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = CHARTS / "sample_size_ranking_stability.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    all_results = []
    for metric_name, report_dir, field, higher in METRICS:
        result = run_metric(metric_name, report_dir, field, higher, rng)
        print_ranking_table(result)
        all_results.append(result)

    f3 = plot_figure3_ranking_stability(all_results)
    print(f"\nSaved {f3}")

    payload = {
        "seed": SEED,
        "n_reps_per_size": N_REPS,
        "sample_sizes": SAMPLE_SIZES,
        "method": "random subsampling without replacement, drawn from the full N=100 corpus",
        "metrics": all_results,
    }
    (OUT / "sample_size_stability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON saved to {OUT / 'sample_size_stability.json'}")


if __name__ == "__main__":
    main()
