"""
Bootstrap confidence-interval analysis on per-document evaluation scores.

For every (metric, engine) pair, resample the N=100 per-document scores with
replacement B=5000 times, recompute the mean each time, and report the
95% percentile bootstrap confidence interval around the reported mean.

Run: python src/evaluation/bootstrap_ci.py
Output: evaluation_reports/statistical/bootstrap_ci.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "evaluation_reports"
OUT = REPORTS / "statistical"

SEED = 42
N_BOOTSTRAP = 5000
SAMPLE_SIZE = 100  # per bootstrap iteration, matches reported N

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

# (display name, report_dir, field, higher_is_better)
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


def load_scores(report_dir: str, field: str) -> dict[str, dict[str, float]]:
    scores = {}
    for path in sorted((REPORTS / report_dir).glob("*_eval_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", path.stem.replace("_eval_report", ""))
        scores[model] = {row["id"]: float(row[field]) for row in data["details"]}
    return scores


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, b: int = N_BOOTSTRAP):
    n = len(values)
    idx = rng.integers(0, n, size=(b, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return boot_means, float(lo), float(hi)


def run_metric(metric_name: str, report_dir: str, field: str, higher: bool, rng: np.random.Generator):
    scores = load_scores(report_dir, field)
    engines = sorted(scores)
    rows = []
    for engine in engines:
        doc_scores = scores[engine]
        doc_ids = sorted(doc_scores)
        values = np.array([doc_scores[d] for d in doc_ids], dtype=float)
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        _, lo, hi = bootstrap_mean_ci(values, rng)
        rows.append({
            "engine": engine,
            "display": label(engine),
            "n_documents": n,
            "sample_size_per_iteration": n,
            "mean": mean,
            "stdev": sd,
            "ci_lower_2.5": lo,
            "ci_upper_97.5": hi,
            "ci_width": hi - lo,
        })

    order_key = (lambda r: -r["mean"]) if higher else (lambda r: r["mean"])
    rows.sort(key=order_key)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    return {
        "metric": metric_name,
        "report_dir": report_dir,
        "field": field,
        "higher_is_better": higher,
        "bootstrap_iterations": N_BOOTSTRAP,
        "sample_size_per_iteration": SAMPLE_SIZE,
        "seed": SEED,
        "engines": rows,
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def print_table(result: dict):
    higher = result["higher_is_better"]
    is_pct = "CER" in result["metric"] or "WER" in result["metric"]
    print(f"\n=== Bootstrap 95% CI - {result['metric']} ({'higher better' if higher else 'lower better'}) ===")
    print(f"{'Rank':<5}{'Model':<20}{'Mean':>10}{'SD':>10}{'95% CI':>22}")
    for row in result["engines"]:
        if is_pct:
            mean_s, sd_s = fmt_pct(row["mean"]), fmt_pct(row["stdev"])
            ci_s = f"{fmt_pct(row['ci_lower_2.5'])} - {fmt_pct(row['ci_upper_97.5'])}"
        else:
            mean_s, sd_s = f"{row['mean']:.4f}", f"{row['stdev']:.4f}"
            ci_s = f"{row['ci_lower_2.5']:.4f} - {row['ci_upper_97.5']:.4f}"
        print(f"{row['rank']:<5}{row['display']:<20}{mean_s:>10}{sd_s:>10}{ci_s:>22}")


def main():
    rng = np.random.default_rng(SEED)
    all_results = []
    for metric_name, report_dir, field, higher in METRICS:
        result = run_metric(metric_name, report_dir, field, higher, rng)
        print_table(result)
        all_results.append(result)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "bootstrap_iterations": N_BOOTSTRAP,
        "sample_size_per_iteration": SAMPLE_SIZE,
        "seed": SEED,
        "method": "percentile bootstrap on the mean, sampling with replacement",
        "metrics": all_results,
    }
    (OUT / "bootstrap_ci.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON saved to {OUT / 'bootstrap_ci.json'}")


if __name__ == "__main__":
    main()
