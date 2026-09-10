"""
Friedman + Wilcoxon tests on per-document scores from evaluation_reports/.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "evaluation_reports"
OUT = REPORTS / "statistical"
ALPHA = 0.05

NAMES = {
    "baidu_ocr": "Baidu Unlimited OCR",
    "deepseekOCR": "DeepSeek-OCR",
    "deepseekOCR2": "DeepSeek-OCR2",
    "dots_mocr": "dots.ocr",
    "glm_ocr": "GLM-OCR",
    "mineru": "MinerU",
    "monkey_ocr": "MonkeyOCR",
    "paddle_vl_1.5": "PaddleVL-1.5",
    "paddle_vl_1.6": "PaddleVL-1.6",
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


def label(model):
    return NAMES.get(model, model)


def load_matrix(report_dir, field):
    scores = {}
    for path in sorted((REPORTS / report_dir).glob("*_eval_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", path.stem.replace("_eval_report", ""))
        scores[model] = {row["id"]: row[field] for row in data["details"]}

    doc_ids = sorted(set.intersection(*[set(s) for s in scores.values()]))
    engines = sorted(scores)
    matrix = np.array([[scores[e][d] for e in engines] for d in doc_ids])
    return doc_ids, engines, matrix


def kendall_w(chi2, n_docs, n_engines):
    return chi2 / (n_docs * (n_engines - 1))


def w_label(w):
    if w < 0.1:
        return "trivial"
    if w < 0.3:
        return "small"
    if w < 0.5:
        return "medium"
    return "large"


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [0.0] * len(pvals)
    prev = 0.0
    m = len(pvals)
    for step, idx in enumerate(order, 1):
        adj = min(1.0, max(prev, pvals[idx] * (m - step + 1)))
        out[idx] = adj
        prev = adj
    return out


def friedman_test(matrix):
    chi2, p = friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    n_docs, n_engines = matrix.shape
    w = kendall_w(chi2, n_docs, n_engines)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "kendall_w": float(w),
        "kendall_w_label": w_label(w),
        "significant": bool(p < ALPHA),
        "n_documents": n_docs,
        "n_engines": n_engines,
    }


def wilcoxon_pairs(matrix, engines, higher_is_better):
    rows = []
    for i, j in combinations(range(len(engines)), 2):
        a, b = matrix[:, i], matrix[:, j]
        if np.allclose(a, b):
            stat, p = 0.0, 1.0
            winner = engines[i]
        else:
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            if higher_is_better:
                winner = engines[i] if np.median(a) >= np.median(b) else engines[j]
            else:
                winner = engines[i] if np.median(a) <= np.median(b) else engines[j]

        rows.append({
            "model_a": engines[i],
            "model_b": engines[j],
            "statistic": float(stat),
            "p_value": float(p),
            "better_model": winner,
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
        })

    for row, adj in zip(rows, holm([r["p_value"] for r in rows])):
        row["adjusted_p_value"] = float(adj)
        row["significant"] = bool(adj < ALPHA)
    return rows


def fmt_p(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def print_friedman_table(results):
    print("\n=== Friedman test ===")
    print(f"{'Metric':<22} {'chi2':>8} {'p':>10} {'W':>7} {'W label':<10} {'sig':>4}")
    print("-" * 65)
    for r in results:
        f = r["friedman"]
        sig = "yes" if f["significant"] else "no"
        print(
            f"{r['metric']:<22} {f['chi2']:8.2f} {fmt_p(f['p_value']):>10} "
            f"{f['kendall_w']:7.3f} {f['kendall_w_label']:<10} {sig:>4}"
        )


def print_wilcoxon(metric_name, pairs):
    sig = [p for p in pairs if p["significant"]]
    print(f"\n--- Wilcoxon ({metric_name}) - {len(sig)} significant pairs ---")
    if not pairs:
        print("(skipped: Friedman not significant)")
        return
    print(f"{'A':<16} {'B':<16} {'stat':>7} {'adj p':>10} {'med A':>8} {'med B':>8} {'winner':<16}")
    for row in sorted(pairs, key=lambda x: x["adjusted_p_value"]):
        if not row["significant"]:
            continue
        print(
            f"{label(row['model_a']):<16} {label(row['model_b']):<16} "
            f"{row['statistic']:7.1f} {fmt_p(row['adjusted_p_value']):>10} "
            f"{row['median_a']:8.4f} {row['median_b']:8.4f} {label(row['better_model']):<16}"
        )


def main():
    all_results = []
    friedman_json = []
    wilcoxon_json = []

    for metric_name, report_dir, field, higher in METRICS:
        doc_ids, engines, matrix = load_matrix(report_dir, field)
        friedman = friedman_test(matrix)
        friedman["metric"] = metric_name

        pairs = wilcoxon_pairs(matrix, engines, higher) if friedman["significant"] else None

        all_results.append({
            "metric": metric_name,
            "report_dir": report_dir,
            "field": field,
            "higher_is_better": higher,
            "n_documents": len(doc_ids),
            "engines": engines,
            "friedman": friedman,
            "pairwise_wilcoxon": pairs,
        })
        friedman_json.append(friedman)
        if pairs:
            wilcoxon_json.extend({"metric": metric_name, **row} for row in pairs)

    print_friedman_table(all_results)
    for r in all_results:
        print_wilcoxon(r["metric"], r["pairwise_wilcoxon"] or [])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "friedman_summary.json").write_text(json.dumps(friedman_json, indent=2), encoding="utf-8")
    (OUT / "wilcoxon_pairwise.json").write_text(json.dumps(wilcoxon_json, indent=2), encoding="utf-8")
    (OUT / "statistical_analysis.json").write_text(
        json.dumps({"alpha": ALPHA, "metrics": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON saved to {OUT}")


if __name__ == "__main__":
    main()
