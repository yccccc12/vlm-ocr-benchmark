"""
Run every evaluation task -- handwritten English, handwritten Chinese, and
table -- in one command.

Each task is delegated to its existing script (eval_handwritten_en.py,
eval_handwritten_zh.py, eval_table.py) so the scoring logic lives in exactly
one place; this just runs them in sequence and reports pass/fail.

Prerequisites (see README.md "Reproducing the evaluation"):
  - Ground truth under data/raw/<task>/gt/  (from src/fetch_data/)
  - Predictions under outputs/<task>/<model>/
    -> run `python src/evaluation/extract_inference_output.py` to populate these
       from inference_output/*.zip, and `python src/engines/tesseract.py`
       for the tesseract baseline (no zip).

Run: python src/evaluation/eval.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "src" / "evaluation"

TASK_SCRIPTS = {
    "handwritten_en": EVAL_DIR / "eval_handwritten_en.py",
    "handwritten_zh": EVAL_DIR / "eval_handwritten_zh.py",
    "table": EVAL_DIR / "eval_table.py",
}


def run_task(name: str, script: Path, extra_args: list[str]) -> bool:
    print("\n" + "=" * 70)
    print(f"Evaluating: {name}  ({script.relative_to(ROOT)})")
    print("=" * 70)
    result = subprocess.run([sys.executable, str(script), *extra_args], cwd=ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run the handwritten English, handwritten Chinese, and table evaluations."
    )
    parser.add_argument(
        "--tasks", default="handwritten_en,handwritten_zh,table",
        help="Comma-separated subset of tasks to run.",
    )
    parser.add_argument(
        "--table-mode", default="both", choices=["level", "overall", "both"],
        help="Passed through to eval_table.py --mode (default: both).",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(tasks) - set(TASK_SCRIPTS)
    if unknown:
        parser.error(f"Unknown task(s): {sorted(unknown)}. Valid: {sorted(TASK_SCRIPTS)}")

    all_ok = True
    for task in tasks:
        extra_args = ["--mode", args.table_mode] if task == "table" else []
        all_ok = run_task(task, TASK_SCRIPTS[task], extra_args) and all_ok

    print("\n" + "=" * 70)
    print("All evaluations finished." if all_ok else "Finished with errors -- see output above.")
    print("Reports written to evaluation_reports/<task>/<model>_eval_report.json")
    print("=" * 70)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
