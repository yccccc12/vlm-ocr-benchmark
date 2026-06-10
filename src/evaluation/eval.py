"""
Table Evaluation Script
======================================================
Evaluates VLM-predicted HTML tables against PubTabNet OTSL ground truth.

Inputs:
  - row_*.json  : PubTabNet ground truth (OTSL + cell tokens)
  - doc_*.md    : VLM prediction (markdown-wrapped HTML)

Metrics:
  - TEDS        : Tree Edit Distance Similarity (structure + content)
  - TEDS-Struct : Tree Edit Distance Similarity (structure only)
  - Cell P/R/F1 : Cell-text multiset precision / recall / F1
"""

import json
import re
import argparse
import contextlib
import io
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import os

from bs4 import BeautifulSoup, NavigableString, Tag

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from apted import APTED, Config
except ImportError:
    APTED = None
    Config = object

try:
    from otsl_to_html import convert_otsl_to_html
except ImportError:
    from .otsl_to_html import convert_otsl_to_html


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 64):
    print()
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


def _sub(title: str):
    print(f"\n  ▸ {title}")


LATEX_SYMBOLS = {
    "times": "×",
    "cdot": "·",
    "pm": "±",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
}


def normalize_cell_text(text: str) -> str:
    """
    Normalize visually equivalent OCR text before comparing table cells.

    Examples:
      '$ \\times $', '\\( \\times \\)', and '×' all become '×'.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("Ã—", "×")

    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)

    for latex, symbol in LATEX_SYMBOLS.items():
        text = re.sub(rf"\\{latex}\b", symbol, text)

    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ground-truth reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_gt_html(row: dict, struct_only: bool = False) -> str:
    """
    Reconstruct the full HTML table from PubTabNet token format.

    'html'     — flat list of structural tokens
                 e.g. ['<thead>', '<tr>', '<td>', '</td>', '<td', ' colspan="2"', '>', ...]
    'cells[0]' — flat list of {tokens, bbox} dicts, one per </td>, in order

    We walk the token list and inject cell text immediately before each '</td>'.
    If struct_only=True we skip the injection (empty cells for TEDS-Struct).
    """
    html_tokens = row["html"]
    cells       = row["cells"][0]

    cell_idx   = 0
    html_parts = []

    for token in html_tokens:
        if token == "</td>":
            if (not struct_only) and cell_idx < len(cells):
                raw = "".join(cells[cell_idx]["tokens"])
                # Strip inline HTML formatting tags (<b>, <i>, etc.)
                text = normalize_cell_text(re.sub(r"<[^>]+>", "", raw))
                html_parts.append(text)
            cell_idx += 1
            html_parts.append("</td>")
        else:
            html_parts.append(token)

    return f"<table>{''.join(html_parts)}</table>"


def _cell_tokens_to_text(cell: dict) -> str:
    raw = "".join(cell.get("tokens", []))
    return normalize_cell_text(re.sub(r"<[^>]+>", "", raw))


def _otsl_token_to_tag(token: str) -> str:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return token
    return f"<{token}>"


def reconstruct_gt_html_from_otsl(row: dict) -> str:
    """
    Reconstruct ground-truth HTML by converting the row's OTSL tokens to HTML.

    PubTabNet stores OTSL as structure tokens and stores cell text separately
    in cells[0]. This function injects each cell's text into the OTSL stream
    before calling convert_otsl_to_html(...).
    """
    if "otsl" not in row:
        raise ValueError("Ground-truth row does not contain an 'otsl' field.")

    cells = row.get("cells", [[]])[0]
    cell_idx = 0
    otsl_parts = []

    for token in row["otsl"]:
        tag = _otsl_token_to_tag(token)
        otsl_parts.append(tag)

        token_name = tag.strip("<>")
        if token_name in {"fcel", "ecel"}:
            if token_name == "fcel" and cell_idx < len(cells):
                otsl_parts.append(_cell_tokens_to_text(cells[cell_idx]))
            cell_idx += 1

    return convert_otsl_to_html("".join(otsl_parts))


def print_gt_html(
    gt_path: str | Path,
    source: str = "otsl",
    pretty: bool = True,
) -> str:
    """
    Print a ground-truth table as HTML in the console and return the HTML string.

    source='otsl' converts row['otsl'] to HTML with cell text injected.
    source='tokens' uses the existing PubTabNet row['html'] reconstruction.
    """
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
        gt_row = gt_data.get("row", gt_data)

    if source == "otsl":
        gt_html = reconstruct_gt_html_from_otsl(gt_row)
    elif source == "tokens":
        gt_html = reconstruct_gt_html(gt_row, struct_only=False)
    else:
        raise ValueError("source must be either 'otsl' or 'tokens'.")

    html_to_print = BeautifulSoup(gt_html, "html.parser").prettify() if pretty else gt_html

    _banner(f"Ground Truth HTML ({source})")
    print(f"  Source file: {gt_path}")
    print()
    print(html_to_print)
    return gt_html


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_html_from_markdown(md_text: str) -> str:
    """
    Pull the HTML table out of a markdown string.
    Tries fenced code block first, then bare <table>…</table>.
    """
    fence = re.search(r"```(?:html)?\s*(.*?)```", md_text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    bare = re.search(r"(<table[\s\S]*?</table>)", md_text, re.IGNORECASE | re.DOTALL)
    if bare:
        return bare.group(1).strip()

    raise ValueError("No HTML table found in markdown input.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTML normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_html(html_str: str, struct_only: bool = False) -> str:
    """
    Canonical form for comparison:
      - Keep only table layout tags: table / tr / td / th
      - Unwrap section wrappers: thead / tbody / tfoot
      - Keep only colspan and rowspan attributes; drop everything else
      - Remove colspan="1" / rowspan="1" (they are the default)
      - Collapse whitespace inside cells
      - If struct_only=True: wipe all cell text
    """
    soup  = BeautifulSoup(html_str, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> element found.")

    KEEP_TAGS  = {"table", "tr", "td", "th"}
    KEEP_ATTRS = {"colspan", "rowspan"}

    def _clean(tag: Tag):
        for child in list(tag.children):
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            if child.name not in KEEP_TAGS:
                _clean(child)
                child.unwrap()
            else:
                _clean(child)

        if hasattr(tag, "attrs"):
            tag.attrs = {k: v for k, v in tag.attrs.items() if k in KEEP_ATTRS}
            for attr in ("colspan", "rowspan"):
                if tag.attrs.get(attr) in ("1", 1):
                    del tag.attrs[attr]

        if struct_only and tag.name in ("td", "th"):
            tag.clear()
            return

        if (not struct_only) and tag.name in ("td", "th"):
            text = normalize_cell_text(tag.get_text(separator=" ", strip=True))
            tag.clear()
            if text:
                tag.string = text

    _clean(table)
    return str(table)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEDS
# ─────────────────────────────────────────────────────────────────────────────

class TableTree:
    def __init__(self, tag: str, text: str = "", attrs: dict = None):
        self.tag      = tag
        self.text     = text.strip() if text else ""
        self.attrs    = attrs or {}
        self.children: list["TableTree"] = []

    def __repr__(self):
        return f"TableTree({self.tag!r}, text={self.text!r}, n_children={len(self.children)})"


def _html_to_tree(html_str: str) -> TableTree | None:
    soup      = BeautifulSoup(html_str, "html.parser")
    table_tag = soup.find("table")
    if table_tag is None:
        return None

    def _convert(bs_node) -> TableTree | None:
        if isinstance(bs_node, str):
            text = bs_node.strip()
            return TableTree("__text__", text=text) if text else None
        node = TableTree(
            tag=bs_node.name,
            attrs={k: str(v) for k, v in bs_node.attrs.items()},
        )
        for child in bs_node.children:
            c = _convert(child)
            if c is not None:
                node.children.append(c)
        return node

    return _convert(table_tag)


class TEDSConfig(Config):
    def rename(self, n1: TableTree, n2: TableTree) -> float:
        if n1.tag != n2.tag:       return 1.0
        if n1.text != n2.text:     return 1.0
        if n1.attrs != n2.attrs:   return 1.0
        return 0.0

    def children(self, node: TableTree):
        return node.children


def _count_nodes(node: TableTree) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def compute_teds(
    pred_html:   str,
    gt_html:     str,
    struct_only: bool = False,
    verbose:     bool = False,
) -> float:
    """
    Compute TEDS (or TEDS-Struct when struct_only=True).

    Formula
    -------
        TEDS = 1 - EditDist(T_pred, T_gt) / max(|T_pred|, |T_gt|)

    The edit distance is computed with the APTED algorithm.
    Node costs: insert = delete = rename = 1 (rename = 0 when nodes are identical).
    """
    label = "TEDS-Struct" if struct_only else "TEDS"

    if APTED is None:
        raise ImportError("Missing dependency 'apted'. Install it to compute TEDS metrics.")

    try:
        pred_norm = normalize_html(pred_html, struct_only=struct_only)
        gt_norm   = normalize_html(gt_html,   struct_only=struct_only)
    except Exception as e:
        print(f"    [WARN] normalisation failed: {e}")
        return 0.0

    pred_tree = _html_to_tree(pred_norm)
    gt_tree   = _html_to_tree(gt_norm)

    if pred_tree is None or gt_tree is None:
        print(f"    [WARN] tree conversion failed")
        return 0.0

    n_pred = _count_nodes(pred_tree)
    n_gt   = _count_nodes(gt_tree)
    denom  = max(n_pred, n_gt)

    apted = APTED(pred_tree, gt_tree, TEDSConfig())
    ted   = apted.compute_edit_distance()

    score = 1.0 - (ted / denom) if denom > 0 else 1.0

    if verbose:
        _sub(f"{label} calculation")
        print(f"      Tree nodes (pred)      : {n_pred}")
        print(f"      Tree nodes (GT)        : {n_gt}")
        print(f"      max(|T_pred|, |T_gt|)  : {denom}")
        print(f"      Edit distance (TED)    : {ted}")
        print(f"      Formula                : 1 - {ted} / {denom}")
        print(f"      {label:<22}= {score:.6f}")

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cell-level P / R / F1
# ─────────────────────────────────────────────────────────────────────────────

def extract_cells(html_str: str, include_empty: bool = False) -> list[str]:
    """
    Return a list of normalised cell-text strings from an HTML table.

    Parameters
    ----------
    include_empty : if False (default), cells whose text is empty after
                    stripping are excluded.  Setting include_empty=True
                    restores the old behaviour and is the root cause of
                    Precision = Recall = F1 when both tables have the
                    same number of empty cells.
    """
    soup  = BeautifulSoup(html_str, "html.parser")
    cells = []
    for tag in soup.find_all(["td", "th"]):
        text = normalize_cell_text(tag.get_text(separator=" ", strip=True)).lower()
        if text or include_empty:
            cells.append(text)
    return cells


def compute_cell_f1(
    pred_html: str,
    gt_html:   str,
    verbose:   bool = False,
) -> dict:
    """
    Multiset precision / recall / F1 over cell text strings.

    Why multiset?
    -------------
    A table can have repeated values (e.g. many cells containing "0" or "-").
    Using a plain set would treat all occurrences as one match; using a multiset
    (Counter) counts each occurrence separately, which is more faithful.

    Formula
    -------
        TP = |pred_cells ∩ gt_cells|   (multiset intersection — sum of mins)
        FP = |pred_cells − gt_cells|   (cells predicted but not in GT)
        FN = |gt_cells   − pred_cells| (GT cells absent from prediction)

        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 x Precision x Recall / (Precision + Recall)

    Note on empty cells
    -------------------
    Empty cells (<td></td>) produce the string "".  If both tables have the
    same number of empty cells they all land in TP and P = R = F1 exactly.
    We therefore EXCLUDE empty-string cells from the comparison by default.
    """
    pred_cells = Counter(extract_cells(pred_html, include_empty=False))
    gt_cells   = Counter(extract_cells(gt_html,   include_empty=False))

    tp = sum((pred_cells & gt_cells).values())   # multiset intersection
    fp = sum((pred_cells - gt_cells).values())   # extra predictions
    fn = sum((gt_cells - pred_cells).values())   # missed GT cells

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    if verbose:
        _sub("Cell-level F1 calculation")
        print(f"      Non-empty cells (pred)   : {sum(pred_cells.values())}")
        print(f"      Non-empty cells (GT)     : {sum(gt_cells.values())}")
        print()
        print(f"      TP (matched cells)       : {tp}")
        print(f"      FP (extra in pred)       : {fp}")
        print(f"      FN (missing from pred)   : {fn}")
        print()
        print(f"      Precision = {tp} / ({tp}+{fp}) = {precision:.6f}")
        print(f"      Recall    = {tp} / ({tp}+{fn}) = {recall:.6f}")
        if (precision + recall) > 0:
            print(f"      F1        = 2·{precision:.4f}·{recall:.4f} / ({precision:.4f}+{recall:.4f}) = {f1:.6f}")
        else:
            print(f"      F1        = 0.0 (precision + recall = 0)")

        # Show top FP and FN to help diagnose errors
        fp_cells = list((pred_cells - gt_cells).elements())
        fn_cells = list((gt_cells - pred_cells).elements())
        if fp_cells:
            sample = fp_cells[:8]
            print(f"\n      Top FP cells (predicted but wrong):")
            for c in sample:
                print(f"        '{c}'")
            if len(fp_cells) > 8:
                print(f"        ... and {len(fp_cells)-8} more")
        if fn_cells:
            sample = fn_cells[:8]
            print(f"\n      Top FN cells (GT cells missed):")
            for c in sample:
                print(f"        '{c}'")
            if len(fn_cells) > 8:
                print(f"        ... and {len(fn_cells)-8} more")

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    gt_path:  str,
    pred_path: str,
    verbose:  bool = True,
    show_gt_html: bool = False,
    gt_html_source: str = "otsl",
    pretty_gt_html: bool = True,
) -> dict:
    """
    Run the full evaluation pipeline for one (GT, prediction) pair.

    Parameters
    ----------
    gt_path   : path to a PubTabNet row_*.json file
    pred_path : path to a VLM output markdown file (doc_*.md)
    verbose   : print step-by-step calculation details
    show_gt_html : print the ground-truth table HTML to the console
    gt_html_source : 'otsl' to convert row['otsl'], or 'tokens' for row['html']
    pretty_gt_html : prettify printed GT HTML for easier reading
    """

    # ── Load ──────────────────────────────────────────────────────
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
        gt_row = gt_data.get("row", gt_data)

    with open(pred_path, encoding="utf-8") as f:
        md_text = f.read()

    # ── Reconstruct GT ────────────────────────────────────────────
    _banner("Step 1 — Ground-Truth Reconstruction")
    gt_html_full   = reconstruct_gt_html(gt_row, struct_only=False)
    gt_html_struct = reconstruct_gt_html(gt_row, struct_only=True)

    if show_gt_html:
        print_gt_html(gt_path, source=gt_html_source, pretty=pretty_gt_html)

    # Count raw cells to show context
    n_gt_cells_all   = sum(1 for t in gt_row["html"] if t == "</td>")
    n_gt_cells_empty = sum(
        1 for c in gt_row["cells"][0]
        if not re.sub(r"<[^>]+>", "", "".join(c["tokens"])).strip()
    )

    print(f"  File          : {gt_row['filename']}")
    print(f"  Table size    : {gt_row['rows']} rows × {gt_row['cols']} cols")
    print(f"  Total cells   : {n_gt_cells_all}")
    print(f"  Empty cells   : {n_gt_cells_empty}  "
          f"← these are EXCLUDED from Cell-F1 to prevent P = R = F1 artefact")
    print(f"  GT HTML chars : {len(gt_html_full)}")

    # ── Extract prediction ────────────────────────────────────────
    _banner("Step 2 — Prediction Extraction")
    pred_html = extract_html_from_markdown(md_text)

    soup_pred = BeautifulSoup(pred_html, "html.parser")
    n_pred_cells_all   = len(soup_pred.find_all(["td", "th"]))
    n_pred_cells_empty = sum(
        1 for tag in soup_pred.find_all(["td", "th"])
        if not tag.get_text(strip=True)
    )
    print(f"  Source        : {pred_path}")
    print(f"  HTML chars    : {len(pred_html)}")
    print(f"  Total cells   : {n_pred_cells_all}")
    print(f"  Empty cells   : {n_pred_cells_empty}")

    # ── Compute metrics ───────────────────────────────────────────
    _banner("Step 3 — TEDS")
    teds = compute_teds(pred_html, gt_html_full, struct_only=False, verbose=verbose)

    _banner("Step 4 — TEDS-Struct")
    teds_struct = compute_teds(pred_html, gt_html_struct, struct_only=True, verbose=verbose)

    _banner("Step 5 — Cell-Level F1")
    cell_scores = compute_cell_f1(pred_html, gt_html_full, verbose=verbose)

    # ── Summary ───────────────────────────────────────────────────
    _banner("Summary")
    print(f"  {'Metric':<30} {'Score':>10}")
    print(f"  {'─'*30} {'─'*10}")
    print(f"  {'TEDS (struct + content)':<30} {teds:>10.4f}")
    print(f"  {'TEDS-Struct (struct only)':<30} {teds_struct:>10.4f}")
    print(f"  {'Cell Precision':<30} {cell_scores['precision']:>10.4f}")
    print(f"  {'Cell Recall':<30} {cell_scores['recall']:>10.4f}")
    print(f"  {'Cell F1':<30} {cell_scores['f1']:>10.4f}")
    print()
    print(f"  TP={cell_scores['tp']}  FP={cell_scores['fp']}  FN={cell_scores['fn']}")

    return {
        "gt_filename": gt_row["filename"],
        "gt_rows":     gt_row["rows"],
        "gt_cols":     gt_row["cols"],
        "TEDS":        teds,
        "TEDS-Struct": teds_struct,
        "Cell-P":      cell_scores["precision"],
        "Cell-R":      cell_scores["recall"],
        "Cell-F1":     cell_scores["f1"],
        "TP":          cell_scores["tp"],
        "FP":          cell_scores["fp"],
        "FN":          cell_scores["fn"],
    }


def main_single():
    GT_PATH   = "PubTabNet_OTSL_train_20/row_16.json"
    PRED_PATH = "experiments/output/otsl_3/doc_0.md"

    result = evaluate(GT_PATH, PRED_PATH, verbose=True)

    out_path = "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {out_path}")

def _empty_summary() -> dict:
    return {
        "count": 0,
        "failed": 0,
        "average_TEDS": 0.0,
        "average_TEDS-Struct": 0.0,
        "average_Cell-P": 0.0,
        "average_Cell-R": 0.0,
        "average_Cell-F1": 0.0,
        "total_TP": 0,
        "total_FP": 0,
        "total_FN": 0,
    }


def _summarize(results: list[dict], failed: int = 0) -> dict:
    summary = _empty_summary()
    summary["count"] = len(results)
    summary["failed"] = failed

    if not results:
        return summary

    metrics = ["TEDS", "TEDS-Struct", "Cell-P", "Cell-R", "Cell-F1"]
    for metric in metrics:
        summary[f"average_{metric}"] = (
            sum(item[metric] for item in results) / len(results)
        )

    summary["total_TP"] = sum(item["TP"] for item in results)
    summary["total_FP"] = sum(item["FP"] for item in results)
    summary["total_FN"] = sum(item["FN"] for item in results)
    return summary


def _evaluate_quiet(
    gt_path: Path,
    pred_path: Path,
    verbose: bool,
    show_gt_html: bool = False,
    gt_html_source: str = "otsl",
    pretty_gt_html: bool = True,
) -> dict:
    if verbose or show_gt_html:
        return evaluate(
            str(gt_path),
            str(pred_path),
            verbose=verbose,
            show_gt_html=show_gt_html,
            gt_html_source=gt_html_source,
            pretty_gt_html=pretty_gt_html,
        )

    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate(str(gt_path), str(pred_path), verbose=False)


MODELS = ["paddle_vl", "monkey_ocr", "mineru", "deepseekOCR2"]


def _parse_models(models_arg: str, pred_root: Path) -> list[str]:
    if models_arg.lower() == "all":
        return sorted(p.name for p in pred_root.iterdir() if p.is_dir())

    models = []
    for raw_model in models_arg.split(","):
        model = raw_model.strip()
        if model and model not in models:
            models.append(model)
    return models


def _resolve_prediction_path(model: str, table_dir: Path, table_id: str) -> Path:
    candidates = {
        "paddle_vl": [
            table_dir / "doc_0.md",
        ],
        "mineru": [
            table_dir / "ocr" / f"{table_id}.md",
        ],
        "deepseekOCR2": [
            table_dir / "result.mmd",
        ],
        "monkey_ocr": [
            table_dir / f"{table_id}.md",
        ],
    }.get(model, [table_dir / "doc_0.md"])

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def evaluate_model(model: str, args: argparse.Namespace) -> dict | None:
    gt_root = Path(args.gt_root)
    pred_model_root = Path(args.pred_root) / model

    if args.out and len(args.models_to_run) == 1:
        out_path = Path(args.out)
    else:
        out_path = Path("evaluation_reports") / "table_by_level" / f"{model}_eval_report.json"

    if not gt_root.is_dir():
        raise FileNotFoundError(f"Ground-truth root not found: {gt_root}")
    if not pred_model_root.is_dir():
        print(f"\nSkipping {model}: prediction model folder not found: {pred_model_root}")
        return None

    report = {
        "model": model,
        "gt_root": str(gt_root),
        "pred_root": str(pred_model_root),
        "summary": _empty_summary(),
        "levels": {},
    }
    all_results = []
    total_failed = 0

    for level_dir in sorted(p for p in pred_model_root.iterdir() if p.is_dir()):
        level_name = level_dir.name
        gt_level_dir = gt_root / level_name / "gt"
        level_results = []
        level_failures = []

        print(f"\nEvaluating {model}/{level_name}...")

        for table_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
            table_id = table_dir.name
            gt_path = gt_level_dir / f"{table_id}.json"
            pred_path = _resolve_prediction_path(model, table_dir, table_id)

            if not gt_path.is_file() or not pred_path.is_file():
                level_failures.append({
                    "id": table_id,
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                    "error": "missing ground-truth or prediction file",
                })
                continue

            try:
                result = _evaluate_quiet(
                    gt_path,
                    pred_path,
                    verbose=args.verbose,
                    show_gt_html=args.print_gt_html,
                    gt_html_source=args.gt_html_source,
                    pretty_gt_html=not args.raw_gt_html,
                )
                result.update({
                    "id": table_id,
                    "level": level_name,
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                })
                level_results.append(result)
                print(f"  {table_id}: TEDS={result['TEDS']:.4f}, Cell-F1={result['Cell-F1']:.4f}")
            except Exception as exc:
                level_failures.append({
                    "id": table_id,
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                    "error": str(exc),
                })
                print(f"  {table_id}: failed ({exc})")

        level_summary = _summarize(level_results, failed=len(level_failures))
        report["levels"][level_name] = {
            "summary": level_summary,
            "details": level_results,
            "failures": level_failures,
        }
        all_results.extend(level_results)
        total_failed += len(level_failures)

        print(
            f"  Summary: count={level_summary['count']}, "
            f"failed={level_summary['failed']}, "
            f"avg TEDS={level_summary['average_TEDS']:.4f}, "
            f"avg Cell-F1={level_summary['average_Cell-F1']:.4f}"
        )

    report["summary"] = _summarize(all_results, failed=total_failed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nOverall summary for {model}")
    print(f"  Evaluated: {report['summary']['count']}")
    print(f"  Failed:    {report['summary']['failed']}")
    print(f"  Avg TEDS:  {report['summary']['average_TEDS']:.4f}")
    print(f"  Avg Cell-F1: {report['summary']['average_Cell-F1']:.4f}")
    print(f"\n  Results saved -> {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate table OCR outputs by table difficulty level.",
    )
    parser.add_argument("--model", default=None, help="Single model folder name. Overrides --models.")
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model names, or 'all'. Defaults to MODELS.")
    parser.add_argument("--gt-root", default="data/raw/table_by_level", help="Ground-truth root containing level_X/gt folders.")
    parser.add_argument("--pred-root", default="outputs/table_by_level", help="Prediction root containing MODEL/level_X folders.")
    parser.add_argument("--out", default=None, help="Output report path. Only used when evaluating one model.")
    parser.add_argument("--verbose", action="store_true", help="Print full diagnostics for every table.")
    parser.add_argument("--print-gt-only", default=None, help="Print one ground-truth JSON table as HTML and exit.")
    parser.add_argument("--print-gt-html", action="store_true", help="Print ground-truth HTML during evaluation.")
    parser.add_argument("--gt-html-source", choices=["otsl", "tokens"], default="otsl", help="Source used when printing GT HTML.")
    parser.add_argument("--raw-gt-html", action="store_true", help="Print GT HTML on one line instead of prettifying it.")
    args = parser.parse_args()

    if args.print_gt_only:
        print_gt_html(args.print_gt_only, source=args.gt_html_source, pretty=not args.raw_gt_html)
        return

    pred_root = Path(args.pred_root)
    models_arg = args.model if args.model else args.models
    args.models_to_run = _parse_models(models_arg, pred_root)

    for model in args.models_to_run:
        evaluate_model(model, args)


if __name__ == "__main__":
    main()
