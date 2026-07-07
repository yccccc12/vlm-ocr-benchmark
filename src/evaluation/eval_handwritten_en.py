import os
import json
import jiwer
import unicodedata
import re
import unicodedata
from statistics import mean, stdev

# --- CONFIGURATION ---
DATASET_TYPE = "handwritten_en"
MODELS = [
    "baidu_ocr",
    "glm_ocr",
    "deepseekOCR",
    "deepseekOCR2",
    "dots_mocr",
    "mineru",
    "monkey_ocr",
    "paddle_vl_1.5",
    "paddle_vl_1.6",
    "tesseract",
]

# Paths
GT_DIR = os.path.join("data", "raw", DATASET_TYPE, "gt")


# -----------------------------
# NORMALIZATION
# -----------------------------
import re
import unicodedata

def normalize(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Remove markdown image tags, e.g. ![alt](path)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Standardize quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Fix contractions
    text = re.sub(r"\b(\w+)\s+'\s*(\w+)\b", r"\1'\2", text)

    # Fix punctuation spacing
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Normalize spaces
    text = " ".join(text.split())

    return text.strip()


# -----------------------------
# EXTRACT PREDICTION HELPERS
# -----------------------------
def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _strip_html(text):
    # Replace tags with spaces so adjacent table cells don't merge together.
    return re.sub(r"<[^>]+>", " ", text)


def _join_content_list(items):
    parts = []
    for item in items:
        for key in ("text", "content", "table_body"):
            val = item.get(key)
            if val:
                parts.append(val)
                break
    return "\n".join(parts)


# -----------------------------
# EXTRACT PREDICTION
# -----------------------------
def extract_pred_text(file_id, model_type, results_dir):

    try:
        sample_dir = os.path.join(results_dir, file_id)

        # -- Baidu Unlimited OCR --
        if model_type == "baidu_ocr":
            return _strip_html(_read_file(os.path.join(sample_dir, "result.md")))

        # -- GLM-OCR --
        if model_type == "glm_ocr":
            return _strip_html(_read_file(os.path.join(sample_dir, file_id, f"{file_id}.md")))

        # -- DeepSeek-OCR / DeepSeek-OCR-2 --
        if model_type in ("deepseekOCR", "deepseekOCR2"):
            return _read_file(os.path.join(sample_dir, "result.mmd"))

        # -- Tesseract --
        if model_type == "tesseract":
            return _read_file(os.path.join(sample_dir, "result.txt"))

        # -- dots.mocr --
        if model_type == "dots_mocr":
            return _read_file(os.path.join(sample_dir, f"{file_id}.md"))

        # -- PaddleOCR-VL (1.5 / 1.6) --
        if model_type in ("paddle_vl_1.5", "paddle_vl_1.6"):
            pred_path = os.path.join(sample_dir, f"{file_id}_res.json")
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            res_list = pred_data.get("parsing_res_list", [])
            return "\n".join(b.get("block_content", "") for b in res_list)

        # -- MinerU2.5-Pro-2605-1.2B --
        if model_type == "mineru":
            pred_path = os.path.join(
                sample_dir, file_id, "vlm", f"{file_id}_content_list.json"
            )
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            return _join_content_list(pred_data)

        # -- MonkeyOCR-pro-3B --
        if model_type == "monkey_ocr":
            pred_path = os.path.join(
                sample_dir, file_id, f"{file_id}_content_list.json"
            )
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            return _strip_html(_join_content_list(pred_data))

    except Exception as e:
        print(f"[WARNING] {model_type} missing file for {file_id}: {e}")
        return ""


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def run_evaluation():
    gt_files = [f for f in os.listdir(GT_DIR) if f.endswith(".json")]

    print(f"Evaluating {len(gt_files)} samples...\n")

    for MODEL_NAME in MODELS:
        print("=" * 40)
        print(f"Evaluating model: {MODEL_NAME}")
        print("=" * 40)

        RESULTS_DIR = os.path.join("outputs", DATASET_TYPE, MODEL_NAME)
        REPORT_PATH = os.path.join(
            "evaluation_reports",
            DATASET_TYPE,
            f"{MODEL_NAME}_eval_report.json"
        )

        results = []
        all_gt = []
        all_pred = []

        for i, gt_file in enumerate(gt_files, 1):
            file_id = os.path.splitext(gt_file)[0]

            print(f"[{i}/{len(gt_files)}] {file_id}")

            # --- Load GT ---
            with open(os.path.join(GT_DIR, gt_file), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_text = normalize(gt_data.get("text", ""))

            # --- Load Prediction ---
            raw_pred_text = extract_pred_text(file_id, MODEL_NAME, RESULTS_DIR)
            pred_text = normalize(raw_pred_text)

            # --- Metrics ---
            cer = jiwer.cer(gt_text, pred_text) if gt_text else 1.0
            wer = jiwer.wer(gt_text, pred_text) if gt_text else 1.0

            results.append({
                "id": file_id,
                "gt": gt_text,
                "pred": pred_text,
                "cer": cer,
                "wer": wer
            })

            all_gt.append(gt_text)
            all_pred.append(pred_text)

        # --- Global Metrics ---
        # corpus_cer / corpus_wer: micro (concatenated corpus) — matches jiwer default behaviour
        corpus_cer = jiwer.cer(all_gt, all_pred)
        corpus_wer = jiwer.wer(all_gt, all_pred)

        # macro mean and stdev over per-sample values
        cer_vals = [r["cer"] for r in results]
        wer_vals = [r["wer"] for r in results]
        macro_cer = mean(cer_vals)
        macro_wer = mean(wer_vals)
        stdev_cer = stdev(cer_vals) if len(cer_vals) > 1 else 0.0
        stdev_wer = stdev(wer_vals) if len(wer_vals) > 1 else 0.0

        print("-" * 30)
        print(f"FINAL RESULTS for {MODEL_NAME}")
        print(f"Corpus CER (micro): {corpus_cer:.4f} ({corpus_cer*100:.2f}%)")
        print(f"Corpus WER (micro): {corpus_wer:.4f} ({corpus_wer*100:.2f}%)")
        print(f"Macro CER:          {macro_cer:.4f}  ±{stdev_cer:.4f}")
        print(f"Macro WER:          {macro_wer:.4f}  ±{stdev_wer:.4f}")
        print("-" * 30)

        # --- Save Report ---
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL_NAME,
                "summary": {
                    "corpus_cer": corpus_cer,
                    "corpus_wer": corpus_wer,
                    "macro_cer": macro_cer,
                    "stdev_cer": stdev_cer,
                    "macro_wer": macro_wer,
                    "stdev_wer": stdev_wer,
                },
                "details": results
            }, f, indent=4)

        print(f"Saved: {REPORT_PATH}\n")

if __name__ == "__main__":
    run_evaluation()
