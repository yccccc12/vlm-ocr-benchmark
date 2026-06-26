# OCR & VLM Document-Parsing Benchmark

A benchmark for comparing modern OCR / Vision-Language Models (VLMs) on document understanding tasks. The project covers the full pipeline: fetching public datasets, running each model to produce predictions, and scoring those predictions with task-appropriate metrics (including a computational-cost comparison).

## Tasks & Metrics

| Task | Dataset | Metrics |
|------|---------|---------|
| Table recognition | [PubTabNet OTSL](https://huggingface.co/datasets/docling-project/PubTabNet_OTSL) | TEDS, TEDS-Struct, Cell P/R/F1 |
| Handwritten English | [IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) | CER, WER |
| Handwritten Chinese | [CASIA-HWDB2-line](https://huggingface.co/datasets/Teklia/CASIA-HWDB2-line) | CER |
| Computational cost | (runtime logs) | Time, CPU, RAM, GPU util / memory / temperature |

- **TEDS / TEDS-Struct** — Tree Edit Distance Similarity over the reconstructed HTML table tree (structure + content, or structure only). Computed with the APTED algorithm.
- **Cell P/R/F1** — multiset precision / recall / F1 over normalised cell-text strings.
- **CER / WER** — character / word error rate (via `jiwer`), after Unicode and markdown normalisation.

## Models Benchmarked

| Key | Official Model Name | Description | Repository |
|------|---------------------|-------------|------------|
| `baidu_ocr` | Baidu Unlimited OCR | Baidu OCR service | [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) |
| `deepseekOCR` | DeepSeek-OCR | First-generation DeepSeek OCR model | [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) |
| `deepseekOCR2` | DeepSeek-OCR-2 | Second-generation DeepSeek OCR model | [deepseek-ai/DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) |
| `dots_mocr` | dots.mocr | 3B parameters VLM | [rednote-hilab/dots.mocr](https://github.com/rednote-hilab/dots.mocr) |
| `mineru` | MinerU2.5-Pro-2605-1.2B | MinerU 2.5 Pro model (1.2B parameters) | [opendatalab/MinerU](https://github.com/opendatalab/MinerU) |
| `monkey_ocr` | MonkeyOCR-pro-3B | MonkeyOCR Pro (3B parameters) | [Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR) |
| `paddle_vl_1.5` | PaddleOCR-VL-1.5 | PaddleOCR-VL version 1.5 (0.9B paramters) | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| `paddle_vl_1.6` | PaddleOCR-VL-1.6 | PaddleOCR-VL version 1.6 (0.9B paramters) | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| `tesseract` | Tesseract OCR | Traditional OCR baseline | [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) |

## Project Structure

```
final-year-project/
├── src/
│   ├── fetch_data/             # Download datasets & build the table-by-level split
│   │   ├── table.py            # Fetch PubTabNet OTSL tables (images + GT)
│   │   ├── select_table.py     # Organise selected tables into level_1..level_4
│   │   ├── handwritten_en.py   # Fetch IAM-line         (English handwriting)
│   │   └── handwritten_zh.py   # Fetch CASIA-HWDB2-line (Chinese handwriting)
│   │ 
│   ├── engines/                # Run each model to generate predictions
│   │   ├── deepseek_ocr.ipynb
│   │   ├── deepseek_ocr2.ipynb
│   │   ├── dots_mocr.ipynb
│   │   ├── mineru_pro.ipynb
│   │   ├── monkey_ocr.ipynb
│   │   ├── paddle_ocr_vl.ipynb
│   │   ├── paddle_ocr_vl_api.py # PaddleOCR-VL via hosted API
│   │   └── tesseract.py         # Tesseract baseline
│   │ 
│   └── evaluation/                # Score predictions
│       ├── eval_table.py          # Table metrics (TEDS / TEDS-Struct / Cell-F1)
│       ├── eval_handwritten_en.py # English CER / WER
│       ├── eval_handwritten_zh.py # Chinese CER
│       ├── eval_computational.py  # Aggregate runtime / resource logs
│       └── otsl_to_html.py        # OTSL -> HTML conversion helper
│ 
├── evaluation_reports/   # Generated metric reports
│   ├── table/  
│   ├── table_by_level/  
│   ├── handwritten_en/  
│   └── handwritten_zh/
│ 
├── data/                 # Datasets (git-ignored)
├── outputs/              # Model predictions (git-ignored)
├── requirements.txt
└── README.md
```

> `data/` and `outputs/` are git-ignored. Run the fetch and engine steps to populate them locally.

## Setup

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

For the **Tesseract** baseline, install the Tesseract binary separately and the `eng` / `chi_sim` language packs ([install guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)).

For the **PaddleOCR-VL API**, create a `.env` file in the project root:

```
PaddleOCR_VL_API_URL=<your-endpoint-url>
PaddleOCR_VL_API_TOKEN=<your-token>
```

## Workflow

All scripts are intended to be run from the project root so that relative paths (`data/`, `outputs/`, `evaluation_reports/`) resolve correctly.

### 1. Fetch datasets

```bash
python src/fetch_data/table.py            # PubTabNet tables -> data/raw/table/
python src/fetch_data/select_table.py     # build data/raw/table_by_level/level_1..4
python src/fetch_data/handwritten_en.py   # IAM-line          -> data/raw/handwritten_en/
python src/fetch_data/handwritten_zh.py   # CASIA-HWDB2-line  -> data/raw/handwritten_zh/
```

### 2. Generate predictions

Run the engine for each model. The notebooks in `src/engines/` produce per-model outputs under `outputs/<dataset>/<model>/...`. Script-based engines:

```bash
python src/engines/tesseract.py            # handwritten_en + handwritten_zh
python src/engines/paddle_ocr_vl_api.py    # PaddleOCR-VL (configure dataset in file)
```

### 3. Evaluate

```bash
# Tables (by-level + overall). Defaults to all configured models.
python src/evaluation/eval_table.py

# A single model, one mode, custom output path:
python src/evaluation/eval_table.py --model paddle_vl_1.6 --mode level --out report.json

# Inspect one ground-truth table as HTML (sanity check, no scoring):
python src/evaluation/eval_table.py --print-gt-only data/raw/table_by_level/level_2/gt/table_0704.json

# Handwritten text
python src/evaluation/eval_handwritten_en.py
python src/evaluation/eval_handwritten_zh.py

# Computational cost (reads outputs/computation_logs/*.json)
python src/evaluation/eval_computational.py
```

Reports are written to `evaluation_reports/<task>/<model>_eval_report.json`.

### `eval_table.py` options

| Flag | Description |
|------|-------------|
| `--model` | Single model folder name (overrides `--models`). |
| `--models` | Comma-separated model names, or `all`. |
| `--mode` | `level`, `overall`, or `both` (default). |
| `--gt-root` / `--pred-root` | By-level ground-truth / prediction roots. |
| `--overall-gt-root` / `--overall-pred-root` | Overall (flat) ground-truth / prediction roots. |
| `--out` | Output report path (single model + single mode only). |
| `--verbose` | Print full per-table diagnostics. |
| `--print-gt-only` | Print one GT JSON as HTML and exit. |
| `--print-gt-html` / `--gt-html-source` / `--raw-gt-html` | Control GT-HTML printing during evaluation. |

## Table Difficulty Levels

The table set is split into four difficulty levels (5 tables each) to analyse how structural complexity affects accuracy:

| Level | Description |
|-------|-------------|
| Level 1 | Simple grid, no merged cells |
| Level 2 | Merged cells |
| Level 3 | Multi-level headers + merged cells |
| Level 4 | Complex layouts |

## Requirements

Key dependencies (see `requirements.txt` for the full list): `datasets`, `transformers`, `beautifulsoup4`, `apted`, `jiwer`, `editdistance`, `opencv-python`, `scikit-image`, `pillow`, `pytesseract`, `python-dotenv`.

## Acknowledgements

This repository was developed as a **Final Year Project (FYP)** for academic purposes.

It builds on the following open-source models and datasets, whose authors and maintainers are gratefully acknowledged:

- **Models** — [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) & [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2), [dots.mocr](https://github.com/rednote-hilab/dots.mocr), [MinerU](https://github.com/opendatalab/MinerU), [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR), [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR), and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
- **Datasets** — [PubTabNet OTSL](https://huggingface.co/datasets/docling-project/PubTabNet_OTSL), [IAM-line](https://huggingface.co/datasets/Teklia/IAM-line), and [CASIA-HWDB2-line](https://huggingface.co/datasets/Teklia/CASIA-HWDB2-line).

All models and datasets remain the property of their respective owners and are used here under their original licenses for non-commercial, educational research only.

