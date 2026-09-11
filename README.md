# OCR & VLM Document-Parsing Benchmark

A benchmark for comparing modern OCR / Vision-Language Models (VLMs) on document understanding tasks. The project covers the full pipeline: fetching public datasets, running each model to produce predictions, and scoring those predictions with task-appropriate metrics (including a computational-cost comparison).

## Tasks & Metrics

| Task | Dataset | Metrics |
|------|---------|---------|
| Table recognition | [PubTabNet OTSL](https://huggingface.co/datasets/docling-project/PubTabNet_OTSL) | TEDS, TEDS-Struct, Cell P/R/F1 |
| Handwritten English | [IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) | CER, WER |
| Handwritten Chinese | [CASIA-HWDB2-line](https://huggingface.co/datasets/Teklia/CASIA-HWDB2-line) | CER |
| Computational cost | (runtime logs) | Time, CPU, RAM, GPU util / memory / temperature |

- **TEDS / TEDS-Struct** — PubTabNet official Tree Edit Distance Similarity ([`src/evaluation/metric.py`](src/evaluation/metric.py)); macro mean per table. GT rebuilt from `html` tokens + `cells`.
- **Cell P/R/F1** — multiset precision / recall / F1 over normalised cell-text strings; macro mean per table in batch reports.
- **CER / WER** — character / word error rate (via `jiwer`), after Unicode and markdown normalisation.

## Models Benchmarked

| Key | Official Model Name | Description | Repository |
|------|---------------------|-------------|------------|
| `baidu_ocr` | Baidu Unlimited OCR | Baidu OCR service | [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) |
| `deepseekOCR` | DeepSeek-OCR | First-generation DeepSeek OCR model | [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) |
| `deepseekOCR2` | DeepSeek-OCR-2 | Second-generation DeepSeek OCR model | [deepseek-ai/DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) |
| `dots_mocr` | dots.mocr | 3B parameters VLM | [rednote-hilab/dots.mocr](https://github.com/rednote-hilab/dots.mocr) |
| `glm_ocr` | GLM-OCR | Zhipu AI GLM-based OCR model | [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) |
| `mineru` | MinerU2.5-Pro-2605-1.2B | MinerU 2.5 Pro model (1.2B parameters) | [opendatalab/MinerU](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) |
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
│   │   ├── handwritten_zh.py   # Fetch CASIA-HWDB2-line (Chinese handwriting)
│   │   └── zip_input_images.py # Package data/raw images -> data/data.zip (engine input)
│   │ 
│   ├── engines/                # Run each model to generate predictions
│   │   ├── deepseek_ocr.ipynb
│   │   ├── deepseek_ocr2.ipynb
│   │   ├── dots_mocr.ipynb
│   │   ├── mineru_pro.ipynb
│   │   ├── monkey_ocr.ipynb
│   │   ├── paddle_ocr_vl.ipynb
│   │   └── tesseract.py         # Tesseract baseline
│   │ 
│   └── evaluation/                       # Score predictions
│       ├── eval.py                       # Runs all three evaluations below in sequence
│       ├── extract_inference_output.py   # Unzip inference_output/*.zip -> outputs/
│       ├── eval_table.py                 # Table metrics (TEDS / TEDS-Struct / Cell-F1)
│       ├── eval_handwritten_en.py        # English CER / WER
│       ├── eval_handwritten_zh.py        # Chinese CER
│       ├── eval_computational.py         # Aggregate runtime / resource logs
│       └── otsl_to_html.py               # OTSL -> HTML conversion helper
│ 
├── evaluation_reports/   # Generated metric reports
│   ├── table/  
│   ├── table_by_level/  
│   ├── handwritten_en/  
│   └── handwritten_zh/
│ 
├── data/                 # Datasets (git-ignored)
├── inference_output/     # Downloaded prediction zips, one per model (git-ignored)
├── outputs/              # Extracted model predictions (git-ignored)
├── requirements.txt
└── README.md
```

> `data/`, `inference_output/`, and `outputs/` are git-ignored. Run the fetch, engine, and extraction steps to populate them locally.

## Setup

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

For the **Tesseract** baseline, install the Tesseract binary separately and the `eng` / `chi_sim` language packs ([install guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)).

## Workflow

All scripts are intended to be run from the project root so that relative paths (`data/`, `outputs/`, `evaluation_reports/`) resolve correctly.

### 1. Fetch datasets

```bash
python src/fetch_data/table.py            # PubTabNet tables -> data/raw/table/
python src/fetch_data/select_table.py     # PubTabNet tables (by-level) -> data/raw/table_by_level/level_1..4
python src/fetch_data/handwritten_en.py   # IAM-line          -> data/raw/handwritten_en/
python src/fetch_data/handwritten_zh.py   # CASIA-HWDB2-line  -> data/raw/handwritten_zh/
```

Or run all four in sequence:

```bash
python src/fetch_data/fetch_all.py
```

This step also produces the ground-truth files used for evaluation (`data/raw/<task>/gt/`).

#### How samples are selected

There is no random sampling and no seed: each script takes the **first N rows of a fixed
split**, in dataset order, via the Hugging Face `datasets-server` API. The sample ID *is* the
row index (`handwritten_en_0042` = row 42 of the IAM-line `test` split), so the 100-sample set
is fully determined by the split and reproducible without any stored manifest. Re-running a
fetch script overwrites the same files.

#### Adjusting how much data is fetched

The fetch scripts have no CLI flags — the parameters are constants at the top of each file:

| Script | Parameter | Default | Meaning |
|---|---|---|---|
| `handwritten_en.py` / `handwritten_zh.py` | `split` | `"test"` | Dataset split to read from |
| | `offset` | `0` | Index of the first row to fetch |
| | `length` | `100` | Rows to fetch — **max 100, API limit** (see below) |
| `table.py` | `split` | `"val"` | Dataset split |
| | `total_samples` | `100` | Total tables to fetch |
| | `batch_size` | `100` | Rows per API request — **must stay ≤ 100** |
| `select_table.py` | `levels` | 5 IDs × 4 levels | Hand-picked row indices for the difficulty split |

Upper bounds (split sizes as of this writing): IAM-line `test` = 2,915 · CASIA-HWDB2-line
`test` = 10,441 · PubTabNet_OTSL `val` = 6,942.

**The API caps every request at 100 rows** — asking for more returns
`422 Parameter 'length' must not be greater than 100`. The scripts handle this differently:

- **`table.py` paginates.** To fetch 500 tables, set `total_samples = 500` and leave
  `batch_size = 100`; it loops over offsets automatically.
- **`handwritten_en.py` / `handwritten_zh.py` make a single request** and will fail if
  `length > 100`. To fetch more than 100 lines, either run the script several times with
  `offset = 0`, `100`, `200`, … (IDs embed the offset, so nothing collides), or add a
  pagination loop mirroring `table.py`.
- **`select_table.py` is independent of `table.py`.** It fetches its 20 tables directly by
  absolute row index, so changing `total_samples` does not affect the difficulty split, and
  it can run on its own. To change the split, edit the `levels` dict; several of its indices
  (e.g. `0704`, `0804`) deliberately lie outside the first 100 rows.

Sample IDs are zero-padded to four digits, so a single fetch supports up to 9,999 samples per
task.

**If you change N:**

- Run every engine on the same samples. The statistical tests compare engines document by
  document, so any document missing from one engine is dropped for all of them.
- The evaluation and statistics scripts work with whatever number of samples you fetched.
  If N > 100, also update `SAMPLE_SIZES` in `src/evaluation/sample_size_stability.py`.
- Results from a different N cannot be compared with the published results.

### 2. Package images for the engines

Bundle the fetched images (no ground truth) into a single upload-ready zip:

```bash
python src/fetch_data/zip_input_images.py    # data/raw/<task> images -> data/data.zip
```

`handwritten_en`, `handwritten_zh`, and `table` are mandatory — the script aborts if any is missing from `data/raw/`. `table_by_level` is optional: if you skipped `select_table.py`, it's simply left out of the zip with a note, and the other three still get packaged.

> To skip fetching + packaging entirely, download the pre-packaged input-images zip used for this project's notebook runs: [Google Drive link](https://drive.google.com/file/d/17fy1xziJWVD7hC1PaFI5KZrRNDWGxIDe/view?usp=sharing). It does not include ground truth (that still comes from step 1).

### 3. Generate predictions

Run the engine for each model. The notebooks in `src/engines/` are Colab-based: upload `data/data.zip` (from step 2), run the notebook, then download the resulting `<model>_output.zip`.

Place every downloaded `<model>_output.zip` into `inference_output/` (create the folder if needed), then extract them all into `outputs/` in one go:

```bash
python src/evaluation/extract_inference_output.py
```

This locates the `handwritten_en` / `handwritten_zh` / `table` / `table_by_level` folders inside each zip (whatever top-level prefix the notebook wrapped them in) and copies each model's predictions to `outputs/<task>/<model>/`, matching the layout the evaluation scripts expect. Computational-cost logs bundled in the same zips are skipped. `handwritten_en`, `handwritten_zh`, and `table` are mandatory per zip — a model missing one is flagged with `[ERROR]` and the script exits non-zero; `table_by_level` is optional and just prints a note if absent. It never modifies `inference_output/` itself (zips are opened read-only); add `--dry-run` to preview counts first, or `--tasks table,table_by_level` to extract a subset. See the script header for the zip-filename -> model-key mapping if you rename a downloaded zip.

Script-based engines write straight to `outputs/` and don't go through `inference_output/`:

```bash
python src/engines/tesseract.py  # handwritten_en + handwritten_zh
```

### 4. Evaluate

Run everything (handwritten English, handwritten Chinese, table) in one command:

```bash
python src/evaluation/eval.py
```

This runs `eval_handwritten_en.py`, `eval_handwritten_zh.py`, and `eval_table.py` (`--mode both`) in sequence. Use `--tasks` to run a subset (e.g. `--tasks table`) and `--table-mode` to change the table mode (`level`, `overall`, `both`).

Or run each metric individually for finer control:

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

Key dependencies are list at `requirements.txt`.

## Acknowledgements

This repository was developed as a **Final Year Project (FYP)** for academic purposes.

It builds on the following open-source models and datasets, whose authors and maintainers are gratefully acknowledged:

- **Models** — [Baidu Unlimited OCR](https://github.com/baidu/Unlimited-OCR), [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) & [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2), [dots.mocr](https://github.com/rednote-hilab/dots.mocr), [GLM-OCR](https://github.com/zai-org/GLM-OCR), [MinerU2.5-Pro](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B), [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR), [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR), and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
- **Datasets** — [PubTabNet OTSL](https://huggingface.co/datasets/docling-project/PubTabNet_OTSL), [IAM-line](https://huggingface.co/datasets/Teklia/IAM-line), and [CASIA-HWDB2-line](https://huggingface.co/datasets/Teklia/CASIA-HWDB2-line).

All models and datasets remain the property of their respective owners and are used here under their original licenses for non-commercial, educational research only.

