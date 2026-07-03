---
license: mit
task: text-generation
tags:
- ocr
- arabic
- medical
- handwriting
- ensemble
- baseline
- omnimedical
language:
- ar
- en
library_name: transformers
---

# Arabic Medical OCR — Baseline Ensemble Reference

> **Note:** This is a reference model card documenting the multi-engine ensemble approach used in the OmniMedical Suite ecosystem. No model weights are hosted here — see "How to Use" below.

## Model Description

The OmniMedical OCR Engine uses a multi-engine ensemble combining:
- **PaddleOCR** (Arabic text detection + recognition)
- **Tesseract 5** (Arabic + English with custom traineddata)
- **EasyOCR** (Arabic handwriting recognition)
- **TrOCR** (Transformer-based OCR, fine-tuned for handwritten text)
- **Surya OCR** (Modern multilingual OCR)

The ensemble uses confidence-based voting: each engine produces text + confidence scores, and a weighted majority vote determines the final output.

## Intended Use

- Arabic medical handwritten document recognition
- Pediatric orthopedic reports (primary domain)
- Post-OCR correction pipeline input

## Inference (Standalone)

Run TrOCR inference directly without the full platform:

```bash
pip install -r requirements.txt

# Single image
python inference.py --input prescription.png --output result.json

# Directory of images
python inference.py --input ./scans/ --output results.json --mode handwritten

# Printed documents with diacritics removal
python inference.py --input ./printed/ --output results.json --mode printed --remove-diacritics
```

**Output format** (JSON):
```json
[
  {
    "filename": "prescription.png",
    "text": "الاسم: أحمد ... الدواء: باراسيتامول 500 ملغ",
    "inference_time_ms": 342.5,
    "model": "microsoft/trocr-base-handwritten"
  }
]
```

## Benchmark

Evaluate OCR quality against ground truth:

```bash
# Prepare benchmark data: images + ground_truth.csv (columns: filename, ground_truth)
# OR: image.png + image.png.txt (paired text files)

python eval_benchmark.py --data ./benchmark_data/ --output benchmark_report.md
python eval_benchmark.py --data ./benchmark_data/ --output results.json --format json
```

The benchmark computes **CER** (Character Error Rate) and **WER** (Word Error Rate) with Arabic text normalization (alef/yaa/taa forms, diacritics removal).

## Data Pipeline Setup

Initialize the 14-day pipeline directory structure:

```bash
python setup_data.py                          # creates ./data/ with all subdirs
python setup_data.py --root /path/to/project  # custom location
```

Creates: `scans_printed_raw/`, `scans_printed_fixed/`, `handwriting_raw/`, `handwriting_labeled/`, `ground_truth_verified/`, `ocr_corrections/`, `training_exports/`, `reports/`, `previews/`, `quarantine/`

## Full Platform

For the complete multi-engine ensemble pipeline:

```bash
git clone https://github.com/DrAbdulmalek/omni-medical-suite.git
cd omni-medical-suite
pip install -r requirements.txt
pip install scanner-fixer  # MANDATORY preprocessing step
make dev
```

## Training Data

- **Primary:** 282-page handwritten pediatric orthopedic study notes (Arabic + English mixed)
- **Ground Truth:** Built from ABBYY FineReader 16 + Readiris 23 consensus ([ocr-groundtruth](https://github.com/DrAbdulmalek/ocr-groundtruth))
- **Correction corpus:** [arabic-medical-ocr-corrections](https://huggingface.co/datasets/DrAbdulmalek/arabic-medical-ocr-corrections) — 160 unique OCR error-correction pairs
- **Medical dictionary:** 900K+ Arabic medical terms ([arabic-dictionaries-collection](https://github.com/DrAbdulmalek/arabic-dictionaries-collection) — private)

## Evaluation

Measured against verified ground truth from ABBYY + Readiris consensus:
- See [medical-ocr-benchmarks](https://github.com/DrAbdulmalek/medical-ocr-benchmarks) for CER/WER metrics
- Benchmarks run on real scanned medical documents, not synthetic data

## Limitations

- Cursive connected Arabic handwriting remains extremely challenging
- Interline insertions and marginal notes are not handled
- Mixed Arabic/English medical terms may cause engine confusion
- No model weights are hosted — this card documents the architecture only

## Ecosystem

| Component | Link | Role |
|-----------|------|------|
| Platform | [omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) | Main platform |
| Preprocessing | [scanner-fixer](https://github.com/DrAbdulmalek/scanner-fixer) | Mandatory image normalization |
| Ground Truth CLI | [ocr-groundtruth](https://github.com/DrAbdulmalek/ocr-groundtruth) | Build GT from ABBYY/Readiris |
| Data Authority | [medical-ocr-ground-truth](https://github.com/DrAbdulmalek/medical-ocr-ground-truth) | SSOT for verified datasets |
| Training Hub | [medical-ocr-training-hub](https://github.com/DrAbdulmalek/medical-ocr-training-hub) | Data ingestion & release pipeline |
| Benchmarks | [medical-ocr-benchmarks](https://github.com/DrAbdulmalek/medical-ocr-benchmarks) | Quality gates |
| Live Demo | [handwriting-ocr](https://huggingface.co/spaces/DrAbdulmalek/handwriting-ocr) | HF Space |

## License

MIT