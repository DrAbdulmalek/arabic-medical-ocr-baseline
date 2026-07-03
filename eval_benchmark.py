#!/usr/bin/env python3
"""
Benchmark Evaluation for Arabic Medical OCR.

Computes CER and WER for OCR output against ground truth text.
Generates a detailed markdown report with per-image and aggregate statistics.

Usage:
    python eval_benchmark.py --data benchmark_data/ --output report.md
    python eval_benchmark.py --data benchmark_data/ --output results.json --format json
"""

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """Per-image benchmark result."""
    filename: str
    ground_truth: str
    predicted: str
    cer: float
    wer: float
    inference_time_ms: float = 0.0
    status: str = "ok"


@dataclass
class AggregateStats:
    """Aggregate benchmark statistics."""
    total_images: int = 0
    mean_cer: float = 0.0
    median_cer: float = 0.0
    std_cer: float = 0.0
    min_cer: float = 0.0
    max_cer: float = 0.0
    mean_wer: float = 0.0
    median_wer: float = 0.0
    std_wer: float = 0.0
    min_wer: float = 0.0
    max_wer: float = 0.0
    mean_inference_ms: float = 0.0
    perfect_matches: int = 0  # CER = 0
    high_error_count: int = 0  # CER > 0.5

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    """Compute Character Error Rate."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    distance = compute_edit_distance(hypothesis, reference)
    return distance / len(reference)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = compute_edit_distance(" ".join(hyp_words), " ".join(ref_words))
    return distance / len(ref_words)


def normalize_arabic_text(text: str, remove_diacritics: bool = True) -> str:
    """Normalize Arabic text for fair comparison."""
    text = text.strip()
    # Normalize alef variants
    text = text.replace("\u0622", "\u0627")  # alef madda → alef
    text = text.replace("\u0623", "\u0627")  # alef hamza above → alef
    text = text.replace("\u0625", "\u0627")  # alef hamza below → alef
    text = text.replace("\u0624", "\u0627")  # waw hamza → alef (rare)
    # Normalize yaa
    text = text.replace("\u064a", "\u0649")  # yaa → alef maqsura
    # Normalize taa marbuta
    text = text.replace("\u0629", "\u0647")  # taa marbuta → haa
    # Remove diacritics (tashkeel)
    if remove_diacritics:
        diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
        text = diacritics.sub("", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def load_ground_truth(data_dir: Path) -> Dict[str, str]:
    """Load ground truth from data directory.

    Supports two formats:
    1. Pair files: image.ext + image.ext.txt (text file has same name as image)
    2. CSV/TSV: columns [filename, ground_truth]
    """
    gt: Dict[str, str] = {}

    # Try CSV/TSV first
    for csv_name in ["ground_truth.csv", "ground_truth.tsv", "metadata.csv", "labels.csv"]:
        csv_path = data_dir / csv_name
        if csv_path.exists():
            delimiter = "\t" if csv_name.endswith(".tsv") else ","
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    fname = row.get("filename", row.get("image", row.get("file", "")))
                    text = row.get("ground_truth", row.get("text", row.get("label", "")))
                    if fname and text:
                        gt[fname] = text.strip()
            log.info("Loaded %d entries from %s", len(gt), csv_name)
            return gt

    # Try paired .txt files
    txt_files = list(data_dir.glob("*.txt"))
    for txt_file in txt_files:
        image_name = txt_file.stem
        # Look for corresponding image
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
            image_path = data_dir / f"{image_name}{ext}"
            if image_path.exists():
                gt[image_path.name] = txt_file.read_text(encoding="utf-8").strip()
                break
    if gt:
        log.info("Loaded %d entries from paired .txt files", len(gt))
    return gt


def run_inference(image_path: Path, model_name: str, processor, model, device) -> Tuple[str, float]:
    """Run TrOCR inference on a single image."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    elapsed_ms = (time.time() - start) * 1000

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip(), elapsed_ms


def generate_markdown_report(
    results: List[ImageResult],
    stats: AggregateStats,
    model_name: str,
    normalize: bool,
) -> str:
    """Generate a markdown benchmark report."""
    lines = [
        "# Arabic Medical OCR — Benchmark Report",
        "",
        f"**Model**: `{model_name}`",
        f"**Images**: {stats.total_images}",
        f"**Normalization**: {'enabled' if normalize else 'disabled'}",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Mean | Median | Std | Min | Max |",
        "|--------|------|--------|-----|-----|-----|",
        f"| CER | {stats.mean_cer:.4f} | {stats.median_cer:.4f} | {stats.std_cer:.4f} | {stats.min_cer:.4f} | {stats.max_cer:.4f} |",
        f"| WER | {stats.mean_wer:.4f} | {stats.median_wer:.4f} | {stats.std_wer:.4f} | {stats.min_wer:.4f} | {stats.max_wer:.4f} |",
        "",
        f"- **Perfect matches** (CER=0): {stats.perfect_matches}/{stats.total_images}",
        f"- **High errors** (CER>0.5): {stats.high_error_count}/{stats.total_images}",
        f"- **Mean inference time**: {stats.mean_inference_ms:.1f} ms",
        "",
        "## Per-Image Results",
        "",
        "| Image | CER | WER | Time (ms) | Status |",
        "|-------|-----|-----|-----------|--------|",
    ]

    for r in sorted(results, key=lambda x: x.cer):
        lines.append(f"| `{r.filename}` | {r.cer:.4f} | {r.wer:.4f} | {r.inference_time_ms:.1f} | {r.status} |")

    lines.extend([
        "",
        "---",
        "*Generated by eval_benchmark.py*",
    ])
    return "\n".join(lines)


def compute_aggregate(results: List[ImageResult]) -> AggregateStats:
    """Compute aggregate statistics from per-image results."""
    if not results:
        return AggregateStats()

    cer_values = [r.cer for r in results]
    wer_values = [r.wer for r in results]
    time_values = [r.inference_time_ms for r in results]

    return AggregateStats(
        total_images=len(results),
        mean_cer=statistics.mean(cer_values),
        median_cer=statistics.median(cer_values),
        std_cer=statistics.stdev(cer_values) if len(cer_values) > 1 else 0.0,
        min_cer=min(cer_values),
        max_cer=max(cer_values),
        mean_wer=statistics.mean(wer_values),
        median_wer=statistics.median(wer_values),
        std_wer=statistics.stdev(wer_values) if len(wer_values) > 1 else 0.0,
        min_wer=min(wer_values),
        max_wer=max(wer_values),
        mean_inference_ms=statistics.mean(time_values) if time_values else 0.0,
        perfect_matches=sum(1 for c in cer_values if c == 0.0),
        high_error_count=sum(1 for c in cer_values if c > 0.5),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Arabic Medical OCR")
    parser.add_argument("--data", "-d", required=True, help="Directory with images + ground truth")
    parser.add_argument("--output", "-o", required=True, help="Output report path (.md or .json)")
    parser.add_argument("--model", "-m", default="microsoft/trocr-base-handwritten", help="TrOCR model name")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    parser.add_argument("--no-normalize", action="store_true", help="Disable Arabic text normalization")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    normalize = not args.no_normalize

    # Load model
    log.info("Loading model: %s", args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()
    log.info("Model loaded on %s", device)

    # Load ground truth
    gt = load_ground_truth(data_dir)
    if not gt:
        log.error("No ground truth found in %s", data_dir)
        sys.exit(1)

    # Run benchmark
    results: List[ImageResult] = []
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    for gt_name, gt_text in gt.items():
        image_path = data_dir / gt_name
        if not image_path.exists():
            # Try finding by stem
            found = False
            for ext in image_extensions:
                candidate = data_dir / f"{Path(gt_name).stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    found = True
                    break
            if not found:
                log.warning("Image not found for %s, skipping", gt_name)
                results.append(ImageResult(
                    filename=gt_name, ground_truth=gt_text, predicted="",
                    cer=1.0, wer=1.0, status="image_not_found",
                ))
                continue

        try:
            predicted, elapsed_ms = run_inference(image_path, args.model, processor, model, device)

            hyp = normalize_arabic_text(predicted, remove_diacritics=normalize)
            ref = normalize_arabic_text(gt_text, remove_diacritics=normalize)

            cer = compute_cer(hyp, ref)
            wer = compute_wer(hyp, ref)

            results.append(ImageResult(
                filename=gt_name, ground_truth=gt_text, predicted=predicted,
                cer=cer, wer=wer, inference_time_ms=elapsed_ms, status="ok",
            ))
            log.info("%s: CER=%.4f WER=%.4f (%.0fms)", gt_name, cer, wer, elapsed_ms)

        except Exception as e:
            log.error("Error processing %s: %s", gt_name, e)
            results.append(ImageResult(
                filename=gt_name, ground_truth=gt_text, predicted="",
                cer=1.0, wer=1.0, status=f"error: {e}",
            ))

    # Compute aggregate
    stats = compute_aggregate(results)

    # Output
    output_path = Path(args.output)
    if args.format == "json":
        report = {
            "model": args.model,
            "normalize": normalize,
            "device": device,
            "aggregate": stats.to_dict(),
            "per_image": [asdict(r) for r in results],
        }
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        report = generate_markdown_report(results, stats, args.model, normalize)
        output_path.write_text(report, encoding="utf-8")

    log.info("Report written to %s", output_path)
    log.info("Mean CER: %.4f | Mean WER: %.4f | Perfect: %d/%d",
             stats.mean_cer, stats.mean_wer, stats.perfect_matches, stats.total_images)


if __name__ == "__main__":
    main()