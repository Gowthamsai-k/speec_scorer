#!/usr/bin/env python3
"""
Proprietary Benchmark Evaluation Script for Indian-English CEFR Speech Assessor
==============================================================================
Evaluates the speech assessment model against a proprietary benchmark dataset
structured with candidate audio recordings, question prompts, and multi-rater consensus
CEFR ground truth labels.

Benchmark Structure:
-------------------
benchmark/
├── metadata.csv
└── audio/
    ├── 00001.wav
    ├── 00002.wav
    └── ...

Usage:
------
    python scripts/evaluate_benchmark.py --benchmark_dir benchmark/ --device cpu
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import SpeechInferenceEngine, get_api

CEFR_MAP = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
REVERSE_CEFR = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}


def compute_metrics(y_true_num: list[float], y_pred_num: list[float], y_true_bands: list[str], y_pred_bands: list[str]) -> dict:
    """
    Calculates comprehensive benchmarking evaluation metrics.
    """
    y_true_num = np.array(y_true_num, dtype=float)
    y_pred_num = np.array(y_pred_num, dtype=float)

    # 1. Exact Match Accuracy
    exact_matches = sum(1 for gt, pred in zip(y_true_bands, y_pred_bands) if gt == pred)
    exact_accuracy = (exact_matches / len(y_true_bands)) * 100.0 if len(y_true_bands) > 0 else 0.0

    # 2. Adjacent Match Accuracy (Within +/- 1 CEFR Band)
    adjacent_matches = sum(1 for gt, pred in zip(y_true_num, y_pred_num) if abs(round(gt) - round(pred)) <= 1)
    adjacent_accuracy = (adjacent_matches / len(y_true_num)) * 100.0 if len(y_true_num) > 0 else 0.0

    # 3. Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(y_true_num - y_pred_num))) if len(y_true_num) > 0 else 0.0

    # 4. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean((y_true_num - y_pred_num) ** 2))) if len(y_true_num) > 0 else 0.0

    # 5. Pearson Correlation
    if len(y_true_num) > 1 and np.std(y_true_num) > 0 and np.std(y_pred_num) > 0:
        pearson_r = float(np.corrcoef(y_true_num, y_pred_num)[0, 1])
    else:
        pearson_r = 1.0 if np.array_equal(y_true_num, y_pred_num) else 0.0

    # 6. Spearman Rank Correlation
    try:
        from scipy.stats import spearmanr
        spearman_rho, _ = spearmanr(y_true_num, y_pred_num)
        spearman_rho = float(spearman_rho) if not np.isnan(spearman_rho) else 0.0
    except Exception:
        spearman_rho = pearson_r

    # 7. Confusion Matrix (6x6 for A1..C2)
    bands = ["A1", "A2", "B1", "B2", "C1", "C2"]
    conf_matrix = {gt_band: {pred_band: 0 for pred_band in bands} for gt_band in bands}
    for gt_band, pred_band in zip(y_true_bands, y_pred_bands):
        if gt_band in conf_matrix and pred_band in conf_matrix[gt_band]:
            conf_matrix[gt_band][pred_band] += 1

    return {
        "sample_count": len(y_true_bands),
        "exact_match_accuracy_pct": round(exact_accuracy, 2),
        "adjacent_within_1_band_accuracy_pct": round(adjacent_accuracy, 2),
        "mae_continuous": round(mae, 4),
        "rmse_continuous": round(rmse, 4),
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman_rho, 4),
        "confusion_matrix": conf_matrix
    }


def print_evaluation_report(metrics: dict, per_sample_results: list[dict]):
    """
    Renders a formatted evaluation report for terminal output.
    """
    W = 68
    print("\n" + "═" * W)
    print("PROPRIETARY BENCHMARK EVALUATION REPORT".center(W))
    print("Indian-English CEFR Speech Assessment Engine".center(W))
    print("═" * W)

    print(f" Total Benchmark Samples  : {metrics['sample_count']}")
    print(f" Exact Match Accuracy     : {metrics['exact_match_accuracy_pct']}%")
    print(f" Adjacent (+/- 1 Band) Acc: {metrics['adjacent_within_1_band_accuracy_pct']}%")
    print(f" Mean Absolute Error (MAE): {metrics['mae_continuous']}")
    print(f" Root Mean Sq Error (RMSE): {metrics['rmse_continuous']}")
    print(f" Pearson Correlation (r)  : {metrics['pearson_r']}")
    print(f" Spearman Correlation (rho): {metrics['spearman_rho']}")
    print("─" * W)

    print("CEFR CONFUSION MATRIX (Row = Ground Truth, Col = Predicted):")
    bands = ["A1", "A2", "B1", "B2", "C1", "C2"]
    header = "      " + " ".join(f"{b:>5}" for b in bands)
    print(header)
    for gt_b in bands:
        row_str = f" {gt_b:>3} |" + " ".join(f"{metrics['confusion_matrix'][gt_b][pred_b]:>5}" for pred_b in bands)
        print(row_str)

    print("─" * W)
    print("SAMPLE EVALUATION BREAKDOWN:")
    print(f" {'ID':<7} | {'GT':<4} | {'PRED':<4} | {'SCORE':<5} | {'PRON':<4} | {'FLU':<4} | {'GRAM':<4} | {'VOCAB':<4} | MATCH")
    print("─" * W)

    for item in per_sample_results:
        match_symbol = "✓ EXACT" if item['exact_match'] else ("~ ADJ" if item['adjacent_match'] else "✗ DIFF")
        print(
            f" {item['id']:<7} | {item['ground_truth']:<4} | {item['predicted_band']:<4} | "
            f"{item['predicted_score']:<5.2f} | {item['pron_score']:<4.1f} | {item['fluency_score']:<4.1f} | "
            f"{item['grammar_score']:<4.1f} | {item['vocab_score']:<4.1f} | {match_symbol}"
        )

    print("═" * W + "\n")


def run_benchmark_eval(benchmark_dir: Path, metadata_csv: str = "metadata.csv", device: str = "cpu") -> tuple[dict, list[dict]]:
    """
    Loads benchmark dataset, runs evaluation, and computes metrics.
    """
    bm_path = Path(benchmark_dir).resolve()
    csv_file = bm_path / metadata_csv
    if not csv_file.exists():
        raise FileNotFoundError(f"Benchmark metadata file not found: {csv_file}")

    print(f"[Benchmark Evaluator]: Loading benchmark dataset from {csv_file}...")
    engine = SpeechInferenceEngine(device=device)

    y_true_bands = []
    y_pred_bands = []
    y_true_num = []
    y_pred_num = []
    per_sample_results = []

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["id"].strip()
            question = row["question"].strip()
            audio_rel = row["audio"].strip()
            gt_cefr = row["cefr"].strip().upper()

            audio_full_path = bm_path / audio_rel
            if not audio_full_path.exists():
                print(f"[Warning]: Audio file for sample {sample_id} not found: {audio_full_path}. Skipping.")
                continue

            # Run Model Inference
            eval_res = engine.predict(audio_path=str(audio_full_path), question=question)

            pred_band = eval_res["scores"]["cefr_band"]
            pred_score = eval_res["scores"]["cefr_continuous"]
            skills = eval_res.get("skill_scores", {})

            gt_num = float(CEFR_MAP.get(gt_cefr, 3.0))
            pred_num = float(pred_score)

            exact_match = (gt_cefr == pred_band)
            adj_match = abs(round(gt_num) - round(CEFR_MAP.get(pred_band, int(round(pred_score))))) <= 1

            y_true_bands.append(gt_cefr)
            y_pred_bands.append(pred_band)
            y_true_num.append(gt_num)
            y_pred_num.append(pred_num)

            per_sample_results.append({
                "id": sample_id,
                "question": question,
                "audio": str(audio_rel),
                "ground_truth": gt_cefr,
                "predicted_band": pred_band,
                "predicted_score": pred_score,
                "exact_match": exact_match,
                "adjacent_match": adj_match,
                "pron_score": skills.get("pronunciation", 0.0),
                "fluency_score": skills.get("fluency", 0.0),
                "grammar_score": skills.get("grammar", 0.0),
                "vocab_score": skills.get("vocabulary", 0.0),
                "full_output": eval_res
            })

    metrics = compute_metrics(y_true_num, y_pred_num, y_true_bands, y_pred_bands)
    return metrics, per_sample_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speech Assessor Engine on Proprietary Benchmark.")
    parser.add_argument(
        "--benchmark_dir", "-b",
        type=str,
        default=str(PROJECT_ROOT / "benchmark"),
        help="Directory containing benchmark metadata.csv and audio/ folder."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="metadata.csv",
        help="Filename of metadata CSV within benchmark directory."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--output_json", "-o",
        type=str,
        default=None,
        help="Optional path to save full benchmark evaluation metrics JSON."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save per-sample prediction detailed CSV."
    )

    args = parser.parse_args()

    metrics, per_sample = run_benchmark_eval(
        benchmark_dir=args.benchmark_dir,
        metadata_csv=args.csv,
        device=args.device
    )

    print_evaluation_report(metrics, per_sample)

    if args.output_json:
        out_json_path = Path(args.output_json).resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "per_sample": per_sample}, f, indent=2)
        print(f"[SUCCESS]: Evaluation JSON metrics saved to -> {out_json_path}")

    if args.output_csv:
        out_csv_path = Path(args.output_csv).resolve()
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "question", "audio", "ground_truth", "predicted_band", "predicted_score", "exact_match", "adjacent_match", "pron_score", "fluency_score", "grammar_score", "vocab_score"]
        with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_sample)
        print(f"[SUCCESS]: Evaluation CSV predictions saved to -> {out_csv_path}")


if __name__ == "__main__":
    main()
