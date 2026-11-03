"""
Author: Tomer Porat

Results saving and reporting
"""

import json
import sys
import os
from datetime import datetime


def save_results(results, log_dir):
    """
    Save results to JSON file

    Args:
        results: dict with structure {M: {mode: result}}
        log_dir: directory to save results
    """
    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)

    summary = {"timestamp": datetime.now().isoformat()}

    for M in results:
        summary[f"M{M}"] = {}
        for mode in results[M]:
            if "error" not in results[M][mode]:
                r = results[M][mode]
                summary[f"M{M}"][mode] = {
                    "accuracy": r.get("accuracy", max(r["phase2"]["epochs"]) if "phase2" in r else None),
                }

    with open(f"{log_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved results: {log_dir}/results.json")


def print_summary(results):
    """
    Print results summary table

    Args:
        results: dict with structure {M: {mode: result}}
    """
    print(f"\nRESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'M':<6} {'Learned':<12} {'DCT':<12} {'Hadamard':<12} {'Random':<12}")
    print("-" * 70)

    for M in sorted(results.keys()):
        row = f"{M:<6}"
        for mode in ["learned", "dct", "hadamard", "random"]:
            # prefer the 2D variant when present (dct2d / hadamard2d superseded
            # the 1D / natural-order legacy modes, see diffusers.py)
            key = "dct2d" if (mode == "dct" and "dct2d" in results[M]) else mode
            key = "hadamard2d" if (mode == "hadamard" and "hadamard2d" in results[M]) else key
            if key in results[M] and "error" not in results[M][key]:
                r = results[M][key]
                if "accuracy" in r:
                    acc = r["accuracy"] * 100
                else:
                    acc = max(r["phase2"]["epochs"]) * 100
                row += f"{acc:>10.2f}%  "
            else:
                row += f"{'ERROR':<12}"
        print(row)

    print("\n" + "="*70 + "\n")


def load_and_print_results(log_file):
    """
    Load and print results from a saved JSON file

    Args:
        log_file: path to results.json file
    """
    with open(log_file, "r") as f:
        results = json.load(f)

    print(f"\nResults from: {log_file}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}\n")

    print(f"{'M':<6} {'Learned':<12} {'DCT':<12} {'Hadamard':<12} {'Random':<12}")
    print("-" * 70)

    for M_key in sorted([k for k in results.keys() if k.startswith("M")]):
        M = M_key[1:]  # Remove 'M' prefix
        row = f"{M:<6}"
        for mode in ["learned", "dct", "hadamard", "random"]:
            key = "dct2d" if (mode == "dct" and "dct2d" in results[M_key]) else mode
            key = "hadamard2d" if (mode == "hadamard" and "hadamard2d" in results[M_key]) else key
            if key in results[M_key]:
                acc = results[M_key][key]["accuracy"] * 100
                row += f"{acc:>10.2f}%  "
            else:
                row += f"{'N/A':<12}"
        print(row)

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        load_and_print_results(sys.argv[1])
    else:
        print("Usage: python results.py <path_to_results.json>")
