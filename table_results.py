#!/usr/bin/env python3
"""
Author: Tomer Porat

Generate ASCII table from results JSON
Usage:
  python table_results.py <results.json>

Example:
  python table_results.py logs_run_all_20260428_162051_cifar/results.json
"""

import sys
import json


def normalize_results(results):
    """Convert results from various formats to consistent {M: {mode: {accuracy: value}}}"""
    if not results:
        return {}

    converted = {}
    for key in results:
        if isinstance(key, str) and key.startswith("M"):
            try:
                M = int(key[1:])
                converted[M] = results[key]
            except ValueError:
                continue
        elif isinstance(key, int):
            converted[key] = results[key]

    return converted if converted else results


def extract_accuracy_from_result(result):
    """Extract accuracy value from different result formats"""
    if isinstance(result, dict):
        if "accuracy" in result:
            return result["accuracy"]
        elif "phase2" in result and "epochs" in result["phase2"]:
            return max(result["phase2"]["epochs"])
    return None


def print_results_table(results):
    """Print results as a formatted ASCII table"""
    results = normalize_results(results)
    m_values = sorted(results.keys())

    if not m_values:
        print("✗ No data to display")
        return False

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'M':<8} {'Learned':<15} {'DCT':<15} {'Hadamard':<15} {'Random':<15}")
    print("-"*80)

    for M in m_values:
        learned_acc = 0
        dct_acc = 0
        hadamard_acc = 0
        random_acc = 0

        if "learned" in results[M] and "error" not in results[M]["learned"]:
            acc = extract_accuracy_from_result(results[M]["learned"])
            if acc is not None:
                learned_acc = acc * 100 if acc < 1 else acc

        dct_key = "dct2d" if "dct2d" in results[M] else "dct"
        if dct_key in results[M] and "error" not in results[M][dct_key]:
            acc = extract_accuracy_from_result(results[M][dct_key])
            if acc is not None:
                dct_acc = acc * 100 if acc < 1 else acc

        had_key = "hadamard2d" if "hadamard2d" in results[M] else "hadamard"
        if had_key in results[M] and "error" not in results[M][had_key]:
            acc = extract_accuracy_from_result(results[M][had_key])
            if acc is not None:
                hadamard_acc = acc * 100 if acc < 1 else acc

        if "random" in results[M] and "error" not in results[M]["random"]:
            acc = extract_accuracy_from_result(results[M]["random"])
            if acc is not None:
                random_acc = acc * 100 if acc < 1 else acc

        print(f"M={M:<6} {learned_acc:>6.2f}% {dct_acc:>12.2f}% {hadamard_acc:>12.2f}% {random_acc:>12.2f}%")

    print("="*80 + "\n")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = sys.argv[1]

    try:
        print(f"Loading results from: {json_path}")
        with open(json_path) as f:
            results = json.load(f)

        print(f"Generating table...")
        if print_results_table(results):
            print("✓ Done!")

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
