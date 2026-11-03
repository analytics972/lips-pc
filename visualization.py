"""
Author: Tomer Porat

Visualization and plotting utilities
"""

import json
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config import DATASET


def extract_accuracy_from_result(result):
    """Extract accuracy value from different result formats"""
    if isinstance(result, dict):
        if "accuracy" in result:
            return result["accuracy"]
        elif "phase2" in result and "epochs" in result["phase2"]:
            return max(result["phase2"]["epochs"])
    return None


def normalize_results(results):
    """Convert results from various formats to consistent {M: {mode: {accuracy: value}}}"""
    if not results:
        return {}

    # Check if results use string keys like "M8", "M16"
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


def get_data_for_plotting(results):
    """Extract accuracy data for plotting across all modes and M values"""
    results = normalize_results(results)
    data = {"learned": [], "dct": [], "hadamard": [], "random": []}
    m_values = []

    for M in sorted(results.keys()):
        m_values.append(M)
        for mode in ["learned", "dct", "hadamard", "random"]:
            # support both "dct"/"hadamard" (old) and "dct2d"/"hadamard2d" (new) key names
            key = "dct2d" if (mode == "dct" and "dct2d" in results[M]) else mode
            key = "hadamard2d" if (mode == "hadamard" and "hadamard2d" in results[M]) else key
            if key in results[M] and "error" not in results[M][key]:
                result = results[M][key]
                acc = extract_accuracy_from_result(result)
                data[mode].append(acc * 100 if acc is not None and acc < 1 else (acc or 0))
            else:
                data[mode].append(0)

    return m_values, data


def print_results_table(results):
    """Print results as a formatted ASCII table"""
    results = normalize_results(results)
    m_values = sorted(results.keys())

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


def plot_comparison(results, output_path):
    """
    Generate comparison plots for all modes across M values

    Args:
        results: dict with structure {M: {mode: result}} or loaded JSON
        output_path: path to save plot
    """
    if not HAS_MATPLOTLIB:
        print("⚠ matplotlib not available - skipping plot generation")
        return

    m_values, data = get_data_for_plotting(results)

    if not m_values:
        print("✗ No data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"learned": "#FFA07A", "dct": "#FF6B6B", "hadamard": "#4ECDC4", "random": "#45B7D1"}
    for mode in ["learned", "dct", "hadamard", "random"]:
        ax.plot(m_values, data[mode], marker='o', linewidth=2.5, label=mode.upper(),
                color=colors[mode], markersize=10)

    ax.set_xlabel("M (Compression Dimension)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
    ax.set_title(f"Measurement Mode Comparison ({DATASET.upper()})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(m_values)

    # Set y-axis to 0-100 range
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved plot: {output_path}")
    plt.close()


def load_results_from_json(json_path):
    """Load results from JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results_json_path> [output_plot_path]")
        print("\nExamples:")
        print("  # Generate plot and print table")
        print("  python visualization.py logs_run_all_20260422_191031/results.json comparison.png")
        print("\n  # Just print table from existing results")
        print("  python visualization.py logs_run_all_20260422_191031/results.json")
        sys.exit(1)

    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading results from: {json_path}")
    results = load_results_from_json(json_path)

    print(f"Generating results table...")
    print_results_table(results)

    if output_path:
        print(f"Generating comparison plot...")
        plot_comparison(results, output_path)
