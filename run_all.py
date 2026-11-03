"""
Author: Tomer Porat

Experiment: Learn diffusers and train classifiers for each mode
No encoder - directly optimize D + classifier for each measurement mode
"""

import os
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DEVICE, M_VALUES, MEASUREMENT_MODES, LOGS_DIR, BATCH_SIZE, DATASET, DATA_DIR
from pipeline import TrainingPipeline
# from pipeline_enhanced import EnhancedTrainingPipeline as TrainingPipeline
from utils import get_dataset_and_transforms
from visualization import plot_comparison, print_results_table
from results import save_results, print_summary


def run_experiment_per_M(M):
    """
    For each M, test all measurement modes:
    - Learn diffuser D (or create fixed matrix)
    - Train classifier on measurements from that diffuser
    - Compare results across modes
    """
    results = {}

    print(f"\n{'='*70}")
    print(f"M={M}: TESTING ALL MEASUREMENT MODES")
    print(f"{'='*70}\n")

    # Prepare data loaders (shared across all modes)
    dataset_class, train_transform, test_transform = get_dataset_and_transforms(DATASET, DATA_DIR)
    train_dataset = dataset_class(DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = dataset_class(DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # For each measurement mode
    for mode in MEASUREMENT_MODES:
        print(f"  Testing {mode.upper()}...")
        # Create fresh pipeline for this mode
        pipeline = TrainingPipeline(M, device=DEVICE)

        try:
            # Extract or create diffuser for this mode
            if mode == "learned":
                diffuser = pipeline.get_diffuser(mode, train_loader=train_loader, test_loader=test_loader)
            else:
                diffuser = pipeline.get_diffuser(mode)

            # Train classifier on this diffuser's measurements.
            # All modes use the same improved Phase 2: full train set +
            # ResNet18Classifier + cosine LR + gradient clipping.
            phase2 = pipeline.phase2_train(
                diffuser, mode=mode,
                train_loader=train_loader,
                test_loader=test_loader,
            )
            results[mode] = {"M": M, "mode": mode, "accuracy": phase2["best"]}
            print(f"  ✓ {mode.upper()}: {phase2['best']:.2%}\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[mode] = {"M": M, "mode": mode, "error": str(e)}

    return results


def main():
    """Run all experiments"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{LOGS_DIR}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nMeasurement Mode Comparison")
    print(f"Device: {DEVICE} | M values: {M_VALUES}")
    print(f"Output: {log_dir}\n")

    # Run experiments
    all_results = {}
    for M in M_VALUES:
        print(f"M={M}")
        all_results[M] = {}
        results_for_M = run_experiment_per_M(M)
        for mode, result in results_for_M.items():
            all_results[M][mode] = result
        print()

    # Save and report
    save_results(all_results, log_dir)
    print_results_table(all_results)
    plot_comparison(all_results, f"{log_dir}/comparison.png")
    print_summary(all_results)


if __name__ == "__main__":
    main()
