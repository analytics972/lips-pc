# LIPS-PC: Learning Illumination Patterns for Single-Pixel Classification

**Author: Tomer Porat**

Code accompanying the paper:

> **"LIPS-PC: Learning Illumination Patterns for Single-Pixel Classification."**
> Submitted to *IEEE Transactions on Computational Imaging*.

## What this is

An end-to-end deep learning pipeline that replaces a fixed, hand-designed sensing transform with a
**learned linear embedding**, trained jointly with a downstream classifier under a single
cross-entropy objective, with no image ever reconstructed along the way. In other words: instead of
compressing data with a generic transform (DCT, Hadamard, random projection) and hoping it keeps
what the task needs, the compression matrix itself is optimized end-to-end for classification
accuracy, under a hard architectural constraint (one fixed linear operator, identical for every
input) that keeps it deployable on real single-pixel/compressive-sensing hardware.

The same pipeline, classifier, and training recipe run unmodified across four datasets, two standard
vision benchmarks (MNIST, Fashion-MNIST) and a real industrial case: RIAWELC, a 24K-image X-ray
weld-defect dataset. Across all three, at the most aggressive compression budget tested (`M=8`, under
1% of the input dimension) the learned operator beats the best fixed transform (2D DCT, 2D Hadamard,
random projection) by 9 to 15 percentage points, reaching an accuracy the best fixed transform only
matches at 2 to 4x more measurements. That advantage shrinks as the measurement budget grows, and on
RIAWELC it reverses by a small margin at intermediate budgets, consistent with the paper above.

![Accuracy of the learned linear measurement operator minus the best fixed transform, versus the measurement budget M (out of D=1024). On a real industrial X-ray dataset and two standard benchmarks the learned operator wins by 9 to 15 percentage points at M=8; the advantage shrinks to zero, or presents a small reversal, as M grows.](figures/margin_vs_m.png)

**Why the low-M regime matters**: in dose- or time-constrained acquisition (ionizing probes such as
X-ray, or any setting where each measurement costs acquisition time), the number of measurements `M`
sets the physical cost directly. Because the learned operator's advantage is concentrated exactly at
small `M`, the same target accuracy can be reached at a smaller `M`, and therefore potentially at a
lower delivered dose or a shorter acquisition, without changing the classifier or the detector
hardware (the learned operator stays a single fixed `M x D` matrix, loadable onto the same modulator
as any fixed transform). The paper flags extending this argument to clinical dose-constrained imaging
as future work, not a claim made here.

## Repository contents

| File | Purpose |
|---|---|
| `config.py` | Central configuration: dataset selection, measurement budgets (`M_VALUES`), diffuser architecture, training hyperparameters. |
| `run_all.py` | Main experiment runner. Sweeps all configured `M` values and measurement modes, trains, evaluates, and saves results. |
| `pipeline.py` | `TrainingPipeline`, orchestrating diffuser (Phase 1) and classifier (Phase 2) training. |
| `models.py` | Diffuser architectures (fixed and learned) and classifier heads. |
| `diffusers.py` | Fixed measurement-matrix construction (2D DCT, 2D Hadamard, random) and the measurement operator `y = Dx`. |
| `datasets_custom.py` | Custom dataset loader for RIAWELC (industrial X-ray weld-defect radiographs). |
| `utils.py` | Dataset/transform loading utilities, dispatching to `torchvision` datasets or `datasets_custom`. |
| `visualization.py` | Generates comparison plots from saved results. |
| `results.py` | Saving, loading, and printing of experiment results (JSON). |
| `table_results.py` | Prints a results JSON file as a formatted table. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. Uses PyTorch; a CUDA or Apple-Silicon (MPS) GPU is used automatically if
available, otherwise it falls back to CPU.

## Data

Select the active dataset via `DATASET` in `config.py` (`"mnist"`, `"fmnist"`, `"cifar10"`, or
`"riawelc"`).

- **MNIST**, **Fashion-MNIST**, **CIFAR-10**: downloaded automatically via `torchvision` on first
  run, into `DATA_DIR` (default `./data`, relative to wherever `run_all.py` is launched from). No
  manual download needed.
  - MNIST: http://yann.lecun.com/exdb/mnist/ (fetched automatically by `torchvision.datasets.MNIST`)
  - Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist (fetched automatically by
    `torchvision.datasets.FashionMNIST`)
  - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html (fetched automatically by
    `torchvision.datasets.CIFAR10`)

- **RIAWELC** (industrial X-ray weld-defect radiographs, 4 classes: ND, CR, LP, PO; not bundled, not
  auto-downloaded):
  1. Download it from the official repository: https://github.com/stefyste/RIAWELC
     (`git clone https://github.com/stefyste/RIAWELC.git`), or see the dataset paper, Totino,
     Spagnolo, Perri, *"RIAWELC: A Novel Dataset of Radiographic Images for Automatic Weld Defects
     Classification,"* IJECER 12(2), 2023, DOI: 10.53375/ijecer.2023.320.
  2. Place the extracted dataset as a **sibling directory of this repository**, named `data_riawelc`,
     containing the four class folders directly inside it:
     ```
     lips-pc/              <- this repository
     data_riawelc/         <- sibling directory
       ND/                 <- no defect
       CR/                 <- crack
       LP/                 <- lack of penetration
       PO/                 <- porosity
     ```
     This matches `config.py`'s `DATA_DIR = "../data_riawelc"` for the `riawelc` branch (a path
     relative to the directory `run_all.py` is launched from). If you keep the dataset elsewhere,
     just edit `DATA_DIR` in `config.py` to point at the folder containing those four class
     subfolders.
  3. `RIAWELC_SPLIT_BY` in `config.py` controls the train/test split granularity (`"crop"`,
     `"radiograph"`, or `"weld"`); see the comment above it in `config.py` for the tradeoffs.

## Running

Select the dataset and sweep parameters in `config.py` (`DATASET`, `M_VALUES`, `MEASUREMENT_MODES`,
`DIFFUSER_ARCH`, `PHASE1_EPOCHS`, `PHASE2_EPOCHS`), then, from inside this directory:

```bash
python run_all.py
```

This trains and evaluates every configured `(measurement mode, M)` combination, writing results and
a comparison plot to a timestamped `logs_run_all_<timestamp>/` directory.

To regenerate a plot or print a table from an existing results file:

```bash
python visualization.py <results.json> <output.png>
python table_results.py <results.json>
```

## Method summary

Given an image `x`, a measurement matrix `D` (fixed or learned) produces a compressed measurement
`y = Dx`. A classifier is trained directly on `y`; there is no image-reconstruction step anywhere in
the loop. The learned embedding (`residual_dct` / `residual_hadamard` in `config.py`) is
parameterized as a fixed, structured base matrix (2D DCT or 2D Hadamard) plus a jointly-trained
low-rank correction `Δ`, kept as a single input-agnostic linear operator throughout (no per-sample
adaptation), matching the constraint that real compressive-sensing hardware cannot alter its sensing
pattern based on content it has not yet sensed.

## License

See repository owner for licensing terms.
