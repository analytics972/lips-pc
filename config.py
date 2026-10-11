"""
Author: Tomer Porat

Configuration for Linear Diffuser Ghost Imaging Pipeline
Dataset-agnostic: Works with CIFAR10, MNIST, Fashion-MNIST

OPTIMIZED (2026-05-14):
- Fixed NonLinearDiffuser to work with all image sizes
- Extended Phase 1/2 epochs for better convergence
- Separate learning rates for diffuser vs classifier
"""

import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ============================================
# DATASET SELECTION
# ============================================
# Choose one: "cifar10", "mnist", "fmnist", "riawelc"
DATASET = "fmnist"
# 2026-08-14: DCT-vs-Hadamard anchor ablation, continuing to Fashion-MNIST.
# RIAWELC result (residual_dct vs residual_hadamard, both +mlp+10ep,
# M=8-128): DCT 81.50/85.80/86.28/86.32/85.82%, Hadamard
# 79.68/85.23/85.41/85.14/85.59% -- Hadamard lost at ALL 5 M values
# (deficits 0.23-1.82pp), small but consistent one-directional pattern.
# MNIST result: DCT 95.27/97.81/98.80/98.98/98.75%, Hadamard
# 95.48/98.12/98.73/98.99/98.93% -- Hadamard WINS or ties at 4/5 M values
# (M=8 +0.21, M=16 +0.31, M=32 -0.07, M=64 +0.01, M=128 +0.18) --
# CONTRADICTS the RIAWELC pattern. Using FMNIST as tiebreaker before
# drawing any conclusion.
# DATASET = "riawelc"  # anchor ablation on RIAWELC complete, see numbers above
# DATASET = "mnist"    # anchor ablation on MNIST complete, see numbers above
# 2026-08-14: diffuser-matrix visualization sweep across all 4 datasets is
# DONE. Now testing the open "DCT-vs-Hadamard anchor" question raised in
# Section 3.3 of the thesis: does anchoring the learned residual correction
# to DCT2D (vs Hadamard2D) actually matter for the LEARNED result, or only
# for which fixed transform is strongest on its own? First run: RIAWELC,
# residual_dct (DCT anchor) + mlp decoder, 10 epochs, M=8-128 -- this exact
# combo (plain residual_dct, not RIAWELC's usual network_residual_dct, at
# only 10 epochs) has not been run before, needed as the matched baseline
# for the residual_hadamard comparison that follows. MNIST/FMNIST already
# have their residual_dct+mlp+10ep baselines from the visualization sweep
# (95.27/97.81/98.80/98.98/98.75% and 84.50/87.92/88.27/88.82/88.95%).
# DATASET = "fmnist"  # Fashion-MNIST diffuser sweep complete, matrices saved M=8-128
# DATASET = "mnist"  # MNIST diffuser sweep complete, matrices saved M=8-128
# DATASET = "riawelc"  # RIAWELC diffuser-visualization sweep complete, matrices saved M=8-128
# 2026-08-11: reverted from the MNIST+proxy_cnn decoder canary (stopped
# before completion, user redirected effort to RIAWELC instead -- that's the
# dataset with an actual known unresolved gap, not just an unconfirmed
# defense of one abstract sentence). Back to RIAWELC to test whether
# tightening the network-generated delta's capacity (NETWORK_RESIDUAL_MAX_RATIO
# below) narrows the M=32-128 deficit vs DCT2D/Hadamard2D, WITHOUT touching
# the decoder -- a decoder change would raise all 4 modes' accuracy together
# and likely leave the RELATIVE gap unchanged, which is exactly the confound
# the user flagged.
# 2026-08-09: back to RIAWELC -- the CIFAR-10 canaries (network_residual_dct
# RGB, and grayscale+network_residual_dct combined) both came back
# unpromising: grayscale consistently trailed RGB by 6-7pp at matched
# epochs (color is genuine class-discriminative signal on CIFAR-10, unlike
# the already-grayscale RIAWELC/MNIST/Fashion-MNIST), and the RGB run itself
# was only tracking toward, not clearly beating, the existing 47.99%
# baseline. Refocusing back on RIAWELC, where network_residual_dct has an
# actual confirmed win -- see the max_ratio=1.0 test queued below, targeted
# at M=64 specifically (the point of peak deficit vs DCT2D).
# DATASET = "cifar10"

# UNIFIED IMAGE SIZE: All datasets resized to 32×32
# This ensures identical pipeline and fair comparison across all datasets
IMAGE_SIZE = 32
FLAT_DIM = 32 * 32  # 1024 for grayscale, 3072 for RGB

# Data configuration (auto-set based on DATASET)
DATA_DIR = "./data"

CIFAR10_GRAYSCALE = False
# 2026-08-14: switched to RGB (D_in=3072) for the CIFAR-10 diffuser-matrix
# visualization sweep -- matches Table 1's actual reported CIFAR-10
# methodology, which is RGB not grayscale.
# 2026-08-09: canary -- converts CIFAR-10 to single-channel (D_in=1024,
# matching MNIST/FMNIST/RIAWELC) instead of RGB (D_in=3072). Motivation:
# isolates whether CIFAR-10's narrow high-M margin and the free-tensor
# delta's overfitting risk (M*D_in params) are driven by D_in size or by
# color/natural-image content specifically -- CIFAR-10 grayscale removes the
# color channel while keeping the same natural-image content, textures, and
# classes. Distinct, single-variable change vs the network_residual_dct
# canary above (which changes the diffuser, not the input). Set to False to
# restore standard RGB CIFAR-10.

if DATASET == "cifar10":
    NUM_CLASSES = 10
    INPUT_CHANNELS = 1 if CIFAR10_GRAYSCALE else 3
    FLAT_DIM = 1024 if CIFAR10_GRAYSCALE else 3072
elif DATASET in ["mnist", "fmnist"]:
    NUM_CLASSES = 10
    INPUT_CHANNELS = 1
    FLAT_DIM = 1024  # 32x32x1 (upsampled from 28x28)
elif DATASET == "riawelc":
    NUM_CLASSES = 4  # ND, CR, LP, PO
    INPUT_CHANNELS = 1  # grayscale X-ray (downsampled from 224x224)
    FLAT_DIM = 1024  # 32x32x1
    DATA_DIR = "../data_riawelc"  # Path to extracted RIAWELC dataset
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

# RIAWELC train/test split granularity (filename hierarchy:
#   RRT-101R = physical weld -> _Img2 = exposure -> [12][10] = crop):
#   "weld" (STRICTEST, recommended), all crops of the same physical weld
#       stay on one side. The whole dataset comes from only 29 welds!
#   "radiograph", groups by exposure (RRT-XXX_ImgN); still leaks 95.8% of
#       test crops via other exposures of the same weld.
#   "crop" (legacy, ROLLBACK), old random crop-level split; 97.8% of test
#       crops shared a source film with training crops (inflated accuracy).
RIAWELC_SPLIT_BY = "crop"
# 2026-08-07: DELIBERATE DEFAULT (decision made, not a temporary diagnostic
# anymore). Started 2026-08-06 as a diagnostic-only A/B; the user reviewed
# the rationale and explicitly decided to adopt "crop" as the reported
# default going forward.
#
# Rationale FOR: 24/29 welds (82.8%) and 95.4% of all crops span more than
# one defect class (verified 2026-08-06 by grouping all 21,964 filenames by
# weld ID against data_riawelc/) -- the simplest leakage shortcut ("weld ID
# predicts label") cannot explain inflated accuracy here, since weld ID
# mostly doesn't determine the label.
#
# Caveat NOT fully resolved by that rationale (record this honestly in the
# thesis methodology/limitations section, don't omit it): the ORIGINAL
# reason for the weld-level split (see riawelc_split_leakage.md,
# 2026-07-15) was a subtler mechanism than "weld ID -> label" -- near-
# duplicate crops. Even different EXPOSURES of the SAME weld were found to
# be "strongly correlated (same weld geometry, same defects re-photographed)",
# which is why the split went all the way to weld-level instead of stopping
# at radiograph-level. A model could still exploit weld-specific nuisance
# correlations (film noise, exposure, background texture) shared between a
# train crop and a near-duplicate test crop from the SAME weld -- something
# the 82.8%/95.4% multi-class statistic does not address, since it only
# rules out the label-shortcut mechanism, not the near-duplicate one. No
# matched "weld" run under the current architecture (residual_dct +
# ProxyHybridCNN) was completed before this decision -- the direct A/B this
# entry originally called for was not carried out.
#
# "weld" split is still used and referenced throughout conference_paper/ and
# most of this project's existing memory/thesis documents; those must be
# reconciled with this new default before RIAWELC numbers are reported
# anywhere as final (see CLAUDE.md 2026-08-07 entry).

# Model Architecture (auto-adjusted based on DATASET)
if DATASET == "riawelc":
    ENCODER_CONFIG = {
        "channels": [1, 64, 128, 256, 512],  # Grayscale input for X-rays
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
    }
else:
    ENCODER_CONFIG = {
        "channels": [INPUT_CHANNELS, 128, 256, 512] if DATASET == "cifar10" else [1, 64, 128, 256],
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
    }

CLASSIFIER_CONFIG = {
    "hidden_dims": [512, 512, 256, 128],
    "dropout_rates": [0.3, 0.3, 0.2, 0.1],
    "use_batch_norm": True,
}

# ============================================
# OPTIMIZED TRAINING PARAMETERS (2026-05-14)
# ============================================
# Extended epochs for better convergence
PHASE1_EPOCHS = 10  # 2026-08-13: MNIST/FMNIST diffuser-visualization sweep -- user
PHASE2_EPOCHS = 10  # requested 10 epochs (not 35) since these datasets converge fast;
# purpose is collecting a representative learned matrix for the thesis figure,
# not chasing Table 1's exact reported accuracy. Revert to 35 for CIFAR-10
# (harder dataset, not covered by this shortcut -- ask user first) or for any
# RIAWELC/accuracy-sensitive run.
# PHASE1_EPOCHS = 35#60#35#20#35#10#10#15#10#5        # Phase 1: Learn diffuser + classifier (proven to work from previous runs)
# PHASE2_EPOCHS = 35#60#35#20#35#10#10#15#10#5         # Phase 2: Train classifier on fixed measurements (proven to work from previous runs)
# 2026-08-12: reverted 60->35 per user's request (cap stays at 35) --
# stopped before finishing. Refocusing on a genuinely different mechanism
# instead of more training time -- see NetworkResidualDiffuser's learnable
# latent below.
# 2026-08-09: bumped 20->35 for the M=8-focused push -- applies equally to
# all 4 modes (learned/dct2d/hadamard2d/random), so the controlled-recipe
# comparison stays fair; this is a "did training actually converge" check,
# not a change specific to the learned diffuser.

# Phase 1 specific parameters (for learned mode)
PHASE1_LR_DIFFUSER = 1e-3      # Separate learning rate for diffuser
PHASE1_LR_CLASSIFIER = 5e-3    # Separate learning rate for classifier

# Learned-diffuser architecture:
#   "resnet"    - ResNet-18-style CNN encoder, ~92%+ on CIFAR10. NOT physically
#                 realizable as GI masks: runs conv layers directly on the raw
#                 image, which no single-pixel bucket detector could do. Use
#                 for an accuracy ceiling / ML-only ablation, not for the
#                 mask-based comparison (proposal §1.1/§4.1).
#   "simple"    - legacy 3-conv NonLinearDiffuser, ~85% ceiling on CIFAR10.
#                 Same physical-realizability caveat as "resnet".
#   "residual"     - proposal §4.1.2 Residual Diffuser: a genuine (M,D) mask
#                    matrix D = base(random, fixed) + delta(learned), applied
#                    via y = D @ x. Physically realizable, input-agnostic.
#   "residual_dct" - same ResidualDiffuser, but `base` is DCT2D instead of
#                    random noise (DCT2D is already a much stronger fixed
#                    baseline than Random -- see 2026-08-02 full-run results
#                    below -- so refining it should beat plain "residual").
#   "generator"    - proposal §4.1.1 Generator-Based Diffuser: small FC
#                    network maps a fixed latent vector to a full (M,D) mask
#                    matrix (network never sees the image). Physically
#                    realizable, input-agnostic. See models.py
#                    GeneratorDiffuser docstring for how it simplifies
#                    proposal Fig. 2 (LayerNorm instead of BatchNorm1d, no
#                    output sigmoid).
#   "generator_distill" - same GeneratorDiffuser network, but trained in two
#                    stages (pipeline.py::_learn_generator_distilled): (1)
#                    train a ResNetDiffuser teacher normally; (2) train the
#                    generator's FC weights so its matrix's measurements
#                    match the frozen teacher's (MSE), plus classification
#                    loss. Literally "a network learned a better diffuser by
#                    studying the successful CNN" -- still outputs one
#                    physically-realizable (M,D) matrix at inference, only
#                    the training signal differs from plain "generator".
#                    ~2x the runtime of the other options (trains teacher +
#                    student sequentially).
#   "frozen_decoder" - Two-stage, decoupled training
#                    (pipeline.py::_learn_frozen_decoder_diffuser): Stage A
#                    trains ProxyCNN on FIXED DCT2D alone (no learned matrix)
#                    so the decoder can't co-adapt to memorize anything
#                    mask-specific; Stage B freezes that decoder and trains
#                    only a rank-16 LowRankResidualDiffuser delta on top of
#                    it. ~3x runtime (Stage A + Stage B + normal Phase 2).
#                    RESOLVED 2026-08-05: a single M=64 run gave 83.46%
#                    (beats DCT2D 81.40%), but a 4-run repeat (see
#                    frozen_decoder_mechanism memory) gives mean+-std
#                    81.36+-1.59% vs DCT2D 80.91+-0.82% vs Hadamard2D
#                    82.22+-1.08% -- a STATISTICAL TIE, not a win. The 83.46%
#                    was the noise distribution's high tail. Do not cite it.
#                    Also loses to free-delta residual_dct at M=8 (56.26% vs
#                    66.84%, single run) -- decoupling itself seems to hurt
#                    at low M where co-adaptation isn't harmful in the first
#                    place (rank=16 isn't even binding when M<16). Not
#                    recommended as a default; kept for reference/comparison.
#   "residual_lowrank" - LowRankResidualDiffuser (delta=U@V, rank=16),
#                    trained JOINTLY with the classifier (no decoupling, 1x
#                    runtime). An earlier attempt at capping delta's capacity
#                    without frozen_decoder's cost; plateaued around ~74% at
#                    M=32 in an abandoned run (roughly tied with residual_dct
#                    and DCT2D there, no clear win). Superseded by
#                    "residual_lowrank8" as the next thing to try.
#   "residual_lowrank8" - Same as "residual_lowrank" but rank=8 instead of
#                    16, still jointly trained (1x runtime, not 3x).
#                    2026-08-05 hypothesis, UNTESTED: at M<=8, rank=8 is
#                    already >= M so the low-rank constraint is barely
#                    binding -- delta should behave close to the free-delta
#                    residual_dct there, preserving its large low-M win
#                    (+13-16pp over DCT2D). At higher M, rank=8 caps delta's
#                    capacity harder than rank=16 did, which might curb the
#                    overfitting-driven loss residual_dct showed at M>=32
#                    (gap widened to -11.8pp at M=128). Goal: one mechanism
#                    that wins at low M and doesn't lose at high M, without
#                    frozen_decoder's 3x cost. NOT YET RUN as of this
#                    writing -- residual_lowrank (rank=16) plateaued at ~74%
#                    (M=32) in an earlier attempt, so there's no guarantee
#                    rank=8 does better; needs single-run screening across
#                    M=8/16/32/64 first, THEN multi-seed on whichever M looks
#                    promising, before trusting any result (see the
#                    frozen_decoder 4-run saga above for why single runs
#                    cannot be trusted here).
#
# 2026-08-02 FULL run (RIAWELC, all 262 train batches, 10+10 epochs,
# DIFFUSER_ARCH="residual"; M=64/128 not yet run):
#   M=8:  LEARNED(residual)=62.88%  DCT2D=46.86%  Hadamard2D=46.72%  Random=38.53%
#   M=16: LEARNED(residual)=63.84%  DCT2D=63.70%  Hadamard2D=57.65%  Random=46.84%
#   M=32: LEARNED(residual)=65.42%  DCT2D=64.34%  Hadamard2D=62.80%  Random=(cut off)
# residual BEATS all fixed baselines at every M tested, largest margin at
# low M -- reproduces proposal §4.4's own key finding ("learned diffusers
# dominate at low measurement rates, DCT approaches near-optimal at higher
# M"). Still far below resnet's ~90%+ ceiling -- that gap is structural
# (linear projection vs. per-image nonlinear CNN encoding), not a tuning
# bug; see CLAUDE.md 2026-08-02 entry and memory
# proposal_vs_code_learned_diffuser_mismatch.md. Earlier "94.27% residual
# lost to 97.60% random" quick-test note was a 4-epoch/6k-subset artifact --
# disregard, full run above supersedes it.
#   "network_residual_dct" - proposal_vs_code combo, 2026-08-08 (user's idea):
#                    DCT2D base like "residual_dct", but the correction delta
#                    is the OUTPUT of a small FC network (like
#                    GeneratorDiffuser's architecture) applied to a fixed
#                    latent vector, instead of a free (M,D_in) tensor. Targets
#                    the SAME overfitting problem as residual_lowrank/
#                    residual_lowrank8 (delta's free-parameter count is the
#                    documented cause, residual_delta_overfitting_fix.md) but
#                    via a network's inductive bias instead of an explicit
#                    low-rank U@V factorization. UNTESTED as of this writing.
#                    See models.py::NetworkResidualDiffuser.
DIFFUSER_ARCH = "residual_hadamard"
# 2026-08-14: DCT-vs-Hadamard anchor ablation. RIAWELC's residual_dct+mlp+
# 10epoch DCT-anchor baseline just completed: 81.50/85.80/86.28/86.32/85.82%
# (M=8-128). Now the matched Hadamard-anchor run, identical config otherwise
# (mlp, 10 epochs, same M sweep) -- single-variable change (DIFFUSER_ARCH only).
# DIFFUSER_ARCH = "residual_dct"  # DCT-anchor baseline done, see numbers above
# 2026-08-13: switched for the MNIST/FMNIST/CIFAR-10 diffuser-matrix
# visualization sweep -- Table 1's methodology for these three datasets uses
# the free-tensor residual diffuser (residual_dct), NOT RIAWELC-specific
# network_residual_dct.
# DIFFUSER_ARCH = "network_residual_dct"  # RIAWELC-specific, sweep complete

# Cap on the network-generated delta's norm relative to the DCT2D base
# (models.py::NetworkResidualDiffuser.get_matrix, _clamp_delta_norm). Was
# hardcoded at 2.0 (the M=8-optimal value found 2026-08-09); the M=64
# max_ratio=1.0 test was queued back then ("targeted at M=64 specifically,
# the point of peak deficit vs DCT2D") but never run. Testing it now: a
# tighter cap gives the delta less room to overfit the ~29-weld training set
# at high M (see residual_delta_overfitting_fix.md), which is the suspected
# cause of the M=32-128 deficit -- without touching the decoder, so any
# change is attributable to the mask's own capacity, not a stronger CNN
# reading it.
NETWORK_RESIDUAL_MAX_RATIO = 2.0
# 2026-08-13: opposite-direction test (max_ratio=1.0, half the cap) RESULT:
# 87.07% -- a tie with the 87.14% baseline (-0.07pp, pure noise). Confusion
# matrix essentially unchanged (ND 98.80%, CR 85.95%, LP 81.11%, PO 81.67%;
# CR<->LP confusions 217 vs baseline's 233 -- also within noise). This is
# the 6th straight experiment (4 capacity-up + focal loss + this capacity-
# down test) landing in the 82-87% band without beating baseline --
# NETWORK_RESIDUAL_MAX_RATIO is not the lever in either direction. Reverted
# to 2.0 (proven baseline). Pivoting to a structurally different mechanism:
# COHERENCE_LOSS_WEIGHT below, now tested at M=8 instead of only M=64.
# NETWORK_RESIDUAL_MAX_RATIO = 1.0  # 2026-08-11 canary, stopped mid-run
# (inconclusive on M=64, the point that mattered) to refocus on the decoder
# test below instead, per user's request.
# 2026-08-09: at user's request, combining network_residual_dct WITH
# CIFAR10_GRAYSCALE in the SAME run rather than isolating each first --
# explicitly a TWO-VARIABLE exploratory canary (grayscale input AND
# network-generated diffuser, both vs the existing RGB+residual_dct Table 1
# baseline: 47.99%/35.67%/32.59%/25.59% at M=8). If this comes back
# surprising in either direction, don't attribute it to either change alone
# -- follow up with the two isolated runs (grayscale+residual_dct, and
# RGB+network_residual_dct) to find out which one actually did it.
# DIFFUSER_ARCH = "residual_dct"
# 2026-08-08: back from "conv_network_residual_dct" -- after 4 fix attempts
# (zero-init, GroupNorm, wider fixed channels, delta-norm clamp) the conv
# variant still trailed network_residual_dct by 15-20pp at the same early
# epochs (M=8: 39.26%/44.67% vs 58.76%/65.45% at epochs 1-2). Deprioritized
# rather than continuing to chase it -- network_residual_dct (FC-based) is
# the one that's actually working (M=8 Phase 2: 84.89%, ~matching the best
# existing RIAWELC number).
# 2026-08-08: TEMPORARY, exploratory canary -- testing network_residual_dct
# (new) TOGETHER with PHASE2_CLASSIFIER="set_transformer_proxy" (also new) in
# the SAME run, at the user's explicit request, despite this being a
# two-variable-at-once change (this assistant flagged the single-variable
# concern first). If this run's numbers are surprising in either direction,
# do NOT attribute the effect to either component alone -- rerun each
# separately (network_residual_dct+mlp, and residual_dct+set_transformer_proxy)
# before drawing any conclusion. Revert to "residual_dct" for the proven
# baseline.
# 2026-08-05: switched from "residual_lowrank8" to "residual_dct" specifically
# to test the CIFAR10 decoder hypothesis below -- residual_dct is the exact
# architecture used in the paper's main validated table (conference_paper/),
# so this run changes ONLY the decoder (PHASE2_CLASSIFIER=proxy_cnn, already
# set below) relative to that table, isolating the decoder as the single
# variable -- matching how the original RIAWELC ProxyCNN finding (69.09%->
# 78.38%, 2026-08-03) was established. Do not also change DIFFUSER_ARCH in
# this run, or the comparison to TransCL (arXiv:2207.11972, ~69% CIFAR10 at
# a similar measurement ratio with learned-linear-sensing + a Transformer
# decoder) is no longer apples-to-apples.
# Phase 1 recipe for the resnet arch: single SGD + OneCycle over all params
PHASE1_RESNET_MAX_LR = 0.1
PHASE1_RESNET_WEIGHT_DECAY = 5e-4

# Delta-overfitting fix -- REVERTED TO SAFE DEFAULT 2026-08-04 (same day):
# a single-change A/B at M=32 showed strong weight decay on the diffuser's
# delta helps THERE (74.32%->76.69%), but a live M=8 rerun with this as a
# blanket default showed WORSE, noisier Phase-1 val_acc (dips to 44-49%,
# vs a clean climb to 62-66% under the old flat 5e-4) -- the fix does not
# generalize across M and should NOT be a blanket default until it's
# validated per-M (delta has M*1024 params, so the "right" weight decay
# plausibly needs to scale with M, not be a single constant). Kept equal to
# PHASE1_RESNET_WEIGHT_DECAY so current behavior is IDENTICAL to
# pre-2026-08-04 (matches logs_run_all_20260803_091922). Only change this
# for a deliberate, isolated experiment -- do not leave it non-default.
PHASE1_DELTA_WEIGHT_DECAY = PHASE1_RESNET_WEIGHT_DECAY

# RIAWELC train-time flip augmentation (RandomHorizontalFlip+RandomVerticalFlip,
# on top of the existing RandomRotation(5)+RandomAffine(translate=0.05)).
# Weld-defect X-ray crops have no canonical orientation (a crack/porosity
# pattern is still a valid instance of its class flipped), so flips are
# physically legitimate, unlike e.g. flipping a MNIST digit. Motivation: the
# effective training-set diversity is tiny (~29 physical welds, crop-split),
# so free extra diversity from augmentation is one of the few totally
# architecture-independent levers left to try -- distinct from every mask/
# decoder-capacity change already tested (all either failed or unproven).
# 2026-08-12, UNTESTED.
RIAWELC_FLIP_AUGMENTATION = False
# RIAWELC_FLIP_AUGMENTATION = True  # stopped mid-run, reverted -- per user's
# request, staying focused on changes to the diffuser network itself, not
# data-side tricks that would need to apply equally to all 4 measurement
# modes to stay a fair comparison (and wouldn't specifically demonstrate an
# improvement to the learned mask/network).

# CIFAR10 train-time augmentation (RandomCrop(32,padding=4)+RandomHorizontalFlip).
# Set False to isolate the architecture/recipe gain from the augmentation's own
# contribution (literature value ~+2-3%) -- not yet run as a controlled ablation.
CIFAR10_AUGMENTATION = False
# 2026-08-06: back to False -- kept OFF deliberately so the run below
# (PHASE2_CLASSIFIER="proxy_resnet18") isolates decoder DEPTH as the only
# variable relative to the existing 58.58% (M=64, "learned", MLP) baseline
# and the failed 57.49% (M=64, "learned", ProxyCNN) attempt. Do not enable
# augmentation in the same run as an architecture change -- keep changes
# single-variable (established practice throughout this project).

# Phase-2 classifier architecture (applies to ALL measurement modes so the
# comparison stays fair):
#   "proxy_cnn" - ProxyCNN over the adjoint proxy image x_hat = D^T y
#                 (models.py::ProxyCNN). Physics-legal (runs after the bucket
#                 detector) and matches proposal S3.2.1 ("the model will be a
#                 CNN"). Broke the ~70% MLP asymptote on RIAWELC: fixed DCT2D
#                 M=32 69.09% -> 78.38% (2026-08-03). Also used as the
#                 Phase-1 classifier for matrix-based DIFFUSER_ARCHs
#                 (end-to-end), so the learned matrix is optimized for the
#                 decoder that will actually read it.
#   "mlp"       - legacy ClassifierResNet on the raw M-vector (rollback /
#                 reproduction of pre-2026-08-03 numbers).
#   "proxy_resnet18" - same adjoint-proxy idea as "proxy_cnn", but with a
#                 full 4-stage ResNet-18-depth decoder (models.py::
#                 ProxyResNet18) instead of ProxyCNN's 3 stages. 2026-08-06:
#                 testing whether ProxyCNN's CIFAR10 regression (M=64,
#                 "learned": 57.49% vs 58.58% MLP baseline) was simply an
#                 undersized decoder rather than the adjoint-proxy approach
#                 being fundamentally unsuited to natural images. Same depth
#                 as ResNetDiffuser (the encoder that took CIFAR10 84%->92%
#                 in this project) -- single-variable change (depth only)
#                 vs "proxy_cnn". RESULT (2026-08-06): tracked almost
#                 identically to "proxy_cnn" (54.89% vs 55.39% at the same
#                 Phase-1 epoch) -- depth/capacity was ruled out as the
#                 bottleneck, run stopped before Phase 2 as uninformative.
#   "proxy_hybrid" - models.py::ProxyHybridCNN. Since decoder depth didn't
#                 matter, tries a different axis: feed the classifier BOTH
#                 the CNN branch over the proxy image AND a small MLP branch
#                 over the raw M-vector y (what the plain "mlp" mode already
#                 uses successfully), concatenated before the final layer --
#                 instead of forcing a choice between representations.
#   "set_transformer" - models.py::GISetClassifier. 2026-08-08: adapted from
#                 a Set-Transformer classifier shared by Gal Katz (contact via
#                 Prof. Cohen) -- treats each of the M raw measurements as one
#                 token (Conv1d tokenizer), runs self-attention BETWEEN
#                 measurements (GISetSAB blocks), then pools permutation-
#                 invariantly (GISetPMA, Set Transformer/Lee et al. 2019)
#                 before a small MLP head. No adjoint/image reshape -- works
#                 with any diffuser, matrix-based or not. UNTESTED as of this
#                 writing.
#   "set_transformer_proxy" - models.py::GISetClassifierProxy. Same SAB/PMA
#                 body as "set_transformer", but tokenizes the adjoint-proxy
#                 image x_hat=D^Ty via ViT-style patches instead of the raw
#                 M-vector -- 2026-08-08 hypothesis (user): a single raw
#                 measurement carries no spatial structure on its own, so a
#                 patch of the reconstructed proxy (local spatial context)
#                 should be a more informative token than a lone scalar.
#                 Requires a genuine matrix like proxy_cnn/proxy_hybrid; CNN
#                 diffusers (resnet/simple) have none and fall back to MLP.
#                 UNTESTED as of this writing.
# Note: requires a genuine matrix to build the proxy; CNN diffusers
# (resnet/simple) have no matrix, so they always use the MLP path.
PHASE2_CLASSIFIER = "mlp"
# 2026-08-13: switched for MNIST/FMNIST/CIFAR-10 diffuser-matrix visualization
# sweep -- Table 1's methodology for these three uses the flat-MLP head, not
# RIAWELC-specific proxy_cnn.
# PHASE2_CLASSIFIER = "proxy_cnn"  # RIAWELC-specific, sweep complete
# 2026-08-11: REVERTED -- the enlarged GISetClassifierProxy transformer
# (d_model=128, 4 layers) LOST to proxy_cnn at M=8 (82.89% vs 87.14%,
# -4.25pp, well outside this project's ~1-2pp noise floor). Decoder-side is
# not the lever; back to the proven proxy_cnn decoder, now trying a
# diffuser-side (mask capacity) change instead -- see
# NETWORK_RESIDUAL_HIDDEN_DIM below.
# PHASE2_CLASSIFIER = "set_transformer_proxy"

# Capacity of GISetClassifierProxy ("set_transformer_proxy" decoder above).
# The 2026-08-08 default (d_model=64, 2 layers) was a small, un-scaled
# adaptation of someone else's code (Gal Katz's GISetClassifier), never
# benchmarked against anything -- not a deliberate capacity choice. TransCL
# (Mou & Zhang, arXiv:2207.11972, the paper already cited in the RIAWELC
# comparison) uses a full transformer backbone over CS-measurement tokens
# and reports near-lossless accuracy even at very low CS ratios, which is
# genuine literature support for trying a bigger transformer decoder here.
# BUT RIAWELC has only ~29 physical welds in training -- a much smaller
# regime than TransCL was evaluated on -- and this project already saw a
# stronger decoder (ProxyCNN) induce new delta-overfitting once
# (residual_delta_overfitting_fix.md). So this is a moderate increase, not a
# jump to TransCL's own scale, with extra dropout as a hedge against that
# known risk. 2026-08-11, UNTESTED: single-variable decoder test at M=64
# only (the current peak deficit) against the proxy_cnn baseline, same
# diffuser (network_residual_dct, NETWORK_RESIDUAL_MAX_RATIO=2.0).
GISET_PROXY_D_MODEL = 128
GISET_PROXY_N_LAYERS = 4
GISET_PROXY_N_HEADS = 4
GISET_PROXY_DROPOUT = 0.2
# RESULT (2026-08-11, M=8, RIAWELC): 82.89% vs proxy_cnn's 87.14% -- LOST by
# 4.25pp. Decoder capacity is not the lever for improving "Learned" here.

# Width of NetworkResidualDiffuser's delta-generating FC network (see
# models.py, default hidden_dim=512). Depth was already tested (2 vs 3
# hidden layers, 3 was worse -- see DIFFUSER_ARCH comment above) but width
# never was. 2026-08-11, UNTESTED: single-variable test at M=8 (fastest,
# and where "Learned" already leads, so any accuracy change is easy to read)
# against the proxy_cnn decoder (proven best) -- isolates the mask-network's
# own capacity as the only variable.
NETWORK_RESIDUAL_HIDDEN_DIM = 512
# NETWORK_RESIDUAL_HIDDEN_DIM = 1024  # tested, no benefit -- see result below

# Size of the fixed random latent vector fed into NetworkResidualDiffuser's
# delta-generating network (default 100). Untested axis, distinct from
# hidden_dim (network width, already tested, no benefit) and depth (already
# tested, worse) -- this is the INPUT dimensionality the network has to work
# with. A bigger latent gives the fixed random seed more independent
# "directions" to expand into the (M, D_in) delta via the network's learned
# weights, which is a different capacity lever than widening the hidden
# layers. Purely a diffuser-network change, not data-side -- doesn't affect
# fairness of the mode comparison. 2026-08-12, UNTESTED.
NETWORK_RESIDUAL_LATENT_DIM = 100
# NETWORK_RESIDUAL_LATENT_DIM = 256  # tested, trending worse (~81-83% vs
# 87.14%), stopped before Phase 2 finished -- another capacity increase that
# didn't help, consistent with the decoder/width results.

# Whether NetworkResidualDiffuser's input latent vector is a learnable
# nn.Parameter (True) or a frozen random buffer (False, original design).
# 2026-08-12, UNTESTED: three capacity increases in a row (decoder size,
# delta_net width, latent_dim) all failed or were neutral -- this is NOT
# another capacity increase (only ~100 extra scalars vs delta_net's ~500K+
# weights). Instead of asking the network's WEIGHTS to map an arbitrary
# fixed input to a good delta, this also lets the INPUT itself be optimized
# -- a different kind of lever, and a much cheaper one than anything tried
# so far. User explicitly asked for directions other than more capacity,
# more epochs, or ensembling; this is the first candidate.
NETWORK_RESIDUAL_LEARNABLE_LATENT = False
# NETWORK_RESIDUAL_LEARNABLE_LATENT = True  # tested, trended to ~84-85%
# (below the 87.14% baseline), stopped before Phase 2 finished -- fourth
# lever in a row that didn't beat baseline. Switching from blind trial-and-
# error to diagnosing the baseline model itself (confusion matrix, per-class
# accuracy, train/val gap) per user's request, before trying more levers.

# FocalLoss gamma (models.py::FocalLoss) for Phase 1 + Phase 2 training.
# 0 = disabled (plain label-smoothed cross-entropy, the proven baseline).
# 2026-08-12: diagnostic run (RIAWELC, M=8, "learned", proven baseline
# config) printed a confusion matrix: ND 98.15% (already saturated) vs
# CR/LP/PO at 82-85%, with 233 of the errors being CR<->LP confusions alone
# (crack vs lack-of-penetration look visually similar under X-ray) --
# concentrated class confusion, not general undertraining or overfitting.
# Focal loss redirects gradient effort away from already-easy ND examples
# toward hard/confused ones, adaptively, without hand-specifying the CR/LP
# pair. gamma=2.0 is the standard RetinaNet default, untuned here.
FOCAL_LOSS_GAMMA = 0.0
# FOCAL_LOSS_GAMMA = 2.0  # tested 2026-08-13: 86.64% vs 87.14% baseline --
# a wash/slightly worse. CR's own accuracy improved (84.86%->86.32%) but
# LP (82.69%->81.11%) and PO (81.67%->80.05%) got worse, and raw CR<->LP
# confusion count went UP (233->251). Reweighting gradient by difficulty
# doesn't fix this confusion -- reverted to isolate the max_ratio test below.
# RESULT (2026-08-11, M=8, RIAWELC): 86.91% vs proxy_cnn+hidden_dim=512's
# 87.14% -- a wash (-0.23pp, within noise). Width is not the lever either.

# Orthogonality/mutual-coherence regularization on the diffuser matrix
# D = base + delta (models.py, added to Phase 1 loss for "learned" mode
# only). Literature-grounded (Wu et al. ICML 2019 "Learning a Compressed
# Sensing Measurement Matrix via Gradient Unrolling"; MMD-based structural-
# constraint learned-matrix papers) -- fixed transforms (DCT2D/Hadamard2D)
# are orthogonal BY CONSTRUCTION, which is exactly what makes them efficient
# at high M (every measurement is non-redundant). The learned delta has no
# such constraint and may be eroding orthogonality to fit the ~29-weld
# training set, which would specifically hurt at HIGH M (near D_in=1024,
# almost the whole basis is in use, so redundant/correlated rows directly
# waste capacity) and barely matter at low M (plenty of basis directions
# unused) -- this mechanism, unlike the decoder/width attempts above,
# actually predicts the M-dependent shape of the observed deficit. 2026-08-12,
# UNTESTED: penalizes ||D_normalized @ D_normalized.T - I||^2 (mean squared
# off-diagonal cosine similarity between measurement rows). Weight untuned;
# 1.0 is a starting guess, not derived from anything.
COHERENCE_LOSS_WEIGHT = 0.0
# 2026-08-13: was set to 1.0 to test at M=8 (previously only scaffolded/
# stopped mid-run at M=64). Six straight training-side/capacity levers
# (decoder size, delta_net width, latent_dim, learnable latent, focal loss,
# max_ratio up AND down) all landed within noise of the 87.14% baseline --
# none touched the CR<->LP confusion that dominates the error pattern. This
# coherence-loss run was killed mid-training THREE times by the OS (laptop
# sleep despite caffeinate -d -i -m -s) and never completed; the user then
# redirected priorities away from the M=8 accuracy hunt toward documenting
# the learned diffuser matrices (visualized, per mode and per M) for the
# thesis -- reverted to 0.0 (proven 87.14% baseline) rather than restarting
# a 4th time. Revisit this experiment later if the accuracy hunt resumes.
# 2026-08-12 rationale for the coherence-loss idea (kept for reference, still
# untested to completion): this is a genuinely
# different mechanism: instead of changing how hard the classifier tries,
# it changes what the 8 measurement rows themselves capture, by penalizing
# ||D_normalized @ D_normalized.T - I||^2 so the rows stay mutually
# decorrelated (literature: Wu et al. ICML 2019 gradient-unrolling CS
# matrices). Motivating logic reconsidered for M=8 specifically: at M=8 the
# delta has the LEAST room to be redundant with only 8 rows out of 1024
# possible directions, so if the CR/LP-relevant information is genuinely
# there but the 8 learned directions overlap/waste capacity on similar
# projections, decorrelating them could recover it -- opposite reasoning
# from the original M=64 motivation (which predicted low-M wouldn't need
# this), but no other single-variable idea has moved the needle at M=8 so
# far, so this is worth actually measuring rather than assuming.
# COHERENCE_LOSS_WEIGHT = 0.0  # disabled, proven M=8 baseline (87.14%), rollback value
# PHASE2_CLASSIFIER = "mlp"  # was used for the CIFAR-10 canary, see above
# 2026-08-09: RESOLVED -- network_residual_dct+proxy_hybrid tested at M=8:
# 85.25%, actually slightly BELOW network_residual_dct+proxy_cnn's 85.57%
# (-0.32pp, within this project's established noise floor but with zero
# upside). Second confirmation (after residual_dct+hybrid vs +mlp: 84.75 vs
# 83.91, also a wash) that the raw-y MLP branch adds nothing on top of the
# adjoint-proxy CNN. Reverted to plain "proxy_cnn" -- simpler and at least as
# good. Do not revisit proxy_hybrid without a new reason to expect it'd help.
# 2026-08-09: testing pure proxy (CNN on x_hat only, no raw-y MLP branch)
# with network_residual_dct, per user's request -- distinct from the earlier
# mlp-vs-proxy_hybrid comparison (which showed almost no difference, 83.91%
# vs 84.75%, under this crop split). Single-variable vs the mlp baseline
# above (still DIFFUSER_ARCH="network_residual_dct", max_ratio=1.0 pending
# separately -- don't change both at once).
# 2026-08-08: switched back from "set_transformer_proxy" to isolate whether
# NetworkResidualDiffuser's slow start (M=8: 45.7%/51.8%/52.6% at epochs
# 1-3, vs 66/68/68% for residual_dct+proxy_hybrid) is caused by the new
# diffuser or the untested Set Transformer decoder -- this run changes only
# the diffuser back to the proven MLP head, single-variable.
# 2026-08-07: TEMPORARY, single-variable control experiment -- RIAWELC_SPLIT_BY
# stays "crop" (unchanged), only the decoder is switched back to the plain MLP
# to isolate whether the crop-split RIAWELC results (84.75/92.01/94.54% at
# M=8/16/32 under proxy_hybrid) come from the decoder or from the split itself.
# Compare this run's RIAWELC numbers against: (a) the old weld+MLP numbers
# (~63.97%/66.84%/67.26% around M=8) -- if close, the split isn't doing much
# on its own; (b) the crop+proxy_hybrid numbers above -- the full gap between
# this run and those is the decoder's contribution. Revert to "proxy_hybrid"
# after this comparison -- this is diagnostic only, not a new default.

# Phase 2 specific parameters (for all modes)
PHASE2_LR = 0.1                # SGD base learning rate
PHASE2_WARMUP_EPOCHS = 5       # Linear warmup for first 5 epochs
PHASE2_WEIGHT_DECAY = 5e-4

# Common hyperparameters
BATCH_SIZE = 64  # Reduced from 512 for MPS GPU (Apple Silicon)
LEARNING_RATE = 1e-2    # Fallback for legacy code
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
NESTEROV = True

# Regularization
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
GRADIENT_CLIP = 1.0
DROPOUT_RATE = 0.15

# M values to test
# 2026-08-05: CANARY RUN -- single M value first, before committing hours to
# a full sweep. M=64 chosen specifically because it's the closest match to
# TransCL's (arXiv:2207.11972) 54-measurement (1.8% ratio) CIFAR10 result
# (~69%), so this one run directly answers "does ProxyCNN close the gap for
# CIFAR10 the way it did for RIAWELC (69.09%->78.38%)?" All 4 modes are still
# run (not just "learned") -- the RIAWELC finding was actually driven by the
# FIXED DCT2D baseline improving, so dropping the fixed modes here would lose
# the ability to tell a real decoder effect from noise. If this looks
# promising, expand to M_VALUES = [8, 16, 32] and/or [64, 128] as separate
# runs (kept small deliberately after today's crash on a 5-M sweep in one
# process -- see the DIFFUSER_ARCH comment above for the root-cause writeup).
M_VALUES = [8, 16, 32, 64, 128]
# 2026-08-13: RIAWELC diffuser-visualization sweep COMPLETE (M=8-128, all 5
# matrices saved, all 5 figures in thesis.tex/thesis_he.html). Now MNIST, all
# 5 M values in one continuous run (residual_dct + mlp, 10+10 epochs -- see
# above) -- user explicitly said not to stop between M values for MNIST/FMNIST,
# only between datasets.
# M_VALUES = [128]  # RIAWELC done -- 99.11%, matrix saved
# M_VALUES = [64]  # done -- 98.54%, matrix saved
# M_VALUES = [32]  # done -- 96.34%, matrix saved
# M_VALUES = [16]  # done -- 93.81%, matrix saved
# M_VALUES = [8]  # done -- 86.66% (baseline reproduction), matrix saved
# 2026-08-13: M=8 accuracy hunt concluded (7 structurally distinct levers
# tried, all tied/failed vs 87.14% -- likely a genuine information ceiling
# at M=8). User redirected priorities: collect the trained "learned" matrix
# for each M in {8,16,32,64,128} (fixed modes' matrices are free, no
# training needed) to visualize alongside DCT2D/Hadamard2D/Random for the
# thesis (paper.tex). M=8 done (86.66%, matrix saved). Now M=16, same
# proven baseline config otherwise (network_residual_dct, proxy_cnn,
# crop split, 35+35 epochs) -- single-variable change (M_VALUES only).
# M_VALUES = [8]  # done -- 86.66% (baseline reproduction), matrix saved
# 2026-08-12: back to M=8 per user's request -- focus on pushing "Learned"
# accuracy up at M=8 specifically (target 90%+, from 87.14%), trying several
# quick methods in sequence today. M=64 coherence test deprioritized, not
# abandoned -- revisit if something here works and there's time left.
# M_VALUES = [64]
# M_VALUES = [8,16,32,64,]
# 2026-08-09: refocused on M=8 specifically, per user's request -- this is
# where the learned-vs-fixed margin is large (+11.38pp) and worth pushing
# further; M=32/64/128 margins are already small/negative, so there's much
# less to gain by tuning there. Combined with the PHASE1_EPOCHS/PHASE2_EPOCHS
# increase below (single variable: more training, applied identically to
# all 4 modes so the comparison stays fair) -- current M=8 Phase-1 training
# curve (network_residual_dct) was still climbing at epoch 20 (85.30% val,
# not clearly plateaued), so there may be unconverged headroom left.
# M_VALUES = [64]  # was the max_ratio=1.0 test, deprioritized
# M_VALUES = [8]
# 2026-08-09: canary, single M -- testing network_residual_dct + proxy_hybrid
# (the one diffuser+decoder combo not yet tried together). The just-completed
# M=8-64(+128 in progress) sweep with network_residual_dct+proxy_cnn gave a
# CONSISTENT margin improvement over the old residual_dct+proxy_hybrid table
# at every M (+11.38/+4.21/-1.58/-1.85 vs +11.10/+3.30/-1.66/-3.14) -- strong,
# not single-run-noise, since it held across 4 M values in one continuous run.
# Separately, residual_dct+proxy_hybrid vs residual_dct+mlp only showed a
# ~0.84pp gap (83.91->84.75, M=8) -- within this project's established noise
# floor (0.8-1.6pp, frozen_decoder saga), so hybrid's own contribution is
# unproven. This run isolates whether hybrid ADDS to network_residual_dct's
# already-confirmed gain, or is redundant with it. M=8 only (cheapest, and
# where any effect would show largest) before expanding to a full sweep.
# M_VALUES = [64, 128]
# M_VALUES = [8, 16, 32, 64, 128]
#
# M_VALUES = [1,2,4,8, 16, 32, 64, 128,256]  # Full range for nightly experiments

# Measurement modes
# "hadamard2d" (sequency/Walsh-ordered, 2D low-frequency selection) replaces
# "hadamard" (natural Sylvester order) as the default, same relationship as
# dct -> dct2d: natural-order Hadamard only samples a narrow band of spatial
# frequencies on one axis (see HadamardDiffuser2D docstring in diffusers.py
# for the exact mechanism). Keep "hadamard" available below for an A/B rerun
# against the legacy ordering, or for exact rollback/reproduction of old results.
MEASUREMENT_MODES = ["learned"]
# 2026-08-11: LEARNED only -- fixed baselines (DCT2D/Hadamard2D/Random) don't
# change between diffuser-capacity experiments, no need to re-run them; we
# already have M=8 fixed-baseline numbers (76.97/71.96/58.10%) to compare
# against. Restore full 4-mode list for any run whose numbers go in the paper.
# MEASUREMENT_MODES = ["learned", "dct2d", "hadamard2d", "random"]
# MEASUREMENT_MODES = ["learned", "hadamard", "hadamard2d"]  # A/B: legacy vs sequency-ordered Hadamard only
# MEASUREMENT_MODES = ["learned"]  # Compare all modes

# Paths
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs_run_all"
