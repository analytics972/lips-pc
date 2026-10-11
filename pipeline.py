"""
Author: Tomer Porat

Training pipeline for two-phase learning:
  Phase 1: Train encoder + classifier jointly
  Phase 2: Extract diffuser and train classifier on measurements
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    PHASE1_EPOCHS, PHASE2_EPOCHS,
    PHASE1_LR_DIFFUSER, PHASE1_LR_CLASSIFIER,
    PHASE2_LR, PHASE2_WARMUP_EPOCHS, PHASE2_WEIGHT_DECAY,
    CHECKPOINT_DIR, DATASET, DATA_DIR, INPUT_CHANNELS, NUM_CLASSES, FLAT_DIM, IMAGE_SIZE,
    LABEL_SMOOTHING, MIXUP_ALPHA, GRADIENT_CLIP
)
from config import DIFFUSER_ARCH, PHASE1_RESNET_MAX_LR, PHASE1_RESNET_WEIGHT_DECAY, PHASE2_CLASSIFIER, PHASE1_DELTA_WEIGHT_DECAY
from config import GISET_PROXY_D_MODEL, GISET_PROXY_N_LAYERS, GISET_PROXY_N_HEADS, GISET_PROXY_DROPOUT
from config import NETWORK_RESIDUAL_HIDDEN_DIM, COHERENCE_LOSS_WEIGHT, NETWORK_RESIDUAL_LATENT_DIM
from config import FOCAL_LOSS_GAMMA
from models import ClassifierResNet, NonLinearDiffuser, ResNetDiffuser, ResidualDiffuser, GeneratorDiffuser, LinearDiffuser, BinaryDiffuser, ProxyCNN, ProxyResNet18, ProxyHybridCNN, LowRankResidualDiffuser, GISetClassifier, GISetClassifierProxy, NetworkResidualDiffuser, ConvNetworkResidualDiffuser, FocalLoss
from diffusers import DiffuserFactory, compute_measurements
from utils import get_dataset_and_transforms


class TrainingPipeline:
    """Two-phase training pipeline"""
    
    def __init__(self, M, device=DEVICE):
        self.M = M
        self.device = device
        # self.encoder = Encoder(M, input_channels=INPUT_CHANNELS).to(device)
        self.classifier = ClassifierResNet(M, num_classes=NUM_CLASSES).to(device)
    

    def get_diffuser(self, mode="learned", train_loader=None, test_loader=None):
        """
        Get or create measurement matrix/network

        Args:
            mode: "learned", "dct", "dct2d", "hadamard", or "random"
            train_loader: (required for "learned") DataLoader with training data
            test_loader: (required for "learned") DataLoader with test data

        Returns:
            torch.Tensor or nn.Module: diffuser matrix (M, FLAT_DIM) or NonLinearDiffuser network
        """
        if mode == "learned":
            if train_loader is None or test_loader is None:
                raise ValueError("train_loader and test_loader required for learned diffuser extraction")
            arch_names = {"resnet": "ResNetDiffuser", "linear": "LinearDiffuser (plain matrix)",
                          "residual": "ResidualDiffuser (matrix)",
                          "residual_dct": "ResidualDiffuser (DCT2D-init matrix)",
                          "residual_lowrank": "LowRankResidualDiffuser (rank=16, joint)",
                          "residual_lowrank8": "LowRankResidualDiffuser (rank=8, joint)",
                          "generator": "GeneratorDiffuser (matrix)",
                          "generator_distill": "GeneratorDiffuser (matrix, distilled from ResNetDiffuser)",
                          "frozen_decoder": "LowRankResidualDiffuser (frozen-decoder two-stage)",
                          "network_residual_dct": "NetworkResidualDiffuser (DCT2D-init, network-generated delta)",
                          "conv_network_residual_dct": "ConvNetworkResidualDiffuser (DCT2D-init, conv-generated delta)"}
            print(f">>> M={self.M}: Using {arch_names.get(DIFFUSER_ARCH, 'NonLinearDiffuser')} (dataset={DATASET}, D={FLAT_DIM})")
            if DIFFUSER_ARCH == "generator_distill":
                diffuser = self._learn_generator_distilled(train_loader, test_loader)
            elif DIFFUSER_ARCH == "frozen_decoder":
                diffuser = self._learn_frozen_decoder_diffuser(train_loader, test_loader)
            else:
                diffuser = self._learn_diffuser_nonlinear(train_loader, test_loader)
            return diffuser
        else:
            # DCT, Hadamard, Random, create fixed matrices with correct D
            print(f">>> M={self.M}: Using {mode.upper()} (dataset={DATASET}, D={FLAT_DIM})")
            return DiffuserFactory.create(mode, self.M, D=FLAT_DIM, device=self.device)

    def _learn_diffuser_nonlinear(self, train_loader, test_loader, epochs=PHASE1_EPOCHS):
        """
        Learn non-linear CNN diffuser jointly with classifier (M=64 only)

        Args:
            train_loader: training data loader
            test_loader: test data loader for validation
            epochs: number of training epochs

        Returns:
            nn.Module: trained NonLinearDiffuser
        """
        # Create non-linear diffuser and classifier
        # Pass image_size for pre-computed feature dimension (efficient)
        if DIFFUSER_ARCH == "resnet":
            diffuser = ResNetDiffuser(M=self.M, input_channels=INPUT_CHANNELS, image_size=IMAGE_SIZE).to(self.device)
        elif DIFFUSER_ARCH == "linear":
            diffuser = LinearDiffuser(M=self.M, D_in=FLAT_DIM).to(self.device)
        elif DIFFUSER_ARCH == "binary":
            diffuser = BinaryDiffuser(M=self.M, D_in=FLAT_DIM).to(self.device)
        elif DIFFUSER_ARCH == "residual":
            diffuser = ResidualDiffuser(M=self.M, D_in=FLAT_DIM).to(self.device)
        elif DIFFUSER_ARCH == "residual_dct":
            # Same ResidualDiffuser, but delta refines a DCT2D base instead of
            # random noise -- DCT2D is already a much stronger fixed baseline
            # than Random (see CLAUDE.md 2026-08-02 RIAWELC results), so this
            # should out-perform plain "residual" without changing anything
            # else about the training recipe.
            dct_base = DiffuserFactory.create("dct2d", self.M, D=FLAT_DIM, device=self.device)
            diffuser = ResidualDiffuser(M=self.M, D_in=FLAT_DIM, base_matrix=dct_base).to(self.device)
        elif DIFFUSER_ARCH == "residual_hadamard":
            # 2026-08-14: single-variable ablation of "residual_dct" -- same
            # ResidualDiffuser, same free-tensor delta, only the anchor swapped
            # from DCT2D to Hadamard2D. Tests whether DCT2D is uniquely good as
            # an anchor for the LEARNED correction, or just happens to be the
            # stronger FIXED baseline (the existing justification in
            # thesis.tex Section 3.3 only establishes the latter). See
            # thesis.tex/thesis_he.html's "DCT-vs-Hadamard anchor" callout.
            hadamard_base = DiffuserFactory.create("hadamard2d", self.M, D=FLAT_DIM, device=self.device)
            diffuser = ResidualDiffuser(M=self.M, D_in=FLAT_DIM, base_matrix=hadamard_base).to(self.device)
        elif DIFFUSER_ARCH == "residual_lowrank":
            # Structural fix for delta overfitting (2026-08-04, see
            # models.py::LowRankResidualDiffuser docstring and
            # residual_delta_overfitting_fix memory): delta = U@V, rank 16,
            # instead of a free (M,1024) tensor. Capacity grows much slower
            # with M than the free version, targeting the M>=32 gap that
            # widened to -7.1pp at M=64 with residual_dct. NOT YET VALIDATED
            # across the full M range as of this writing -- check
            # riawelc_lowrank_results before trusting any single M's number.
            dct_base = DiffuserFactory.create("dct2d", self.M, D=FLAT_DIM, device=self.device)
            diffuser = LowRankResidualDiffuser(M=self.M, D_in=FLAT_DIM, rank=16, base_matrix=dct_base).to(self.device)
        elif DIFFUSER_ARCH == "residual_lowrank8":
            # 2026-08-05: tighter capacity cap than residual_lowrank (rank=8
            # instead of 16), trained JOINTLY (no frozen-decoder decoupling,
            # so 1x runtime not 3x). Motivation: at low M (<=8), rank=16 is
            # already >= M so LowRankResidualDiffuser's "low-rank" constraint
            # isn't actually binding there -- it behaves like a free delta,
            # which is why frozen_decoder's M=8 loss (56.26% vs 66.84% free
            # delta) can't be blamed on rank; it was decoupling itself that
            # hurt at low M. rank=8 keeps the constraint effectively inactive
            # at very low M (M<=8) while capping delta's capacity harder than
            # rank=16 as M grows past it -- goal: preserve the large free-delta
            # win at low M while curbing the overfitting-driven loss at high M,
            # in ONE mechanism, without frozen_decoder's 3x cost. UNVALIDATED:
            # a similar joint-trained rank=16 variant plateaued around 74% at
            # M=32 in an earlier abandoned run (no clear win over residual_dct
            # or DCT2D there) -- this is a hypothesis, not a proven fix.
            dct_base = DiffuserFactory.create("dct2d", self.M, D=FLAT_DIM, device=self.device)
            diffuser = LowRankResidualDiffuser(M=self.M, D_in=FLAT_DIM, rank=8, base_matrix=dct_base).to(self.device)
        elif DIFFUSER_ARCH == "generator":
            diffuser = GeneratorDiffuser(M=self.M, D_in=FLAT_DIM).to(self.device)
        elif DIFFUSER_ARCH == "network_residual_dct":
            # 2026-08-08, user's idea: delta generated by a small network
            # (like GeneratorDiffuser) anchored to a DCT2D base (like
            # residual_dct), instead of a free (M,D_in) tensor -- see
            # models.py::NetworkResidualDiffuser docstring.
            dct_base = DiffuserFactory.create("dct2d", self.M, D=FLAT_DIM, device=self.device)
            diffuser = NetworkResidualDiffuser(M=self.M, D_in=FLAT_DIM, base_matrix=dct_base, hidden_dim=NETWORK_RESIDUAL_HIDDEN_DIM, latent_dim=NETWORK_RESIDUAL_LATENT_DIM).to(self.device)
        elif DIFFUSER_ARCH == "conv_network_residual_dct":
            # 2026-08-08: same idea as network_residual_dct, but the delta
            # network is convolutional (Mor Hale's Gen_conv_3 style) instead
            # of flat FC -- see models.py::ConvNetworkResidualDiffuser.
            # Single-channel only (grayscale D_in==image_size**2).
            dct_base = DiffuserFactory.create("dct2d", self.M, D=FLAT_DIM, device=self.device)
            diffuser = ConvNetworkResidualDiffuser(M=self.M, D_in=FLAT_DIM, base_matrix=dct_base, image_size=IMAGE_SIZE).to(self.device)
        else:
            diffuser = NonLinearDiffuser(M=self.M, input_channels=INPUT_CHANNELS, image_size=IMAGE_SIZE).to(self.device)

        # Proxy-CNN decoder (see models.py::ProxyCNN) needs a genuine matrix
        # to build x_hat = D^T y; CNN diffusers (resnet/simple) have none, so
        # they keep the MLP head regardless of the toggle.
        use_hybrid = PHASE2_CLASSIFIER == "proxy_hybrid" and hasattr(diffuser, "get_matrix")
        use_proxy = PHASE2_CLASSIFIER in ("proxy_cnn", "proxy_resnet18") and hasattr(diffuser, "get_matrix")
        use_set_proxy = PHASE2_CLASSIFIER == "set_transformer_proxy" and hasattr(diffuser, "get_matrix")
        if use_hybrid:
            classifier = ProxyHybridCNN(M=self.M, num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(self.device)
        elif use_proxy:
            proxy_cls = ProxyResNet18 if PHASE2_CLASSIFIER == "proxy_resnet18" else ProxyCNN
            classifier = proxy_cls(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(self.device)
        elif use_set_proxy:
            classifier = GISetClassifierProxy(
                num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS,
                d_model=GISET_PROXY_D_MODEL, n_layers=GISET_PROXY_N_LAYERS,
                n_heads=GISET_PROXY_N_HEADS, dropout=GISET_PROXY_DROPOUT,
            ).to(self.device)
        elif PHASE2_CLASSIFIER == "set_transformer":
            classifier = GISetClassifier(M=self.M, num_classes=NUM_CLASSES).to(self.device)
        else:
            classifier = ClassifierResNet(self.M, num_classes=NUM_CLASSES).to(self.device)

        def decode(measurements):
            if use_hybrid:
                x_hat = measurements @ diffuser.get_matrix()
                return classifier(x_hat.view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), measurements)
            if use_proxy or use_set_proxy:
                x_hat = measurements @ diffuser.get_matrix()
                return classifier(x_hat.view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
            return classifier(measurements)

        if DIFFUSER_ARCH in ("resnet", "linear", "binary", "residual", "residual_dct", "residual_hadamard", "residual_lowrank", "residual_lowrank8", "generator", "network_residual_dct", "conv_network_residual_dct"):
            # Proven recipe (originally tuned for ResNetDiffuser, drove the
            # 84%->92% CIFAR10 gain): single SGD over diffuser+classifier,
            # OneCycle LR (stepped per batch), label smoothing. Reused here
            # for the matrix-based diffusers instead of the weaker legacy
            # Adam+per-epoch-cosine recipe below -- same classifier either
            # way, so there's no reason the matrix diffusers should get a
            # worse-tuned optimizer than the CNN one did.
            # Delta gets its own param group with stronger weight decay if the
            # diffuser has one (ResidualDiffuser/residual_dct): under the
            # ProxyCNN decoder, an unrestrained delta (trained on only ~20
            # welds) overfits and lets fixed DCT2D overtake the learned mask
            # at M>=32. Single-change A/B (M=32, 2026-08-04): wd 5e-4 ->
            # 74.32%, wd 5e-3 -> 76.69% (vs fixed DCT2D 77.30%). See
            # config.py::PHASE1_DELTA_WEIGHT_DECAY.
            if hasattr(diffuser, "delta"):
                other_diffuser_params = [p for n, p in diffuser.named_parameters() if n != "delta"]
                param_groups = [
                    {"params": [diffuser.delta], "weight_decay": PHASE1_DELTA_WEIGHT_DECAY},
                    {"params": other_diffuser_params + list(classifier.parameters()),
                     "weight_decay": PHASE1_RESNET_WEIGHT_DECAY},
                ]
                max_lrs = [PHASE1_RESNET_MAX_LR, PHASE1_RESNET_MAX_LR]
            elif hasattr(diffuser, "delta_parameters"):
                # NetworkResidualDiffuser / ConvNetworkResidualDiffuser: delta
                # is a NETWORK's output, not a free tensor, and its gradient
                # touches many matrix entries at once through shared weights
                # (a conv kernel update perturbs all M output channels'
                # receptive fields simultaneously). A synthetic-data test
                # (2026-08-08) found delta's norm explodes to ~4400 at the
                # default max_lr=0.1 (vs. a DCT2D base norm of ~3), completely
                # swamping the anchor the whole "residual" design relies on.
                # Give this group both the stronger weight decay AND a much
                # lower peak LR (1/10th) -- weight decay alone wasn't the
                # lever that mattered in the synthetic test, LR was.
                delta_params = diffuser.delta_parameters()
                delta_ids = {id(p) for p in delta_params}
                other_diffuser_params = [p for p in diffuser.parameters() if id(p) not in delta_ids]
                param_groups = [
                    {"params": delta_params, "weight_decay": PHASE1_DELTA_WEIGHT_DECAY},
                    {"params": other_diffuser_params + list(classifier.parameters()),
                     "weight_decay": PHASE1_RESNET_WEIGHT_DECAY},
                ]
                max_lrs = [PHASE1_RESNET_MAX_LR * 0.1, PHASE1_RESNET_MAX_LR]
            else:
                param_groups = [{"params": list(diffuser.parameters()) + list(classifier.parameters()),
                                 "weight_decay": PHASE1_RESNET_WEIGHT_DECAY}]
                max_lrs = PHASE1_RESNET_MAX_LR
            optimizer_d = optim.SGD(
                param_groups, lr=PHASE1_RESNET_MAX_LR / 25, momentum=0.9, nesterov=True)
            optimizer_c = None
            scheduler_d = optim.lr_scheduler.OneCycleLR(
                optimizer_d, max_lr=max_lrs,
                epochs=epochs, steps_per_epoch=len(train_loader))
            scheduler_c = None
            criterion = (FocalLoss(gamma=FOCAL_LOSS_GAMMA, label_smoothing=LABEL_SMOOTHING)
                         if FOCAL_LOSS_GAMMA > 0 else nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING))
        else:
            # Legacy recipe: separate Adam optimizers, cosine annealing per epoch
            optimizer_d = optim.Adam(diffuser.parameters(), lr=PHASE1_LR_DIFFUSER, weight_decay=1e-5)
            optimizer_c = optim.Adam(classifier.parameters(), lr=PHASE1_LR_CLASSIFIER, weight_decay=1e-4)

            scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=epochs, eta_min=1e-6)
            scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=epochs, eta_min=1e-5)

            criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        best_diffuser_state = None

        pbar_epochs = tqdm(range(epochs), desc="Learning NonLinear Diffuser + Classifier", leave=True)
        for epoch in pbar_epochs:
            # Training phase
            diffuser.train()
            classifier.train()
            total_acc = 0
            n_batches = 0

            pbar_batches = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs} (train)", leave=False, position=0, ncols=100)
            for images, labels in pbar_batches:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                measurements = diffuser(images)
                logits = decode(measurements)
                loss = criterion(logits, labels)

                # Orthogonality/coherence regularization -- penalize
                # correlated measurement rows, which fixed transforms avoid
                # by construction (see COHERENCE_LOSS_WEIGHT in config.py).
                if COHERENCE_LOSS_WEIGHT > 0 and hasattr(diffuser, "get_matrix"):
                    D = diffuser.get_matrix()
                    D_normalized = D / (D.norm(dim=1, keepdim=True) + 1e-8)
                    gram = D_normalized @ D_normalized.T
                    identity = torch.eye(self.M, device=self.device)
                    coherence_loss = ((gram - identity) ** 2).sum() / (self.M * (self.M - 1))
                    loss = loss + COHERENCE_LOSS_WEIGHT * coherence_loss

                # Backward pass
                optimizer_d.zero_grad()
                if optimizer_c is not None:
                    optimizer_c.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(diffuser.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.5)

                optimizer_d.step()
                if optimizer_c is not None:
                    optimizer_c.step()
                if isinstance(scheduler_d, optim.lr_scheduler.OneCycleLR):
                    scheduler_d.step()  # OneCycle steps per batch

                # Accuracy
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                total_acc += acc
                n_batches += 1

                pbar_batches.set_postfix(acc=f"{acc:.2%}", loss=f"{loss.item():.4f}")

            train_acc = total_acc / n_batches

            # Validation phase (batch-wise to avoid GPU OOM)
            diffuser.eval()
            classifier.eval()
            val_correct = 0
            val_total = 0

            pbar_val = tqdm(test_loader, desc=f"  Epoch {epoch+1}/{epochs} (val)", leave=False, position=0, ncols=100)
            with torch.no_grad():
                for images, labels in pbar_val:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    measurements_val = diffuser(images)
                    logits_val = decode(measurements_val)
                    preds_val = logits_val.argmax(dim=1)
                    batch_correct = (preds_val == labels).sum().item()
                    val_correct += batch_correct
                    val_total += labels.size(0)
                    batch_acc = batch_correct / len(labels)
                    pbar_val.set_postfix(acc=f"{batch_acc:.2%}")

            val_acc = val_correct / val_total if val_total > 0 else 0

            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_diffuser_state = diffuser.state_dict().copy()

            pbar_epochs.set_postfix(train_acc=f"{train_acc:.2%}", val_acc=f"{val_acc:.2%}")
            # Plain newline-terminated summary so epoch history survives in log files
            # (tqdm bars overwrite themselves with \r)
            tqdm.write(f"[Phase 1] epoch {epoch+1}/{epochs}: train_acc={train_acc:.2%} "
                       f"val_acc={val_acc:.2%} best={best_val_acc:.2%}")

            # Update learning rates (OneCycle already stepped per batch)
            if not isinstance(scheduler_d, optim.lr_scheduler.OneCycleLR):
                scheduler_d.step()
            if scheduler_c is not None:
                scheduler_c.step()

        # Restore best diffuser
        if best_diffuser_state is not None:
            diffuser.load_state_dict(best_diffuser_state)

        # Persist the final learned measurement matrix, independent of
        # accuracy -- 2026-08-13, user request: visualize the learned
        # diffusers per method/per M for the thesis. Not previously saved
        # anywhere (only self.encoder.state_dict() in _save_checkpoint,
        # unrelated to this matrix-based diffuser family).
        if hasattr(diffuser, "get_matrix"):
            os.makedirs("diffuser_matrices", exist_ok=True)
            torch.save(diffuser.get_matrix().detach().cpu(),
                       f"diffuser_matrices/M{self.M}_{DIFFUSER_ARCH}_learned.pt")

        return diffuser

    def _learn_generator_distilled(self, train_loader, test_loader, epochs=PHASE1_EPOCHS):
        """
        Reverse-engineer a physically-realizable matrix out of the strong
        ResNetDiffuser: train GeneratorDiffuser's matrix-producing FC network
        to reproduce the CNN's measurements (MSE distillation), jointly with
        classification loss. The deployed diffuser is still a single global
        (M, D) matrix (y = D @ x, input-agnostic) -- only *how the network
        arrived at it* differs from plain "generator" (matched against a
        strong teacher's behaviour, not just classification loss alone).

        Two stages, each `epochs` long:
          1. Train a ResNetDiffuser + classifier "teacher" (same recipe as
             DIFFUSER_ARCH="resnet").
          2. Freeze the teacher; train GeneratorDiffuser + a fresh classifier
             with loss = CE(classifier(student_matrix @ x), label)
                        + MSE(student_matrix @ x, teacher(x))
             so the generator network is explicitly taught to approximate
             what the CNN learned to measure, while remaining a genuine
             linear mask matrix at inference time.

        Still bounded by the same linear-projection ceiling as "generator"/
        "residual" -- distillation can only transfer the LINEAR component of
        what the CNN does, not its per-image nonlinearity -- but this is the
        best-informed matrix we can construct, since it's guided by what the
        successful nonlinear model actually discovered rather than a generic
        fixed transform or plain classification loss on a blank-slate matrix.
        """
        print(f">>> M={self.M}: Stage 1/2, training ResNetDiffuser teacher ({epochs} epochs)")
        teacher = ResNetDiffuser(M=self.M, input_channels=INPUT_CHANNELS, image_size=IMAGE_SIZE).to(self.device)
        teacher_classifier = ClassifierResNet(self.M, num_classes=NUM_CLASSES).to(self.device)
        optimizer_t = optim.SGD(
            list(teacher.parameters()) + list(teacher_classifier.parameters()),
            lr=PHASE1_RESNET_MAX_LR / 25, momentum=0.9, nesterov=True,
            weight_decay=PHASE1_RESNET_WEIGHT_DECAY)
        scheduler_t = optim.lr_scheduler.OneCycleLR(
            optimizer_t, max_lr=PHASE1_RESNET_MAX_LR, epochs=epochs, steps_per_epoch=len(train_loader))
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        teacher.train()
        teacher_classifier.train()
        pbar_teacher = tqdm(range(epochs), desc="  Teacher (ResNetDiffuser)", leave=True)
        for epoch in pbar_teacher:
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = teacher_classifier(teacher(images))
                loss = criterion_ce(logits, labels)
                optimizer_t.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(teacher.parameters()) + list(teacher_classifier.parameters()), max_norm=2.0)
                optimizer_t.step()
                scheduler_t.step()
            pbar_teacher.set_postfix(loss=f"{loss.item():.4f}")

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        print(f">>> M={self.M}: Stage 2/2, distilling GeneratorDiffuser from teacher ({epochs} epochs)")
        student = GeneratorDiffuser(M=self.M, D_in=FLAT_DIM).to(self.device)
        classifier = ClassifierResNet(self.M, num_classes=NUM_CLASSES).to(self.device)
        optimizer_s = optim.SGD(
            list(student.parameters()) + list(classifier.parameters()),
            lr=PHASE1_RESNET_MAX_LR / 25, momentum=0.9, nesterov=True,
            weight_decay=PHASE1_RESNET_WEIGHT_DECAY)
        scheduler_s = optim.lr_scheduler.OneCycleLR(
            optimizer_s, max_lr=PHASE1_RESNET_MAX_LR, epochs=epochs, steps_per_epoch=len(train_loader))
        criterion_mse = nn.MSELoss()

        best_val_acc = 0
        best_state = None
        pbar_student = tqdm(range(epochs), desc="  Student (GeneratorDiffuser, distilled)", leave=True)
        for epoch in pbar_student:
            student.train()
            classifier.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    teacher_meas = teacher(images)
                student_meas = student(images)
                logits = classifier(student_meas)
                loss = criterion_ce(logits, labels) + criterion_mse(student_meas, teacher_meas)
                optimizer_s.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(classifier.parameters()), max_norm=2.0)
                optimizer_s.step()
                scheduler_s.step()

            student.eval()
            classifier.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = classifier(student(images))
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total if total > 0 else 0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = student.state_dict().copy()
            pbar_student.set_postfix(val_acc=f"{val_acc:.2%}", best=f"{best_val_acc:.2%}")
            tqdm.write(f"[Phase 1/distill] epoch {epoch+1}/{epochs}: val_acc={val_acc:.2%} best={best_val_acc:.2%}")

        if best_state is not None:
            student.load_state_dict(best_state)

        return student

    def _learn_frozen_decoder_diffuser(self, train_loader, test_loader, epochs=PHASE1_EPOCHS, rank=16):
        """
        Two-stage training that decouples the decoder from the mask, instead
        of training them jointly (2026-08-04 -- see residual_delta_overfitting_fix
        and proxy_cnn_decoder_breakthrough memories for why joint training
        under ProxyCNN lets a free/low-rank delta overfit the small training
        set and lose to fixed transforms at M>=32, with the gap widening as
        M grows, e.g. -11.8pp at M=128).

        Stage A: train ProxyCNN on a FIXED DCT2D matrix alone (no learned
                 matrix at all) to convergence. This decoder never sees a
                 learned mask, so it cannot co-adapt to memorize anything
                 mask-specific -- it becomes a strong, generic decoder.
        Stage B: FREEZE that decoder. Train only a low-rank delta
                 (rank=`rank`, see models.py::LowRankResidualDiffuser) on top
                 of the frozen decoder -- delta's only way to reduce loss is
                 to produce measurements the already-generalizing decoder can
                 read better; it cannot change the decoder to fit its own
                 noise.

        Verified result (RIAWELC, M=64, clean split, 2026-08-04): fixed
        DCT2D=81.40%, Hadamard2D=83.54%, joint-trained residual_dct=76.40%.
        This mechanism -> 83.46% after Phase 2 (beats DCT2D by +2.1pp, within
        noise of Hadamard2D's 83.54%). Not yet validated at other M or other
        datasets -- run and check before trusting a new M/dataset's number.

        Returns the trained LowRankResidualDiffuser (Stage B's result); the
        caller still runs the normal phase2_train on it for the final,
        comparable number (Stage C in the original standalone script).
        """
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        print(f">>> M={self.M}: Stage A, training ProxyCNN on fixed DCT2D alone ({epochs} epochs)")
        Dmat = DiffuserFactory.create("dct2d", self.M, D=FLAT_DIM, device=self.device)
        decoder = ProxyCNN(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(self.device)
        opt_a = optim.SGD(decoder.parameters(), lr=PHASE1_RESNET_MAX_LR / 25, momentum=0.9,
                          nesterov=True, weight_decay=PHASE1_RESNET_WEIGHT_DECAY)
        sched_a = optim.lr_scheduler.OneCycleLR(opt_a, max_lr=PHASE1_RESNET_MAX_LR,
                                                epochs=epochs, steps_per_epoch=len(train_loader))
        best_a, best_a_state = 0, None
        pbar_a = tqdm(range(epochs), desc="  Stage A (decoder on fixed DCT2D)", leave=True)
        for epoch in pbar_a:
            decoder.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                x_flat = images.view(images.size(0), -1)
                y = x_flat @ Dmat.T
                x_hat = (y @ Dmat).view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
                loss = criterion(decoder(x_hat), labels)
                opt_a.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=2.0)
                opt_a.step()
                sched_a.step()
            decoder.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    x_flat = images.view(images.size(0), -1)
                    y = x_flat @ Dmat.T
                    x_hat = (y @ Dmat).view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
                    correct += (decoder(x_hat).argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total if total > 0 else 0
            if acc > best_a:
                best_a, best_a_state = acc, decoder.state_dict().copy()
            pbar_a.set_postfix(val_acc=f"{acc:.2%}", best=f"{best_a:.2%}")
            tqdm.write(f"[Stage A] epoch {epoch+1}/{epochs}: val_acc={acc:.2%} best={best_a:.2%}")
        decoder.load_state_dict(best_a_state)
        print(f">>> Stage A done: decoder-only best = {best_a:.2%}")

        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False

        print(f">>> M={self.M}: Stage B, training rank-{rank} delta with decoder FROZEN ({epochs} epochs)")
        diffuser = LowRankResidualDiffuser(M=self.M, D_in=FLAT_DIM, rank=rank,
                                           base_matrix=Dmat.cpu()).to(self.device)
        opt_b = optim.SGD(diffuser.parameters(), lr=PHASE1_RESNET_MAX_LR / 25, momentum=0.9,
                          nesterov=True, weight_decay=PHASE1_RESNET_WEIGHT_DECAY)
        sched_b = optim.lr_scheduler.OneCycleLR(opt_b, max_lr=PHASE1_RESNET_MAX_LR,
                                                epochs=epochs, steps_per_epoch=len(train_loader))
        best_b, best_b_state = 0, None
        pbar_b = tqdm(range(epochs), desc="  Stage B (delta, decoder frozen)", leave=True)
        for epoch in pbar_b:
            diffuser.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                meas = diffuser(images)
                x_hat = (meas @ diffuser.get_matrix()).view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
                loss = criterion(decoder(x_hat), labels)
                opt_b.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffuser.parameters(), max_norm=2.0)
                opt_b.step()
                sched_b.step()
            diffuser.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    meas = diffuser(images)
                    x_hat = (meas @ diffuser.get_matrix()).view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
                    correct += (decoder(x_hat).argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total if total > 0 else 0
            if acc > best_b:
                best_b, best_b_state = acc, diffuser.state_dict().copy()
            pbar_b.set_postfix(val_acc=f"{acc:.2%}", best=f"{best_b:.2%}", decoder_ref=f"{best_a:.2%}")
            tqdm.write(f"[Stage B] epoch {epoch+1}/{epochs}: val_acc={acc:.2%} best={best_b:.2%} (decoder-only ref={best_a:.2%})")
        diffuser.load_state_dict(best_b_state)
        print(f">>> Stage B done: frozen-decoder + learned mask best = {best_b:.2%}  (decoder-only was {best_a:.2%})")

        return diffuser

    def phase2_train(self, diffuser, epochs=PHASE2_EPOCHS, batch_size=BATCH_SIZE, mode="learned",
                     train_loader=None, test_loader=None):
        """
        Phase 2: Train classifier on measurements from this diffuser mode.

        ALL modes (learned, dct, hadamard, random) use the optimized approach when
        train_loader and test_loader are supplied:
          - Trains on the FULL 50K training set
          - ClassifierResNet MLP (handles all M values via projection)
          - SGD+Nesterov, lr=0.1, weight_decay=5e-4
          - 5-epoch linear warmup then CosineAnnealingLR (T_max=epochs-5, eta_min=1e-4)
          - Mixup(alpha=0.2) on measurement vectors
          - label_smoothing=0.1, gradient_clip=1.0

        This unified recipe applies to ALL measurement modes for direct comparison:
        - Learned mode: NonLinearDiffuser (CNN encoder, trained in Phase 1)
        - Fixed modes: DCT, Hadamard, Random matrices (no Phase 1 needed)

        Args:
            diffuser:     measurement matrix (M, D) tensor OR nn.Module (NonLinearDiffuser)
            epochs:       number of training epochs
            batch_size:   batch size
            mode:         "learned", "dct", "hadamard", "random"
            train_loader: (required) full training DataLoader
            test_loader:  (required) test DataLoader

        Returns:
            dict: {"epochs": [val_acc per epoch], "best": best_val_acc}
        """
        # ── Optimized path: ALL modes with proper train/test split ────────────
        # For learned mode, diffuser is nn.Module (NonLinearDiffuser)
        # For fixed modes, diffuser is torch.Tensor
        # _phase2_train_fixed handles both via isinstance check and _compute_measurements_on_loader
        if train_loader is not None and test_loader is not None:
            return self._phase2_train_fixed(diffuser, train_loader, test_loader, epochs, mode)

        # ── Legacy path: original behaviour (learned mode or no loaders) ─────
        print(f"Phase 2: Classifier on {mode.upper()} (M={self.M})")

        measurements, labels = self._compute_measurements_on_test(diffuser, mode=mode)

        num_test = len(measurements)
        num_train = int(0.8 * num_test)
        train_meas, train_labels = measurements[:num_train], labels[:num_train]
        val_meas,   val_labels   = measurements[num_train:], labels[num_train:]

        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(train_meas, train_labels)
        val_dataset   = TensorDataset(val_meas,   val_labels)
        tr_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        vl_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        classifier_phase2 = ClassifierResNet(self.M, num_classes=NUM_CLASSES).to(self.device)
        optimizer = optim.Adam(classifier_phase2.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        phase2_epochs = []

        pbar_epochs = tqdm(range(epochs), desc=f"Classifier on {mode.upper()}", leave=True)
        for epoch in pbar_epochs:
            classifier_phase2.train()
            total_acc = 0
            for meas, batch_labels in tr_loader:
                meas = meas.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = classifier_phase2(meas)
                loss = criterion(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                preds = logits.argmax(dim=1)
                total_acc += (preds == batch_labels).sum().item() / len(batch_labels)

            train_acc = total_acc / len(tr_loader)
            classifier_phase2.eval()
            val_acc = 0
            with torch.no_grad():
                for meas, batch_labels in vl_loader:
                    meas = meas.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    logits = classifier_phase2(meas)
                    val_acc += (logits.argmax(1) == batch_labels).sum().item() / len(batch_labels)

            val_acc /= len(vl_loader)
            phase2_epochs.append(val_acc)
            pbar_epochs.set_postfix(train_acc=f"{train_acc:.2%}", val_acc=f"{val_acc:.2%}")
            if val_acc > best_acc:
                best_acc = val_acc

        return {"epochs": phase2_epochs, "best": best_acc}

    def _phase2_train_fixed(self, diffuser, train_loader, test_loader, epochs, mode):
        """
        Phase-2 training for ALL measurement modes (learned, dct, dct2d, hadamard, random).

        Unified classifier and training recipe for all modes so results are directly comparable:
          - Classifier: ClassifierResNet (M→512→512→512→10 MLP with residuals)
          - Optimizer: SGD with lr=0.1, Nesterov momentum=0.9, weight_decay=5e-4
          - Learning rate: 5-epoch linear warmup (0→0.1) then CosineAnnealingLR (eta_min=1e-4)
          - Regularization: Mixup(alpha=0.2) on measurement vectors + label_smoothing=0.1
          - Gradient: clip to max_norm=1.0

        Works with:
          - Learned mode: diffuser is nn.Module (NonLinearDiffuser from Phase 1)
          - Fixed modes: diffuser is torch.Tensor (DCT/Hadamard/Random matrices)
        """
        import numpy as np

        # Use config parameters for Phase 2
        WARMUP      = PHASE2_WARMUP_EPOCHS
        BASE_LR     = PHASE2_LR
        WEIGHT_DECAY_PHASE2 = PHASE2_WEIGHT_DECAY

        diffuser_type = "NonLinearDiffuser" if isinstance(diffuser, nn.Module) else "Fixed Matrix"

        # Freeze diffuser for learned mode (eval mode + no gradients)
        if isinstance(diffuser, nn.Module):
            diffuser.eval()
            for param in diffuser.parameters():
                param.requires_grad = False

        # Proxy-CNN decoder: classify the adjoint proxy image x_hat = D^T y
        # instead of the raw M-vector (see models.py::ProxyCNN). Applies to
        # every mode that has a genuine matrix -- fixed matrices and
        # matrix-based learned diffusers alike -- so comparisons stay fair.
        # CNN diffusers (resnet/simple) have no matrix: MLP fallback.
        if isinstance(diffuser, torch.Tensor):
            proxy_matrix = diffuser
        elif hasattr(diffuser, "get_matrix"):
            with torch.no_grad():
                proxy_matrix = diffuser.get_matrix().detach()
        else:
            proxy_matrix = None
        use_hybrid = PHASE2_CLASSIFIER == "proxy_hybrid" and proxy_matrix is not None
        use_proxy = PHASE2_CLASSIFIER in ("proxy_cnn", "proxy_resnet18") and proxy_matrix is not None
        use_set_proxy = PHASE2_CLASSIFIER == "set_transformer_proxy" and proxy_matrix is not None

        use_settransformer = PHASE2_CLASSIFIER == "set_transformer"
        head_name = "ProxyHybridCNN" if use_hybrid else ("ProxyResNet18" if (use_proxy and PHASE2_CLASSIFIER == "proxy_resnet18") else ("ProxyCNN" if use_proxy else ("GISetClassifierProxy" if use_set_proxy else ("GISetClassifier" if use_settransformer else "ClassifierResNet"))))
        print(f"Phase 2: {head_name} + SGD/Warmup/Mixup on {mode.upper()} ({diffuser_type}, M={self.M})")

        # Precompute all test measurements once (no augmentation on test set)
        test_meas, test_labels = self._compute_measurements_on_loader(diffuser, test_loader)
        test_meas_dev   = test_meas.to(self.device)
        test_labels_dev = test_labels.to(self.device)

        if use_hybrid:
            classifier = ProxyHybridCNN(M=self.M, num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(self.device)
        elif use_proxy:
            proxy_cls = ProxyResNet18 if PHASE2_CLASSIFIER == "proxy_resnet18" else ProxyCNN
            classifier = proxy_cls(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(self.device)
        elif use_set_proxy:
            classifier = GISetClassifierProxy(
                num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS,
                d_model=GISET_PROXY_D_MODEL, n_layers=GISET_PROXY_N_LAYERS,
                n_heads=GISET_PROXY_N_HEADS, dropout=GISET_PROXY_DROPOUT,
            ).to(self.device)
        elif use_settransformer:
            classifier = GISetClassifier(M=self.M, num_classes=NUM_CLASSES).to(self.device)
        else:
            classifier = ClassifierResNet(M=self.M, num_classes=NUM_CLASSES).to(self.device)

        def head(meas):
            if use_hybrid:
                x_hat = meas @ proxy_matrix
                return classifier(x_hat.view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), meas)
            if use_proxy or use_set_proxy:
                x_hat = meas @ proxy_matrix
                return classifier(x_hat.view(-1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
            return classifier(meas)
        optimizer  = optim.SGD(classifier.parameters(), lr=BASE_LR,
                               momentum=0.9, weight_decay=WEIGHT_DECAY_PHASE2, nesterov=True)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - WARMUP), eta_min=1e-4)
        criterion  = (FocalLoss(gamma=FOCAL_LOSS_GAMMA, label_smoothing=LABEL_SMOOTHING)
                      if FOCAL_LOSS_GAMMA > 0 else nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING))

        best_acc      = 0
        phase2_epochs = []

        pbar = tqdm(range(epochs), desc=f"Phase 2: {mode.upper()}", leave=True)
        for epoch in pbar:
            # Linear LR warmup
            if epoch < WARMUP:
                for pg in optimizer.param_groups:
                    pg['lr'] = BASE_LR * (epoch + 1) / WARMUP

            classifier.train()
            pbar_train = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs} (train)", leave=False, position=0, ncols=100)
            train_acc = 0
            for images, labels in pbar_train:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch  = images.size(0)

                with torch.no_grad():
                    if isinstance(diffuser, nn.Module):
                        meas = diffuser(images)
                    else:
                        x_flat = images.view(batch, -1)
                        meas   = (diffuser @ x_flat.T).T

                # Mixup on measurement vectors
                lam  = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(batch, device=self.device)
                meas_mix = lam * meas + (1 - lam) * meas[perm]
                logits = head(meas_mix)
                loss   = lam * criterion(logits, labels) + \
                         (1 - lam) * criterion(logits, labels[perm])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), GRADIENT_CLIP)
                optimizer.step()

                # Accuracy
                acc = (logits.argmax(1) == labels).float().mean().item()
                train_acc = acc
                pbar_train.set_postfix(acc=f"{acc:.2%}", loss=f"{loss.item():.4f}")

            # Evaluate on precomputed test measurements
            classifier.eval()
            correct = 0
            epoch_preds = []
            pbar_val = tqdm(range(0, len(test_meas_dev), 512), desc=f"  Epoch {epoch+1}/{epochs} (val)", leave=False, position=0, ncols=100)
            with torch.no_grad():
                for start in pbar_val:
                    mb = test_meas_dev[start:start + 512]
                    lb = test_labels_dev[start:start + 512]
                    preds = head(mb).argmax(1)
                    epoch_preds.append(preds.cpu())
                    batch_correct = (preds == lb).sum().item()
                    correct += batch_correct
                    batch_acc = batch_correct / len(lb)
                    pbar_val.set_postfix(acc=f"{batch_acc:.2%}")

            val_acc = correct / len(test_labels_dev)
            phase2_epochs.append(val_acc)
            train_acc_for_log = train_acc

            if epoch >= WARMUP:
                scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_preds = torch.cat(epoch_preds)

            pbar.set_postfix(val_acc=f"{val_acc:.2%}", best=f"{best_acc:.2%}")
            # Plain newline-terminated summary so epoch history survives in log files
            tqdm.write(f"[Phase 2/{mode}] epoch {epoch+1}/{epochs}: "
                       f"train_acc={train_acc_for_log:.2%} val_acc={val_acc:.2%} best={best_acc:.2%}")

        # Diagnostics for the "learned" mode only, per user's request to
        # understand what's actually limiting accuracy (confusion matrix,
        # per-class accuracy) instead of more blind architecture guesses.
        if mode == "learned":
            self._print_diagnostics(best_preds, test_labels_dev.cpu())

        return {"epochs": phase2_epochs, "best": best_acc}

    def _print_diagnostics(self, preds, labels):
        """Confusion matrix + per-class accuracy for the best Phase-2 epoch."""
        class_names = ["ND", "CR", "LP", "PO"] if NUM_CLASSES == 4 else [str(i) for i in range(NUM_CLASSES)]
        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        for t, p in zip(labels.tolist(), preds.tolist()):
            cm[t, p] += 1
        tqdm.write("\n=== DIAGNOSTICS (best epoch, learned mode) ===")
        tqdm.write("Confusion matrix (rows=true, cols=predicted): " + " ".join(class_names))
        for i, name in enumerate(class_names):
            tqdm.write(f"  {name}: " + " ".join(f"{cm[i, j].item():5d}" for j in range(NUM_CLASSES)) +
                       f"   (n={cm[i].sum().item()}, acc={cm[i, i].item() / max(cm[i].sum().item(), 1):.2%})")
        tqdm.write("=== END DIAGNOSTICS ===\n")

    def _compute_measurements_on_loader(self, diffuser, loader):
        """Compute measurements for all batches in a DataLoader."""
        all_meas, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                if isinstance(diffuser, nn.Module):
                    meas = diffuser(images)
                else:
                    x_flat = images.view(images.size(0), -1)
                    meas   = (diffuser @ x_flat.T).T
                all_meas.append(meas.cpu())
                all_labels.append(labels)
        return torch.cat(all_meas), torch.cat(all_labels)

    def _compute_measurements_on_test(self, diffuser, mode="learned"):
        """Helper: Compute measurements on test data

        Handles both linear (matrix) and non-linear (CNN) diffusers.
        """
        # Load test data
        dataset_class, _, test_transform = get_dataset_and_transforms(DATASET, DATA_DIR)
        test_dataset = dataset_class(DATA_DIR, train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

        # Compute measurements
        all_meas = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Computing measurements"):
                images = images.to(self.device)

                # Check if diffuser is a neural network (NonLinearDiffuser) or matrix
                if isinstance(diffuser, nn.Module):
                    # Non-linear diffuser: direct forward pass
                    meas = diffuser(images)
                else:
                    # Linear diffuser: matrix multiplication
                    meas = compute_measurements(diffuser.to(self.device), images)

                all_meas.append(meas.cpu())
                all_labels.append(labels)

        measurements = torch.cat(all_meas, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return measurements, labels
    
    def _save_checkpoint(self, epoch, phase):
        """Save model checkpoint"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = f"{CHECKPOINT_DIR}/M{self.M}_{phase}_epoch{epoch}.pth"
        torch.save(self.encoder.state_dict(), path)
