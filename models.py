"""
Author: Tomer Porat

Model architectures for encoder and classifier
Dataset-agnostic: Works with CIFAR10, MNIST, Fashion-MNIST, etc.
"""

import torch
import torch.nn as nn

from config import NETWORK_RESIDUAL_MAX_RATIO, NETWORK_RESIDUAL_LEARNABLE_LATENT


class FocalLoss(nn.Module):
    """Standard focal loss (Lin et al. 2017, RetinaNet): down-weights
    already-easy/confident examples, up-weights hard/misclassified ones,
    adaptively per-example based on the model's own current confidence --
    no need to hand-specify which classes are hard.

    2026-08-12: motivated by RIAWELC (RIAWELC, M=8, "learned") diagnostics
    showing ND at 98% (already saturated) while CR/LP/PO sit at 82-85%, with
    233 of the errors being CR<->LP confusions specifically (crack vs
    lack-of-penetration look visually similar in X-ray). Standard
    cross-entropy spends equal gradient budget on easy ND examples and hard
    CR/LP examples; focal loss should redirect training effort toward the
    confused pairs automatically.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ResidualBlock(nn.Module):
    """Residual block for ResNet classifier"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.skip = nn.Sequential()
        if in_dim != out_dim:
            self.skip = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        identity = self.skip(identity)
        out = out + identity
        out = self.relu(out)
        return out


class ClassifierResNet(nn.Module):
    """ResNet-style classifier with skip connections"""

    def __init__(self, M, num_classes=10, hidden_dim=None, num_blocks=3):
        super().__init__()

        # Keep hidden_dim fixed - avoids 53× parameter explosion
        # Previous scaling (512 * (1 + (M-8)/16)) created 117M params for M=128
        # Fixed size allows fair comparison and faster training
        if hidden_dim is None:
            hidden_dim = 512

        self.input_proj = nn.Sequential(
            nn.Linear(M, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate=0.15)
            for _ in range(num_blocks)
        ])

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x)
        return x

# class Classifier(nn.Module):
#     """Classification head that operates on M-dimensional measurements"""
#
#     def __init__(self, M, num_classes=10, hidden_dims=None, dropout_rates=None):
#         super().__init__()
#
#         if hidden_dims is None:
#             hidden_dims = [512, 512, 256, 128]
#         if dropout_rates is None:
#             dropout_rates = [0.3, 0.3, 0.2, 0.1]
#
#         layers = []
#         in_dim = M
#
#         # Hidden layers
#         for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.Dropout(dropout_rate))
#             in_dim = hidden_dim
#
#         # Output layer
#         layers.append(nn.Linear(in_dim, num_classes))
#
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.net(x)


class ConvBlock(nn.Module):
    """Residual basic block (3x3 conv + BN) for the ResNet diffuser encoder"""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetDiffuser(nn.Module):
    """ResNet-18-style non-linear diffuser for learned measurement encoding.

    Replaces the plain 3-conv NonLinearDiffuser (no BN, no residuals, ~85%
    ceiling on CIFAR10) with a standard CIFAR ResNet-18 backbone. The final
    Linear(512 -> M) readout keeps the M-dimensional measurement bottleneck,
    and BatchNorm1d on the measurements keeps their scale stable without the
    manual weight-scaling hacks the old diffuser needed.
    """
    def __init__(self, M=64, input_channels=3, image_size=None):
        super().__init__()
        self.M = M
        self.input_channels = input_channels

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))
        self.layer2 = nn.Sequential(ConvBlock(64, 128, stride=2), ConvBlock(128, 128))
        self.layer3 = nn.Sequential(ConvBlock(128, 256, stride=2), ConvBlock(256, 256))
        self.layer4 = nn.Sequential(ConvBlock(256, 512, stride=2), ConvBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, M)
        self.meas_bn = nn.BatchNorm1d(M)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).flatten(1)
        measurements = self.meas_bn(self.fc(out))
        return measurements


class NonLinearDiffuser(nn.Module):
    """CNN-based non-linear diffuser for learned measurement encoding

    Replaces the linear measurement matrix D with a small CNN that learns
    to extract task-specific hierarchical features from images.
    Achieves 85%+ accuracy on CIFAR10 with M=64 (vs 51.5% for linear D).

    Fixed (2026-05-13): Removed AdaptiveAvgPool2d bottleneck that prevented
    M from scaling. Now preserves 256×8×8 spatial features for CIFAR10,
    allowing diffuser capacity to scale with M (87.93%→90.31% for M=8→128).

    Fixed (2026-05-14): Made dataset-agnostic by computing feature dimension
    dynamically based on input size. Works for CIFAR10 (32×32), MNIST (28×28),
    and other image sizes.
    """
    def __init__(self, M=64, input_channels=3, image_size=None):
        super().__init__()
        self.M = M
        self.input_channels = input_channels

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Compute feature dimension dynamically based on image size
        if image_size is None:
            # Infer from input: will use dummy forward to compute size
            self._feature_dim = None
        else:
            # Pre-compute feature dimension for known image size
            # After 2 MaxPool2d (stride 2): image_size // 4
            # Spatial dims: (image_size // 4) × (image_size // 4)
            # Channels: 256
            spatial_size = image_size // 4
            self._feature_dim = 256 * spatial_size * spatial_size

        # Create FC layer if feature dim is known
        if self._feature_dim is not None:
            self.fc = nn.Linear(self._feature_dim, M)
            # Initialize FC layer with larger weights to prevent output collapse
            nn.init.kaiming_normal_(self.fc.weight, a=0.1, mode='fan_in', nonlinearity='linear')
            with torch.no_grad():
                self.fc.weight.mul_(4.0)  # Scale up to ensure variance
        else:
            self.fc = None  # Will be created after first forward pass

        # Learnable scaling to prevent measurement collapse
        self.output_scale = nn.Parameter(torch.ones(1) * 5.0)

    def forward(self, x):
        features = self.conv(x)

        # Lazy initialization of FC layer if needed
        if self.fc is None:
            feature_dim = features.shape[1]
            self.fc = nn.Linear(feature_dim, self.M).to(x.device)
            nn.init.kaiming_normal_(self.fc.weight, a=0.1, mode='fan_in', nonlinearity='linear')
            with torch.no_grad():
                self.fc.weight.mul_(10.0)

        measurements = self.fc(features)

        # Scale measurements to prevent collapse (learnable parameter)
        measurements = measurements * self.output_scale

        return measurements


class LinearDiffuser(nn.Module):
    """Physically-realizable learned diffuser, plain linear sensing layer.

    y = D @ x via a single unconstrained nn.Linear(D_in, M, bias=False). No
    base/delta split, no generator indirection, no orthogonality constraint.

    Added 2026-08-02 after checking the published literature on task-driven
    learned sensing matrices (Adler et al., "Compressed Learning: A Deep
    Neural Network Approach", arXiv:1610.09615; "Fully Learnable Model for
    Task-Driven Image Compressed Sensing", PMC8309481): both report that a
    plain linear layer trained end-to-end with the downstream classifier
    outperforms random/fixed matrices, and that orthogonality constraints or
    structured init are NOT necessary -- joint task-driven optimization is
    what matters, not architectural cleverness on the matrix side. This
    directly contradicts the earlier (untested) orthogonality-regularization
    proposal for ResidualDiffuser -- kept here as the literature-grounded
    baseline to compare against the more complex residual/generator variants.
    """
    def __init__(self, M, D_in, input_channels=None, image_size=None):
        super().__init__()
        self.M = M
        self.D_in = D_in
        self.linear = nn.Linear(D_in, M, bias=False)

    def get_matrix(self):
        return self.linear.weight

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.linear(x_flat)


class BinaryDiffuser(nn.Module):
    """Physically-realizable learned diffuser, binary (+-1) mask matrix.

    y = D @ x where D = sign(Z) / sqrt(D_in), Z a continuous trainable
    parameter. Trained via a Straight-Through Estimator (STE): forward pass
    uses the true binary sign(Z), backward pass passes the gradient through
    as if sign() were the identity (`z + (sign(z) - z).detach()`).

    Added 2026-08-02 after checking the literature on binary sensing
    patterns for single-pixel imaging ("Learning Binary Sampling Patterns
    for Single-Pixel Imaging using Bilevel Optimisation", arXiv:2508.19068;
    "Non-imaging single-pixel sensing with optimized binary modulation",
    arXiv:1909.11498 -- 96%+ on MNIST). Binary masks are arguably a BETTER
    match to physical realizability than the continuous-valued matrices
    used elsewhere in this file: a real DMD mirror is literally on/off, not
    a continuous transmittance value, so this is closer to genuine hardware
    constraints, not just a training trick.
    """
    def __init__(self, M, D_in, input_channels=None, image_size=None):
        super().__init__()
        self.M = M
        self.D_in = D_in
        self.Z = nn.Parameter(torch.randn(M, D_in) * 0.1)

    def get_matrix(self):
        z = self.Z
        binary = z + (torch.sign(z) - z).detach()  # STE: forward=sign(z), backward=identity
        return binary / (self.D_in ** 0.5)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return x_flat @ self.get_matrix().T


class ResidualDiffuser(nn.Module):
    """Physically-realizable learned diffuser, proposal §4.1.2 (Residual Diffuser).

    Unlike ResNetDiffuser/NonLinearDiffuser (which run conv layers on the raw
    image, not realizable by a single-pixel bucket detector), this diffuser
    IS a single (M, D_in) mask matrix: y = D @ x, D = base + delta, where
    `base` is a fixed matrix and `delta` is the only learned parameter.
    Input-agnostic (forward never lets the matrix depend on the sample), so
    it is a valid computational-GI mask set, unlike the CNN diffusers.
    Residual parameterization keeps optimization stable by limiting
    deviation from an already-informative init (proposal: "preserves the
    favorable properties of [the] projection... while enabling fine-grained
    task-specific refinement").

    `base_matrix` lets the init come from a structured basis (e.g. DCT2D)
    instead of pure Gaussian noise: empirically (RIAWELC, M=8/16/32,
    2026-08-02) DCT2D already beats Random by a wide margin as a fixed
    baseline, so starting `delta`'s refinement from that stronger basin
    should out-perform refining a random init from scratch. Falls back to a
    random base (proposal's original spec) when `base_matrix` is None.
    """
    def __init__(self, M, D_in, input_channels=None, image_size=None, base_matrix=None):
        super().__init__()
        self.M = M
        self.D_in = D_in
        if base_matrix is None:
            base_matrix = torch.randn(M, D_in) / (D_in ** 0.5)
        self.register_buffer("base", base_matrix.clone().detach())
        self.delta = nn.Parameter(torch.zeros(M, D_in))

    def get_matrix(self):
        return self.base + self.delta

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return x_flat @ self.get_matrix().T


class GeneratorDiffuser(nn.Module):
    """Physically-realizable learned diffuser, proposal §4.1.1 (Generator-Based Diffuser).

    A small FC network maps a fixed latent noise vector to a full (M, D_in)
    mask matrix. The network never sees the input image, only its own
    weights are optimized (via gradients through the classification loss),
    so the resulting matrix is a single global, input-agnostic measurement
    operator, exactly like the proposal's Fig. 2 network. Simplification vs.
    the proposal: LayerNorm instead of BatchNorm1d (the proposal's Fig. 2 net
    runs on a single latent vector, i.e. batch size 1, which plain
    BatchNorm1d cannot normalize in training mode) and no output sigmoid
    (keeps matrix entries real-valued, consistent with the DCT/Hadamard/
    Random matrices already used elsewhere in this codebase, for a fair
    comparison).
    """
    def __init__(self, M, D_in, latent_dim=100, hidden_dim=512, input_channels=None, image_size=None):
        super().__init__()
        self.M = M
        self.D_in = D_in
        self.register_buffer("latent", torch.randn(1, latent_dim))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, M * D_in),
        )

    def get_matrix(self):
        return self.net(self.latent).view(self.M, self.D_in)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return x_flat @ self.get_matrix().T


def _clamp_delta_norm(delta, base, max_ratio=2.0):
    # 2026-08-09: settled here after testing 1.0/2.0/4.0 on RIAWELC M=8 --
    # results (83.80%/84.89%/83.52%) show no real trend, all within normal
    # run-to-run noise. 2.0 happened to score highest of the three sampled
    # points; kept as the default without further tuning (would need
    # multi-seed repeats to justify picking one value over another).
    """Rescale `delta` so ||delta|| <= max_ratio * ||base||, by construction --
    a structural guarantee, not a hope that weight decay / a lower LR will
    keep the network-generated correction from swamping the fixed anchor.

    2026-08-08: added after a synthetic-data test showed neither a 10x lower
    LR nor matching ResidualDiffuser's weight decay was enough on their own
    -- delta's norm still reached ~950 (LR fix alone) against a base norm of
    ~3, down from ~4400 unclamped, but still two orders of magnitude too
    large. Only clamping the norm directly enforces the "small correction on
    top of a strong anchor" design the whole ResidualDiffuser family relies
    on, independent of however large the network's raw output happens to be.
    """
    delta_norm = delta.norm()
    max_norm = base.norm() * max_ratio
    if delta_norm > max_norm:
        delta = delta * (max_norm / (delta_norm + 1e-8))
    return delta


class NetworkResidualDiffuser(nn.Module):
    """Residual diffuser (base + delta), but `delta` is the OUTPUT of a
    small network applied to a fixed latent vector -- like GeneratorDiffuser
    -- rather than a free (M, D_in) tensor like ResidualDiffuser's `delta`.

    2026-08-08, user's idea: ResidualDiffuser's `delta = nn.Parameter(zeros(M,
    D_in))` is exactly the "arbitrary weights" alternative to a structured
    generator -- an M*D_in-parameter free tensor with no inductive bias,
    which is the documented cause of the overfitting problem in
    `residual_delta_overfitting_fix.md` (delta memorizes RIAWELC's ~20
    training welds, worse as M grows since delta's parameter count grows
    with it). GeneratorDiffuser already shows a *network* can produce a
    matrix -- but it searches the whole matrix space from scratch, with no
    anchor point, which is the reason it lost to ResidualDiffuser+DCT2D in
    earlier testing (see CLAUDE.md "generator" vs "residual" history). This
    class combines both ideas instead of treating them as alternatives:
    anchor to a strong fixed base (DCT2D) like ResidualDiffuser, but generate
    the correction through a small network like GeneratorDiffuser, rather
    than leaving it as a free unconstrained tensor. The network's own
    architecture (depth, width, or convolutional structure) becomes the
    lever for controlling delta's effective capacity, as an alternative to
    LowRankResidualDiffuser's explicit U@V low-rank factorization.
    """
    def __init__(self, M, D_in, base_matrix=None, latent_dim=100, hidden_dim=512,
                 num_hidden_layers=2, input_channels=None, image_size=None):
        super().__init__()
        # 2026-08-08: reverted default 3->2 hidden layers -- the 3-hidden-layer
        # run scored 83.25% Phase 2 final (M=8), slightly WORSE than the
        # original 2-hidden-layer version's 84.89%. Depth isn't the lever;
        # kept configurable but back to the proven default so the max_ratio
        # change below is tested as a single variable against 84.89%, not
        # confounded with the depth change too.
        self.M = M
        self.D_in = D_in
        if base_matrix is None:
            base_matrix = torch.randn(M, D_in) / (D_in ** 0.5)
        self.register_buffer("base", base_matrix.clone().detach())
        # 2026-08-12: latent made LEARNABLE (nn.Parameter, not a frozen
        # buffer) when NETWORK_RESIDUAL_LEARNABLE_LATENT is set -- see
        # config.py. Previously the network had to produce a good delta from
        # an arbitrary FIXED random input, using only its weights; letting
        # the ~100-scalar input itself be optimized too is a cheap
        # (negligible extra parameter count vs delta_net's ~500K+) extra
        # degree of freedom, a different kind of lever than every capacity
        # increase tried so far (all of which widen/deepen the WEIGHTS, not
        # the input they're fed).
        if NETWORK_RESIDUAL_LEARNABLE_LATENT:
            self.latent = nn.Parameter(torch.randn(1, latent_dim))
        else:
            self.register_buffer("latent", torch.randn(1, latent_dim))

        # 2026-08-08: num_hidden_layers made configurable (default raised
        # 2->3) at the user's request to try more capacity, independent of
        # the (deprioritized) conv variant -- still flat FC, just deeper.
        layers = [nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, M * D_in))
        self.delta_net = nn.Sequential(*layers)

        # Zero-init the final layer so delta_net(latent) == 0 at init, i.e.
        # get_matrix() starts EXACTLY at `base` (DCT2D) -- matching
        # ResidualDiffuser's zero-initialized free delta. Without this, the
        # network's random initial weights perturb the matrix away from the
        # strong DCT2D starting point before training even begins, undermining
        # the whole anchoring rationale (2026-08-08, found via a bad first run).
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

    def get_matrix(self):
        delta = self.delta_net(self.latent).view(self.M, self.D_in)
        # 2026-08-11: max_ratio now config-driven (NETWORK_RESIDUAL_MAX_RATIO)
        # to run the queued M=64 tighter-cap test (1.0 vs the proven M=8
        # default of 2.0) without editing this file per run.
        delta = _clamp_delta_norm(delta, self.base, max_ratio=NETWORK_RESIDUAL_MAX_RATIO)
        return self.base + delta

    def delta_parameters(self):
        """Params that produce the correction on top of `base` -- give these
        their own (stronger-weight-decay / lower-LR) optimizer param group,
        the same way ResidualDiffuser's free `delta` tensor already gets one,
        to keep the correction from swamping the DCT2D anchor (2026-08-08:
        found via a synthetic test that delta's norm explodes to ~4400 at the
        project's default max_lr=0.1, vs a base matrix norm of ~3)."""
        return list(self.delta_net.parameters())

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return x_flat @ self.get_matrix().T


class AdaptiveDiffuser(nn.Module):
    """Adaptive/sequential learned diffuser -- NOT a fixed (M,D) matrix.

    Every other diffuser in this file (Residual/Generator/Linear/their
    distilled variants) reduces to one fixed matrix D applied identically to
    every sample: y = D @ x. All of them cluster around the same accuracy
    ceiling (measured empirically, RIAWELC M=64: 64-67% regardless of how D
    is produced) because they're all bounded by the same "single fixed
    linear projection" argument.

    This diffuser is genuinely different: masks are generated ONE AT A TIME
    by a GRU controller, and each mask depends on the measurements already
    taken FROM THAT SAME SAMPLE so far (not on any other sample, not on the
    label -- only on that object's own prior bucket readings). This is
    still physically realizable: a real single-pixel/DMD setup with a fast
    controller (e.g. FPGA) can read the previous bucket intensity and choose
    the next mask pattern in real time, before the next flash. This is the
    "Adaptive Compressed Sensing" paradigm in the literature -- distinct
    from, and not limited by the same ceiling as, a fixed sensing matrix.

    forward() unrolls M steps: mask_t (a length-D_in vector) dot the sample
    -> scalar measurement m_t -> GRU hidden state update -> mask_{t+1}. The
    whole sequence is differentiable (dot products + GRU + linear), so it
    trains end-to-end with backprop exactly like the fixed-matrix diffusers.
    """
    def __init__(self, M, D_in, hidden_dim=128, input_channels=None, image_size=None):
        super().__init__()
        self.M = M
        self.D_in = D_in
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(1, hidden_dim)
        self.mask_head = nn.Linear(hidden_dim, D_in)
        self.init_mask = nn.Parameter(torch.randn(D_in) / (D_in ** 0.5))

    def forward(self, x):
        batch = x.size(0)
        x_flat = x.view(batch, -1)
        device = x.device
        h = torch.zeros(batch, self.hidden_dim, device=device)
        mask = self.init_mask.unsqueeze(0).expand(batch, -1)

        measurements = []
        for _ in range(self.M):
            m_t = (mask * x_flat).sum(dim=1, keepdim=True)
            measurements.append(m_t)
            h = self.gru(m_t, h)
            mask = self.mask_head(h)

        return torch.cat(measurements, dim=1)


class ProxyCNN(nn.Module):
    """CNN classifier over the adjoint proxy image x_hat = D^T y.

    The physics constrains only the measurement (y = D @ x, bucket detector);
    everything after y is computation, and the proposal itself (S3.2.1)
    specifies a CNN classifier. Instead of feeding an MLP the raw M-vector,
    embed the measurements back to image space via the adjoint (x_hat = D^T y,
    a linear, information-preserving reindexing) and classify the proxy image
    with a ResNet-style CNN, recovering the spatial inductive bias the MLP
    lacks. Standard compressed-learning decoder (Adler et al. 1610.09615).

    Empirical effect (RIAWELC, clean weld split, 2026-08-03): fixed DCT2D at
    M=32 went 69.09% (MLP decoder, 35 epochs) -> 78.38% (this decoder, 20
    epochs) -- broke the ~70% asymptote every MLP-decoded linear method
    saturated at. At M=D the proxy is exact, so accuracy provably approaches
    the raw-image CNN ceiling (~90-93%) as M grows -- restoring the monotone
    accuracy-vs-M curve.
    """
    def __init__(self, num_classes=4, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))
        self.layer2 = nn.Sequential(ConvBlock(64, 128, stride=2), ConvBlock(128, 128))
        self.layer3 = nn.Sequential(ConvBlock(128, 256, stride=2), ConvBlock(256, 256))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x_hat):
        out = self.stem(x_hat)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return self.fc(self.avgpool(out).flatten(1))


class ProxyResNet18(nn.Module):
    """Deeper adjoint-proxy decoder -- standard 4-stage ResNet-18 depth.

    2026-08-06: ProxyCNN (above) is only 3 stages (64->128->256, 8x8 final
    map for 32x32 input) -- lighter than a "standard" ResNet-18. It failed
    to help CIFAR10 (M=64: 57.49% vs 58.58% MLP baseline, a regression) even
    though the same mechanism gave +9.3pp on RIAWELC. Before concluding the
    adjoint-proxy decoder approach doesn't work for natural images at all,
    test whether it was simply undersized: this class adds the 4th stage
    (256->512, stride 2 -> 4x4 final map) that ProxyCNN lacks, exactly
    matching ResNetDiffuser's proven encoder depth (the same architecture
    that took CIFAR10 learned-mode from 84%->92% as an encoder, see
    RESNET_DIFFUSER_REPORT.md). Reuses the same ConvBlock building block for
    a controlled, single-variable comparison (depth only, nothing else
    changes) against ProxyCNN.
    """
    def __init__(self, num_classes=4, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))
        self.layer2 = nn.Sequential(ConvBlock(64, 128, stride=2), ConvBlock(128, 128))
        self.layer3 = nn.Sequential(ConvBlock(128, 256, stride=2), ConvBlock(256, 256))
        self.layer4 = nn.Sequential(ConvBlock(256, 512, stride=2), ConvBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x_hat):
        out = self.stem(x_hat)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.fc(self.avgpool(out).flatten(1))


class ProxyHybridCNN(nn.Module):
    """Adjoint-proxy CNN + raw measurement vector, combined.

    2026-08-06: both ProxyCNN (3-stage) and ProxyResNet18 (4-stage) failed
    to beat the plain MLP on CIFAR10 (M=64, "learned": 57.49% and ~55%
    respectively vs 58.58% MLP) -- and gave nearly IDENTICAL results to each
    other, meaning decoder depth/capacity was not the bottleneck. Working
    hypothesis: reshaping y into the blurry adjoint-proxy image x_hat=D^Ty
    for a natural, texture-heavy dataset like CIFAR10 may destroy usable
    signal that the raw M-vector still carries (unlike RIAWELC, where large
    low-frequency defect shapes survive the same blur -- see the ProxyCNN
    docstring above and CLAUDE.md's 2026-08-03 finding). Rather than forcing
    a choice between the two representations, feed the classifier BOTH: a
    CNN branch over the proxy image (spatial structure, when useful) and a
    small MLP branch over the raw M-vector (whatever the CNN branch loses),
    concatenated before the final classification layer. Reuses the same
    ConvBlock stages as ProxyCNN (not the deeper ProxyResNet18, since depth
    was already ruled out) to isolate "add the raw-vector branch" as the one
    new variable.
    """
    def __init__(self, M, num_classes=4, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))
        self.layer2 = nn.Sequential(ConvBlock(64, 128, stride=2), ConvBlock(128, 128))
        self.layer3 = nn.Sequential(ConvBlock(128, 256, stride=2), ConvBlock(256, 256))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.meas_branch = nn.Sequential(
            nn.Linear(M, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256 + 256, num_classes)

    def forward(self, x_hat, y):
        img_feat = self.stem(x_hat)
        img_feat = self.layer1(img_feat)
        img_feat = self.layer2(img_feat)
        img_feat = self.layer3(img_feat)
        img_feat = self.avgpool(img_feat).flatten(1)
        meas_feat = self.meas_branch(y)
        combined = torch.cat([img_feat, meas_feat], dim=1)
        return self.fc(combined)


class LowRankResidualDiffuser(nn.Module):
    """Physically-realizable learned diffuser -- rank-limited residual correction.

    Same idea as ResidualDiffuser (D = base + delta, y = D @ x, input-agnostic
    matrix), but `delta` is factored as a low-rank product `U @ V` (U: (M,r),
    V: (r,D_in), r << D_in) instead of a free (M,D_in) tensor.

    Motivation (2026-08-04): the free-delta ResidualDiffuser overfits
    RIAWELC's ~20 training welds under the ProxyCNN decoder, and the
    overfitting WORSENS as M grows (delta's parameter count is M*D_in, so
    bigger M means strictly more capacity to memorize the same small
    training set) -- gap to fixed transforms widened from -3.1pp at M=32 to
    -7.1pp at M=64. A per-M weight-decay tweak validated only at M=32 made
    M=8 worse when deployed globally (reverted, see
    proposal_vs_code_learned_diffuser_mismatch / residual_delta_overfitting_fix
    memories) -- a structural capacity cap is a more principled fix than
    re-tuning a scalar hyperparameter per M.

    With rank r fixed, delta's per-row capacity is bounded by r regardless of
    M (V is shared across all M rows, only U scales with M), so the
    memorization risk should grow much more slowly with M than the free-delta
    version -- testable directly instead of assumed. U is zero-initialized
    (so the matrix starts exactly at `base`, matching ResidualDiffuser's
    init) and V is randomly initialized (standard low-rank-adapter pattern).
    """
    def __init__(self, M, D_in, rank=16, input_channels=None, image_size=None, base_matrix=None):
        super().__init__()
        self.M = M
        self.D_in = D_in
        self.rank = rank
        if base_matrix is None:
            base_matrix = torch.randn(M, D_in) / (D_in ** 0.5)
        self.register_buffer("base", base_matrix.clone().detach())
        self.U = nn.Parameter(torch.zeros(M, rank))
        self.V = nn.Parameter(torch.randn(rank, D_in) / (D_in ** 0.5))

    def get_matrix(self):
        return self.base + self.U @ self.V

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return x_flat @ self.get_matrix().T


class GISetTokenizer(nn.Module):
    """Turns each of the M raw measurements into a d_model-dim token.

    2026-08-08: adapted from a Set-Transformer classifier for GI measurements
    shared by Gal Katz (contact via Prof. Cohen), originally class name
    `Tokenizer`. Renamed with a `GISet` prefix here to avoid colliding with
    unrelated `Tokenizer`/`SAB`/`PMA` names elsewhere. Logic unchanged.
    """
    def __init__(self, d_model, p, use_conv=True, k=7):
        super().__init__()
        self.use_conv = bool(use_conv)
        if self.use_conv:
            kk = int(max(3, k))
            if kk % 2 == 0:
                kk += 1
            self.conv = nn.Conv1d(1, d_model, kernel_size=kk, padding=kk // 2, bias=True)
            self.post = nn.Sequential(nn.GELU(), nn.Dropout(p))
            self.ln = nn.LayerNorm(d_model)
        else:
            self.net = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(p),
            )

    def forward(self, y):
        if not self.use_conv:
            return self.net(y)
        x = y.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = self.post(x)
        return x


class GISetSAB(nn.Module):
    """Self-Attention Block: standard pre-LN Transformer encoder layer,
    applied over the M measurement-tokens (attention *between* measurements).
    See GISetTokenizer docstring for provenance."""
    def __init__(self, d_model, n_heads, p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(p)

    def forward(self, x):
        z, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.drop(z)
        z = self.ffn(self.ln2(x))
        x = x + self.drop(z)
        return x


class GISetPMA(nn.Module):
    """Pooling by Multihead Attention (Set Transformer, Lee et al. 2019):
    a single learned query vector attends over all M tokens, producing one
    permutation-invariant pooled vector. See GISetTokenizer docstring for
    provenance."""
    def __init__(self, d_model, n_heads, p, k=1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d_model))
        self.ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B = x.size(0)
        seed = self.seed.expand(B, -1, -1)
        z, _ = self.mha(self.ln(seed), self.ln(x), self.ln(x), need_weights=False)
        return seed + self.drop(z)


class GISetClassifier(nn.Module):
    """Set-Transformer classifier over the raw M-dimensional measurement
    vector: treats each measurement as a token (no adjoint/image reshape,
    unlike ProxyCNN/ProxyHybridCNN), lets tokens attend to each other
    (GISetSAB), pools permutation-invariantly (GISetPMA), then classifies.

    2026-08-08: adapted from Gal Katz's GISetClassifier (see GISetTokenizer
    docstring). Only change from the original: the output head's final layer
    is `Linear(d_model // 2, num_classes)` instead of `Linear(d_model // 2, 1)`
    (the original returned a single logit, presumably for a binary task; this
    project's datasets are multi-class). Matches this project's classifier
    constructor convention, `(M, num_classes=...)`, used by ClassifierResNet
    et al., and its `forward(self, y)` signature where `y` is the raw
    (batch, M) measurement vector -- reshaped to (batch, M, 1) internally.
    """
    def __init__(self, M, num_classes=4, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.1, use_conv_tok=True, conv_k=7):
        super().__init__()
        self.M = M
        self.tok = GISetTokenizer(d_model, dropout, use_conv=use_conv_tok, k=conv_k)
        self.blocks = nn.ModuleList([GISetSAB(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.pma = GISetPMA(d_model, n_heads, dropout, k=1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, y):
        y = y.view(y.size(0), self.M, 1)
        x = self.tok(y)
        for blk in self.blocks:
            x = blk(x)
        pooled = self.pma(x).squeeze(1)
        return self.head(pooled)


class GISetPatchTokenizer(nn.Module):
    """ViT-style patch embedding: splits the adjoint-proxy image x_hat=D^T y
    into non-overlapping patches and linearly projects each to a d_model
    token, instead of GISetTokenizer's one-token-per-raw-measurement scheme.
    A single strided conv both extracts and projects patches (standard ViT
    patch-embedding trick)."""
    def __init__(self, d_model, in_channels=1, patch_size=4, p=0.1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(d_model)
        self.post = nn.Sequential(nn.GELU(), nn.Dropout(p))

    def forward(self, x_hat):
        x = self.proj(x_hat)              # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        x = self.ln(x)
        x = self.post(x)
        return x


class GISetClassifierProxy(nn.Module):
    """Same Set-Transformer body as GISetClassifier (GISetSAB + GISetPMA +
    classification head), but tokenizes the adjoint-proxy image
    $\\hat{x}=D^\\top y$ via ViT-style patches instead of tokenizing the raw
    M measurements directly.

    2026-08-08: user hypothesis -- a single raw measurement (e.g. one DCT
    coefficient) carries no spatial structure on its own, so attention over
    M such scalars may be attending over relatively uninformative tokens.
    Patches of the adjoint-reconstructed proxy image, by contrast, each
    carry local spatial context (like ProxyCNN/ProxyHybridCNN's motivation),
    so self-attention here operates *between spatial regions* rather than
    between frequency-domain scalars. Requires a genuine diffuser matrix to
    build x_hat (like ProxyCNN/ProxyHybridCNN); not usable with CNN
    diffusers (resnet/simple), which have no matrix.
    """
    def __init__(self, num_classes=4, in_channels=1, patch_size=4,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.tok = GISetPatchTokenizer(d_model, in_channels, patch_size, dropout)
        self.blocks = nn.ModuleList([GISetSAB(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.pma = GISetPMA(d_model, n_heads, dropout, k=1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x_hat):
        x = self.tok(x_hat)
        for blk in self.blocks:
            x = blk(x)
        pooled = self.pma(x).squeeze(1)
        return self.head(pooled)


class ConvNetworkResidualDiffuser(nn.Module):
    """Like NetworkResidualDiffuser, but the delta-generating network is
    convolutional instead of flat FC -- inspired by Mor Hale's Gen_conv_3
    (only_code_12_10_24_Mor_Hale_ORIGINAL/Model.py, the DO-CGI reconstruction
    codebase cited as \\cite{docgi}), which reshapes a latent vector into a
    spatial grid and generates the mask set via Conv2d layers rather than
    flat FC. Anchored to a fixed base (e.g. DCT2D) like ResidualDiffuser /
    NetworkResidualDiffuser -- only the delta-generation mechanism differs.

    2026-08-08. Currently assumes a single input channel and
    D_in == image_size**2 (i.e. grayscale, like RIAWELC/MNIST/Fashion-MNIST
    in this project's unified 32x32 pipeline); would need a channel-aware
    reshape to support CIFAR-10's 3-channel D_in=3072.
    """
    def __init__(self, M, D_in, base_matrix=None, latent_dim=100, image_size=32,
                 input_channels=None, hidden_channels=64):
        super().__init__()
        self.M = M
        self.D_in = D_in
        self.image_size = image_size
        assert D_in == image_size * image_size, (
            "ConvNetworkResidualDiffuser currently only supports single-channel "
            "D_in == image_size**2 (e.g. grayscale RIAWELC/MNIST/Fashion-MNIST)."
        )
        if base_matrix is None:
            base_matrix = torch.randn(M, D_in) / (D_in ** 0.5)
        self.register_buffer("base", base_matrix.clone().detach())
        self.register_buffer("latent", torch.randn(1, latent_dim))

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, image_size * image_size),
            nn.LayerNorm(image_size * image_size),
            nn.ReLU(inplace=True),
        )
        # GroupNorm(1, C) == LayerNorm over (C,H,W) -- no batch statistics
        # needed, unlike BatchNorm2d. This network always runs on a batch of
        # size 1 (the single fixed latent), the same anti-pattern
        # GeneratorDiffuser's own docstring already warned about re:
        # BatchNorm1d (2026-08-08: found the hard way, via a bad first run).
        #
        # Hidden channel count is now FIXED (default 64), not derived from M
        # like Mor Hale's Gen_conv_3 (c1=M//4, c2=M//2) -- that formula
        # assumes M large enough for the resulting channel counts to carry
        # real capacity (his reconstruction task uses far more measurements).
        # At this project's M range (8/16/32), M//4 and M//2 give absurdly
        # narrow hidden layers (2-4 channels at M=8) -- likely insufficient
        # capacity, unlike NetworkResidualDiffuser's FC path, which keeps a
        # constant hidden_dim=512 regardless of M. Fixing hidden_channels
        # keeps capacity comparable across M, matching that design principle
        # (2026-08-08: found via a second bad run, after the GroupNorm fix
        # alone didn't resolve the gap to NetworkResidualDiffuser).
        c = max(1, int(hidden_channels))
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.GroupNorm(1, c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(1, c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, M, kernel_size=3, padding=1),
        )
        # Zero-init the final conv so delta starts at exactly 0, i.e.
        # get_matrix() starts at `base` -- see NetworkResidualDiffuser's
        # docstring for why this matters (2026-08-08 bad-first-run finding).
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def get_matrix(self):
        x = self.fc(self.latent).view(1, 1, self.image_size, self.image_size)
        delta = self.conv(x)                    # (1, M, H, W)
        delta = delta.view(self.M, self.D_in)    # (M, D_in)
        delta = _clamp_delta_norm(delta, self.base)
        return self.base + delta

    def delta_parameters(self):
        """See NetworkResidualDiffuser.delta_parameters -- same rationale,
        applies equally here (this network's delta is even more prone to
        exploding, since a single shared conv kernel update perturbs all M
        output channels' receptive fields at once)."""
        return list(self.fc.parameters()) + list(self.conv.parameters())

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return x_flat @ self.get_matrix().T
