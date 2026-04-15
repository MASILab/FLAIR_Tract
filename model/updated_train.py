from aim import Image as AimImage
import matplotlib
import gc
import time


matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from pathlib import Path
import warnings

# Aim import for experiment tracking
from aim import Run

from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning
import numpy as np
import math
import os
import configparser

from utils import default_context
from data import DT1Dataset, unload
from modules import (
    DetStepLoss,
    DetRNN,
    DetFCLoss,
    DetCNNFake,
    DetConvProj,
)


warnings.simplefilter("ignore", category=UserWarning)


def maybe_wrap_ddp(model, device_ids, output_device):
    """Wrap model with DDP only if it has parameters requiring gradients."""
    if any(p.requires_grad for p in model.parameters()):
        return DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=True,
            static_graph=True,
        )
    return model


def unwrap_model(model):
    """Return the underlying module, handling both DDP-wrapped and plain models."""
    return model.module if isinstance(model, DDP) else model


def visualize_predictions(
    fod_pred,
    fod_step_pred,
    step,
    mask,
    stage,
    t1_pred=None,
    t1_step_pred=None,
    save_path=None,
    epoch=None,
    slice_dim=2,
    rotation_k=1,  # 270° clockwise = 1 × 90° counter-clockwise
):
    """Generate and save visualization of 3D CNN predictions and 1D/2D RNN outputs.

    Args:
        fod_pred: FOD CNN predictions [B, C, D, H, W] or [B, D, H, W].
        fod_step_pred: FOD RNN step predictions [B, num_steps, 3] or [B, 3, num_steps].
        step: Ground truth step directions [B, num_steps, 3] or [B, 3, num_steps].
        mask: Valid streamline mask [B, D, H, W] or [B, 1, D, H, W].
        t1_pred: T1 CNN predictions (Stage 1 only).
        t1_step_pred: T1 RNN step predictions (Stage 1 only).
        save_path: Path to save the figure.
        epoch: Current epoch number.
        slice_dim: Which dimension to use for middle slice extraction (for 3D volumes).
        rotation_k: Number of 90° counter-clockwise rotations for 3D slices.
    """

    def get_middle_slice(tensor_3d, dim):
        """Extract middle slice from 3D/4D tensor along specified dimension."""
        if tensor_3d.ndim == 4:  # [C, D, H, W]
            if dim == 0:  # axial: middle of depth
                return tensor_3d[:, tensor_3d.shape[1] // 2, :, :].mean(axis=0)
            elif dim == 1:  # sagittal: middle of height
                return tensor_3d[:, :, tensor_3d.shape[2] // 2, :].mean(axis=0)
            else:  # coronal: middle of width
                return tensor_3d[:, :, :, tensor_3d.shape[3] // 2].mean(axis=0)
        elif tensor_3d.ndim == 3:  # [D, H, W]
            if dim == 0:
                return tensor_3d[tensor_3d.shape[0] // 2, :, :]
            elif dim == 1:
                return tensor_3d[:, tensor_3d.shape[1] // 2, :]
            else:
                return tensor_3d[:, :, tensor_3d.shape[2] // 2]
        return tensor_3d

    def normalize_for_viz(arr):
        """Normalize array to [0, 1] for visualization."""
        arr = np.asarray(arr)

        if arr.size == 0:
            return arr

        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)

    def rotate_slice(arr, k):
        """Rotate slice by k * 90 degrees counter-clockwise."""
        return np.rot90(arr, k=k)

    def plot_streamline_components(
        ax, step_data, title, color_pred="blue", color_gt="red"
    ):
        """Plot streamline direction components (x, y, z) as line plots."""
        # step_data shape: [num_steps, 3] or [3, num_steps]
        if step_data.shape[0] == 3 and step_data.ndim == 2:
            # Transpose to [num_steps, 3]
            step_data = step_data.T

        num_steps = step_data.shape[0]
        steps = np.arange(num_steps)

        # Plot each component
        ax.plot(steps, step_data[:, 0], label="X", color=color_pred, linewidth=1)
        ax.plot(steps, step_data[:, 1], label="Y", color=color_gt, linewidth=1)
        ax.plot(steps, step_data[:, 2], label="Z", color="green", linewidth=1)

        ax.set_xlabel("Streamline Step")
        ax.set_ylabel("Direction Component")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, num_steps)

    def plot_streamline_heatmap(ax, step_data, title, cmap="viridis"):
        """Plot streamline data as a 2D heatmap."""
        if step_data.shape[0] == 3 and step_data.ndim == 2:
            step_data = step_data.T  # [num_steps, 3]

        step_norm = normalize_for_viz(step_data)
        im = ax.imshow(step_norm.T, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xlabel("Streamline Step")
        ax.set_ylabel("Component (X/Y/Z)")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["X", "Y", "Z"])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def plot_orthogonal_3d(ax_row, data, title_prefix):
        """Plot 3D volume as three orthogonal middle slices with rotation."""
        for col, dim in enumerate([0, 1, 2]):
            slice_data = get_middle_slice(data, dim)
            slice_rotated = rotate_slice(slice_data, rotation_k)
            slice_norm = normalize_for_viz(slice_rotated)
            im = ax_row[col].imshow(slice_norm, cmap="gist_yarg", vmin=0.0, vmax=1.0)
            ax_row[col].set_title(
                f"{title_prefix} - {'Sagittal' if col == 0 else 'Coronal' if col == 1 else 'Axial'}"
            )
            ax_row[col].axis("off")
            plt.colorbar(im, ax=ax_row[col], fraction=0.046, pad=0.04)

    # Detach and move to CPU
    fod_pred_np = fod_pred.detach().cpu().numpy()
    fod_step_pred_np = fod_step_pred.detach().cpu().numpy()
    step_np = step.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    t1_pred_np = t1_pred.detach().cpu().numpy() if t1_pred is not None else None
    t1_step_pred_np = (
        t1_step_pred.detach().cpu().numpy() if t1_step_pred is not None else None
    )

    # Detect data types by shape
    is_fod_step_1d = (
        fod_step_pred_np[0].ndim <= 2 and fod_step_pred_np[0].shape[-1] == 3
    )
    is_step_1d = step_np[0].ndim <= 2 and step_np[0].shape[-1] == 3
    is_t1_step_1d = (
        t1_step_pred_np is not None
        and t1_step_pred_np[0].ndim <= 2
        and t1_step_pred_np[0].shape[-1] == 3
    )

    # Create figure: 4 rows (FOD CNN, FOD RNN, GT, T1 CNN/RNN) x 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    # Row 0: FOD CNN predictions (3D volume - orthogonal slices)
    plot_orthogonal_3d(axes[0], fod_pred_np[0], "FOD CNN Pred")

    # Row 1: FOD RNN step predictions (1D streamline - line plot or heatmap)
    if is_fod_step_1d and fod_step_pred_np[0].size > 0:
        # Use line plot for directional components
        plot_streamline_components(
            axes[1, 0], fod_step_pred_np[0], "FOD RNN: X/Y/Z Components"
        )
        # Use heatmap for intensity view
        plot_streamline_heatmap(axes[1, 1], fod_step_pred_np[0], "FOD RNN: Heatmap")
        # Show magnitude
        magnitudes = (
            np.linalg.norm(fod_step_pred_np[0], axis=-1)
            if fod_step_pred_np[0].shape[-1] == 3
            else fod_step_pred_np[0].squeeze()
        )
        if magnitudes.size > 0:
            axes[1, 2].plot(
                np.arange(len(magnitudes)), magnitudes, color="purple", linewidth=1.5
            )
            axes[1, 2].set_xlabel("Streamline Step")
            axes[1, 2].set_ylabel("Magnitude")
            axes[1, 2].set_title("FOD RNN: Direction Magnitude")
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, "No magnitude data", ha="center", va="center")
            axes[1, 2].axis("off")
    else:
        # Fallback to orthogonal slices if not 1D or empty
        plot_orthogonal_3d(axes[1], fod_step_pred_np[0], "FOD RNN Pred")

    # Row 2: Ground Truth Step (same visualization as predictions for comparison)
    if is_step_1d and step_np[0].size > 0:
        plot_streamline_components(
            axes[2, 0],
            step_np[0],
            "Ground Truth: X/Y/Z Components",
            color_pred="red",
            color_gt="blue",
        )
        plot_streamline_heatmap(axes[2, 1], step_np[0], "Ground Truth: Heatmap")
        magnitudes = (
            np.linalg.norm(step_np[0], axis=-1)
            if step_np[0].shape[-1] == 3
            else step_np[0].squeeze()
        )
        if magnitudes.size > 0:
            axes[2, 2].plot(
                np.arange(len(magnitudes)), magnitudes, color="orange", linewidth=1.5
            )
            axes[2, 2].set_xlabel("Streamline Step")
            axes[2, 2].set_ylabel("Magnitude")
            axes[2, 2].set_title("Ground Truth: Direction Magnitude")
            axes[2, 2].grid(True, alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, "No magnitude data", ha="center", va="center")
            axes[2, 2].axis("off")
    else:
        plot_orthogonal_3d(axes[2], step_np[0], "Ground Truth Step")

    # Row 3: T1 predictions (Stage 1 only) or mask
    if stage == 1 and t1_pred is not None:
        if (
            is_t1_step_1d
            and t1_step_pred_np is not None
            and t1_step_pred_np[0].size > 0
        ):
            # T1 RNN step predictions (1D streamline - line plot or heatmap)
            # Use line plot for directional components
            plot_streamline_components(
                axes[3, 0], t1_step_pred_np[0], "T1 RNN: X/Y/Z Components"
            )
            # Use heatmap for intensity view
            plot_streamline_heatmap(axes[3, 1], t1_step_pred_np[0], "T1 RNN: Heatmap")
            # Show magnitude
            magnitudes = (
                np.linalg.norm(t1_step_pred_np[0], axis=-1)
                if t1_step_pred_np[0].shape[-1] == 3
                else t1_step_pred_np[0].squeeze()
            )
            if magnitudes.size > 0:
                axes[3, 2].plot(
                    np.arange(len(magnitudes)), magnitudes, color="teal", linewidth=1.5
                )
                axes[3, 2].set_xlabel("Streamline Step")
                axes[3, 2].set_ylabel("Magnitude")
                axes[3, 2].set_title("T1 RNN: Direction Magnitude")
                axes[3, 2].grid(True, alpha=0.3)
            else:
                axes[3, 2].text(0.5, 0.5, "No magnitude data", ha="center", va="center")
                axes[3, 2].axis("off")
        else:
            # Fallback: Show T1 CNN orthogonal slices if no RNN step data
            plot_orthogonal_3d(axes[3], t1_pred_np[0], "T1 CNN Pred")
    else:
        for col in range(3):
            axes[3, col].text(
                0.5,
                0.5,
                f"Stage 0\nMask: {mask_np[0].shape}",
                ha="center",
                va="center",
                fontsize=12,
                style="italic",
            )
            axes[3, col].axis("off")

    plt.suptitle(f"Epoch {epoch} - Validation Predictions", fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    del fod_pred_np, fod_step_pred_np, step_np, mask_np
    if t1_pred_np is not None:
        del t1_pred_np
    if t1_step_pred_np is not None:
        del t1_step_pred_np

    gc.collect()

    return save_path


def epoch_losses(
    data_loader,
    fod_cnn,
    t1_cnn,
    fod_rnn,
    t1_rnn,
    fc_criterion,
    rnn_criterion,
    device,
    optimizer,
    scheduler,
    label,
    stage,
    scaler,
    use_amp,
    rank,
):
    """Compute losses over one epoch for training or validation.

    Args:
        data_loader: DataLoader for the dataset.
        fod_cnn: FOD CNN model.
        t1_cnn: T1 CNN model.
        fod_rnn: FOD RNN model.
        t1_rnn: T1 RNN model.
        fc_criterion: Criterion for fully connected loss.
        rnn_criterion: Criterion for RNN loss.
        device: Torch device.
        optimizer: Optimizer.
        scheduler: Scheduler.
        label: Label for progress bar (e.g., 'Train', 'Validation').
        stage: Training stage (0 or 1).
        scaler: GradScaler for AMP.
        use_amp: Boolean flag for Automatic Mixed Precision.
        rank: Process rank for distributed training.

    Returns:
        Tuple of average losses.
    """
    # epoch_start = time.time()

    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    train = "train" in label.lower()

    epoch_loss = 0
    epoch_fod_dot_loss = 0
    epoch_fod_cum_loss = 0
    epoch_t1_dot_loss = 0
    epoch_t1_cum_loss = 0
    epoch_fc_loss = 0

    epoch_iters = len(data_loader)

    show_batch_progress = rank == 0

    if train:
        optimizer.zero_grad(set_to_none=True)
        if stage == 0:
            fod_cnn.train()
            fod_rnn.train()
        else:
            fod_cnn.eval()
            fod_rnn.eval()
            for param in fod_cnn.parameters():
                param.requires_grad = False
            for param in fod_rnn.parameters():
                param.requires_grad = False
            t1_cnn.train()
            t1_rnn.train()
    else:
        fod_cnn.eval()
        t1_cnn.eval()
        fod_rnn.eval()
        t1_rnn.eval()

    if stage == 0:

        def loss_fxn(
            fod_dot_loss,
            fod_cum_loss,
            t1_dot_loss,
            t1_cum_loss,
            fc_loss,
        ):
            return fod_dot_loss

    else:

        def loss_fxn(
            fod_dot_loss,
            fod_cum_loss,
            t1_dot_loss,
            t1_cum_loss,
            fc_loss,
        ):
            return fc_loss + t1_dot_loss

    if train and use_amp:
        ctx_fod = autocast(device_type="cuda")
    elif stage == 1 or (stage == 0 and not train):
        ctx_fod = torch.no_grad()
    else:
        ctx_fod = default_context()

    if stage == 1:
        if train and use_amp:
            ctx_t1 = autocast(device_type="cuda")
        elif train:
            ctx_t1 = default_context()
        else:
            ctx_t1 = torch.no_grad()

    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    batch_times = []
    for bidx, dt1_item in tqdm(
        enumerate(data_loader),
        total=epoch_iters,
        desc=label,
        leave=False,
        disable=not show_batch_progress,
    ):
        # batch_start = time.time()
        # ten_2mm, fod, brain, step, trid, trii, mask = unload(*dt1_item)
        # t_load_start = time.time()
        ten_2mm, fod, brain, step, trid, trii, mask = unload(*dt1_item)
        # t_load_done = time.time()
        # t_fwd_start = time.time()

        with ctx_fod:
            fod_pred = fod_cnn(fod.to(device))

            fod_step_pred, _, _, _, fod_fc = fod_rnn(fod_pred, trid.to(device), trii)
            fod_dot_loss, fod_cum_loss = rnn_criterion(
                fod_step_pred, step.to(device), mask.to(device)
            )
            t1_dot_loss = torch.Tensor([0]).detach()
            t1_cum_loss = torch.Tensor([0]).detach()
            fc_loss = torch.Tensor([0]).detach()
        # t_fwd_done = time.time()

        # t_t1_fwd_start = time.time()
        if stage == 1:
            with ctx_t1:
                if train and use_amp:
                    amp_status = (
                        "ENABLED" if torch.is_autocast_enabled() else "NOT ENABLED"
                    )
                    # print(f"[DEBUG] AMP for T1: {amp_status}")
                t1_pred = t1_cnn(ten_2mm.to(device))
                t1_step_pred, _, _, _, t1_fc = t1_rnn(t1_pred, trid.to(device), trii)
                t1_dot_loss, t1_cum_loss = rnn_criterion(
                    t1_step_pred, step.to(device), mask.to(device)
                )
                fc_loss = fc_criterion(t1_fc, fod_fc)

        loss = loss_fxn(
            fod_dot_loss,
            fod_cum_loss,
            t1_dot_loss,
            t1_cum_loss,
            fc_loss,
        )

        # t_t1_fwd_done = time.time()

        # t_bwd_start = time.time()
        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
        # t_bwd_done = time.time()

        epoch_loss += loss.item()
        epoch_fod_dot_loss += fod_dot_loss.item()
        epoch_fod_cum_loss += fod_cum_loss.item()
        epoch_t1_dot_loss += t1_dot_loss.item()
        epoch_t1_cum_loss += t1_cum_loss.item()
        epoch_fc_loss += fc_loss.item()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _ = gc.collect()

        # batch_time = time.time() - batch_start
        # print(f"[{label}] Batch {bidx}: \n",
        #       f"Load={t_load_done-t_load_start:.2f}s | \n",
        #       f"FWD={t_fwd_done-t_fwd_start:.2f}s | \n",
        #       f"T1_FWD={t_t1_fwd_done-t_t1_fwd_start:.2f}s | \n",
        #       f"BWD={t_bwd_done-t_bwd_start:.2f}s | \n",
        #       f"TOTAL={batch_time:.2f}s")

    epoch_loss /= epoch_iters
    epoch_fod_dot_loss /= epoch_iters
    epoch_fod_cum_loss /= epoch_iters
    epoch_t1_dot_loss /= epoch_iters
    epoch_t1_cum_loss /= epoch_iters
    epoch_fc_loss /= epoch_iters

    # epoch_time = time.time() - epoch_start
    # avg_batch_time = np.mean(batch_times)
    # print(f"[{label}]: avg batch: {avg_batch_time:.2f}s")

    return (
        epoch_loss,
        epoch_fod_dot_loss,
        epoch_fod_cum_loss,
        epoch_t1_dot_loss,
        epoch_t1_cum_loss,
        epoch_fc_loss,
    )


def dot2ang(dot):
    """Convert dot product loss to angle loss in degrees."""
    return 180 / np.pi * np.arccos(1 - dot)


def initialize_t1_rnn(t1_rnn, fod_rnn_weights):
    """Initialize T1 RNN weights from FOD RNN weights."""
    for weights_name in list(fod_rnn_weights.keys()):
        if "fc" in weights_name:
            del fod_rnn_weights[weights_name]
    t1_rnn.load_state_dict(fod_rnn_weights, strict=False)
    for param in (
        list(t1_rnn.rnn.parameters())
        + list(t1_rnn.azi.parameters())
        + list(t1_rnn.ele.parameters())
    ):
        param.requires_grad = False


def main():
    """Main training loop with distributed support."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=180))
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_distributed = True

        if rank == 0:
            print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
            print(f"Available CPU cores: {os.cpu_count()}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_distributed = False

    parser = configparser.ConfigParser()
    parser.read("train.v2.ini")
    config = parser["Training"]

    out_dir = Path(config["out_dir"])
    resume = config.getboolean("resume")
    start_epoch = config.getint("start_epoch")
    best_fod_epoch = config.getint("best_fod_epoch")
    best_t1_epoch = config.getint("best_t1_epoch")
    stage = config.getint("stage")

    stage_max_epochs = config.getint("stage_max_epochs")
    total_max_epochs = (2 - stage) * stage_max_epochs
    stage_num_epochs_no_change = config.getint("stage_num_epochs_no_change")

    val_tolerence = 0.02

    assert stage == 0 or stage == 1

    if start_epoch > 0:
        assert resume

    if stage == 1:
        assert best_fod_epoch < start_epoch

    fod_cnn_file = out_dir / "fod_cnn_{}.pt"
    fod_rnn_file = out_dir / "fod_rnn_{}.pt"
    t1_cnn_file = out_dir / "t1_cnn_{}.pt"
    t1_rnn_file = out_dir / "t1_rnn_{}.pt"

    fod_optimizer_file = out_dir / "fod_opt_{}.pt"
    fod_scheduler_file = out_dir / "fod_sch_{}.pt"
    t1_optimizer_file = out_dir / "t1_opt_{}.pt"
    t1_scheduler_file = out_dir / "t1_sch_{}.pt"

    train_dirs_file = config["train_dirs"]
    val_dirs_file = config["val_dirs"]
    derivatives_data_path = config["deriv_data_path"]
    cache_root = config["cache_root"]

    num_streamlines = 1000000
    batch_size = 1000
    num_batches = np.ceil(num_streamlines / batch_size).astype(int)

    with open(train_dirs_file, "r") as file:
        train_dirs = file.read().splitlines()

    with open(val_dirs_file, "r") as file:
        val_dirs = file.read().splitlines()

    train_dataset = DT1Dataset(
        train_dirs,
        num_batches,
        cache_root=cache_root + "_train",
        base_data_path=derivatives_data_path,
    )
    val_dataset = DT1Dataset(
        val_dirs,
        num_batches,
        cache_root=cache_root + "_val",
        base_data_path=derivatives_data_path,
    )

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=train_sampler,
            num_workers=3,
            drop_last=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler,
            num_workers=1,
            persistent_workers=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=1, num_workers=4, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)

    fod_cnn = DetCNNFake().to(device)
    fod_rnn = DetRNN(45, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2).to(
        device
    )
    t1_cnn = DetConvProj(123, 512, kernel_size=3).to(device)
    t1_rnn = DetRNN(512, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2).to(
        device
    )

    if is_distributed:
        fod_cnn = maybe_wrap_ddp(
            fod_cnn, device_ids=[local_rank], output_device=local_rank
        )
        fod_rnn = maybe_wrap_ddp(
            fod_rnn, device_ids=[local_rank], output_device=local_rank
        )
        t1_cnn = maybe_wrap_ddp(
            t1_cnn, device_ids=[local_rank], output_device=local_rank
        )
        t1_rnn = maybe_wrap_ddp(
            t1_rnn, device_ids=[local_rank], output_device=local_rank
        )

    epochs_up = 5  # Half of 10-epoch cycle
    step_size_up = num_batches * epochs_up  # 1000 * 5 = 5000

    opt0 = optim.AdamW(
        list(fod_cnn.parameters()) + list(fod_rnn.parameters()), lr=3e-4, eps=1e-6
    )
    scheduler0 = torch.optim.lr_scheduler.CyclicLR(
        opt0,
        base_lr=1e-4,
        max_lr=1e-3,
        step_size_up=step_size_up,
        mode="triangular2",
        scale_mode="cycle",
        cycle_momentum=False,
    )

    opt1 = optim.AdamW(
        list(t1_cnn.parameters()) + list(unwrap_model(t1_rnn).fc.parameters()),
        lr=3e-4,
        eps=1e-6,
    )
    scheduler1 = torch.optim.lr_scheduler.CyclicLR(
        opt1,
        base_lr=1e-4,
        max_lr=1e-3,
        step_size_up=step_size_up,
        mode="triangular2",
        scale_mode="cycle",
        cycle_momentum=False,
    )

    opts = [opt0, opt1]
    scheds = [scheduler0, scheduler1]

    if resume:
        if stage == 0:
            opt0.load_state_dict(
                torch.load(
                    str(fod_optimizer_file).format(start_epoch - 1),
                    map_location=device,
                    weights_only=False,
                )
            )
            scheduler0.load_state_dict(
                torch.load(
                    str(fod_scheduler_file).format(start_epoch - 1),
                    map_location=device,
                    weights_only=False,
                )
            )
            unwrap_model(fod_cnn).load_state_dict(
                torch.load(
                    str(fod_cnn_file).format(start_epoch - 1),
                    map_location=device,
                    weights_only=False,
                )
            )
            fod_rnn_weights = torch.load(
                str(fod_rnn_file).format(start_epoch - 1),
                map_location=device,
                weights_only=False,
            )
            unwrap_model(fod_rnn).load_state_dict(fod_rnn_weights)
        else:
            unwrap_model(fod_cnn).load_state_dict(
                torch.load(
                    str(fod_cnn_file).format(best_fod_epoch),
                    map_location=device,
                    weights_only=True,
                )
            )
            fod_rnn_weights = torch.load(
                str(fod_rnn_file).format(best_fod_epoch),
                map_location=device,
                weights_only=True,
            )
            unwrap_model(fod_rnn).load_state_dict(fod_rnn_weights)

            # optimizer_path = str(t1_optimizer_file).format(start_epoch - 1)
            # scheduler_path = str(t1_scheduler_file).format(start_epoch - 1)
            # previous_rnn_path = str(t1_rnn_file).format(start_epoch - 1)
            optimizer_path = str(t1_optimizer_file).format(best_t1_epoch)
            scheduler_path = str(t1_scheduler_file).format(best_t1_epoch)
            previous_rnn_path = str(t1_rnn_file).format(best_t1_epoch)

            if (
                Path(optimizer_path).exists()
                and Path(previous_rnn_path).exists()
                and Path(scheduler_path)
            ):
                opt1.load_state_dict(
                    torch.load(optimizer_path, map_location=device, weights_only=False)
                )
                scheduler1.load_state_dict(
                    torch.load(scheduler_path, map_location=device, weights_only=False)
                )
                unwrap_model(t1_rnn).load_state_dict(
                    torch.load(
                        previous_rnn_path, map_location=device, weights_only=False
                    )
                )
            else:
                initialize_t1_rnn(unwrap_model(t1_rnn), fod_rnn_weights)

    fc_criterion = DetFCLoss()
    rnn_criterion = DetStepLoss()

    best_loss = math.inf
    best_epoch = "-"
    if rank == 0:
        run = Run(experiment="det_rnn_training", repo=out_dir)
        run["hparams"] = {
            "stage": stage,
            "batch_size": batch_size,
            "out_dir": str(out_dir),
            "use_amp": True,
            "distributed": is_distributed,
            "world_size": world_size,
            "save_visualizations": True,
            "viz_interval": 10,  # Save every 10 epochs
        }
    else:
        run = None

    scaler = GradScaler()
    use_amp = True

    stage_last_epoch = start_epoch + stage_num_epochs_no_change - 1

    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    epoch_bar = tqdm(
        range(start_epoch, total_max_epochs),
        leave=True,
        disable=(rank != 0),
    )

    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    for epoch in epoch_bar:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_bar.set_description(
            f"Stage: {stage} | Best Epoch: {best_epoch} | Current Epoch"
        )

        (
            train_loss,
            train_fod_dot_loss,
            train_fod_cum_loss,
            train_t1_dot_loss,
            train_t1_cum_loss,
            train_fc_loss,
        ) = epoch_losses(
            train_loader,
            fod_cnn,
            t1_cnn,
            fod_rnn,
            t1_rnn,
            fc_criterion,
            rnn_criterion,
            device,
            opts[stage],
            scheds[stage],
            "Train",
            stage,
            scaler,
            use_amp,
            rank,
        )
        (
            val_loss,
            val_fod_dot_loss,
            val_fod_cum_loss,
            val_t1_dot_loss,
            val_t1_cum_loss,
            val_fc_loss,
        ) = epoch_losses(
            val_loader,
            fod_cnn,
            t1_cnn,
            fod_rnn,
            t1_rnn,
            fc_criterion,
            rnn_criterion,
            device,
            opts[stage],
            scheds[stage],
            "Validation",
            stage,
            scaler,
            use_amp,
            rank,
        )
        # Save and track prediction visualizations (rank 0 only, every 10 epochs)
        if False: #  rank == 0 and epoch % 8 == 0:
            viz_dir = out_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            viz_path = viz_dir / f"predictions_epoch_{epoch}.png"

            # Get a single validation batch for visualization
            with torch.no_grad():
                val_iter = iter(val_loader)
                val_item = next(val_iter)
                ten_2mm, fod, brain, step, trid, trii, mask = unload(*val_item)

                fod_pred = fod_cnn(fod.to(device))
                fod_step_pred, _, _, _, fod_fc = fod_rnn(
                    fod_pred, trid.to(device), trii
                )

                t1_pred = None
                t1_step_pred = None
                if stage == 1:
                    t1_pred = t1_cnn(ten_2mm.to(device))
                    t1_step_pred, _, _, _, t1_fc = t1_rnn(
                        t1_pred, trid.to(device), trii
                    )

                fod_pred_cpu = fod_pred.detach().cpu()
                fod_step_pred_cpu = fod_step_pred.detach().cpu()
                step_cpu = step.detach().cpu()
                mask_cpu = mask.detach().cpu()
                t1_pred_cpu = t1_pred.detach().cpu() if t1_pred is not None else None
                t1_step_pred_cpu = (
                    t1_step_pred.detach().cpu() if t1_step_pred is not None else None
                )

                # Clear GPU cache before visualization
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                visualize_predictions(
                    fod_pred_cpu,
                    fod_step_pred_cpu,
                    step_cpu,
                    mask_cpu,
                    stage=stage,
                    t1_pred=t1_pred_cpu,
                    t1_step_pred=t1_step_pred_cpu,
                    save_path=viz_path,
                    epoch=epoch,
                )

                # Clear CPU memory after visualization
                del fod_pred_cpu, fod_step_pred_cpu, step_cpu, mask_cpu
                if t1_pred_cpu is not None:
                    del t1_pred_cpu
                if t1_step_pred_cpu is not None:
                    del t1_step_pred_cpu
                gc.collect()

                # Track image with Aim
                run.track(
                    AimImage(str(viz_path)),
                    name="Prediction Visualization",
                    step=epoch,
                    context={"subset": "validation"},
                )

        train_fod_ang_loss = dot2ang(train_fod_dot_loss)
        val_fod_ang_loss = dot2ang(val_fod_dot_loss)
        train_t1_ang_loss = dot2ang(train_t1_dot_loss)
        val_t1_ang_loss = dot2ang(val_t1_dot_loss)

        if rank == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()


            run.track(
                train_loss, name="Total Loss", step=epoch, context={"subset": "train"}
            )
            run.track(
                val_loss,
                name="Total Loss",
                step=epoch,
                context={"subset": "validation"},
            )
            run.track(
                train_fod_dot_loss,
                name="FOD Dot Loss",
                step=epoch,
                context={"subset": "train"},
            )
            run.track(
                val_fod_dot_loss,
                name="FOD Dot Loss",
                step=epoch,
                context={"subset": "validation"},
            )
            run.track(
                train_fod_ang_loss,
                name="FOD Angle Loss",
                step=epoch,
                context={"subset": "train"},
            )
            run.track(
                val_fod_ang_loss,
                name="FOD Angle Loss",
                step=epoch,
                context={"subset": "validation"},
            )
            run.track(
                train_fod_cum_loss,
                name="FOD Cum Loss",
                step=epoch,
                context={"subset": "train"},
            )
            run.track(
                val_fod_cum_loss,
                name="FOD Cum Loss",
                step=epoch,
                context={"subset": "validation"},
            )
            run.track(
                train_t1_dot_loss,
                name="T1 Dot Loss",
                step=epoch,
                context={"subset": "train"},
            )
            run.track(
                val_t1_dot_loss,
                name="T1 Dot Loss",
                step=epoch,
                context={"subset": "validation"},
            )
            run.track(
                train_t1_ang_loss,
                name="T1 Angle Loss",
                step=epoch,
                context={"subset": "train"},
            )
            run.track(
                val_t1_ang_loss,
                name="T1 Angle Loss",
                step=epoch,
                context={"subset": "validation"},
            )
            run.track(
                train_t1_cum_loss,
                name="T1 Cum Loss",
                step=epoch,
                context={"subset": "train"},
            )
            run.track(
                val_t1_cum_loss,
                name="T1 Cum Loss",
                step=epoch,
                context={"subset": "validation"},
            )
            run.track(
                train_fc_loss, name="FC Loss", step=epoch, context={"subset": "train"}
            )
            run.track(
                val_fc_loss,
                name="FC Loss",
                step=epoch,
                context={"subset": "validation"},
            )

        if epoch % 1 == 0 and rank == 0:
            if stage == 0:
                torch.save(
                    unwrap_model(fod_cnn).state_dict(), str(fod_cnn_file).format(epoch)
                )
                torch.save(
                    unwrap_model(fod_rnn).state_dict(), str(fod_rnn_file).format(epoch)
                )
                torch.save(opt0.state_dict(), str(fod_optimizer_file).format(epoch))
                torch.save(
                    scheduler0.state_dict(), str(fod_scheduler_file).format(epoch)
                )
            else:
                torch.save(
                    unwrap_model(t1_cnn).state_dict(), str(t1_cnn_file).format(epoch)
                )
                torch.save(
                    unwrap_model(t1_rnn).state_dict(), str(t1_rnn_file).format(epoch)
                )
                torch.save(opt1.state_dict(), str(t1_optimizer_file).format(epoch))
                torch.save(
                    scheduler1.state_dict(), str(t1_scheduler_file).format(epoch)
                )

        if val_loss - val_tolerence < best_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                if rank == 0:
                    if stage == 0:
                        torch.save(
                            unwrap_model(fod_cnn).state_dict(),
                            str(fod_cnn_file).format("best"),
                        )
                        torch.save(
                            unwrap_model(fod_rnn).state_dict(),
                            str(fod_rnn_file).format("best"),
                        )
                        torch.save(
                            opt0.state_dict(), str(fod_optimizer_file).format("best")
                        )
                        torch.save(
                            scheduler0.state_dict(),
                            str(fod_scheduler_file).format("best"),
                        )
                    else:
                        torch.save(
                            unwrap_model(t1_cnn).state_dict(),
                            str(t1_cnn_file).format("best"),
                        )
                        torch.save(
                            unwrap_model(t1_rnn).state_dict(),
                            str(t1_rnn_file).format("best"),
                        )
                        torch.save(
                            opt1.state_dict(), str(t1_optimizer_file).format("best")
                        )
                        torch.save(
                            scheduler1.state_dict(),
                            str(t1_scheduler_file).format("best"),
                        )
                stage_last_epoch = np.min(
                    (
                        epoch + stage_num_epochs_no_change - 1,
                        stage_max_epochs - 1,
                        total_max_epochs - 1,
                    )
                )

        if epoch == stage_last_epoch:
            if rank == 0:
                print(
                    f"\nStage: {stage} | Best Epoch: {best_epoch} | Last Epoch: {epoch}"
                )
            stage += 1
            if stage == 1:
                unwrap_model(fod_cnn).load_state_dict(
                    torch.load(
                        str(fod_cnn_file).format("best"),
                        map_location=device,
                        weights_only=True,
                    )
                )
                fod_rnn_weights = torch.load(
                    str(fod_rnn_file).format("best"),
                    map_location=device,
                    weights_only=True,
                )
                unwrap_model(fod_rnn).load_state_dict(fod_rnn_weights)
                initialize_t1_rnn(unwrap_model(t1_rnn), fod_rnn_weights)
            if stage > 1:
                break
            stage_max_epochs = epoch + stage_max_epochs + 1
            best_loss = math.inf
            best_epoch = "-"

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
