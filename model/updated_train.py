import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

# Aim import for experiment tracking
from aim import Run

from tqdm import tqdm
import numpy as np
import math
import os
import configparser

from utils import default_context
from data import DT1Dataset, unload
from modules import (
    DetStepLoss,
    DetRNN,
    # DetFODLoss,
    DetFCLoss,
    DetCNNFake,
    DetConvProj,
    # DetConvProjMulti,
)


# Helper Functions

def maybe_wrap_ddp(model, device_ids, output_device):
    """Wrap model with DDP only if it has parameters requiring gradients."""
    if any(p.requires_grad for p in model.parameters()):
        return DDP(model, device_ids=device_ids, output_device=output_device, find_unused_parameters=True)
    return model


def unwrap_model(model):
    """Return the underlying module, handling both DDP-wrapped and plain models."""
    return model.module if isinstance(model, DDP) else model


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
    label,
    stage,
    scaler,
    use_amp,
    rank,
):
    train = "train" in label.lower()
    epoch_loss = 0
    epoch_fod_dot_loss = 0
    epoch_fod_cum_loss = 0
    epoch_t1_dot_loss = 0
    epoch_t1_cum_loss = 0
    # epoch_fod_loss = 0
    epoch_fc_loss = 0
    # epoch_mid_loss = 0

    epoch_iters = len(data_loader)

    show_batch_progress = (rank == 0)
    
    for _, dt1_item in tqdm(
        enumerate(data_loader), 
        total=epoch_iters, 
        desc=label, 
        leave=False,  # Don't leave batch bars after completion
        disable=not show_batch_progress
    ):
        # ten, step, trid, trii, mask, tdi = unload(*dt1_item)
        # ten, fod, brain, step, trid, trii, mask = unload(*dt1_item)
        # for convprojmulti
        # ten_1mm, ten_2mm, fod, brain, step, trid, trii, mask = unload(*dt1_item)
        ten_2mm, fod, brain, step, trid, trii, mask = unload(*dt1_item)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if stage == 0:
                fod_cnn.train()
                fod_rnn.train()
            else:  # stage == 1
                fod_cnn.eval()
                fod_rnn.eval()
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
                # fod_loss,
                fc_loss,
                # mid_loss,
            ):
                return fod_dot_loss

        else:  # stage == 1

            def loss_fxn(
                fod_dot_loss,
                fod_cum_loss,
                t1_dot_loss,
                t1_cum_loss,
                # fod_loss,
                fc_loss,
                # mid_loss,
            ):
                return fc_loss + t1_dot_loss  # + mid_loss

        ## Context management for forward passes with AMP support
        # FOD models forward pass context
        if train and use_amp:
            ctx_fod = autocast(device_type='cuda')
        elif stage == 1 or (stage == 0 and not train):
            ctx_fod = torch.no_grad()
        else:
            ctx_fod = default_context()
        
        with ctx_fod:
            fod_pred = fod_cnn(fod.to(device))

            fod_step_pred, _, _, _, fod_fc = fod_rnn(fod_pred, trid.to(device), trii)
            fod_dot_loss, fod_cum_loss = rnn_criterion(
                fod_step_pred, step.to(device), mask.to(device)
            )
            t1_dot_loss = torch.Tensor([0]).detach()
            t1_cum_loss = torch.Tensor([0]).detach()
            fc_loss = torch.Tensor([0]).detach()
            
        if stage == 1:
            # T1 models forward pass context
            if train and use_amp:
                ctx_t1 = autocast(device_type='cuda')
            elif train:
                ctx_t1 = default_context()
            else:
                ctx_t1 = torch.no_grad()
                
            with ctx_t1:
                # t1_pred = t1_cnn(ten_1mm.to(device), ten_2mm.to(device))
                t1_pred = t1_cnn(ten_2mm.to(device))
                t1_step_pred, _, _, _, t1_fc = t1_rnn(t1_pred, trid.to(device), trii)
                t1_dot_loss, t1_cum_loss = rnn_criterion(
                    t1_step_pred, step.to(device), mask.to(device)
                )
                # cnn_criterion(t1_pred, fod_pred, brain.to(device))
                # fod_loss = torch.Tensor([0])
                fc_loss = fc_criterion(t1_fc, fod_fc)
                # fc_loss, _ = fc_criterion(t1_step_pred, fod_step_pred, mask.to(device))
                # * trying "end loss" to impose contrastive at output instead of in the middle
                # mid_loss = torch.Tensor([0])  # mid_criterion(t1_fc, fod_fc)

        loss = loss_fxn(
            fod_dot_loss,
            fod_cum_loss,
            t1_dot_loss,
            t1_cum_loss,
            # fod_loss,
            fc_loss,
            # mid_loss,
        )

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item()
        epoch_fod_dot_loss += fod_dot_loss.item()
        epoch_fod_cum_loss += fod_cum_loss.item()
        epoch_t1_dot_loss += t1_dot_loss.item()
        epoch_t1_cum_loss += t1_cum_loss.item()
        # epoch_fod_loss += fod_loss.item()
        epoch_fc_loss += fc_loss.item()
        # epoch_mid_loss += mid_loss.item()

    epoch_loss /= epoch_iters
    epoch_fod_dot_loss /= epoch_iters
    epoch_fod_cum_loss /= epoch_iters
    epoch_t1_dot_loss /= epoch_iters
    epoch_t1_cum_loss /= epoch_iters
    # epoch_fod_loss /= epoch_iters
    epoch_fc_loss /= epoch_iters
    # epoch_mid_loss /= epoch_iters

    return (
        epoch_loss,
        epoch_fod_dot_loss,
        epoch_fod_cum_loss,
        epoch_t1_dot_loss,
        epoch_t1_cum_loss,
        # epoch_fod_loss,
        epoch_fc_loss,
        # epoch_mid_loss,
    )


def dot2ang(dot):
    return 180 / np.pi * np.arccos(1 - dot)


def initialize_t1_rnn(t1_rnn, fod_rnn_weights):
    for weights_name in list(fod_rnn_weights.keys()):
        if "fc" in weights_name:
            del fod_rnn_weights[weights_name]
    # load weights from fod_rnn!
    t1_rnn.load_state_dict(fod_rnn_weights, strict=False)
    for param in (
        list(t1_rnn.rnn.parameters())
        + list(t1_rnn.azi.parameters())
        + list(t1_rnn.ele.parameters())
    ):
        param.requires_grad = False


def main():
    # Check if running in distributed mode via environment variables
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Increase timeout to 30 minutes to prevent premature NCCL timeouts
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_distributed = True
        
        # Debug: Print OMP_NUM_THREADS setting
        if rank == 0:
            print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
            print(f"Available CPU cores: {os.cpu_count()}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        is_distributed = False
    
    parser = configparser.ConfigParser()
    parser.read("train.ini")
    config = parser["Training"]

    # Inputs
    out_dir = config["out_dir"]
    resume = config.getboolean("resume")
    start_epoch = config.getint("start_epoch")
    best_fod_epoch = config.getint("best_fod_epoch")
    stage = config.getint("stage")  # 0 or 1

    stage_max_epochs = config.getint("stage_max_epochs")
    total_max_epochs = (2 - stage) * stage_max_epochs
    stage_num_epochs_no_change = config.getint("stage_num_epochs_no_change")

    # there is randomness in the validation process
    # i.e. each time, difference streamlines in the validation subject are used
    # Thus, I add a tolerence here. If the loss are close, the model parameters are saved
    val_tolerence = 0.02

    assert stage == 0 or stage == 1

    if start_epoch > 0:
        assert resume

    if stage == 1:
        assert best_fod_epoch < start_epoch

    fod_cnn_file = os.path.join(out_dir, "fod_cnn_{}.pt")
    fod_rnn_file = os.path.join(out_dir, "fod_rnn_{}.pt")
    t1_cnn_file = os.path.join(out_dir, "t1_cnn_{}.pt")
    t1_rnn_file = os.path.join(out_dir, "t1_rnn_{}.pt")

    fod_optimizer_file = os.path.join(out_dir, "fod_opt_{}.pt")
    t1_optimizer_file = os.path.join(out_dir, "t1_opt_{}.pt")

    train_dirs_file = config["train_dirs"]
    val_dirs_file = config["val_dirs"]
    # test_dirs_file = config["test_dirs"]

    num_streamlines = 1000000
    batch_size = 1000
    num_batches = np.ceil(num_streamlines / batch_size).astype(int)

    # Prepare data
    with open(train_dirs_file, "r") as file:
        train_dirs = file.read().splitlines()

    with open(val_dirs_file, "r") as file:
        val_dirs = file.read().splitlines()

    # with open(test_dirs_file, "r") as test_dirs_fobj:
    #     test_dirs = test_dirs_fobj.read().splitlines()

    train_dataset = DT1Dataset(train_dirs, num_batches)
    val_dataset = DT1Dataset(val_dirs, num_batches)
    # test_dataset = DT1Dataset(test_dirs, num_batches)

    # Use DistributedSampler for DDP
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        # drop_last=True ensures all ranks have the same number of batches to prevent NCCL timeout
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    # Train
    # device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")

    # Initialize models
    fod_cnn = DetCNNFake().to(device)
    fod_rnn = DetRNN(45, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2).to(device)
    t1_cnn = DetConvProj(123, 512, kernel_size=7).to(device)
    t1_rnn = DetRNN(512, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2).to(device)

    # Wrap with DDP only if distributed AND model has trainable parameters
    if is_distributed:
        fod_cnn = maybe_wrap_ddp(fod_cnn, device_ids=[local_rank], output_device=local_rank)
        fod_rnn = maybe_wrap_ddp(fod_rnn, device_ids=[local_rank], output_device=local_rank)
        t1_cnn = maybe_wrap_ddp(t1_cnn, device_ids=[local_rank], output_device=local_rank)
        t1_rnn = maybe_wrap_ddp(t1_rnn, device_ids=[local_rank], output_device=local_rank)

    # * The original learning rate is 1e-3
    opt0 = optim.AdamW(list(fod_cnn.parameters()) + list(fod_rnn.parameters()), lr=1e-3)
    opt1 = optim.AdamW(list(t1_cnn.parameters()) + list(unwrap_model(t1_rnn).fc.parameters()), lr=1e-3)
    opts = [opt0, opt1]

    if resume:
        if stage == 0:
            opt0.load_state_dict(
                torch.load(
                    fod_optimizer_file.format(start_epoch - 1), map_location=device
                )
            )
            unwrap_model(fod_cnn).load_state_dict(
                torch.load(fod_cnn_file.format(start_epoch - 1), map_location=device)
            )
            fod_rnn_weights = torch.load(
                fod_rnn_file.format(start_epoch - 1), map_location=device
            )
            unwrap_model(fod_rnn).load_state_dict(fod_rnn_weights)
        else:  # stage == 1
            unwrap_model(fod_cnn).load_state_dict(
                torch.load(fod_cnn_file.format(best_fod_epoch), map_location=device)
            )
            fod_rnn_weights = torch.load(
                fod_rnn_file.format(best_fod_epoch), map_location=device
            )
            unwrap_model(fod_rnn).load_state_dict(fod_rnn_weights)

            optimizer_path = t1_optimizer_file.format(start_epoch - 1)
            previous_rnn_path = t1_rnn_file.format(start_epoch - 1)

            if os.path.exists(optimizer_path) and os.path.exists(previous_rnn_path):
                opt1.load_state_dict(torch.load(optimizer_path, map_location=device))
                unwrap_model(t1_rnn).load_state_dict(
                    torch.load(previous_rnn_path, map_location=device)
                )
            else:
                initialize_t1_rnn(unwrap_model(t1_rnn), fod_rnn_weights)

    # cnn_criterion = DetFODLoss()
    fc_criterion = DetFCLoss()
    # trying "end loss" to impose contrastive at output instead of in the middle
    # fc_criterion  = DetStepLoss()
    rnn_criterion = DetStepLoss()
    # mid_criterion = torch.nn.L1Loss()

    # sch0 = MultiStepLR(opt0, [10, 100], gamma=0.1) # start at 1e-3
    # sch1 = MultiStepLR(opt1, [10, 100], gamma=0.1) # start at 1e-3
    # schs = [sch0, sch1]

    best_loss = math.inf
    best_epoch = "-"

    # Initialize Aim run for experiment tracking (only on rank 0)
    if rank == 0:
        run = Run(experiment="det_rnn_training", repo=out_dir)
        run['hparams'] = {
            'stage': stage,
            'batch_size': batch_size,
            'out_dir': out_dir,
            'use_amp': True,
            'distributed': is_distributed,
            'world_size': world_size,
        }
    else:
        run = None

    # Initialize GradScaler for AMP
    scaler = GradScaler()
    use_amp = True  # Set to False to disable automatic mixed precision

    # Initialize stage_last_epoch to avoid undefined variable error
    stage_last_epoch = start_epoch + stage_num_epochs_no_change - 1

    epoch_bar = tqdm(range(start_epoch, total_max_epochs), leave=True, disable=(rank != 0))

    for epoch in epoch_bar:
        # Set epoch for DistributedSampler to ensure proper shuffling
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        epoch_bar.set_description(
            "Stage: {} | Best Epoch: {} | Current Epoch".format(stage, best_epoch)
        )

        (
            train_loss,
            train_fod_dot_loss,
            train_fod_cum_loss,
            train_t1_dot_loss,
            train_t1_cum_loss,
            # train_fod_loss,
            train_fc_loss,
            # train_mid_loss,
        ) = epoch_losses(
            train_loader,
            fod_cnn,
            t1_cnn,
            fod_rnn,
            t1_rnn,
            # cnn_criterion,
            # mid_criterion,
            fc_criterion,
            rnn_criterion,
            device,
            opts[stage],
            "Train",
            stage,
            scaler,
            use_amp,
            rank
        )
        (
            val_loss,
            val_fod_dot_loss,
            val_fod_cum_loss,
            val_t1_dot_loss,
            val_t1_cum_loss,
            # val_fod_loss,
            val_fc_loss,
            # val_mid_loss,
        ) = epoch_losses(
            val_loader,
            fod_cnn,
            t1_cnn,
            fod_rnn,
            t1_rnn,
            # cnn_criterion,
            # mid_criterion,
            fc_criterion,
            rnn_criterion,
            device,
            opts[stage],
            "Validation",
            stage,
            scaler,
            use_amp,
            rank
        )

        train_fod_ang_loss = dot2ang(train_fod_dot_loss)
        val_fod_ang_loss = dot2ang(val_fod_dot_loss)
        train_t1_ang_loss = dot2ang(train_t1_dot_loss)
        val_t1_ang_loss = dot2ang(val_t1_dot_loss)

        # Log metrics with Aim (only on rank 0 to avoid duplicates)
        if rank == 0:
            run.track(train_loss, name='Total Loss', step=epoch, context={'subset': 'train'})
            run.track(val_loss, name='Total Loss', step=epoch, context={'subset': 'validation'})
            run.track(train_fod_dot_loss, name='FOD Dot Loss', step=epoch, context={'subset': 'train'})
            run.track(val_fod_dot_loss, name='FOD Dot Loss', step=epoch, context={'subset': 'validation'})
            run.track(train_fod_ang_loss, name='FOD Angle Loss', step=epoch, context={'subset': 'train'})
            run.track(val_fod_ang_loss, name='FOD Angle Loss', step=epoch, context={'subset': 'validation'})
            run.track(train_fod_cum_loss, name='FOD Cum Loss', step=epoch, context={'subset': 'train'})
            run.track(val_fod_cum_loss, name='FOD Cum Loss', step=epoch, context={'subset': 'validation'})
            run.track(train_t1_dot_loss, name='T1 Dot Loss', step=epoch, context={'subset': 'train'})
            run.track(val_t1_dot_loss, name='T1 Dot Loss', step=epoch, context={'subset': 'validation'})
            run.track(train_t1_ang_loss, name='T1 Angle Loss', step=epoch, context={'subset': 'train'})
            run.track(val_t1_ang_loss, name='T1 Angle Loss', step=epoch, context={'subset': 'validation'})
            run.track(train_t1_cum_loss, name='T1 Cum Loss', step=epoch, context={'subset': 'train'})
            run.track(val_t1_cum_loss, name='T1 Cum Loss', step=epoch, context={'subset': 'validation'})
            run.track(train_fc_loss, name='FC Loss', step=epoch, context={'subset': 'train'})
            run.track(val_fc_loss, name='FC Loss', step=epoch, context={'subset': 'validation'})
            # writer.add_scalars(
            #     "FOD Loss", {"Train": train_fod_loss, "Validation": val_fod_loss}, epoch
            # )
            # writer.add_scalars(
            #     "Mid Loss", {"Train": train_mid_loss, "Validation": val_mid_loss}, epoch
            # )

        if epoch % 50 == 0 and rank == 0:
            if stage == 0:
                torch.save(unwrap_model(fod_cnn).state_dict(), fod_cnn_file.format(epoch))
                torch.save(unwrap_model(fod_rnn).state_dict(), fod_rnn_file.format(epoch))
                torch.save(opt0.state_dict(), fod_optimizer_file.format(epoch))
            else:
                torch.save(unwrap_model(t1_cnn).state_dict(), t1_cnn_file.format(epoch))
                torch.save(unwrap_model(t1_rnn).state_dict(), t1_rnn_file.format(epoch))
                torch.save(opt1.state_dict(), t1_optimizer_file.format(epoch))

        if val_loss - val_tolerence < best_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                # always computed in first epoch of new stage since best_loss = inf
                # -1 to account for 0 indexing
                if rank == 0:  # Only save from rank 0
                    if stage == 0:
                        torch.save(unwrap_model(fod_cnn).state_dict(), fod_cnn_file.format("best"))
                        torch.save(unwrap_model(fod_rnn).state_dict(), fod_rnn_file.format("best"))
                        torch.save(opt0.state_dict(), fod_optimizer_file.format("best"))
                    else:
                        torch.save(unwrap_model(t1_cnn).state_dict(), t1_cnn_file.format("best"))
                        torch.save(unwrap_model(t1_rnn).state_dict(), t1_rnn_file.format("best"))
                        torch.save(opt1.state_dict(), t1_optimizer_file.format("best"))
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
                    "\nStage: {} | Best Epoch: {} | Last Epoch: {}".format(
                        stage, best_epoch, epoch
                    )
                )
            stage += 1
            if stage == 1:
                unwrap_model(fod_cnn).load_state_dict(
                    torch.load(fod_cnn_file.format("best"), map_location=device)
                )
                fod_rnn_weights = torch.load(
                    fod_rnn_file.format("best"), map_location=device
                )
                unwrap_model(fod_rnn).load_state_dict(fod_rnn_weights)
                initialize_t1_rnn(unwrap_model(t1_rnn), fod_rnn_weights)
            if stage > 1:
                break
            # * +1 to account for -1 above for zero indexing
            stage_max_epochs = epoch + stage_max_epochs + 1
            best_loss = math.inf
            best_epoch = "-"
    
    # Cleanup distributed training only if initialized
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
