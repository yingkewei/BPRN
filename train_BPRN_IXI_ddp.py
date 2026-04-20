# train_BPRN_IXI_ddp.py
# -*- coding: utf-8 -*-
"""
DDP training script dedicated to IXI for BPRN_model.BPRN.

Usage example:
  torchrun --nproc_per_node=2 RDP-main/train_BPRN_IXI_ddp.py --batch_size 6 --use_amp

Notes:
- batch_size is PER-GPU batch size in DDP.
- global batch size = batch_size * world_size.
"""

import os
import time
import glob
import math
from datetime import datetime, timedelta
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torchvision import transforms as tv_transforms
import torch.distributed as dist

import losses
import utils
from data import datasets, trans
from BPRN_model import BPRN


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def is_main_process():
    return get_rank() == 0


def setup_ddp_from_env():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # 增加超时时间到2小时，避免因为验证阶段计算慢导致误杀
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=7200)
    )
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    if is_dist():
        # 移除 barrier，避免异常退出时卡住
        dist.destroy_process_group()


class TrainingMonitor:
    def __init__(self, save_dir, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.log_fp = open(os.path.join(save_dir, "training_log.txt"), "a")
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        if not self.enabled:
            return
        self.epoch_start_time = None
        self.train_loss_sum = 0.0
        self.train_ncc_sum = 0.0
        self.train_reg_sum = 0.0
        self.train_count = 0

    def start_epoch(self, epoch):
        if not self.enabled:
            return
        self.reset_epoch_stats()
        self.epoch_start_time = time.time()
        self.epoch = epoch
        msg = f"\n=== Epoch {epoch} start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
        print(msg)
        self.log_fp.write(msg + "\n")
        self.log_fp.flush()

    def update_train_batch(self, loss, ncc, reg, batch_idx, total_batches, lr):
        if not self.enabled:
            return
        self.train_loss_sum += loss
        self.train_ncc_sum += ncc
        self.train_reg_sum += reg
        self.train_count += 1

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            avg_loss = self.train_loss_sum / max(1, self.train_count)
            avg_ncc = self.train_ncc_sum / max(1, self.train_count)
            avg_reg = self.train_reg_sum / max(1, self.train_count)
            lr_str = f"{lr:.6e}" if lr is not None else "N/A"
            msg = (f"  Batch {batch_idx+1:4d}/{total_batches:4d} | "
                   f"Loss={avg_loss:.6f} | NCC={avg_ncc:.6f} | Reg={avg_reg:.6f} | LR={lr_str}")
            print(msg)
            self.log_fp.write(msg + "\n")
            self.log_fp.flush()

    def end_epoch(self, val_metrics):
        if not self.enabled:
            return

        epoch_time = time.time() - self.epoch_start_time
        assd_str = "N/A" if val_metrics['val_assd'] == -1.0 else (
            "inf" if math.isinf(val_metrics['val_assd']) else f"{val_metrics['val_assd']:.4f}"
        )
        hd95_str = "N/A" if val_metrics['val_hd95'] == -1.0 else (
            "inf" if math.isinf(val_metrics['val_hd95']) else f"{val_metrics['val_hd95']:.4f}"
        )

        msg = (f"\nEpoch {self.epoch} finished in {epoch_time:.1f}s | "
               f"Val Loss={val_metrics['val_loss']:.6f} | "
               f"Val NCC={val_metrics['val_ncc']:.6f} | Val Reg={val_metrics['val_reg']:.6f} | "
               f"Val Dice={val_metrics['val_dice']:.4f} | "
               f"ASSD={assd_str} | HD95={hd95_str}")
        print(msg)
        self.log_fp.write(msg + "\n")
        self.log_fp.flush()

    def close(self):
        if self.enabled and hasattr(self, 'log_fp'):
            self.log_fp.close()


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, min_lr=1e-6):
    warmup = max(2, max_epoch // 20)
    if epoch < warmup:
        lr = init_lr * (0.5 + 0.5 * (epoch + 1) / warmup)
    else:
        progress = (epoch - warmup) / max(1, (max_epoch - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (init_lr - min_lr) * cosine

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(args):
    local_rank, rank, world_size = setup_ddp_from_env()
    device = torch.device(f"cuda:{local_rank}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main_process() else None
    if is_main_process():
        log_dir = os.path.join("logs", f"train_BPRN_IXI_ddp_{timestamp}")
        ckpt_dir = os.path.join("checkpoints", f"train_BPRN_IXI_ddp_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"DDP initialized | world_size={world_size} | rank={rank} | local_rank={local_rank}")
        print(f"IXI DDP on inshape={args.inshape}, per_gpu_batch={args.batch_size}, global_batch={args.batch_size * world_size}")
    else:
        log_dir, ckpt_dir = None, None

    monitor = TrainingMonitor(log_dir, enabled=is_main_process())

    if args.data_root is not None:
        data_root = args.data_root
    else:
        data_root = "./affineIXI_data" if args.use_affine_data else "./rawIXI_data"

    train_dir = args.train_dir or os.path.join(data_root, "Train")
    val_dir = args.val_dir or os.path.join(data_root, "Test")

    if is_main_process():
        print(f"Using IXI data root: {data_root}")
        print(f"Train dir: {train_dir}")
        print(f"Val dir: {val_dir}")

    voi_labels = utils.get_voi_labels('IXI')
    surface_voxelspacing = (2.0, 2.0, 2.0)

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.pkl")))

    if len(train_files) == 0:
        raise RuntimeError(f"No training files found in {train_dir}")
    if len(val_files) == 0 and is_main_process():
        print(f"Warning: no validation files found in {val_dir}")

    train_aug_list = []
    if args.aug_flip:
        train_aug_list.append(trans.RandomFlip())
    if args.aug_affine:
        train_aug_list.append(
            trans.RandomAffine3D(
                degrees=args.aug_affine_degrees,
                translate=args.aug_affine_translate,
                scale=args.aug_affine_scale,
                mode=args.aug_affine_mode,
                order=1,
                cval=0.0,
            )
        )
    if args.aug_gamma:
        train_aug_list.append(
            trans.RandomGamma(
                gamma_range=(args.aug_gamma_min, args.aug_gamma_max),
                same_on_pair=args.aug_gamma_same_on_pair,
            )
        )
    if args.aug_noise:
        train_aug_list.append(
            trans.RandomNoise(
                sigma_range=(args.aug_noise_min, args.aug_noise_max),
                same_on_pair=args.aug_noise_same_on_pair,
            )
        )

    train_trans = tv_transforms.Compose(
        train_aug_list + [trans.NumpyType((np.float32, np.float32))]
    )
    val_trans = tv_transforms.Compose([
        trans.Seg_norm(dataset='IXI'),
        trans.NumpyType((np.float32, np.int16))
    ])

    total_all_pairs = len(train_files) * max(0, (len(train_files) - 1))
    if args.train_pair_mode == 'all':
        train_set = datasets.LPBABrainDatasetS2S(train_files, transforms=train_trans)
    else:
        train_set = datasets.LPBABrainRandomPairDatasetS2S(
            train_files, transforms=train_trans, pairs_per_epoch=args.train_pairs_per_epoch
        )

    val_set = datasets.LPBABrainInferDatasetS2S(val_files, transforms=val_trans) if len(val_files) > 0 else None

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Validation on rank0 only (simple + avoids distributed metric gather complexity)
    val_loader = None
    if val_set is not None and is_main_process():
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True
        )

    if is_main_process():
        if args.train_pair_mode == 'all':
            print(f"Train pair mode: all (N*(N-1)={total_all_pairs})")
        else:
            print(f"Train pair mode: random ({args.train_pairs_per_epoch} pairs/epoch)")
            if total_all_pairs > 0 and args.train_pairs_per_epoch > total_all_pairs:
                print("[Info] train_pairs_per_epoch > all unique ordered pairs; repeated pairs will appear.")
        print(f"Train samples (dataset len): {len(train_set)} | Val samples: {len(val_set) if val_set is not None else 0}")

    model = BPRN(
        inshape=args.inshape,
        in_channel=1,
        channels=args.channels,
        use_lightweight_sacb=True
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {total_params:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    criterion_sim = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')
    criterion_bend = losses.BendingEnergy3d() if args.w_bend > 0 else None
    reg_model = utils.register_model(args.inshape, mode='nearest').to(device)
    scaler = GradScaler("cuda", enabled=args.use_amp)

    # Optional resume: load checkpoint weights/optimizer/scaler and auto-set start epoch.
    if args.resume is not None:
        if is_main_process():
            print(f"Loading checkpoint for resume: {args.resume}")
        map_location = {f"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(args.resume, map_location=map_location)

        state_dict = ckpt.get('state_dict', ckpt)
        model.module.load_state_dict(state_dict, strict=True)

        if isinstance(ckpt, dict) and ('optimizer' in ckpt):
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                if is_main_process():
                    print("Optimizer state restored from checkpoint.")
            except Exception as e:
                if is_main_process():
                    print(f"Warning: failed to restore optimizer state: {e}")

        if isinstance(ckpt, dict) and ('scaler' in ckpt):
            try:
                scaler.load_state_dict(ckpt['scaler'])
                if is_main_process():
                    print("GradScaler state restored from checkpoint.")
            except Exception as e:
                if is_main_process():
                    print(f"Warning: failed to restore scaler state: {e}")

        if isinstance(ckpt, dict) and ('val_metrics' in ckpt):
            try:
                best_val_dice = float(ckpt['val_metrics'].get('val_dice', -1.0))
            except Exception:
                best_val_dice = -1.0
        else:
            best_val_dice = -1.0

        if isinstance(ckpt, dict) and ('epoch' in ckpt):
            resume_epoch = int(ckpt['epoch'])
            if args.resume_auto_start_epoch:
                args.start_epoch = resume_epoch + 1
                if is_main_process():
                    print(f"Auto resume epoch enabled: start_epoch set to {args.start_epoch}")
            else:
                if is_main_process():
                    print(f"Resume loaded from epoch {resume_epoch}, keeping user start_epoch={args.start_epoch}")
        else:
            if is_main_process():
                print("Resume checkpoint has no 'epoch' field; keeping current start_epoch.")
    else:
        best_val_dice = -1.0

    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)

        monitor.start_epoch(epoch)
        model.train()
        epoch_lr = adjust_learning_rate(optimizer, epoch, args.max_epoch, args.lr, min_lr=args.min_lr)

        total_batches = len(train_loader)
        for batch_idx, data in enumerate(train_loader):
            x = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            if args.use_amp:
                # Training阶段必须始终调用 DDP 封装后的 model(x, y)，不能用 model.module，
                # 否则会破坏各 rank 的同步，导致 NCCL 超时。
                with autocast(device_type='cuda', enabled=True):
                    output = model(x, y)
                    y_moved, flow = output[0], output[1]
                    sim_loss = criterion_sim(y_moved.float(), y.float()) * args.w_sim
                    reg_loss = criterion_reg(flow.float(), y.float()) * args.w_reg
                    bend_loss = (criterion_bend(flow.float()) * args.w_bend) if criterion_bend is not None else 0.0
                    loss = sim_loss + reg_loss + bend_loss
            else:
                output = model(x, y)
                y_moved, flow = output[0], output[1]
                sim_loss = criterion_sim(y_moved.float(), y.float()) * args.w_sim
                reg_loss = criterion_reg(flow.float(), y.float()) * args.w_reg
                bend_loss = (criterion_bend(flow.float()) * args.w_bend) if criterion_bend is not None else 0.0
                loss = sim_loss + reg_loss + bend_loss

            # ---- gather loss stats across ranks for stable global reporting ----
            local_loss = float(loss.detach().cpu().item())
            local_sim = float(sim_loss.detach().cpu().item())
            local_reg = float(reg_loss.detach().cpu().item()) + (
                float(bend_loss.detach().cpu().item()) if isinstance(bend_loss, torch.Tensor) else float(bend_loss)
            )

            if not math.isfinite(local_loss):
                if is_main_process():
                    print(f"[Warning] Epoch {epoch} Batch {batch_idx+1}: Non-finite loss detected (loss={local_loss:.6f}), skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Lightweight debug：在 rank0 上周期性查看 flow 大小，判断网络是否在“动”
            if is_main_process() and (batch_idx % 200 == 0 or batch_idx == total_batches - 1):
                try:
                    flow_max = float(flow.detach().abs().max().cpu().item())
                except Exception:
                    flow_max = float("nan")
                print(f"[Debug] Epoch {epoch} Batch {batch_idx+1}/{total_batches} | "
                      f"flow_max={flow_max:.3e} | local_sim={local_sim:.4f} | local_reg={local_reg:.4e}")

            # Compute global (averaged) loss across all ranks
            if is_dist():
                try:
                    stats_tensor = torch.tensor(
                        [local_loss, local_sim, local_reg],
                        device=device,
                        dtype=torch.float32,
                    )
                    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                    stats_tensor /= float(world_size)
                    loss_val, sim_val, reg_val = (
                        float(stats_tensor[0].item()),
                        float(stats_tensor[1].item()),
                        float(stats_tensor[2].item()),
                    )
                except Exception as e:
                    # 如果 all_reduce 失败，使用本地值并打印错误
                    print(f"[Rank {get_rank()}] Warning: all_reduce failed at batch {batch_idx}: {e}")
                    loss_val, sim_val, reg_val = local_loss, local_sim, local_reg
            else:
                loss_val, sim_val, reg_val = local_loss, local_sim, local_reg

            if is_main_process():
                monitor.update_train_batch(
                    loss_val,
                    sim_val,
                    reg_val,
                    batch_idx,
                    total_batches,
                    epoch_lr
                )

        # All ranks sync before rank0 validation
        dist.barrier()

        val_metrics = {
            'val_loss': 0.0,
            'val_ncc': 0.0,
            'val_reg': 0.0,
            'val_dice': 0.0,
            'val_assd': 0.0,
            'val_hd95': 0.0,
        }

        if is_main_process() and val_loader is not None:
            try:
                model.eval()
                val_count = 0
                dice_list = []
                val_loss_acc = 0.0
                val_ncc_acc = 0.0
                val_reg_acc = 0.0

                compute_surface_metrics = (epoch in args.surface_metric_epochs)
                if compute_surface_metrics:
                    assd_list = []
                    hd95_list = []

                with torch.no_grad():
                    for v_idx, data in enumerate(val_loader):
                        x = data[0].to(device, non_blocking=True)
                        y = data[1].to(device, non_blocking=True)
                        x_seg = data[2].to(device, non_blocking=True)
                        y_seg = data[3].to(device, non_blocking=True)

                        # Validation runs on rank0 only; use underlying module to avoid DDP forward hooks.
                        output = model.module(x, y)
                        y_moved, flow = output[0], output[1]

                        v_sim = criterion_sim(y_moved.float(), y.float()) * args.w_sim
                        v_reg = criterion_reg(flow.float(), y.float()) * args.w_reg
                        v_bend = (criterion_bend(flow.float()) * args.w_bend) if criterion_bend is not None else 0.0
                        v_loss = v_sim + v_reg + v_bend

                        val_loss_acc += float(v_loss.detach().cpu().item())
                        val_ncc_acc += float(v_sim.detach().cpu().item())
                        v_reg_total = v_reg + (v_bend if isinstance(v_bend, torch.Tensor) else 0.0)
                        val_reg_acc += float(v_reg_total.detach().cpu().item())

                        def_seg = reg_model([x_seg.float(), flow])
                        dsc = utils.dice_val_VOI(def_seg.long(), y_seg.long(), voi_labels=voi_labels)
                        dice_list.append(float(dsc))

                        if compute_surface_metrics:
                            try:
                                assd_v = utils.calculate_assd(
                                    def_seg, y_seg, voxelspacing=surface_voxelspacing, voi_labels=voi_labels
                                )
                                hd95_v = utils.calculate_hd95(
                                    def_seg, y_seg, voxelspacing=surface_voxelspacing, voi_labels=voi_labels
                                )
                            except Exception:
                                assd_v = float('inf')
                                hd95_v = float('inf')
                            assd_list.append(float(assd_v if np.isfinite(assd_v) else np.nan))
                            hd95_list.append(float(hd95_v if np.isfinite(hd95_v) else np.nan))

                        val_count += 1
                        if (v_idx + 1) % 10 == 0:
                            print(f"    Validation: {v_idx+1}/{len(val_loader)}")

                if val_count > 0:
                    val_metrics['val_loss'] = val_loss_acc / val_count
                    val_metrics['val_ncc'] = val_ncc_acc / val_count
                    val_metrics['val_reg'] = val_reg_acc / val_count
                    val_metrics['val_dice'] = float(np.nanmean(dice_list))

                    if compute_surface_metrics:
                        valid_assd = [v for v in assd_list if not math.isnan(v)]
                        valid_hd95 = [v for v in hd95_list if not math.isnan(v)]
                        val_metrics['val_assd'] = float(np.mean(valid_assd)) if len(valid_assd) > 0 else float('inf')
                        val_metrics['val_hd95'] = float(np.mean(valid_hd95)) if len(valid_hd95) > 0 else float('inf')
                    else:
                        val_metrics['val_assd'] = -1.0
                        val_metrics['val_hd95'] = -1.0
            except Exception as e:
                print(f"[Error] Validation failed at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                # 使用默认值，避免阻塞其他进程

        if is_main_process():
            monitor.end_epoch(val_metrics)

            current_val_dice = val_metrics['val_dice']
            state_dict = model.module.state_dict()

            save_name = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_dice_{current_val_dice:.4f}.pth")
            try:
                torch.save({
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if args.use_amp else None,
                    'val_metrics': val_metrics,
                }, save_name)
            except Exception as e:
                print(f"Warning: failed to save checkpoint: {e}")

            if current_val_dice > best_val_dice:
                best_val_dice = current_val_dice
                best_name = os.path.join(ckpt_dir, f"best_model_dice_{best_val_dice:.4f}.pth")
                try:
                    torch.save({
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if args.use_amp else None,
                        'val_metrics': val_metrics,
                    }, best_name)
                    print(f"  Saved new best model: {best_name}")
                except Exception as e:
                    print(f"Warning: failed to save best checkpoint: {e}")

        dist.barrier()

    monitor.close()
    cleanup_ddp()

    if is_main_process():
        print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IXI DDP training for BPRN")

    parser.add_argument("--data_root", type=str, default=None,
                        help="IXI data root containing Train/ and Test/. If unset, choose by --use_affine_data")
    parser.add_argument("--use_affine_data", action='store_true',
                        help="Use ./affineIXI_data as default data root (otherwise ./rawIXI_data)")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Optional explicit train directory (overrides --data_root)")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Optional explicit val/test directory (overrides --data_root)")
    parser.add_argument("--inshape", nargs=3, type=int, default=[80, 96, 112],
                        help="IXI target shape for 2mm setup (2mm, pre-resampled)")

    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--train_pair_mode", type=str, default="random", choices=["random", "all"],
                        help="random=fixed random pairs per epoch, all=all ordered pairs N*(N-1)")
    parser.add_argument("--train_pairs_per_epoch", type=int, default=12000,
                        help="Used only when train_pair_mode=random")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_epoch", type=int, default=40)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--w_sim", type=float, default=1.0)
    parser.add_argument("--w_reg", type=float, default=0.5)
    parser.add_argument("--w_bend", type=float, default=0.0,
                        help="Optional bending energy (2nd-order) regularization weight (default: 0=off)")

    # --- Data augmentation (train only) ---
    parser.add_argument("--aug_flip", action="store_true", help="Enable random flip augmentation")
    parser.add_argument("--aug_gamma", action="store_true", help="Enable random gamma augmentation")
    parser.add_argument("--aug_gamma_min", type=float, default=0.7)
    parser.add_argument("--aug_gamma_max", type=float, default=1.5)
    parser.add_argument("--aug_gamma_same_on_pair", action="store_true",
                        help="If set, apply same gamma to (x,y) pair (default: False)")

    parser.add_argument("--aug_noise", action="store_true", help="Enable random Gaussian noise augmentation")
    parser.add_argument("--aug_noise_min", type=float, default=0.0)
    parser.add_argument("--aug_noise_max", type=float, default=0.03)
    parser.add_argument("--aug_noise_same_on_pair", action="store_true",
                        help="If set, apply same noise level to (x,y) pair (default: False)")

    parser.add_argument("--aug_affine", action="store_true", help="Enable random affine augmentation (light)")
    parser.add_argument("--aug_affine_mode", type=str, default="moving_only",
                        choices=["same", "moving_only", "fixed_only"],
                        help="Which image(s) to apply affine: same / moving_only / fixed_only")
    parser.add_argument("--aug_affine_degrees", type=float, default=5.0)
    parser.add_argument("--aug_affine_translate", type=float, default=2.0,
                        help="Translation in voxels")
    parser.add_argument("--aug_affine_scale", type=float, default=0.05,
                        help="Scale jitter, e.g. 0.05 => [0.95, 1.05]")

    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--surface_metric_epochs", type=int, nargs='+', default=[])

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--resume_auto_start_epoch", action='store_true',
                        help="If set, start from checkpoint epoch + 1")

    args = parser.parse_args()
    args.inshape = tuple(args.inshape)

    main(args)

