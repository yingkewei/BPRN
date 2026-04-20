# train_BPRN_model_parallel.py
# -*- coding: utf-8 -*-
"""
Model Parallel Training script for BPRN_model.BPRN
Splits the model across 2 GPUs to handle large memory requirements

Strategy:
- GPU 0: Encoders (moving + fixed)
- GPU 1: Decoder + Flow prediction

This allows training with batch_size=1 when a single GPU is insufficient.
"""

import os
import time
import glob
import math
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms as tv_transforms

# Project modules
import losses
import utils
from data import datasets, trans
from BPRN_model import BPRN


# ---------------------------
# Model Parallel Wrapper
# ---------------------------
class BPRN_ModelParallel(nn.Module):
    """
    Model Parallel version of BPRN
    Splits model across 2 GPUs:
    - GPU 0: Encoders (moving + fixed)
    - GPU 1: Decoder (bottleneck + all decoder layers)
    """
    
    def __init__(self, inshape=(160, 192, 160), flow_multiplier=1., in_channel=1, 
                 channels=16, use_lightweight_sacb=True, sacb_clusters=4,
                 sacb_fusion_clusters=3, use_sacb_in_encoder=True,
                 device0='cuda:0', device1='cuda:1'):
        super().__init__()
        
        self.device0 = torch.device(device0)
        self.device1 = torch.device(device1)
        self.inshape = inshape
        self.channels = channels
        
        print(f"Initializing Model Parallel:")
        print(f"  - Encoders on {device0}")
        print(f"  - Decoder on {device1}")
        
        # Create full model first
        full_model = BPRN(
            inshape=inshape,
            flow_multiplier=flow_multiplier,
            in_channel=in_channel,
            channels=channels,
            use_lightweight_sacb=use_lightweight_sacb,
            sacb_clusters=sacb_clusters,
            sacb_fusion_clusters=sacb_fusion_clusters,
            use_sacb_in_encoder=use_sacb_in_encoder
        )
        
        # Split model components
        # GPU 0: Encoders
        self.encoder_moving = full_model.encoder_moving.to(self.device0)
        self.encoder_fixed = full_model.encoder_fixed.to(self.device0)
        
        # GPU 1: All decoder components
        self.upsample = full_model.upsample.to(self.device1)
        self.upsample_trilin = full_model.upsample_trilin.to(self.device1)
        
        # Warp and Diff modules
        self.warp = nn.ModuleList([w.to(self.device1) for w in full_model.warp])
        self.diff = nn.ModuleList([d.to(self.device1) for d in full_model.diff])
        
        # Bottleneck and decoder layers
        self.cconv_4 = full_model.cconv_4.to(self.device1)
        self.defconv4 = full_model.defconv4.to(self.device1)
        self.dconv4 = full_model.dconv4.to(self.device1)
        
        self.upconv3 = full_model.upconv3.to(self.device1)
        self.adaptive_fusion_3 = full_model.adaptive_fusion_3.to(self.device1)
        self.adaptive_fusion_3_recur = full_model.adaptive_fusion_3_recur.to(self.device1)
        self.defconv3 = full_model.defconv3.to(self.device1)
        self.dconv3 = full_model.dconv3.to(self.device1)
        
        self.upconv2 = full_model.upconv2.to(self.device1)
        self.adaptive_fusion_2_first = full_model.adaptive_fusion_2_first.to(self.device1)
        self.adaptive_fusion_2_recur = full_model.adaptive_fusion_2_recur.to(self.device1)
        self.defconv2 = full_model.defconv2.to(self.device1)
        self.dconv2_first = full_model.dconv2_first.to(self.device1)
        self.dconv2_recur = full_model.dconv2_recur.to(self.device1)
        
        self.upconv1 = full_model.upconv1.to(self.device1)
        self.adaptive_fusion_1 = full_model.adaptive_fusion_1.to(self.device1)
        self.defconv1 = full_model.defconv1.to(self.device1)
        
        del full_model  # Free memory
        
    def forward(self, moving, fixed):
        """
        Forward pass with model parallelism
        """
        # Ensure inputs are on device0
        moving = moving.to(self.device0, non_blocking=True)
        fixed = fixed.to(self.device0, non_blocking=True)
        
        # ========== Encoding on GPU 0 ==========
        M1, M2, M3, M4 = self.encoder_moving(moving)
        F1, F2, F3, F4 = self.encoder_fixed(fixed)
        
        # Transfer encoder outputs to GPU 1
        M1 = M1.to(self.device1, non_blocking=True)
        M2 = M2.to(self.device1, non_blocking=True)
        M3 = M3.to(self.device1, non_blocking=True)
        M4 = M4.to(self.device1, non_blocking=True)
        F1 = F1.to(self.device1, non_blocking=True)
        F2 = F2.to(self.device1, non_blocking=True)
        F3 = F3.to(self.device1, non_blocking=True)
        F4 = F4.to(self.device1, non_blocking=True)
        
        # Also transfer original images for final warp
        moving = moving.to(self.device1, non_blocking=True)
        
        # ========== Decoding on GPU 1 ==========
        # Layer 4 (bottleneck)
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)
        flow = self.defconv4(C4)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1))
        v = self.defconv4(C4)
        w = self.diff[3](v)
        
        # Layer 3 (2 recursions)
        D3 = self.upconv3(C4)
        flow = self.upsample_trilin(2 * (self.warp[3](flow, w) + w))
        
        # 1st recursion - SACB
        warped = self.warp[2](M3, flow)
        C3 = self.adaptive_fusion_3(F3, warped, D3)
        v = self.defconv3(C3)
        w = self.diff[2](v)
        flow = self.warp[2](flow, w) + w
        
        # 2nd recursion - Standard
        warped = self.warp[2](M3, flow)
        D3 = self.dconv3(C3)
        C3 = self.adaptive_fusion_3_recur(torch.cat([F3, warped, D3], dim=1))
        v = self.defconv3(C3)
        w = self.diff[2](v)
        
        # Layer 2 (3 recursions)
        D2 = self.upconv2(C3)
        flow = self.upsample_trilin(2 * (self.warp[2](flow, w) + w))
        
        # 1st recursion - SACB
        warped = self.warp[1](M2, flow)
        C2 = self.adaptive_fusion_2_first(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        
        # 2nd recursion - Standard
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2_first(C2)
        C2 = self.adaptive_fusion_2_recur(torch.cat([F2, warped, D2], dim=1))
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        
        # 3rd recursion - Standard
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2_recur(C2)
        C2 = self.adaptive_fusion_2_recur(torch.cat([F2, warped, D2], dim=1))
        v = self.defconv2(C2)
        w = self.diff[1](v)
        
        # Layer 1 (final)
        D1 = self.upconv1(C2)
        flow = self.upsample_trilin(2 * (self.warp[1](flow, w) + w))
        warped = self.warp[0](M1, flow)
        C1 = self.adaptive_fusion_1(torch.cat([F1, warped, D1], dim=1))
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w) + w
        
        # Final warp
        y_moved = self.warp[0](moving, flow)
        
        return y_moved, flow


# ---------------------------
# Training monitor
# ---------------------------
class TrainingMonitor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.log_fp = open(os.path.join(save_dir, "training_log.txt"), "a")
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        self.epoch_start_time = None
        self.train_loss_sum = 0.0
        self.train_ncc_sum = 0.0
        self.train_reg_sum = 0.0
        self.train_count = 0

    def start_epoch(self, epoch):
        self.reset_epoch_stats()
        self.epoch_start_time = time.time()
        self.epoch = epoch
        print(f"\n=== Epoch {epoch} start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        self.log_fp.write(f"\n=== Epoch {epoch} start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.log_fp.flush()

    def update_train_batch(self, loss, ncc, reg, batch_idx, total_batches, lr):
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
        epoch_time = time.time() - self.epoch_start_time
        
        # Format ASSD/HD95 with special handling for -1 (not computed) and inf (failed)
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
        self.log_fp.close()


# ---------------------------
# LR schedule
# ---------------------------
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


# ---------------------------
# Main training routine
# ---------------------------
def main(args):
    # Check GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Model parallel training requires at least 2 GPUs.")
    
    device0 = f'cuda:{args.gpu_ids[0]}'
    device1 = f'cuda:{args.gpu_ids[1]}'
    
    print(f"Using Model Parallelism:")
    print(f"  GPU {args.gpu_ids[0]}: Encoders")
    print(f"  GPU {args.gpu_ids[1]}: Decoder")

    # Prepare directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"train_BPRN_mp_{timestamp}")
    ckpt_dir = os.path.join("checkpoints", f"train_BPRN_mp_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    monitor = TrainingMonitor(log_dir)

    # ---------------------------
    # Dataset & DataLoader
    # ---------------------------
    # Dataset-specific defaults
    if args.dataset == 'LPBA':
        default_train_dir = "./LPBA_data/Train/"
        default_val_dir = "./LPBA_data/Val/"
        voi_labels = utils.get_voi_labels('LPBA')
        surface_voxelspacing = None
    elif args.dataset == 'ABDOMENCTCT':
        default_train_dir = "./AbdomenCTCT_data/Train/"
        default_val_dir = "./AbdomenCTCT_data/Val/"
        voi_labels = utils.get_voi_labels('ABDOMENCTCT')
        surface_voxelspacing = None
    elif args.dataset == 'OASIS':
        default_train_dir = "./OASIS_L2R_2021_task03/All/"
        default_val_dir = "./OASIS_L2R_2021_task03/Test/"
        voi_labels = utils.get_voi_labels('OASIS')
        surface_voxelspacing = None
    else:  # IXI
        default_train_dir = "./IXI_data/Train/"
        default_val_dir = "./IXI_data/Val/"
        voi_labels = utils.get_voi_labels('IXI')
        # IXI setup requested: 2x2x2 mm voxel spacing
        surface_voxelspacing = (2.0, 2.0, 2.0)

    train_dir = args.train_dir or default_train_dir
    val_dir = args.val_dir or default_val_dir

    print("Loading datasets...")
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.pkl")))

    if len(train_files) == 0:
        raise RuntimeError(f"No training files found in {train_dir}")
    if len(val_files) == 0:
        print(f"Warning: no validation files found in {val_dir}")

    if args.dataset == 'IXI':
        train_trans = tv_transforms.Compose([
            trans.Resample3D(args.inshape, seg_indices=set()),
            trans.NumpyType((np.float32, np.float32))
        ])
        val_trans = tv_transforms.Compose([
            trans.Resample3D(args.inshape, seg_indices={1}),
            trans.Seg_norm(dataset='IXI'),
            trans.NumpyType((np.float32, np.int16))
        ])
    else:
        train_trans = tv_transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_trans = tv_transforms.Compose([
            trans.Seg_norm(dataset=args.dataset),
            trans.NumpyType((np.float32, np.int16))
        ])

    total_all_pairs = len(train_files) * max(0, (len(train_files) - 1))
    if args.train_pair_mode == 'all':
        train_set = datasets.LPBABrainDatasetS2S(train_files, transforms=train_trans)
        print(f"Train pair mode: all (N*(N-1) = {total_all_pairs})")
    else:
        train_set = datasets.LPBABrainRandomPairDatasetS2S(
            train_files, transforms=train_trans, pairs_per_epoch=args.train_pairs_per_epoch
        )
        print(f"Train pair mode: random ({args.train_pairs_per_epoch} pairs/epoch)")
        if total_all_pairs > 0 and args.train_pairs_per_epoch > total_all_pairs:
            print("[Info] train_pairs_per_epoch is larger than all unique ordered pairs; random mode will include repeated pairs.")

    if args.dataset == 'OASIS':
        val_set = datasets.OASISBrainInferDatasetS2S(val_files, transforms=val_trans) if len(val_files) > 0 else None
    else:
        val_set = datasets.LPBABrainInferDatasetS2S(val_files, transforms=val_trans) if len(val_files) > 0 else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=max(1, args.num_workers // 2), pin_memory=True) if val_set is not None else None

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set) if val_set is not None else 0}")

    # ---------------------------
    # Model (Model Parallel)
    # ---------------------------
    print("Initializing model with Model Parallelism...")
    model = BPRN_ModelParallel(
        inshape=args.inshape,
        in_channel=1,
        channels=args.channels,
        use_lightweight_sacb=args.use_lightweight_sacb,
        sacb_clusters=args.sacb_clusters,
        sacb_fusion_clusters=args.sacb_fusion_clusters,
        use_sacb_in_encoder=args.use_sacb_in_encoder,
        device0=device0,
        device1=device1
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M")
    print(f"GPU {args.gpu_ids[0]} memory: {torch.cuda.memory_allocated(args.gpu_ids[0])/1e9:.2f} GB")
    print(f"GPU {args.gpu_ids[1]} memory: {torch.cuda.memory_allocated(args.gpu_ids[1])/1e9:.2f} GB")

    # ---------------------------
    # Optimizer & Loss
    # ---------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    criterion_sim = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')
    
    # Register model on device1 (where final output is)
    reg_model = utils.register_model(args.inshape, mode='nearest').to(device1)
    
    # Note: Mixed precision with model parallel can be tricky
    # Disable for stability, or use carefully
    scaler = GradScaler(enabled=args.use_amp)

    best_val_dice = -1.0

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(args.start_epoch, args.max_epoch):
        monitor.start_epoch(epoch)
        model.train()
        epoch_lr = adjust_learning_rate(optimizer, epoch, args.max_epoch, args.lr, min_lr=args.min_lr)

        total_batches = len(train_loader)
        for batch_idx, data in enumerate(train_loader):
            # Data will be moved to correct devices in model.forward()
            x = data[0]
            y = data[1]

            # Forward pass (model handles device placement)
            if args.use_amp:
                with autocast(enabled=True):
                    y_moved, flow = model(x, y)
                    # Move y to device1 for loss computation
                    y = y.to(device1, non_blocking=True)
                    sim_loss = criterion_sim(y_moved.float(), y.float()) * args.w_sim
                    reg_loss = criterion_reg(flow.float(), y.float()) * args.w_reg
                    loss = sim_loss + reg_loss
            else:
                y_moved, flow = model(x, y)
                y = y.to(device1, non_blocking=True)
                sim_loss = criterion_sim(y_moved.float(), y.float()) * args.w_sim
                reg_loss = criterion_reg(flow.float(), y.float()) * args.w_reg
                loss = sim_loss + reg_loss

            loss_val = float(loss.detach().cpu().item())
            if not math.isfinite(loss_val):
                print(f"  ⚠️ Non-finite loss at batch {batch_idx}, skipping.")
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

            monitor.update_train_batch(
                loss_val,
                float(sim_loss.detach().cpu().item()),
                float(reg_loss.detach().cpu().item()),
                batch_idx,
                total_batches,
                epoch_lr
            )

        # ---------------------------
        # Validation
        # ---------------------------
        val_metrics = {
            'val_loss': 0.0,
            'val_ncc': 0.0,
            'val_reg': 0.0,
            'val_dice': 0.0,
            'val_assd': 0.0,
            'val_hd95': 0.0
        }
        
        if val_loader is not None:
            model.eval()
            val_count = 0
            dice_list = []
            val_loss_acc = 0.0
            val_ncc_acc = 0.0
            val_reg_acc = 0.0
            
            # Only initialize ASSD/HD95 lists if we're computing them this epoch
            compute_surface_metrics = (epoch in args.surface_metric_epochs)
            if compute_surface_metrics:
                assd_list = []
                hd95_list = []
                print(f"  Computing VOI-wise surface metrics (ASSD/HD95) - this will take several minutes...")

            with torch.no_grad():
                for v_idx, data in enumerate(val_loader):
                    x = data[0]
                    y = data[1]
                    x_seg = data[2].to(device1, non_blocking=True)
                    y_seg = data[3].to(device1, non_blocking=True)

                    y_moved, flow = model(x, y)
                    y = y.to(device1, non_blocking=True)

                    v_sim = criterion_sim(y_moved.float(), y.float()) * args.w_sim
                    v_reg = criterion_reg(flow.float(), y.float()) * args.w_reg
                    v_loss = v_sim + v_reg

                    val_loss_acc += float(v_loss.detach().cpu().item())
                    val_ncc_acc += float(v_sim.detach().cpu().item())
                    val_reg_acc += float(v_reg.detach().cpu().item())

                    def_seg = reg_model([x_seg.float(), flow])
                    dsc = utils.dice_val_VOI(def_seg.long(), y_seg.long(), voi_labels=voi_labels)
                    dice_list.append(float(dsc))

                    # Only compute ASSD/HD95 if this is a metric epoch
                    if compute_surface_metrics:
                        try:
                            assd_v = utils.calculate_assd(
                                def_seg, y_seg, voxelspacing=surface_voxelspacing, voi_labels=voi_labels
                            )
                            hd95_v = utils.calculate_hd95(
                                def_seg, y_seg, voxelspacing=surface_voxelspacing, voi_labels=voi_labels
                            )
                        except Exception as e:
                            assd_v = float('inf')
                            hd95_v = float('inf')
                        assd_list.append(float(assd_v if np.isfinite(assd_v) else np.nan))
                        hd95_list.append(float(hd95_v if np.isfinite(hd95_v) else np.nan))
                    
                    val_count += 1
                    
                    # Print progress every 10 samples
                    if (v_idx + 1) % 10 == 0:
                        print(f"    Validation: {v_idx+1}/{len(val_loader)} samples processed...")

            if val_count > 0:
                val_metrics['val_loss'] = val_loss_acc / val_count
                val_metrics['val_ncc'] = val_ncc_acc / val_count
                val_metrics['val_reg'] = val_reg_acc / val_count
                val_metrics['val_dice'] = float(np.nanmean(dice_list))
                
                # Only compute ASSD/HD95 if we calculated them this epoch
                if compute_surface_metrics:
                    # Filter out nan but keep inf (inf means prediction failed for that VOI)
                    valid_assd = [v for v in assd_list if not math.isnan(v)]
                    valid_hd95 = [v for v in hd95_list if not math.isnan(v)]
                    
                    if len(valid_assd) > 0:
                        # If all values are inf, mean will be inf (correct behavior)
                        val_metrics['val_assd'] = float(np.mean(valid_assd))
                    else:
                        val_metrics['val_assd'] = float('inf')
                    
                    if len(valid_hd95) > 0:
                        val_metrics['val_hd95'] = float(np.mean(valid_hd95))
                    else:
                        val_metrics['val_hd95'] = float('inf')
                else:
                    # Use -1.0 to indicate not computed this epoch
                    val_metrics['val_assd'] = -1.0
                    val_metrics['val_hd95'] = -1.0

        monitor.end_epoch(val_metrics)

        # Save checkpoint
        current_val_dice = val_metrics['val_dice']
        save_name = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_dice_{current_val_dice:.4f}.pth")
        try:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, save_name)
        except Exception as e:
            print(f"Warning: failed to save checkpoint: {e}")

        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            best_name = os.path.join(ckpt_dir, f"best_model_dice_{best_val_dice:.4f}.pth")
            try:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_metrics': val_metrics
                }, best_name)
                print(f"  ✅ Saved new best model: {best_name}")
            except Exception as e:
                print(f"Warning: failed to save best checkpoint: {e}")

    monitor.close()
    print("Training finished.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0,1",
                       help="GPU IDs for model parallel (need 2 GPUs), e.g., '0,1'")
    parser.add_argument("--dataset", type=str, default="OASIS", choices=["LPBA", "IXI", "ABDOMENCTCT", "OASIS"],
                       help="Dataset preset. OASIS uses 160x192x224, LPBA uses 160x192x160, IXI uses 80x96x112 (2mm), AbdomenCTCT uses 192x160x256.")
    parser.add_argument("--train_dir", type=str, default=None,
                       help="Optional custom training directory. If omitted, uses dataset preset path.")
    parser.add_argument("--val_dir", type=str, default=None,
                       help="Optional custom validation directory. If omitted, uses dataset preset path.")
    parser.add_argument("--inshape", nargs=3, type=int, default=[160, 192, 224],
                       help="Input shape. OASIS default is 160 192 224; IXI will auto-adjust to 80 96 112 when using LPBA/OASIS shape.")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--use_lightweight_sacb", action='store_true',
                       help="Use lightweight SACB blocks (default: standard SimplifiedSACB)")
    parser.add_argument("--sacb_clusters", type=int, default=4,
                       help="Cluster count for SACB refinement blocks")
    parser.add_argument("--sacb_fusion_clusters", type=int, default=3,
                       help="Cluster count for SACB adaptive fusion blocks")
    parser.add_argument("--use_sacb_in_encoder", action='store_true',
                       help="Enable SACB in encoder at 1/4 and 1/8 scales (default: disabled)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (keep at 1 for model parallel)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_pair_mode", type=str, default="all", choices=["random", "all"],
                       help="Training pair strategy: random=fixed random pairs per epoch, all=all ordered pairs N*(N-1).")
    parser.add_argument("--train_pairs_per_epoch", type=int, default=10000,
                       help="Number of random training pairs (steps) per epoch when --train_pair_mode random.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_epoch", type=int, default=40)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--w_sim", type=float, default=1.0)
    parser.add_argument("--w_reg", type=float, default=0.5)
    parser.add_argument("--use_amp", action='store_true',
                       help="Use automatic mixed precision (may be unstable with model parallel)")
    parser.add_argument("--surface_metric_epochs", type=int, nargs='+', default=[1, 20, 39],
                       help="Specific epochs to compute ASSD/HD95 (default: [1, 20, 39])")
    args = parser.parse_args()

    # Parse GPU IDs
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    if len(args.gpu_ids) != 2:
        raise ValueError("Model parallel requires exactly 2 GPUs. Use --gpu_ids '0,1'")

    args.inshape = tuple(args.inshape)
    if args.dataset == "IXI" and args.inshape in [(160, 192, 160), (160, 192, 224)]:
        args.inshape = (80, 96, 112)
        print("[Info] Auto-set inshape to IXI default: (80, 96, 112) with voxel spacing (2.0, 2.0, 2.0)")
    if args.dataset == "ABDOMENCTCT" and args.inshape in [(160, 192, 160), (160, 192, 224)]:
        args.inshape = (192, 160, 256)
        print("[Info] Auto-set inshape to AbdomenCTCT default: (192, 160, 256)")
    if args.dataset == "LPBA" and args.inshape == (160, 192, 224):
        args.inshape = (160, 192, 160)
        print("[Info] Auto-set inshape to LPBA default: (160, 192, 160)")
    if args.dataset == "OASIS" and args.inshape == (160, 192, 160):
        args.inshape = (160, 192, 224)
        print("[Info] Auto-set inshape to OASIS default: (160, 192, 224)")

    main(args)
