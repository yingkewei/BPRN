import argparse
import glob
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use('Agg')  # no X11 needed for saving figures (nohup / SSH / headless)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from natsort import natsorted

import utils
from data import datasets, trans
from BPRN_model import BPRN as BPRN_TBN
from BPRN_model1 import BPRN as BPRN_TBN1


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


same_seeds(24)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def resolve_dataset_defaults(dataset):
    if dataset == 'LPBA':
        return './LPBA_data/Val/', (160, 192, 160), None
    if dataset == 'ABDOMENCTCT':
        return './AbdomenCTCT_data/Val/', (192, 160, 256), None
    if dataset == 'OASIS':
        return './OASIS_L2R_2021_task03/Test/', (160, 192, 224), None
    return './rawIXI_data/Test/', (80, 96, 112), (2.0, 2.0, 2.0)


def find_checkpoint(ckpt_dir):
    best_ckpts = natsorted(glob.glob(os.path.join(ckpt_dir, 'best_model_dice_*.pth')))
    if best_ckpts:
        return best_ckpts[-1]
    epoch_ckpts = natsorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*_dice_*.pth')))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    generic_ckpts = natsorted([p for p in glob.glob(os.path.join(ckpt_dir, '*')) if p.endswith('.pth') or p.endswith('.tar')])
    return generic_ckpts[-1] if generic_ckpts else None


def load_state_dict_flexible(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'Warning: missing keys when loading checkpoint: {len(missing)}')
    if unexpected:
        print(f'Warning: unexpected keys when loading checkpoint: {len(unexpected)}')


def _slice_by_plane(vol, plane='coronal'):
    if plane == 'axial':
        return vol[vol.shape[0] // 2, :, :]
    if plane == 'coronal':
        return vol[:, vol.shape[1] // 2, :]
    return vol[:, :, vol.shape[2] // 2]


def _flow_uv_by_plane(flow_3d, plane='coronal'):
    fz, fy, fx = flow_3d[0], flow_3d[1], flow_3d[2]
    if plane == 'axial':
        z = fz.shape[0] // 2
        return fx[z, :, :], fy[z, :, :]
    if plane == 'coronal':
        y = fy.shape[1] // 2
        return fx[:, y, :], fz[:, y, :]
    x = fx.shape[2] // 2
    return fy[:, :, x], fz[:, :, x]


def _flow_to_rgb(u, v):
    import matplotlib.colors as mcolors
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)
    h = (ang + np.pi) / (2 * np.pi)
    m95 = np.percentile(mag, 95) + 1e-6
    s = np.clip(mag / m95, 0, 1)
    hsv = np.stack([h, s, s], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def _checkerboard(a, b, block=16):
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // block + xx // block) % 2 == 0)
    out = np.where(mask, a, b)
    return out


def _make_grid_img(h, w, step=8, thickness=1):
    """Create a binary grid image (white lines on black background)."""
    step = max(1, int(step))
    thickness = max(1, int(thickness))
    img = np.zeros((h, w), dtype=np.float32)
    half = thickness // 2
    for y in range(0, h, step):
        y0 = max(0, y - half)
        y1 = min(h, y0 + thickness)
        img[y0:y1, :] = 1.0
    for x in range(0, w, step):
        x0 = max(0, x - half)
        x1 = min(w, x0 + thickness)
        img[:, x0:x1] = 1.0
    return img


def _vis_rc_context(figure_dpi):
    """Matplotlib rc for consistent figure/save DPI (works across mpl 3.x)."""
    import matplotlib as mpl
    return mpl.rc_context({
        'figure.dpi': figure_dpi,
        'savefig.dpi': figure_dpi,
    })


def _warp_grid_img_2d(grid_img, u, v):
    """Warp a 2D grid image with a 2D displacement field (u,v) using bilinear sampling."""
    import torch.nn.functional as F

    h, w = grid_img.shape
    src = torch.from_numpy(grid_img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx + u.astype(np.float32)
    y = yy + v.astype(np.float32)

    # normalize to [-1, 1] for grid_sample, order is (x, y)
    x = 2.0 * (x / (w - 1.0)) - 1.0
    y = 2.0 * (y / (h - 1.0)) - 1.0
    grid = np.stack([x, y], axis=-1)  # [H,W,2]
    grid = torch.from_numpy(grid).unsqueeze(0)  # [1,H,W,2]

    warped = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped.squeeze(0).squeeze(0).numpy()



def save_oasis_style_panel(x, y, x_def, flow, out_path, plane='coronal', grid_step=8,
                           figure_dpi=300, grid_line_thickness=1):
    import matplotlib.pyplot as plt

    x_np = x.detach().cpu().numpy()[0, 0]
    y_np = y.detach().cpu().numpy()[0, 0]
    xdef_np = x_def.detach().cpu().numpy()[0, 0]
    flow_np = flow.detach().cpu().numpy()[0]

    fixed_sl = _slice_by_plane(y_np, plane)
    moving_sl = _slice_by_plane(x_np, plane)
    warped_sl = _slice_by_plane(xdef_np, plane)

    # checkerboard for clean alignment view
    checker = _checkerboard(fixed_sl, warped_sl, block=max(8, int(grid_step * 2)))

    # flow rgb
    u, v = _flow_uv_by_plane(flow_np, plane=plane)
    flow_rgb = _flow_to_rgb(u, v)

    with _vis_rc_context(figure_dpi):
        fig = plt.figure(figsize=(10, 9))
        axs = [fig.add_subplot(3, 2, i + 1) for i in range(6)]

        axs[0].imshow(fixed_sl, cmap='gray', interpolation='nearest')
        axs[0].set_title('Fixed')
        axs[0].axis('off')
        axs[1].imshow(moving_sl, cmap='gray', interpolation='nearest')
        axs[1].set_title('Moving')
        axs[1].axis('off')
        axs[2].imshow(warped_sl, cmap='gray', interpolation='nearest')
        axs[2].set_title('Warped')
        axs[2].axis('off')
        axs[3].imshow(flow_rgb, interpolation='bilinear')
        axs[3].set_title('Flow RGB')
        axs[3].axis('off')
        axs[4].imshow(checker, cmap='gray', interpolation='nearest')
        axs[4].set_title('Checkerboard (Fixed/Warped)')
        axs[4].axis('off')

        # Warped grid: bilinear imshow softens stair-steps when saving at high DPI.
        h, w = u.shape
        t = max(1, int(grid_line_thickness))
        base_grid = _make_grid_img(h, w, step=max(6, int(grid_step)), thickness=t)
        warped_grid = _warp_grid_img_2d(base_grid, u, v)
        axs[5].imshow(warped_grid, cmap='gray', vmin=0.0, vmax=1.0, interpolation='bilinear')
        axs[5].set_title('Warped Grid')
        axs[5].axis('off')

        fig.tight_layout()
        fig.savefig(out_path, dpi=figure_dpi, bbox_inches='tight')
        plt.close(fig)


def save_oasis_style_panel_split(x, y, x_def, flow, out_dir, plane='coronal', grid_step=8,
                                 figure_dpi=300, grid_line_thickness=1):
    """
    Save the 6 subplots from save_oasis_style_panel as 6 separate PNG files.
    Output folder should already encode (moving,fixed) identity.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    x_np = x.detach().cpu().numpy()[0, 0]
    y_np = y.detach().cpu().numpy()[0, 0]
    xdef_np = x_def.detach().cpu().numpy()[0, 0]
    flow_np = flow.detach().cpu().numpy()[0]

    fixed_sl = _slice_by_plane(y_np, plane)
    moving_sl = _slice_by_plane(x_np, plane)
    warped_sl = _slice_by_plane(xdef_np, plane)

    checker = _checkerboard(fixed_sl, warped_sl, block=max(8, int(grid_step * 2)))

    u, v = _flow_uv_by_plane(flow_np, plane=plane)
    flow_rgb = _flow_to_rgb(u, v)

    h, w = u.shape
    t = max(1, int(grid_line_thickness))
    base_grid = _make_grid_img(h, w, step=max(6, int(grid_step)), thickness=t)
    warped_grid = _warp_grid_img_2d(base_grid, u, v)

    def _save_gray(im2d, path, interpolation='nearest'):
        with _vis_rc_context(figure_dpi):
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(im2d, cmap='gray', interpolation=interpolation)
            plt.axis('off')
            fig.tight_layout()
            fig.savefig(path, dpi=figure_dpi, bbox_inches='tight')
            plt.close(fig)

    def _save_rgb(im3d, path):
        with _vis_rc_context(figure_dpi):
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(im3d, interpolation='bilinear')
            plt.axis('off')
            fig.tight_layout()
            fig.savefig(path, dpi=figure_dpi, bbox_inches='tight')
            plt.close(fig)

    suffix = plane
    _save_gray(fixed_sl, os.path.join(out_dir, f'fixed_{suffix}.png'))
    _save_gray(moving_sl, os.path.join(out_dir, f'moving_{suffix}.png'))
    _save_gray(warped_sl, os.path.join(out_dir, f'warped_{suffix}.png'))
    _save_rgb(flow_rgb, os.path.join(out_dir, f'flow_rgb_{suffix}.png'))
    _save_gray(checker, os.path.join(out_dir, f'checkerboard_{suffix}.png'))
    _save_gray(warped_grid, os.path.join(out_dir, f'warped_grid_{suffix}.png'), interpolation='bilinear')


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for inference.')

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)

    default_val_dir, default_shape, surface_voxelspacing = resolve_dataset_defaults(args.dataset)
    val_dir = args.val_dir or default_val_dir
    inshape = tuple(args.inshape) if args.inshape is not None else default_shape

    if args.dataset == 'IXI' and inshape in [(160, 192, 160), (160, 192, 224)]:
        inshape = (80, 96, 112)
    if args.dataset == 'ABDOMENCTCT' and inshape in [(160, 192, 160), (160, 192, 224)]:
        inshape = (192, 160, 256)
    if args.dataset == 'LPBA' and inshape == (160, 192, 224):
        inshape = (160, 192, 160)
    if args.dataset == 'OASIS' and inshape == (160, 192, 160):
        inshape = (160, 192, 224)

    print(f'Inference dataset: {args.dataset}')
    print(f'Validation directory: {val_dir}')
    print(f'Input shape: {inshape}')

    test_files = sorted(glob.glob(os.path.join(val_dir, '*.pkl')))
    print(f'Found .pkl files: {len(test_files)}')
    if not test_files:
        print('ERROR: no .pkl files found, please check --val_dir')
        return
    n_cases = len(test_files)

    def _case_id_from_path(p):
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        return stem

    def _pair_case_indices_from_sample_idx(sample_idx, n):
        """
        Match LPBABrainInferDatasetS2S pairing scheme:
          x_index = idx // (n-1)
          s = idx % (n-1)
          y_index = s+1 if s >= x_index else s
        Returns (moving_index, fixed_index).
        """
        if n < 2:
            return 0, 0
        x_index = int(sample_idx // (n - 1))
        s = int(sample_idx % (n - 1))
        y_index = int(s + 1 if s >= x_index else s)
        return x_index, y_index

    if args.dataset == 'OASIS':
        model_cls = BPRN_TBN1
        model_source = 'BPRN_model1.py'
    else:
        model_cls = BPRN_TBN
        model_source = 'BPRN_model.py'

    if args.dataset in ('LPBA', 'IXI'):
        model = model_cls(inshape=inshape, in_channel=1, channels=args.channels,
                          use_lightweight_sacb=args.use_lightweight_sacb,
                          use_bea_refine=args.use_bea_refine, bea_alpha=args.bea_alpha).to(device)
    else:
        model = model_cls(inshape=inshape, in_channel=1, channels=args.channels,
                          use_lightweight_sacb=args.use_lightweight_sacb,
                          use_bea_refine=args.use_bea_refine, bea_alpha=args.bea_alpha,
                          sacb_clusters=args.sacb_clusters,
                          sacb_fusion_clusters=args.sacb_fusion_clusters,
                          use_sacb_in_encoder=args.use_sacb_in_encoder).to(device)

    model_label = model.__class__.__name__
    print(f'Model source: {model_source}')

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        patterns = ['checkpoints/train_BPRN_mp_*']
        if args.dataset == 'IXI':
            patterns = ['checkpoints/train_BPRN_IXI_ddp_*', 'checkpoints/train_BPRN_mp_*']
        elif args.dataset == 'OASIS':
            patterns = ['checkpoints/train_BPRN_*', 'checkpoints/train_BPRN_mp_*']
        roots = []
        for p in patterns:
            roots.extend([d for d in glob.glob(p) if os.path.isdir(d)])
        roots = natsorted(list(set(roots)))
        if not roots:
            print(f'ERROR: no checkpoints found for dataset={args.dataset}. Please set --ckpt_dir explicitly.')
            return
        ckpt_dir = roots[-1]

    checkpoint_path = find_checkpoint(ckpt_dir)
    if checkpoint_path is None:
        print(f'ERROR: no checkpoint file found in {ckpt_dir}')
        return

    load_state_dict_flexible(model, checkpoint_path, device)
    print(f'Loaded checkpoint: {checkpoint_path}')

    reg_model = utils.register_model(inshape, 'nearest').to(device)

    if args.dataset == 'IXI':
        test_composed = transforms.Compose([trans.Seg_norm(dataset='IXI'), trans.NumpyType((np.float32, np.int16))])
    else:
        test_composed = transforms.Compose([trans.Seg_norm(dataset=args.dataset), trans.NumpyType((np.float32, np.int16))])

    if args.dataset == 'OASIS':
        test_set = datasets.OASISBrainInferDatasetS2S(test_files, transforms=test_composed)
    else:
        test_set = datasets.LPBABrainInferDatasetS2S(test_files, transforms=test_composed)

    infer_batch_size = args.batch_size if args.dataset == 'IXI' else 1
    test_loader = DataLoader(test_set, batch_size=infer_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    voi_labels = utils.get_voi_labels(args.dataset)
    eval_dsc_def, eval_dsc_raw, eval_det = AverageMeter(), AverageMeter(), AverageMeter()
    eval_assd, eval_hd95 = AverageMeter(), AverageMeter()

    vis_root = os.path.join('visualizations', args.dataset.lower())
    vis_dir = os.path.join(vis_root, model_label.lower())
    os.makedirs(vis_dir, exist_ok=True)
    panel_dir = os.path.join(vis_dir, 'panels') if args.save_panel else None
    legacy_panel_dir = os.path.join(vis_dir, 'oasis_panels') if args.save_panel else None
    if panel_dir:
        os.makedirs(panel_dir, exist_ok=True)
    # Backward compatibility: if legacy dir already exists, also write there.
    write_legacy_panel = bool(legacy_panel_dir and os.path.isdir(legacy_panel_dir))

    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.time()
    num_samples = 0
    per_pair_results = []
    per_fixed_results = {}  # fixed_id -> list[dict]
    per_fixed_csv_rows = {}  # fixed_id -> list[dict] for CSV output
    save_all_vis = bool(args.save_all_visualizations)
    visualize_set = set(args.visualize_indices) if not save_all_vis else set()

    model.eval()
    with torch.no_grad():
        stdy_idx = 0
        total_pairs = len(test_set)
        for data in test_loader:
            x, y, x_seg, y_seg = [t.to(device, non_blocking=True) for t in data]
            x_def, flow = model(x, y)
            def_out = reg_model([x_seg.float(), flow])

            bsz = x.size(0)
            num_samples += bsz
            for b in range(bsz):
                sample_idx = stdy_idx + b
                flow_b = flow[b:b+1]
                jac_det = utils.jacobian_determinant_vxm(flow_b.detach().cpu().numpy()[0, :, :, :, :])
                det_ratio = float(np.sum(jac_det <= 0) / np.prod(y[b:b+1].shape[2:]))

                def_b, xseg_b, yseg_b = def_out[b:b+1], x_seg[b:b+1], y_seg[b:b+1]
                dsc_trans = float(utils.dice_val_VOI(def_b.long(), yseg_b.long(), voi_labels=voi_labels))
                dsc_raw = float(utils.dice_val_VOI(xseg_b.long(), yseg_b.long(), voi_labels=voi_labels))
                assd = float(utils.calculate_assd(def_b, yseg_b, voxelspacing=surface_voxelspacing, voi_labels=voi_labels))
                hd95 = float(utils.calculate_hd95(def_b, yseg_b, voxelspacing=surface_voxelspacing, voi_labels=voi_labels))

                moving_id = None
                fixed_id = None
                if args.dataset == 'IXI' and n_cases >= 2:
                    mov_i, fix_i = _pair_case_indices_from_sample_idx(sample_idx, n_cases)
                    moving_id = _case_id_from_path(test_files[mov_i])
                    fixed_id = _case_id_from_path(test_files[fix_i])

                if save_all_vis or sample_idx in visualize_set:
                    # Always save 6 separate figures (one plane: --panel_plane):
                    #   fixed / moving / warped / flow_rgb / checkerboard / warped_grid
                    if args.dataset == 'IXI' and n_cases >= 2:
                        vis_prefix = f'{moving_id}__to__{fixed_id}'
                        pair_folder = os.path.join(vis_dir, 'panels', f'fixed_{fixed_id}', vis_prefix)
                    else:
                        vis_prefix = f'{model_label.lower()}_sample_{sample_idx:03d}'
                        pair_folder = os.path.join(vis_dir, 'panels', vis_prefix)
                    save_oasis_style_panel_split(
                        x[b:b+1], y[b:b+1], x_def[b:b+1], flow_b,
                        out_dir=pair_folder,
                        plane=args.panel_plane,
                        grid_step=args.panel_grid_step,
                        figure_dpi=args.panel_figure_dpi,
                        grid_line_thickness=args.panel_grid_line_thickness,
                    )
                    if panel_dir is not None:
                        panel_path = os.path.join(pair_folder, f'panel_{args.panel_plane}.png')
                        save_oasis_style_panel(
                            x[b:b+1], y[b:b+1], x_def[b:b+1], flow_b,
                            out_path=panel_path,
                            plane=args.panel_plane,
                            grid_step=args.panel_grid_step,
                            figure_dpi=args.panel_figure_dpi,
                            grid_line_thickness=args.panel_grid_line_thickness,
                        )
                        if write_legacy_panel and legacy_panel_dir is not None:
                            legacy_panel_path = os.path.join(legacy_panel_dir, f'{vis_prefix}_panel.png')
                            save_oasis_style_panel(
                                x[b:b+1], y[b:b+1], x_def[b:b+1], flow_b,
                                out_path=legacy_panel_path,
                                plane=args.panel_plane,
                                grid_step=args.panel_grid_step,
                                figure_dpi=args.panel_figure_dpi,
                                grid_line_thickness=args.panel_grid_line_thickness,
                            )

                eval_det.update(det_ratio, 1)
                eval_dsc_def.update(dsc_trans, 1)
                eval_dsc_raw.update(dsc_raw, 1)
                eval_assd.update(assd, 1)
                eval_hd95.update(hd95, 1)
                pair_row = {
                    'sample_idx': int(sample_idx),
                    'moving_id': moving_id,
                    'fixed_id': fixed_id,
                    'trans_dice': dsc_trans,
                    'raw_dice': dsc_raw,
                    'assd': assd,
                    'hd95': hd95,
                    'folding_ratio_det_le_0': det_ratio,
                }
                per_pair_results.append(pair_row)
                if args.dataset == 'IXI' and fixed_id is not None:
                    per_fixed_results.setdefault(fixed_id, []).append(pair_row)

                    # CSV: VOI-wise deformed DSC + non_jec
                    # Keep VOI column order identical to utils.get_voi_labels('IXI')
                    pred = def_b.long().detach().cpu().numpy()[0, 0, ...]
                    true = yseg_b.long().detach().cpu().numpy()[0, 0, ...]

                    voi_ds = []
                    for lbl in voi_labels:
                        pred_i = (pred == int(lbl))
                        true_i = (true == int(lbl))
                        intersection = np.logical_and(pred_i, true_i).sum()
                        union = pred_i.sum() + true_i.sum()
                        dsc = (2.0 * intersection) / (union + 1e-5)
                        voi_ds.append(float(dsc))

                    per_fixed_csv_rows.setdefault(fixed_id, []).append({
                        'sample_idx': int(sample_idx),
                        'moving_id': moving_id,
                        'voi_dscs': voi_ds,
                        'non_jec': det_ratio,
                    })

                # Progress log: print periodically and at the end.
                done = sample_idx + 1
                if done == 1 or done % max(1, int(args.log_interval)) == 0 or done == total_pairs:
                    pct = 100.0 * done / max(1, total_pairs)
                    if args.dataset == 'IXI' and moving_id is not None and fixed_id is not None:
                        print(f'[{done}/{total_pairs} | {pct:6.2f}%] moving={moving_id} -> fixed={fixed_id} | '
                              f'dsc={dsc_trans:.4f}, assd={assd:.4f}, hd95={hd95:.4f}, non_jec={det_ratio:.6f}')
                    else:
                        print(f'[{done}/{total_pairs} | {pct:6.2f}%] sample_idx={sample_idx} | '
                              f'dsc={dsc_trans:.4f}, assd={assd:.4f}, hd95={hd95:.4f}, non_jec={det_ratio:.6f}')
            stdy_idx += bsz

        total_wall = time.time() - wall_start
        avg_time_per_sample = total_wall / max(1, num_samples)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    num_params_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)

    summary = {
        'dataset': args.dataset,
        'model_class': model_label,
        'model_source': model_source,
        'checkpoint_dir': ckpt_dir,
        'checkpoint_file': checkpoint_path,
        'inshape': list(inshape),
        'batch_size': infer_batch_size,
        'num_samples': int(num_samples),
        'metrics': {
            'deformed_dsc_mean': float(eval_dsc_def.avg), 'deformed_dsc_std': float(eval_dsc_def.std),
            'raw_dsc_mean': float(eval_dsc_raw.avg), 'raw_dsc_std': float(eval_dsc_raw.std),
            'assd_mean': float(eval_assd.avg), 'assd_std': float(eval_assd.std),
            'hd95_mean': float(eval_hd95.avg), 'hd95_std': float(eval_hd95.std),
            'folding_ratio_mean': float(eval_det.avg), 'folding_ratio_std': float(eval_det.std),
        },
        'benchmark': {
            'avg_time_per_sample_sec': float(avg_time_per_sample),
            'peak_memory_gb': float(peak_mem_gb),
            'parameter_mb': float(num_params_mb),
        },
    }

    save_dir = args.metrics_out_dir if args.metrics_out_dir is not None else ckpt_dir
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, f"infer_metrics_{args.dataset.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        payload = {'summary': summary, 'per_pair': per_pair_results}
        if args.dataset == 'IXI':
            payload['per_fixed'] = per_fixed_results
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'Per-pair and summary metrics saved to: {metrics_path}')

    # TransMorph-like CSV output (IXI only):
    # - write one CSV per fixed_id
    # - columns: moving_id, sample_idx, VOI labels (in order), non_jec
    if args.dataset == 'IXI':
        csv_root = os.path.join(save_dir, 'Quantitative_Results')
        csv_fixed_dir = os.path.join(csv_root, 'fixed')
        os.makedirs(csv_fixed_dir, exist_ok=True)

        csv_label_cols = [f'voi_{int(lbl)}_dsc' for lbl in voi_labels]
        header = ['moving_id', 'sample_idx'] + csv_label_cols + ['non_jec']

        for fixed_id, rows in per_fixed_csv_rows.items():
            out_csv = os.path.join(csv_fixed_dir, f'fixed_{fixed_id}.csv')
            with open(out_csv, 'w', encoding='utf-8') as f:
                f.write(','.join(header) + '\n')
                # sort by sample_idx to keep deterministic order
                rows_sorted = sorted(rows, key=lambda r: r['sample_idx'])
                for r in rows_sorted:
                    vals = [
                        str(r['moving_id']),
                        str(r['sample_idx']),
                    ] + [f'{x:.6f}' for x in r['voi_dscs']] + [f'{r["non_jec"]:.6f}']
                    f.write(','.join(vals) + '\n')

        print(f'Fixed-group VOI CSVs saved to: {csv_fixed_dir}')

    # Final output summary for quick inspection after long runs.
    print('========== Inference Finished ==========')
    print(f'Dataset: {args.dataset}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'JSON metrics: {metrics_path}')
    if args.dataset == 'IXI':
        print(f'Fixed-group CSV dir: {os.path.join(save_dir, "Quantitative_Results", "fixed")}')
    print(f'Six-figure outputs (per visualized sample): {os.path.join(vis_dir, "panels")}/<pair_or_sample>/')
    if args.save_panel:
        print(f'Combined 3x2 panel (--save_panel): panel_<plane>.png beside the six figures in each folder.')
    print(f'Visualization root: {vis_dir}')
    print('========================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='OASIS', choices=['LPBA', 'IXI', 'ABDOMENCTCT', 'OASIS'])
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)

    parser.add_argument('--inshape', nargs=3, type=int, default=None)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--use_lightweight_sacb', action='store_true')
    parser.add_argument('--sacb_clusters', type=int, default=4)
    parser.add_argument('--sacb_fusion_clusters', type=int, default=3)
    parser.add_argument('--use_sacb_in_encoder', action='store_true')
    parser.add_argument('--use_bea_refine', action='store_true')
    parser.add_argument('--bea_alpha', type=float, default=0.01)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--visualize_indices', type=int, nargs='+', default=[1])
    parser.add_argument('--save_all_visualizations', action='store_true',
                        help='If set, save visualizations for every inference pair. '
                             'For IXI all-pairs inference, outputs are grouped by fixed case.')
    parser.add_argument('--save_panel', action='store_true',
                        help='Also save one combined 3x2 figure (panel_<plane>.png) next to the six split PNGs.')
    parser.add_argument('--panel_plane', type=str, default='sagittal', choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--panel_grid_step', type=int, default=8,
                        help='Spacing in pixels between grid lines on the 2D slice. '
                             'Larger (e.g. 12-16): fewer lines, less busy in a small paper figure. '
                             'Smaller (e.g. 4-6): denser mesh, better local detail; needs high DPI to avoid clutter.')
    parser.add_argument('--panel_figure_dpi', type=int, default=300,
                        help='PNG export resolution. Use 300 for most papers; 600 for print.')
    parser.add_argument('--panel_grid_line_thickness', type=int, default=1,
                        help='Grid line thickness in pixels before warping. Try 2 if lines look too thin after downscaling.')
    parser.add_argument('--metrics_out_dir', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=200,
                        help='Print progress every N pairs (default: 200).')

    args = parser.parse_args()
    main(args)
