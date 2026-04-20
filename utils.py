import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import pystrum.pynd.ndutils as nd

LPBA_VOI_LABELS = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54
], dtype=np.int16)

ABDO_VOI_LABELS = np.arange(1, 14, dtype=np.int16)
OASIS_VOI_LABELS = np.arange(1, 36, dtype=np.int16)

# NOTE:
# Seg_norm(dataset='IXI') remaps original IXI labels
# [0, 1, ..., 17, 20, ..., 33] -> contiguous indices [0..31].
# Metrics should use remapped indices (exclude background=0), i.e. [1..31].
IXI_VOI_LABELS = np.arange(1, 32, dtype=np.int16)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', device=None):
        super().__init__()

        self.mode = mode
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        # create sampling grid
        vectors = [torch.arange(0, s, device=self.device) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).unsqueeze(0).float()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear', device=None):
        super(register_model, self).__init__()
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.spatial_trans = SpatialTransformer(img_size, mode, device=self.device)

    def forward(self, x):
        # ensure inputs and grid are on the same device
        img = x[0].to(self.spatial_trans.grid.device)
        flow = x[1].to(self.spatial_trans.grid.device)
        return self.spatial_trans(img, flow)


def get_voi_labels(dataset='LPBA'):
    """Get VOI labels for a dataset."""
    if dataset is None:
        dataset = 'LPBA'
    dataset = str(dataset).upper()
    if dataset == 'IXI':
        return IXI_VOI_LABELS
    if dataset == 'OASIS':
        return OASIS_VOI_LABELS
    if dataset in ('ABD', 'ABDOMEN', 'ABDOMENCTCT', 'ABDOMEN_CTCT'):
        return ABDO_VOI_LABELS
    return LPBA_VOI_LABELS


def dice_val_VOI(y_pred, y_true, voi_labels=None):
    """VOI-wise Dice score - only computes labels present in GT."""
    labels = np.asarray(voi_labels if voi_labels is not None else LPBA_VOI_LABELS)

    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]

    dices = []
    for lbl in labels:
        pred_i = (pred == lbl)
        true_i = (true == lbl)

        # Skip labels not present in GT
        if true_i.sum() == 0:
            continue

        intersection = np.logical_and(pred_i, true_i).sum()
        union = pred_i.sum() + true_i.sum()
        dsc = (2.0 * intersection) / (union + 1e-5)
        dices.append(dsc)

    if len(dices) == 0:
        return 0.0
    return float(np.mean(dices))

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


# =====================
# Metrics: ASSD & HD95 (VOI-wise)
# =====================
def _to_numpy(x):
    """Convert tensor to numpy array."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def _binary_mask(label_map: np.ndarray, label: int) -> np.ndarray:
    """Create binary mask for a given label."""
    return (label_map == label).astype(np.uint8)


def _surface(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Extract surface voxels from a binary mask."""
    from scipy.ndimage import binary_erosion, generate_binary_structure
    
    if mask.sum() == 0:
        return mask
    
    structure = generate_binary_structure(mask.ndim, connectivity)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    surface = mask ^ eroded
    
    # Extremely small objects fallback
    if surface.sum() == 0:
        surface = mask
    return surface.astype(np.uint8)


def _surface_distances(a_mask: np.ndarray, b_mask: np.ndarray, voxelspacing=None, connectivity: int = 1):
    """Symmetric surface distances."""
    from scipy.ndimage import distance_transform_edt
    
    a_surf = _surface(a_mask, connectivity)
    b_surf = _surface(b_mask, connectivity)
    
    if a_surf.sum() == 0 or b_surf.sum() == 0:
        return np.array([]), np.array([])
    
    b_dt = distance_transform_edt(1 - b_surf, sampling=voxelspacing)
    a_dt = distance_transform_edt(1 - a_surf, sampling=voxelspacing)
    
    d_a2b = b_dt[a_surf > 0]
    d_b2a = a_dt[b_surf > 0]
    
    return d_a2b.astype(np.float64), d_b2a.astype(np.float64)


def calculate_assd(pred, target, voxelspacing=None, connectivity: int = 1, voi_labels=None) -> float:
    """VOI-wise Average Symmetric Surface Distance (ASSD).

    Computes ASSD for each VOI label separately and returns the mean.
    Only VOIs present in GT are considered.
    """
    labels = np.asarray(voi_labels if voi_labels is not None else LPBA_VOI_LABELS)

    pred_np = _to_numpy(pred)[0, 0]
    tgt_np = _to_numpy(target)[0, 0]

    assd_list = []
    for lbl in labels:
        pred_i = _binary_mask(pred_np, lbl)
        tgt_i = _binary_mask(tgt_np, lbl)

        # Skip labels not in GT
        if tgt_i.sum() == 0:
            continue

        # If prediction is empty for this label
        if pred_i.sum() == 0:
            assd_list.append(np.inf)
            continue

        d1, d2 = _surface_distances(pred_i, tgt_i, voxelspacing, connectivity)
        if d1.size == 0 or d2.size == 0:
            assd_list.append(np.inf)
            continue

        assd_list.append((d1.mean() + d2.mean()) / 2.0)

    # Filter out infinite values
    valid = [v for v in assd_list if np.isfinite(v)]
    if len(valid) == 0:
        return float('inf')
    return float(np.mean(valid))


def calculate_hd95(pred, target, voxelspacing=None, connectivity: int = 1, voi_labels=None) -> float:
    """VOI-wise 95th percentile Hausdorff Distance.

    Computes HD95 for each VOI label separately and returns the mean.
    Only VOIs present in GT are considered.
    """
    labels = np.asarray(voi_labels if voi_labels is not None else LPBA_VOI_LABELS)

    pred_np = _to_numpy(pred)[0, 0]
    tgt_np = _to_numpy(target)[0, 0]

    hd_list = []
    for lbl in labels:
        pred_i = _binary_mask(pred_np, lbl)
        tgt_i = _binary_mask(tgt_np, lbl)

        # Skip labels not in GT
        if tgt_i.sum() == 0:
            continue

        # If prediction is empty for this label
        if pred_i.sum() == 0:
            hd_list.append(np.inf)
            continue

        d1, d2 = _surface_distances(pred_i, tgt_i, voxelspacing, connectivity)
        if d1.size == 0 or d2.size == 0:
            hd_list.append(np.inf)
            continue

        hd = max(np.percentile(d1, 95), np.percentile(d2, 95))
        hd_list.append(hd)

    # Filter out infinite values
    valid = [v for v in hd_list if np.isfinite(v)]
    if len(valid) == 0:
        return float('inf')
    return float(np.mean(valid))


# =====================
# Visualization helpers
# =====================
def _mid_slice(vol: np.ndarray, plane: str):
    """Return a middle slice in the requested plane."""
    if plane == 'axial':  # z
        idx = vol.shape[0] // 2
        return vol[idx, :, :]
    if plane == 'coronal':  # y
        idx = vol.shape[1] // 2
        return vol[:, idx, :]
    if plane == 'sagittal':  # x
        idx = vol.shape[2] // 2
        return vol[:, :, idx]
    raise ValueError(f'Unsupported plane: {plane}')


def _flow_components_for_plane(flow_3d: np.ndarray, plane: str):
    """Select 2D flow vectors (u, v) for plotting on a plane slice.

    flow_3d shape: [3, D, H, W] in (z, y, x) displacement order.
    """
    fz, fy, fx = flow_3d[0], flow_3d[1], flow_3d[2]

    if plane == 'axial':
        z = fz.shape[0] // 2
        # on H-W plane, show x/y components
        u = fx[z, :, :]
        v = fy[z, :, :]
    elif plane == 'coronal':
        y = fy.shape[1] // 2
        # on D-W plane, show x/z components
        u = fx[:, y, :]
        v = fz[:, y, :]
    elif plane == 'sagittal':
        x = fx.shape[2] // 2
        # on D-H plane, show y/z components
        u = fy[:, :, x]
        v = fz[:, :, x]
    else:
        raise ValueError(f'Unsupported plane: {plane}')

    return u, v


def save_sample_visualizations(x, y, x_def, flow, out_dir, prefix='sample', planes=('axial',), stride=8):
    """Save brain slice comparison and 2D flow visualization.

    Args:
        x: moving image tensor, shape [B,1,D,H,W]
        y: fixed image tensor, shape [B,1,D,H,W]
        x_def: warped moving image tensor, shape [B,1,D,H,W]
        flow: displacement tensor, shape [B,3,D,H,W]
        out_dir: output directory
        prefix: filename prefix
        planes: iterable of {'axial','coronal','sagittal'}
        stride: subsample stride for quiver arrows

    Returns:
        dict: {'brain': [paths...], 'flow2d': [paths...]}
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    x_np = _to_numpy(x)[0, 0]
    y_np = _to_numpy(y)[0, 0]
    xdef_np = _to_numpy(x_def)[0, 0]
    flow_np = _to_numpy(flow)[0]

    brain_paths = []
    flow_paths = []

    for plane in planes:
        # 1) brain comparison
        mov_sl = _mid_slice(x_np, plane)
        fix_sl = _mid_slice(y_np, plane)
        def_sl = _mid_slice(xdef_np, plane)

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.imshow(mov_sl, cmap='gray')
        ax1.set_title('Moving')
        ax1.axis('off')

        ax2.imshow(fix_sl, cmap='gray')
        ax2.set_title('Fixed')
        ax2.axis('off')

        ax3.imshow(def_sl, cmap='gray')
        ax3.set_title('Warped')
        ax3.axis('off')

        fig.tight_layout()
        brain_path = os.path.join(out_dir, f'{prefix}_{plane}_brain.png')
        fig.savefig(brain_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        brain_paths.append(brain_path)

        # 2) flow quiver
        u, v = _flow_components_for_plane(flow_np, plane)
        h, w = u.shape
        yy, xx = np.mgrid[0:h, 0:w]

        fig2 = plt.figure(figsize=(6, 6))
        ax = fig2.add_subplot(1, 1, 1)
        bg = _mid_slice(xdef_np, plane)
        ax.imshow(bg, cmap='gray')
        ax.quiver(
            xx[::stride, ::stride],
            yy[::stride, ::stride],
            u[::stride, ::stride],
            v[::stride, ::stride],
            color='r',
            angles='xy',
            scale_units='xy',
            scale=1.0,
            width=0.002,
        )
        ax.set_title(f'Flow 2D ({plane})')
        ax.axis('off')
        fig2.tight_layout()

        flow_path = os.path.join(out_dir, f'{prefix}_{plane}_flow2d.png')
        fig2.savefig(flow_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        flow_paths.append(flow_path)

    return {'brain': brain_paths, 'flow2d': flow_paths}
