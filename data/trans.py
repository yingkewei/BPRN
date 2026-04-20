# import math
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform


class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

class RandomFlip(Base):
    """
    与 UTSRMorph 对齐的 3D 随机翻转：
    - 在 x/y/z 三个轴上分别独立采样是否翻转
    - 通过 Base.__call__ 的 reuse 机制，保证同一对样本 (x, y) 使用相同的翻转参数
    """

    def __init__(self, axis=0):
        # 保留 axis 参数以兼容旧调用方式，实际固定对 (1,2,3) 轴执行翻转（N,H,W,D）
        self.axis = (1, 2, 3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        self.z_buffer = np.random.choice([True, False])
        return list(shape)

    def tf(self, img, k=0):
        if self.x_buffer:
            img = np.flip(img, axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img, axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img, axis=self.axis[2])
        return img


class RandomGamma(Base):
    """
    随机 Gamma 强度变换（适用于 MRI 强度扰动）。

    - 支持对 pair (x, y) 采样相同或不同的 gamma
    - 假设输入为 NCHWD (本项目中通常是 [1, H, W, D])
    """
    def __init__(self, gamma_range=(0.7, 1.5), same_on_pair=True, clip=True, eps=1e-6):
        self.gmin = float(gamma_range[0])
        self.gmax = float(gamma_range[1])
        self.same_on_pair = bool(same_on_pair)
        self.clip = bool(clip)
        self.eps = float(eps)
        self._gammas = {}

    def sample(self, *shape):
        g0 = np.random.uniform(self.gmin, self.gmax)
        if self.same_on_pair:
            g1 = g0
        else:
            g1 = np.random.uniform(self.gmin, self.gmax)
        self._gammas = {0: float(g0), 1: float(g1)}
        return list(shape)

    def tf(self, img, k=0):
        if img.ndim < 4:
            return img
        g = self._gammas.get(k, self._gammas.get(0, 1.0))
        x = img.astype(np.float32, copy=False)
        x_min = float(x.min())
        x_max = float(x.max())
        if (x_max - x_min) < self.eps:
            return img
        x01 = (x - x_min) / (x_max - x_min + self.eps)
        x01 = np.power(x01, g)
        out = x01 * (x_max - x_min) + x_min
        if self.clip:
            out = np.clip(out, x_min, x_max)
        return out.astype(img.dtype, copy=False)


class RandomNoise(Base):
    """
    随机加性高斯噪声（相对动态范围）。
    - sigma = U(sigma_range) * (max - min)
    """
    def __init__(self, sigma_range=(0.0, 0.03), same_on_pair=False, eps=1e-6):
        self.smin = float(sigma_range[0])
        self.smax = float(sigma_range[1])
        self.same_on_pair = bool(same_on_pair)
        self.eps = float(eps)
        self._sigmas = {}

    def sample(self, *shape):
        s0 = np.random.uniform(self.smin, self.smax)
        if self.same_on_pair:
            s1 = s0
        else:
            s1 = np.random.uniform(self.smin, self.smax)
        self._sigmas = {0: float(s0), 1: float(s1)}
        return list(shape)

    def tf(self, img, k=0):
        if img.ndim < 4:
            return img
        sigma_rel = self._sigmas.get(k, self._sigmas.get(0, 0.0))
        if sigma_rel <= 0:
            return img
        x = img.astype(np.float32, copy=False)
        x_min = float(x.min())
        x_max = float(x.max())
        scale = max(self.eps, (x_max - x_min))
        noise = np.random.normal(loc=0.0, scale=sigma_rel * scale, size=x.shape).astype(np.float32)
        out = x + noise
        return out.astype(img.dtype, copy=False)


class RandomAffine3D(Base):
    """
    3D 随机仿射（小幅度），支持只对 moving 或 fixed 应用：
    - mode = 'same' | 'moving_only' | 'fixed_only'
    - moving/fixed 对应 pair 中 k=0 / k=1

    注意：这是“轻量级增强”，用于增加初始错位/插值鲁棒性。
    """
    def __init__(
        self,
        degrees=5.0,
        translate=2.0,
        scale=0.05,
        mode='moving_only',
        order=1,
        cval=0.0,
    ):
        self.deg = float(degrees)
        self.trans = float(translate)
        self.scale = float(scale)
        self.mode = str(mode)
        self.order = int(order)
        self.cval = float(cval)
        self._A = None
        self._t = None

    def sample(self, *shape):
        # sample small rotations (degrees) around x/y/z
        rx, ry, rz = np.deg2rad(np.random.uniform(-self.deg, self.deg, size=3))
        sx = 1.0 + np.random.uniform(-self.scale, self.scale)
        sy = 1.0 + np.random.uniform(-self.scale, self.scale)
        sz = 1.0 + np.random.uniform(-self.scale, self.scale)
        tx, ty, tz = np.random.uniform(-self.trans, self.trans, size=3)

        cx, sxr = np.cos(rx), np.sin(rx)
        cy, syr = np.cos(ry), np.sin(ry)
        cz, szr = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sxr],
                       [0, sxr, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, syr],
                       [0, 1, 0],
                       [-syr, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -szr, 0],
                       [szr, cz, 0],
                       [0, 0, 1]], dtype=np.float32)

        R = (Rz @ Ry @ Rx).astype(np.float32)
        S = np.diag([sx, sy, sz]).astype(np.float32)
        A = (R @ S).astype(np.float32)  # output->input mapping matrix (we'll invert below)

        self._A = A
        self._t = np.array([tx, ty, tz], dtype=np.float32)
        return list(shape)

    def _should_apply(self, k):
        if self.mode == 'same':
            return True
        if self.mode == 'moving_only':
            return k == 0
        if self.mode == 'fixed_only':
            return k == 1
        return False

    def tf(self, img, k=0):
        if not self._should_apply(k):
            return img
        if img.ndim < 4:
            return img

        # img: [N, H, W, D] or [C, H, W, D] (project uses [1, H, W, D])
        x = img
        # apply per-channel (first dim)
        out = np.empty_like(x)
        # center in voxel coordinates
        H, W, D = x.shape[1], x.shape[2], x.shape[3]
        center = np.array([(H - 1) / 2.0, (W - 1) / 2.0, (D - 1) / 2.0], dtype=np.float32)

        # scipy affine_transform maps output coords to input coords via matrix M and offset b:
        # input_coord = M @ output_coord + b
        # We want to apply transform around center: x' = A (x - c) + c + t  (forward)
        # For affine_transform we need inverse mapping (output->input):
        # x = A^{-1} (x' - c - t) + c
        A = self._A
        t = self._t
        A_inv = np.linalg.inv(A).astype(np.float32)
        offset = (center - A_inv @ (center + t)).astype(np.float32)

        for ch in range(x.shape[0]):
            out[ch] = affine_transform(
                x[ch],
                matrix=A_inv,
                offset=offset,
                order=self.order,
                mode='constant',
                cval=self.cval,
                prefilter=(self.order > 1),
            )
        return out


class Seg_norm(Base):
    def __init__(self, dataset='LPBA'):
        dataset = str(dataset).upper()
        if dataset == 'IXI':
            self.seg_table = np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
            ], dtype=np.int16)
        elif dataset in ('ABD', 'ABDOMEN', 'ABDOMENCTCT', 'ABDOMEN_CTCT'):
            self.seg_table = np.array([0] + list(range(1, 14)), dtype=np.int16)
        elif dataset == 'OASIS':
            self.seg_table = np.array([
                0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51,
                52, 53, 54, 58, 60, 62, 63, 72, 77, 80, 85, 251, 252,
                253, 254, 255
            ], dtype=np.int16)
        else:
            self.seg_table = np.array([
                0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65,
                66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                101, 102, 121, 122, 161, 162, 163, 164, 165, 166
            ], dtype=np.int16)

    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out


class Resample3D(Base):
    def __init__(self, out_shape, seg_indices=None):
        self.out_shape = tuple(int(v) for v in out_shape)
        self.seg_indices = set(seg_indices) if seg_indices is not None else set()

    def tf(self, img, k=0):
        if img.ndim < 4:
            return img

        in_shape = img.shape[1:4]
        if tuple(in_shape) == self.out_shape:
            return img

        zoom_factors = [1.0] + [o / i for o, i in zip(self.out_shape, in_shape)]
        order = 0 if k in self.seg_indices else 1
        return zoom(img, zoom=zoom_factors, order=order)


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)

