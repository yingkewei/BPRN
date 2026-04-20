import os, glob
import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import numpy as np


def pkload(fname):
    """
    安全加载 pkl 文件，同时在出错时给出更详细的提示，方便定位数据问题。
    """
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"\n❌ 加载文件失败: {fname}")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        if os.path.exists(fname):
            print(f"   文件大小: {os.path.getsize(fname)} bytes")
        else:
            print(f"   注意: 文件不存在，请检查路径是否正确")
        print(f"\n💡 建议: 检查数据集是否完整、路径是否正确，或重新生成该 pkl 文件")
        raise

class LPBABrainDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class LPBABrainRandomPairDatasetS2S(Dataset):
    """Random-pair dataset to control training steps per epoch.

    Each __getitem__ samples one random (x, y) pair with x != y.
    Dataset length is fixed by pairs_per_epoch, enabling faster epochs.
    """
    def __init__(self, data_path, transforms, pairs_per_epoch=10000):
        self.paths = data_path
        self.transforms = transforms
        self.pairs_per_epoch = int(pairs_per_epoch)
        if len(self.paths) < 2:
            raise ValueError("LPBABrainRandomPairDatasetS2S requires at least 2 samples.")

    def __getitem__(self, index):
        x_index = np.random.randint(0, len(self.paths))
        y_index = np.random.randint(0, len(self.paths) - 1)
        if y_index >= x_index:
            y_index += 1

        path_x = self.paths[x_index]
        path_y = self.paths[y_index]

        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)

        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return self.pairs_per_epoch


class OASISBrainKRandomPairDatasetS2S(Dataset):
    """OASIS-style training dataset: each source is used, with k random targets per source per epoch.

    - Epoch size is N * targets_per_source.
    - For each item, source index is deterministic from global index.
    - Target is sampled randomly from all other cases (target != source).
    """

    def __init__(self, data_path, transforms, targets_per_source=1):
        self.paths = data_path
        self.transforms = transforms
        self.targets_per_source = int(targets_per_source)
        if len(self.paths) < 2:
            raise ValueError("OASISBrainKRandomPairDatasetS2S requires at least 2 samples.")
        if self.targets_per_source < 1:
            raise ValueError("targets_per_source must be >= 1.")

    def __getitem__(self, index):
        n = len(self.paths)
        x_index = index // self.targets_per_source
        x_index = int(x_index % n)

        y_index = np.random.randint(0, n - 1)
        if y_index >= x_index:
            y_index += 1

        path_x = self.paths[x_index]
        path_y = self.paths[y_index]

        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)

        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths) * self.targets_per_source


class LPBABrainInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index%(len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        # print(os.path.basename(path_x), os.path.basename(path_y))
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class OASISBrainInferDatasetS2S(Dataset):
    """
    OASIS 验证 / 推理数据集：
    - 输入文件为 Test 目录下的 pair pkl
    - 每个 pkl 包含 (x, y, x_seg, y_seg)
    """

    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(x_seg),
            torch.from_numpy(y_seg),
        )
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class IXIBrainDatasetS2S(Dataset):
    """
    IXI 训练用数据集（对齐 UTSRMorph 的配置）：
    - x 始终来自 atlas.pkl（moving image）
    - y 来自 Train 目录中的某个 subject（fixed image）
    - 只返回 (x, y)，不在训练阶段使用分割
    """

    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        # atlas 作为 moving image，与每一个 subject 配准（与 UTSRMorph 保持一致）
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)

        # transforms 期望输入为 nhwtc，这里先在最前面加 batch 维
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDatasetS2S(Dataset):
    """
    IXI 验证 / 推理用数据集（对齐 UTSRMorph 的配置）：
    - x 始终来自 atlas.pkl
    - y 为某个 Val subject
    - 同时返回 x_seg, y_seg 以便在验证阶段计算 Dice / ASSD / HD95
    """

    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        # 与 UTSRMorph 一致：对 (x, x_seg) 和 (y, y_seg) 分别做相同的变换
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(x_seg),
            torch.from_numpy(y_seg),
        )
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class LPBABrainHalfDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    def half_pair(self,pair):
        return pair[0][::2,::2,::2], pair[1][::2,::2,::2]

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = self.half_pair(pkload(path_x))
        y, y_seg = self.half_pair(pkload(path_y))

        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class LPBABrainHalfInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    def half_pair(self,pair):
        return pair[0][::2,::2,::2], pair[1][::2,::2,::2]
    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index%(len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        # print(os.path.basename(path_x), os.path.basename(path_y))
        x, x_seg = self.half_pair(pkload(path_x))
        y, y_seg = self.half_pair(pkload(path_y))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)