import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset as TorchDataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import random
from scipy import ndimage
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
class FundusDataset(TorchDataset):
    _freq_mean_cache = {}

    def __init__(self, client_idx=None, freq_site_idx=None, split='train', data_root='dataset',
                 transform=None, freq_strategy='mean', freq_perturb_alpha=0.5,
                 freq_peer_strategy='all', freq_peer_top_k=1, freq_distance_l=0.01):
        self.transform = transform
        self.client_name = [f'client{i}' for i in range(4)]
        self.client_idx = client_idx
        self.freq_site_index = freq_site_idx or []
        self.freq_strategy = freq_strategy
        self.freq_perturb_alpha = freq_perturb_alpha
        self.freq_peer_strategy = freq_peer_strategy
        self.freq_peer_top_k = freq_peer_top_k
        self.freq_distance_l = freq_distance_l
        self.image_list = []
        self.freq_list_clients = []
        self.freq_mean_clients = []
        self.freq_proto_clients = []

        if split == 'train':
            self.image_list = glob(os.path.join(data_root, self.client_name[client_idx], 'data_npy', '*.npy'))
            for name in self.client_name:
                freq_list = sorted(glob(os.path.join(data_root, name, 'freq_amp_npy', '*.npy')))
                if len(freq_list) == 0:
                    self.freq_list_clients.append([])
                    self.freq_mean_clients.append(None)
                    continue
                self.freq_list_clients.append(freq_list)
                cache_key = (os.path.abspath(data_root), name)
                if cache_key not in FundusDataset._freq_mean_cache:
                    FundusDataset._freq_mean_cache[cache_key] = _compute_mean_amp(freq_list)
                self.freq_mean_clients.append(FundusDataset._freq_mean_cache[cache_key])
            self.freq_proto_clients = list(self.freq_mean_clients)
            self.freq_site_index = self._select_frequency_peers(self.freq_site_index)
        print(f"total {len(self.image_list)} slices")
        print(f"client{client_idx} frequency peers: {self.freq_site_index}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        raw_file = self.image_list[idx]

        mask_patches = []

        raw_inp = np.load(raw_file)
        image_patch = raw_inp[..., 0:3]
        mask_patch = raw_inp[..., 3:]
        image_patches = image_patch.copy()

        # image_patches =
        # print (image_patch.dtype)
        # print (mask_patch.dtype)
        disc_contour, disc_bg, cup_contour, cup_bg = _get_coutour_sample(mask_patch)
        # print ('raw', np.min(image_patch), np.max(image_patch))

        # 论文 Section 3.2: "for each external client n≠k, sample an amplitude spectrum"
        # 即遍历所有外部 client, 按指定策略生成目标频谱并变换图像
        # 生成 K-1 个变换图像 (K-1 = len(freq_site_index))
        for tar_freq_domain in self.freq_site_index:
            tar_freq = self._sample_target_freq(tar_freq_domain)
            if tar_freq is None:
                continue
            image_patch_freq = source_to_target_freq(image_patch, tar_freq[...], L=0.01)
            image_patch_freq = np.clip(image_patch_freq, 0, 255)
            image_patches = np.concatenate([image_patches, image_patch_freq], axis=-1)
        image_patches = image_patches.transpose(2, 0, 1)
        mask_patches = mask_patch.transpose(2, 0, 1)
        # contour_bg_mask = np.concatenate(contour_bg_mask, axis=-1)

        sample = {"image": image_patches.astype(np.float32), "label": mask_patches.astype(np.float32),
        "disc_contour":disc_contour, "disc_bg":disc_bg, "cup_contour":cup_contour, "cup_bg":cup_bg}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _sample_target_freq(self, tar_freq_domain):
        freq_list = self.freq_list_clients[tar_freq_domain]
        mean_freq = self.freq_mean_clients[tar_freq_domain]
        proto_freq = self.freq_proto_clients[tar_freq_domain]

        if len(freq_list) == 0:
            return None

        if self.freq_strategy == 'random':
            return np.load(random.choice(freq_list)).astype(np.float32)

        if mean_freq is None:
            return None

        if self.freq_strategy == 'mean':
            return mean_freq

        if self.freq_strategy == 'mean_perturb':
            sample_freq = np.load(random.choice(freq_list)).astype(np.float32)
            target_freq = mean_freq + self.freq_perturb_alpha * (sample_freq - mean_freq)
            return np.maximum(target_freq, 0.0).astype(np.float32)

        if self.freq_strategy == 'ema_mean':
            return proto_freq

        if self.freq_strategy == 'ema_mean_perturb':
            sample_freq = np.load(random.choice(freq_list)).astype(np.float32)
            target_freq = proto_freq + self.freq_perturb_alpha * (sample_freq - proto_freq)
            return np.maximum(target_freq, 0.0).astype(np.float32)

        raise ValueError('Unsupported freq_strategy: {}'.format(self.freq_strategy))

    def get_initial_freq_prototypes(self):
        prototypes = []
        for mean_freq in self.freq_mean_clients:
            prototypes.append(None if mean_freq is None else mean_freq.copy())
        return prototypes

    def set_freq_prototypes(self, freq_proto_clients):
        self.freq_proto_clients = freq_proto_clients

    def sample_client_freq_mean(self, client_idx, sample_size):
        freq_list = self.freq_list_clients[client_idx]
        if len(freq_list) == 0:
            return None

        if sample_size <= 0 or sample_size >= len(freq_list):
            sampled_freq_list = freq_list
        else:
            sampled_freq_list = random.sample(freq_list, sample_size)
        return _compute_mean_amp(sampled_freq_list)

    def _select_frequency_peers(self, candidate_indices):
        if self.freq_peer_strategy == 'all' or len(candidate_indices) <= self.freq_peer_top_k:
            return candidate_indices

        src_mean = self.freq_mean_clients[self.client_idx]
        if src_mean is None:
            return candidate_indices

        scored_candidates = []
        for candidate_idx in candidate_indices:
            candidate_mean = self.freq_mean_clients[candidate_idx]
            if candidate_mean is None:
                continue
            distance = _low_freq_log_l2_distance(src_mean, candidate_mean, self.freq_distance_l)
            scored_candidates.append((candidate_idx, distance))

        if len(scored_candidates) == 0:
            return candidate_indices

        reverse = self.freq_peer_strategy == 'farthest'
        scored_candidates = sorted(scored_candidates, key=lambda item: item[1], reverse=reverse)
        top_k = max(1, min(self.freq_peer_top_k, len(scored_candidates)))
        return [candidate_idx for candidate_idx, _ in scored_candidates[:top_k]]


def _compute_mean_amp(freq_list):
    mean_amp = None
    for idx, freq_path in enumerate(freq_list):
        freq_amp = np.load(freq_path).astype(np.float32)
        if mean_amp is None:
            mean_amp = np.zeros_like(freq_amp, dtype=np.float32)
        mean_amp += (freq_amp - mean_amp) / float(idx + 1)
    return mean_amp


def _low_freq_log_l2_distance(amp_a, amp_b, L=0.01):
    a = np.fft.fftshift(np.log1p(amp_a), axes=(-2, -1))
    b = np.fft.fftshift(np.log1p(amp_b), axes=(-2, -1))

    _, h, w = a.shape
    radius = int(np.floor(np.amin((h, w)) * L))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))

    h1 = c_h - radius
    h2 = c_h + radius + 1
    w1 = c_w - radius
    w2 = c_w + radius + 1

    diff = a[:, h1:h2, w1:w2] - b[:, h1:h2, w1:w2]
    return float(np.sqrt(np.mean(diff * diff)))


def _get_coutour_sample(y_true):
    disc_mask = np.expand_dims(y_true[..., 0], axis=2)

    disc_erosion = ndimage.binary_erosion(disc_mask[..., 0], iterations=1).astype(disc_mask.dtype)
    disc_dilation = ndimage.binary_dilation(disc_mask[..., 0], iterations=5).astype(disc_mask.dtype)
    disc_contour = np.expand_dims(disc_mask[..., 0] - disc_erosion, axis = 2)
    disc_bg = np.expand_dims(disc_dilation - disc_mask[..., 0], axis = 2)
    cup_mask = np.expand_dims(y_true[..., 1], axis=2)

    cup_erosion = ndimage.binary_erosion(cup_mask[..., 0], iterations=1).astype(cup_mask.dtype)
    cup_dilation = ndimage.binary_dilation(cup_mask[..., 0], iterations=5).astype(cup_mask.dtype)
    cup_contour = np.expand_dims(cup_mask[..., 0] - cup_erosion, axis = 2)
    cup_bg = np.expand_dims(cup_dilation - cup_mask[..., 0], axis = 2)

    return [disc_contour.transpose(2, 0, 1), disc_bg.transpose(2, 0, 1), cup_contour.transpose(2, 0, 1), cup_bg.transpose(2, 0, 1)]

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    """
    Continuous frequency space interpolation (论文 Eq.2).
    对低频区域: A_new = (1-λ)*A_src + λ*A_trg
    对高频区域: 保持 A_src 不变
    λ 从 [0.1, 1.0] 动态采样以实现连续插值
    """
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # 论文核心: 动态采样插值比率 λ ∈ [0.1, 1.0], 实现连续频率空间插值
    lam = random.randint(1,10)/10.0

    # 论文 Eq.2: 低频区域进行连续插值, 而非直接替换
    a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * (1 - lam) + a_trg[:,h1:h2,w1:w2] * lam

    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)
