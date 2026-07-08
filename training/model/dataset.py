import numpy as np
import torch


class CsrSplit:
    def __init__(self, npz, prefix):
        self.stm_idx = npz[f"{prefix}_stm_idx"]
        self.stm_off = npz[f"{prefix}_stm_off"]
        self.ntm_idx = npz[f"{prefix}_ntm_idx"]
        self.ntm_off = npz[f"{prefix}_ntm_off"]
        self.targets = npz[f"{prefix}_targets"]
        self.buckets = npz[f"{prefix}_buckets"]
        self.n = len(self.targets)

    def _gather_batch(self, indices_arr, offsets_arr, sample_ids):
        chunks = [indices_arr[offsets_arr[i]:offsets_arr[i + 1]] for i in sample_ids]
        lengths = [len(c) for c in chunks]
        flat = np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)
        local_offsets = np.zeros(len(sample_ids), dtype=np.int64)
        np.cumsum(lengths[:-1], out=local_offsets[1:]) if len(lengths) > 1 else None
        return torch.from_numpy(flat.astype(np.int64)), torch.from_numpy(local_offsets)

    def batches(self, batch_size, shuffle, rng=None):
        order = np.arange(self.n)
        if shuffle:
            rng.shuffle(order)
        for start in range(0, self.n, batch_size):
            sample_ids = order[start:start + batch_size]
            stm_idx, stm_off = self._gather_batch(self.stm_idx, self.stm_off, sample_ids)
            ntm_idx, ntm_off = self._gather_batch(self.ntm_idx, self.ntm_off, sample_ids)
            targets = torch.from_numpy(self.targets[sample_ids])
            buckets = torch.from_numpy(self.buckets[sample_ids].astype(np.int64))
            yield stm_idx, stm_off, ntm_idx, ntm_off, buckets, targets


def load_dataset(path):
    npz = np.load(path)
    return CsrSplit(npz, "train"), CsrSplit(npz, "val")
