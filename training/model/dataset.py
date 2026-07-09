import numpy as np
import torch


class CsrSplit:
    def __init__(self, npz=None, prefix=None):
        if npz is None:
            return
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


def _concat_csr(splits):
    """Physically concatenate several CsrSplits into one fast CsrSplit.
    `splits` may repeat the same split object to oversample it."""
    out = CsrSplit()
    stm_idx_parts, ntm_idx_parts = [], []
    stm_off_parts, ntm_off_parts = [], []
    tgt_parts, buck_parts = [], []
    stm_base = 0
    ntm_base = 0
    first = True
    for sp in splits:
        stm_idx_parts.append(sp.stm_idx)
        ntm_idx_parts.append(sp.ntm_idx)
        # offsets: keep the leading 0 only for the first split, shift the rest
        so = sp.stm_off if first else sp.stm_off[1:]
        no = sp.ntm_off if first else sp.ntm_off[1:]
        stm_off_parts.append(so + stm_base)
        ntm_off_parts.append(no + ntm_base)
        stm_base += int(sp.stm_off[-1])
        ntm_base += int(sp.ntm_off[-1])
        tgt_parts.append(sp.targets)
        buck_parts.append(sp.buckets)
        first = False
    out.stm_idx = np.concatenate(stm_idx_parts)
    out.ntm_idx = np.concatenate(ntm_idx_parts)
    out.stm_off = np.concatenate(stm_off_parts)
    out.ntm_off = np.concatenate(ntm_off_parts)
    out.targets = np.concatenate(tgt_parts)
    out.buckets = np.concatenate(buck_parts)
    out.n = len(out.targets)
    return out


def load_dataset(path):
    npz = np.load(path)
    return CsrSplit(npz, "train"), CsrSplit(npz, "val")


def load_dataset_with_aux(path, aux_path, aux_repeat):
    base_train, base_val = load_dataset(path)
    aux_npz = np.load(aux_path)
    aux_train = CsrSplit(aux_npz, "train")
    merged_train = _concat_csr([base_train] + [aux_train] * aux_repeat)
    return merged_train, base_val
