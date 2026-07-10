import argparse
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.net import NnueNet, NUM_FEATURES

MAGIC = 0x4E4E5546


def export(checkpoint_path, out_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden = ckpt["hidden"]
    model = NnueNet(hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    num_buckets = model.num_buckets

    with open(out_path, "wb") as f:
        f.write(struct.pack("<iii", MAGIC, hidden, num_buckets))

        emb = model.embedding.weight.detach().numpy()
        assert emb.shape == (NUM_FEATURES, hidden)
        f.write(emb.astype("<f4").tobytes())

        for bucket in range(num_buckets):
            for w_param, b_param in ((model.fc1_w, model.fc1_b), (model.fc2_w, model.fc2_b),
                                      (model.fc3_w, model.fc3_b)):
                w = w_param[bucket].detach().numpy()
                b = b_param[bucket].detach().numpy()
                f.write(w.astype("<f4").tobytes())
                f.write(b.astype("<f4").tobytes())

    print(f"exported hidden={hidden} num_buckets={num_buckets} to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/checkpoints/m2.pt")
    parser.add_argument("--out", default="engine/limit/nnue_weights.bin")
    args = parser.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
