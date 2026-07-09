import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import load_dataset, load_dataset_with_aux
from model.net import NnueNet

WDL_LOSS_EXPONENT = 2.6


def combined_loss(pred_logit, target_logit, mse_weight):
    pred_wdl = torch.sigmoid(pred_logit)
    target_wdl = torch.sigmoid(target_logit)
    wdl = torch.mean(torch.abs(pred_wdl - target_wdl) ** WDL_LOSS_EXPONENT)
    if mse_weight <= 0.0:
        return wdl
    # MSE in eval (logit/cp) space gives the network a direct incentive to get
    # magnitudes right on decisive positions, which pure WDL loss squashes away.
    mse = torch.mean((pred_logit - target_logit) ** 2)
    return wdl + mse_weight * mse


def evaluate(model, split, batch_size, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for stm_idx, stm_off, ntm_idx, ntm_off, bucket, tgt in split.batches(batch_size, shuffle=False):
            out = model(stm_idx.to(device), stm_off.to(device), ntm_idx.to(device), ntm_off.to(device),
                        bucket.to(device))
            preds.append(out.cpu().numpy())
            targets.append(tgt.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = float(np.mean(np.abs(preds - targets)))
    corr = float(np.corrcoef(preds, targets)[0, 1])
    return mae, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/data/processed/m1_dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-decay-epoch", type=int, default=6)
    parser.add_argument("--lr-decay-factor", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--out", default="training/checkpoints/m2.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-minutes", type=float, default=0,
                         help="wall-clock budget; 0 disables (epochs/early-stop govern instead)")
    parser.add_argument("--init-from", default="",
                         help="warm-start weights from an existing checkpoint (must match --hidden)")
    parser.add_argument("--mse-weight", type=float, default=0.0,
                         help="weight of eval-space MSE term added to WDL loss (0 = pure WDL)")
    parser.add_argument("--aux-dataset", default="",
                         help="optional decisive/imbalance dataset to oversample")
    parser.add_argument("--aux-repeat", type=int, default=1,
                         help="how many times to repeat the aux dataset per epoch")
    args = parser.parse_args()
    run_start = time.time()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.aux_dataset:
        train_split, val_split = load_dataset_with_aux(args.dataset, args.aux_dataset, args.aux_repeat)
        print(f"oversampling aux dataset {args.aux_dataset} x{args.aux_repeat}; "
              f"merged train size={train_split.n}", flush=True)
    else:
        train_split, val_split = load_dataset(args.dataset)
    model = NnueNet(hidden=args.hidden).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"warm-started from {args.init_from} (epoch {ckpt.get('epoch')}, "
              f"val_mae={ckpt.get('val_mae')})", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay_factor)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for stm_idx, stm_off, ntm_idx, ntm_off, bucket, tgt in train_split.batches(args.batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad()
            out = model(stm_idx.to(device), stm_off.to(device), ntm_idx.to(device), ntm_off.to(device),
                        bucket.to(device))
            loss = combined_loss(out, tgt.to(device), args.mse_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_mae, val_corr = evaluate(model, val_split, args.batch_size, device)
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({"hidden": args.hidden, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_mae": val_mae, "val_corr": val_corr}, args.out)
        else:
            epochs_since_best += 1

        print(f"epoch {epoch}/{args.epochs} train_loss={total_loss / n_batches:.6f} "
              f"val_mae={val_mae:.4f} val_corr={val_corr:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"time={time.time() - t0:.1f}s{'  <-- best, saved' if is_best else ''}", flush=True)

        if epochs_since_best >= args.early_stop_patience:
            print(f"early stopping: no val_mae improvement for {args.early_stop_patience} epochs "
                  f"(best was epoch {best_epoch})")
            break

        if args.max_minutes > 0 and (time.time() - run_start) / 60 >= args.max_minutes:
            print(f"wall-clock budget of {args.max_minutes:.0f}min reached after epoch {epoch}, stopping "
                  f"(best was epoch {best_epoch})")
            break

    print(f"best checkpoint: epoch {best_epoch} (val_mae={best_val_mae:.4f}) saved to {args.out}")


if __name__ == "__main__":
    main()
