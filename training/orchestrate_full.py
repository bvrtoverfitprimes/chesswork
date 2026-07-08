import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0]
PY = os.path.join(REPO, "training", ".venv", "Scripts", "python.exe")
os.chdir(REPO)

# Bash's /tmp (used by the shell that launched the earlier relabeling job) maps to this real
# Windows path, NOT to C:\tmp — native Windows Python resolves "/tmp/x" as drive-root-relative,
# a different location, so all cross-language log/status files must use this explicit path.
TMP_DIR = r"C:\Users\notu7\AppData\Local\Temp"


def tmp(name):
    return os.path.join(TMP_DIR, name)

TOTAL_BUDGET_MIN = 330  # ~5.5h remaining, accounting for time already spent on setup
PHASE2_MIN = 240  # ~4h of continuous self-play generation + incremental fine-tuning, first
FINAL_BUFFER_MIN = 15
N_SELFPLAY_WORKERS = 3
CYCLE_MIN = 25
HIDDEN = 1024

LOG = open(tmp("orchestrate_full.log"), "a", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


def run(cmd, **kw):
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def wait_for_relabel():
    log("waiting for depth-2 relabeling of the base 6.1M corpus to finish (runs in parallel "
        "with phase 2, needed only for the phase-1 fine-tune afterward)...")
    while True:
        if os.path.exists(tmp("relabel.log")):
            with open(tmp("relabel.log")) as f:
                content = f.read()
            if "\ndone:" in content or content.startswith("done:"):
                log("base relabeling finished")
                return
        time.sleep(20)


def phase2_selfplay_and_finetune(t_start, budget_min):
    """Phase 2, now first: continuous Stockfish self-play (2-best-then-random moves, depth-2
    teacher) starting from a freshly initialized network, with incremental fine-tune cycles
    folding in fresh data every CYCLE_MIN minutes. Establishes the base checkpoint."""
    stream_paths = [f"training/data/raw/stream_w{i}.jsonl" for i in range(N_SELFPLAY_WORKERS)]

    log(f"phase 2: launching {N_SELFPLAY_WORKERS} self-play workers for {budget_min:.1f}min "
        f"(depth-2 teacher, 2-best-then-random move pattern)...")
    procs = []
    for i, path in enumerate(stream_paths):
        p = subprocess.Popen([PY, "training/data/selfplay_stream.py",
                               "--minutes", str(budget_min),
                               "--out", path,
                               "--worker-id", str(i),
                               "--seed", "100"],
                              stdout=open(tmp(f"selfplay_w{i}.log"), "w"),
                              stderr=subprocess.STDOUT)
        procs.append(p)

    # Let the very first cycle accumulate a real chunk of data before the first training call
    # (which also has to initialize the network from scratch this time, no warm start yet).
    first_cycle_min = max(CYCLE_MIN, 15)
    offsets = {p: 0 for p in stream_paths}
    cycle = 0
    current_ckpt = None
    phase2_deadline = t_start + budget_min * 60

    while time.time() < phase2_deadline - 60:
        wait_min = first_cycle_min if cycle == 0 else CYCLE_MIN
        sleep_s = min(wait_min * 60, max(30, phase2_deadline - time.time()))
        time.sleep(sleep_s)

        cycle += 1
        cycle_jsonl = f"training/data/raw/cycle{cycle}.jsonl"
        total_new = 0
        with open(cycle_jsonl, "w") as out_f:
            for path in stream_paths:
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    f.seek(offsets[path])
                    new_data = f.read()
                    offsets[path] = f.tell()
                if new_data:
                    out_f.write(new_data)
                    total_new += new_data.count("\n")

        if total_new < 500:
            log(f"cycle {cycle}: only {total_new} new positions, skipping this fine-tune cycle")
            continue

        log(f"cycle {cycle}: {total_new} new self-play positions, building dataset + training...")
        try:
            run([PY, "training/data/build_dataset.py",
                 "--input", cycle_jsonl,
                 "--out", f"training/data/processed/cycle{cycle}.npz"])
            out_ckpt = f"training/checkpoints/shallow2_cycle{cycle}.pt"
            train_cmd = [PY, "training/model/train.py",
                         "--dataset", f"training/data/processed/cycle{cycle}.npz",
                         "--hidden", str(HIDDEN),
                         "--out", out_ckpt]
            if current_ckpt is None:
                # First cycle: no checkpoint yet, train fresh (this establishes the base model).
                train_cmd += ["--lr", "2e-3", "--epochs", "40",
                              "--early-stop-patience", "4",
                              "--max-minutes", str(first_cycle_min - 3)]
            else:
                train_cmd += ["--init-from", current_ckpt, "--lr", "3e-4", "--epochs", "3",
                              "--early-stop-patience", "2",
                              "--max-minutes", str(CYCLE_MIN - 3)]
            run(train_cmd)
            current_ckpt = out_ckpt
            log(f"cycle {cycle}: checkpoint -> {out_ckpt}")
        except subprocess.CalledProcessError as e:
            log(f"cycle {cycle}: FAILED ({e}), keeping previous checkpoint and continuing")

    log("phase 2 loop finished, stopping self-play workers")
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()

    return current_ckpt


def main():
    t_start = time.time()

    current_ckpt = phase2_selfplay_and_finetune(t_start, PHASE2_MIN)
    if current_ckpt is None:
        log("ERROR: phase 2 never produced a checkpoint, aborting")
        return

    elapsed_min = (time.time() - t_start) / 60
    log(f"phase 2 complete after {elapsed_min:.1f}min, checkpoint={current_ckpt}")

    # Phase 1, now second: high-quality fine-tune on the deep-Stockfish-relabeled (depth-2,
    # same teacher depth for consistency) 6.1M real-game + curriculum corpus, warm-started
    # from the phase-2 self-play model.
    wait_for_relabel()

    log("building base dataset from shallow2_labeled.jsonl...")
    run([PY, "training/data/build_dataset.py",
         "--input", "training/data/raw/shallow2_labeled.jsonl",
         "--out", "training/data/processed/shallow2_dataset.npz"])

    remaining_min = max(10, TOTAL_BUDGET_MIN - (time.time() - t_start) / 60 - FINAL_BUFFER_MIN)
    log(f"phase 1 (fine-tune on base corpus): hidden={HIDDEN}, warm-start from {current_ckpt}, "
        f"budget={remaining_min:.1f}min")
    run([PY, "training/model/train.py",
         "--dataset", "training/data/processed/shallow2_dataset.npz",
         "--hidden", str(HIDDEN),
         "--init-from", current_ckpt,
         "--lr", "5e-4",
         "--epochs", "100",
         "--early-stop-patience", "4",
         "--max-minutes", str(remaining_min),
         "--out", "training/checkpoints/shallow2_final.pt"])
    current_ckpt = "training/checkpoints/shallow2_final.pt"

    log(f"ALL DONE. final checkpoint: {current_ckpt}")
    with open("training/checkpoints/LATEST_SHALLOW2.txt", "w") as f:
        f.write(current_ckpt + "\n")


if __name__ == "__main__":
    main()
