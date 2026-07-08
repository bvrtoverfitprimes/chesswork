import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0]
PY = os.path.join(REPO, "training", ".venv", "Scripts", "python.exe")
os.chdir(REPO)

TMP_DIR = r"C:\Users\notu7\AppData\Local\Temp"


def tmp(name):
    return os.path.join(TMP_DIR, name)


HIDDEN = 1024
PHASE1_DATASET_INPUT = "training/data/raw/shallow2_labeled.jsonl"  # ~4.04M depth-2-graded positions
PHASE1_MAX_MIN = 40

N_SELFPLAY_WORKERS = 3
PHASE2_MIN = 180  # 3 hours, per user instruction
CYCLE_MIN = 25

PHASE3_WORKERS = 6
PHASE3_DEPTH = 4
PHASE3_OUT = "training/data/raw/shallow4_labeled.jsonl"

PHASE4_MAX_MIN = 60

LOG = open(tmp("orchestrate_4phase.log"), "a", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


def run(cmd, **kw):
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def phase1():
    log("=== PHASE 1: train initial model on existing ~4.04M depth-2-graded positions ===")
    run([PY, "training/data/build_dataset.py",
         "--input", PHASE1_DATASET_INPUT,
         "--out", "training/data/processed/phase1_dataset.npz"])
    run([PY, "training/model/train.py",
         "--dataset", "training/data/processed/phase1_dataset.npz",
         "--hidden", str(HIDDEN),
         "--epochs", "100",
         "--early-stop-patience", "4",
         "--max-minutes", str(PHASE1_MAX_MIN),
         "--out", "training/checkpoints/phase1.pt"])
    log("phase 1 complete -> training/checkpoints/phase1.pt")
    return "training/checkpoints/phase1.pt"


def phase2(current_ckpt):
    log(f"=== PHASE 2: {PHASE2_MIN}min of synthetic self-play generation + depth-{4} grading, "
        f"incremental fine-tune every {CYCLE_MIN}min, warm-started from {current_ckpt} ===")
    stream_paths = [f"training/data/raw/phase2_stream_w{i}.jsonl" for i in range(N_SELFPLAY_WORKERS)]

    procs = []
    for i, path in enumerate(stream_paths):
        p = subprocess.Popen([PY, "training/data/selfplay_stream.py",
                               "--minutes", str(PHASE2_MIN),
                               "--out", path,
                               "--worker-id", str(i),
                               "--seed", "200"],
                              stdout=open(tmp(f"phase2_selfplay_w{i}.log"), "w"),
                              stderr=subprocess.STDOUT)
        procs.append(p)

    offsets = {p: 0 for p in stream_paths}
    cycle = 0
    t_start = time.time()
    deadline = t_start + PHASE2_MIN * 60

    while time.time() < deadline - 60:
        sleep_s = min(CYCLE_MIN * 60, max(30, deadline - time.time()))
        time.sleep(sleep_s)

        cycle += 1
        cycle_jsonl = f"training/data/raw/phase2_cycle{cycle}.jsonl"
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
            log(f"phase2 cycle {cycle}: only {total_new} new positions, skipping this fine-tune cycle")
            continue

        log(f"phase2 cycle {cycle}: {total_new} new positions, building dataset + fine-tuning...")
        try:
            run([PY, "training/data/build_dataset.py",
                 "--input", cycle_jsonl,
                 "--out", f"training/data/processed/phase2_cycle{cycle}.npz"])
            out_ckpt = f"training/checkpoints/phase2_cycle{cycle}.pt"
            run([PY, "training/model/train.py",
                 "--dataset", f"training/data/processed/phase2_cycle{cycle}.npz",
                 "--hidden", str(HIDDEN),
                 "--init-from", current_ckpt,
                 "--lr", "3e-4",
                 "--epochs", "3",
                 "--early-stop-patience", "2",
                 "--max-minutes", str(CYCLE_MIN - 3),
                 "--out", out_ckpt])
            current_ckpt = out_ckpt
            log(f"phase2 cycle {cycle}: checkpoint -> {out_ckpt}")
        except subprocess.CalledProcessError as e:
            log(f"phase2 cycle {cycle}: FAILED ({e}), keeping previous checkpoint and continuing")

    log("phase 2 loop finished, stopping self-play workers")
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()

    log(f"phase 2 complete -> {current_ckpt}")
    return current_ckpt


def phase3():
    log(f"=== PHASE 3: regrade full 6.12M base corpus at depth={PHASE3_DEPTH} ===")
    run([PY, "training/data/relabel_shallow.py",
         "--depth", str(PHASE3_DEPTH),
         "--workers", str(PHASE3_WORKERS),
         "--out", PHASE3_OUT])
    log(f"phase 3 complete -> {PHASE3_OUT}")


def phase4(current_ckpt):
    log(f"=== PHASE 4: final fine-tune on depth-{PHASE3_DEPTH}-regraded 6.12M corpus, "
        f"warm-started from {current_ckpt} ===")
    run([PY, "training/data/build_dataset.py",
         "--input", PHASE3_OUT,
         "--out", "training/data/processed/phase4_dataset.npz"])
    run([PY, "training/model/train.py",
         "--dataset", "training/data/processed/phase4_dataset.npz",
         "--hidden", str(HIDDEN),
         "--init-from", current_ckpt,
         "--lr", "5e-4",
         "--epochs", "100",
         "--early-stop-patience", "4",
         "--max-minutes", str(PHASE4_MAX_MIN),
         "--out", "training/checkpoints/final_model.pt"])
    log("phase 4 complete -> training/checkpoints/final_model.pt")
    return "training/checkpoints/final_model.pt"


def main():
    t0 = time.time()
    ckpt = phase1()
    ckpt = phase2(ckpt)
    phase3()
    ckpt = phase4(ckpt)

    total_hours = (time.time() - t0) / 3600
    log(f"ALL 4 PHASES DONE in {total_hours:.2f}h. final checkpoint: {ckpt}")
    with open("training/checkpoints/LATEST_4PHASE.txt", "w") as f:
        f.write(ckpt + "\n")


if __name__ == "__main__":
    main()
