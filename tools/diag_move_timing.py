import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(REPO, "tools", "uci_engine.exe")

p = subprocess.Popen([EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      text=True, bufsize=1)


def send(line):
    p.stdin.write(line + "\n")
    p.stdin.flush()


def read_until(tag):
    while True:
        l = p.stdout.readline()
        if tag in l:
            return l


send("uci")
read_until("uciok")
send("setoption name Threads value 6")
send("isready")
read_until("readyok")

moves = []
for i in range(6):
    pos_cmd = "position startpos" if not moves else "position startpos moves " + " ".join(moves)
    t0 = time.time()
    send(pos_cmd)
    send("go movetime 800")
    line = read_until("bestmove")
    dt = time.time() - t0
    mv = line.split()[1]
    moves.append(mv)
    print(f"move {i+1}: requested=800ms actual={dt*1000:.0f}ms bestmove={mv}", flush=True)

send("quit")
p.wait()
