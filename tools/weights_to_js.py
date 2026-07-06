import sys

IN_SIZE = 20
H1_SIZE = 128
H2_SIZE = 64
H3_SIZE = 32


def read_floats(path):
    with open(path) as f:
        return [float(x) for x in f.read().split()]


def take(vals, idx, count):
    return vals[idx:idx + count], idx + count


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "engine/human_limit/weights.txt"
    vals = read_floats(weights_path)
    idx = 0

    w1 = []
    for _ in range(H1_SIZE):
        row, idx = take(vals, idx, IN_SIZE)
        w1.append(row)
    b1, idx = take(vals, idx, H1_SIZE)

    w2 = []
    for _ in range(H2_SIZE):
        row, idx = take(vals, idx, H1_SIZE)
        w2.append(row)
    b2, idx = take(vals, idx, H2_SIZE)

    w3 = []
    for _ in range(H3_SIZE):
        row, idx = take(vals, idx, H2_SIZE)
        w3.append(row)
    b3, idx = take(vals, idx, H3_SIZE)

    w4, idx = take(vals, idx, H3_SIZE)
    b4 = vals[idx]
    idx += 1

    assert idx == len(vals), f"expected to consume {idx} values, file has {len(vals)}"

    def fmt_matrix(m):
        return "[" + ",".join("[" + ",".join(repr(v) for v in row) + "]" for row in m) + "]"

    def fmt_vector(v):
        return "[" + ",".join(repr(x) for x in v) + "]"

    print(f"const NN_IN={IN_SIZE}, NN_H1={H1_SIZE}, NN_H2={H2_SIZE}, NN_H3={H3_SIZE};")
    print(f"const NN_W1={fmt_matrix(w1)}, NN_B1={fmt_vector(b1)}, "
          f"NN_W2={fmt_matrix(w2)}, NN_B2={fmt_vector(b2)}, "
          f"NN_W3={fmt_matrix(w3)}, NN_B3={fmt_vector(b3)}, "
          f"NN_W4={fmt_vector(w4)}, NN_B4={b4!r};")


if __name__ == "__main__":
    main()
