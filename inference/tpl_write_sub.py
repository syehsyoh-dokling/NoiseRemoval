"""Write x - sum(gT) for every test clip (float32 WAV; clips without occurrences = the input) for pass-2 discovery.

    python tpl_write_sub.py --tpl tpl_v40tpl.pkl --out mix_v40sub
"""
import argparse
import glob
import os
import pickle

import numpy as np
import soundfile as sf

W = os.environ.get("T2_WORK", os.path.dirname(os.path.abspath(__file__)))


def apply(x, occ, T):
    est = np.zeros(len(x), np.float64)
    for st, g, tid in occ:
        t = T[tid]; a, b = max(0, st), min(len(x), st + len(t))
        est[a:b] += g * t[a - st:b - st]
    return (x - est).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpl", nargs="+", required=True)
    ap.add_argument("--src", default="phase2_test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = [pickle.load(open(os.path.join(W, p), "rb")) for p in args.tpl]
    out = os.path.join(W, args.out); os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(W, args.src, "**", "*.wav"), recursive=True))
    n = 0
    for p in files:
        k = os.path.basename(p)
        x, sr = sf.read(p, dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        for d in sets:
            if d["occ"].get(k):
                x = apply(x, d["occ"][k], d["templates"]); n += 1
        sf.write(os.path.join(out, k), x, sr, subtype="FLOAT")
    print(f"wrote {len(files)} clips to {out} ({n} template applications)")


if __name__ == "__main__":
    main()
