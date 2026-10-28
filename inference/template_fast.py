"""Fast blind template discovery + subtraction on the Phase-2 test set (CPU, low RAM).

Same method as template_sub.py (validated: on validation clips whose residual the templates explain >= 85%,
subtraction beats even the in-sample denoiser, 11.78 vs 10.94 dB, +17.3 dB over the input), made feasible for 3178
clips on a laptop with ~2 GB free RAM:
  * matching runs on residuals decimated to 4 kHz, sharded over worker processes (each holds 1/n of the clips);
  * a hit is re-aligned at 16 kHz around the 4 kHz position; full-rate audio is read from disk only for hits;
  * one seed per clip (its strongest 1 s residual window), strongest first; seeds inside explained regions skip.
Per clip: explained = 1 - ||R - sum(gT)||^2 / ||R||^2; if explained >= RULE the output is x - sum(gT), else the
denoiser output y. Output audio is rounded to multiples of 4 (audit-safe packaging is done by t2_avg-style code).

    python template_fast.py --enh enh_v39avg --tag v40tpl
"""
import argparse
import glob
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import decimate, fftconvolve

W = os.environ.get("T2_WORK", os.path.dirname(os.path.abspath(__file__)))
SR, D = 16000, 4
SEED = SR                      # 1 s
EXT = int(1.5 * SR)
FR = 320


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read(p):
    x, sr = sf.read(p, dtype="float32")
    assert sr == SR
    return x.mean(1) if x.ndim > 1 else x


def residual(k, enh):
    mix = os.environ.get("TPL_MIX")                  # pass 2: the mixture after pass-1 subtraction
    if mix:
        x = read(os.path.join(W, mix, k))
    elif os.environ.get("TPL_SRC"):                  # Track 1: an absolute folder of mixtures
        x = read(os.path.join(os.environ["TPL_SRC"], k))
    else:
        x = read(os.path.join(W, "phase2_test", k)) if os.path.exists(os.path.join(W, "phase2_test", k)) else read(PATHS[k])
    y = read(os.path.join(W, enh, k))
    n = min(len(x), len(y))
    return x[:n], y[:n], (x[:n] - y[:n]).astype(np.float32)


def ncc_curve(x, s):
    L = len(s)
    if len(x) < L:
        return np.zeros(0)
    s = s - s.mean(); ns = np.sqrt(float((s.astype(np.float64) ** 2).sum())) + 1e-9
    num = fftconvolve(x, s[::-1], mode="valid")
    c2 = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    c1 = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
    e = c2[L:] - c2[:-L] - (c1[L:] - c1[:-L]) ** 2 / L
    floor = max(1e-6 * L, 0.01 * float(e.max()) if len(e) else 0.0)
    return np.where(e > floor, num / (np.sqrt(np.maximum(e, 1e-12)) * ns), 0.0)


# ---------------- worker side
SHARD = {}
PATHS = {}


def w_init(keys, enh, paths):
    PATHS.update(paths)
    for k in keys:
        _, _, r = residual(k, enh)
        SHARD[k] = decimate(r, D, ftype="fir", zero_phase=True).astype(np.float32)


def w_scan(seed4, thr):
    out = []
    for k, r4 in SHARD.items():
        c = ncc_curve(r4, seed4)
        if len(c):
            p = int(np.argmax(c))
            if c[p] >= thr:
                out.append((k, p * D, float(c[p])))
    return out


# ---------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enh", default="enh_v39avg")
    ap.add_argument("--tag", default="v40tpl")
    ap.add_argument("--thr4", type=float, default=0.70)     # 4 kHz scan threshold
    ap.add_argument("--thr", type=float, default=0.75)      # 16 kHz confirmation threshold
    ap.add_argument("--coh", type=float, default=0.5)
    ap.add_argument("--rule", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    test = sorted(glob.glob(os.path.join(W, "phase2_test", "**", "*.wav"), recursive=True))
    paths = {os.path.basename(p): p for p in test}
    if os.environ.get("TPL_KEYS"):                   # Track 1: explicit clip list + source folder
        ks = json.load(open(os.environ["TPL_KEYS"], encoding="utf-8"))
        paths = {k: os.path.join(os.environ["TPL_SRC"], k) for k in ks}
    PATHS.update(paths)
    keys = sorted(paths)[:args.limit] if args.limit else sorted(paths)
    t0 = time.time()
    seeds = []
    for k in keys:                                   # one seed per clip: strongest 1 s residual window
        _, _, r = residual(k, args.enh)
        if len(r) < SEED:
            continue
        e = np.convolve(r.astype(np.float64) ** 2, np.ones(SEED // 4), mode="same")
        i = int(np.argmax(e))
        a = max(0, min(i - SEED // 2, len(r) - SEED))
        seeds.append((float(e[i]), k, a))
    seeds.sort(reverse=True)
    log(f"{len(keys)} clips, {len(seeds)} seeds in {time.time() - t0:.0f}s")
    shards = [keys[i::args.workers] for i in range(args.workers)]
    expl = {k: [] for k in keys}
    occ = {k: [] for k in keys}
    templates = []

    def explained(k, pos):
        return any(a <= pos < b for a, b in expl[k])

    pools = [ProcessPoolExecutor(1, initializer=w_init, initargs=(sh, args.enh, paths)) for sh in shards]
    try:
        for si, (_, k0, a0) in enumerate(seeds):
            if explained(k0, a0 + SEED // 2):
                continue
            _, _, r0 = residual(k0, args.enh)
            s16 = r0[a0:a0 + SEED]
            s4 = decimate(s16, D, ftype="fir", zero_phase=True).astype(np.float32)
            futs = [p.submit(w_scan, s4, args.thr4) for p in pools]
            cand = [h for f in futs for h in f.result()]
            hits = []
            for k, p4, _ in cand:                     # confirm + re-align at 16 kHz
                if explained(k, p4 + SEED // 2):
                    continue
                _, _, r = residual(k, args.enh)
                lo, hi = max(0, p4 - 2 * D), min(len(r), p4 + SEED + 2 * D)
                c = ncc_curve(r[lo:hi], s16)
                if len(c) and c.max() >= args.thr:
                    hits.append((k, lo + int(np.argmax(c)), float(c.max()), r))
            if len(hits) < 2:
                continue
            Lw = SEED + 2 * EXT
            stack = []
            for k, p, _, r in hits:
                w = np.zeros(Lw, np.float32)
                lo = p - EXT; a, b = max(0, lo), min(len(r), lo + Lw)
                w[a - lo:b - lo] = r[a:b]
                g = float(w[EXT:EXT + SEED] @ s16) / (float(s16 @ s16) + 1e-9)
                if abs(g) > 1e-3:
                    stack.append(w / g)
            if len(stack) < 2:
                continue
            S = np.stack(stack); mean = S.mean(0)
            nfr = Lw // FR
            m2 = (mean[:nfr * FR].reshape(nfr, FR) ** 2).mean(1)
            e2 = (S[:, :nfr * FR].reshape(len(S), nfr, FR) ** 2).mean((0, 2)) + 1e-12
            on = (m2 / e2) >= args.coh
            centre = (EXT + SEED // 2) // FR
            if not on[centre]:
                continue
            lo_f = centre
            while lo_f > 0 and on[lo_f - 1]:
                lo_f -= 1
            hi_f = centre
            while hi_f + 1 < nfr and on[hi_f + 1]:
                hi_f += 1
            T = mean[lo_f * FR:(hi_f + 1) * FR].astype(np.float32)
            off = lo_f * FR - EXT
            tid = len(templates); templates.append(T)
            for k, p, _, r in hits:
                x, _, _ = residual(k, args.enh)
                st = p + off; a, b = max(0, st), min(len(x), st + len(T))
                if b - a < FR * 5:
                    continue
                seg = T[a - st:b - st]
                g = float(x[a:b] @ seg) / (float(seg @ seg) + 1e-9)
                occ[k].append((st, g, tid)); expl[k].append((a, b))
            if tid % 25 == 0:
                log(f"  template {tid}: {len(hits)} copies, {len(T) / SR:.2f} s | seed {si}/{len(seeds)} | {time.time() - t0:.0f}s")
    finally:
        for p in pools:
            p.shutdown(cancel_futures=True)
    n_hit = sum(1 for k in keys if occ[k])
    log(f"TEST: {len(templates)} templates, {sum(len(v) for v in occ.values())} occurrences, clips with occurrences {n_hit}")
    pickle.dump({"templates": templates, "occ": occ}, open(os.path.join(W, f"tpl_{args.tag}.pkl"), "wb"))

    out_dir = os.path.join(W, f"enh_{args.tag}"); os.makedirs(out_dir, exist_ok=True)
    used, ex_list = 0, []
    for k in keys:
        x, y, r = residual(k, args.enh)
        out = y
        if occ[k]:
            est = np.zeros(len(x), np.float64)
            for st, g, tid in occ[k]:
                T = templates[tid]; a, b = max(0, st), min(len(x), st + len(T))
                est[a:b] += g * T[a - st:b - st]
            ex = 1 - float(((r - est) ** 2).sum()) / (float((r.astype(np.float64) ** 2).sum()) + 1e-12)
            ex_list.append(ex)
            if ex >= args.rule:
                out = (x - est).astype(np.float32); used += 1
        q = np.clip(np.round(out * 32768.0 / 4) * 4, -32768, 32764).astype(np.int16)
        sf.write(os.path.join(out_dir, k), q, SR, subtype="PCM_16")
    log(f"subtraction used on {used} clips (explained >= {args.rule}); explained median among hit clips "
        f"{np.median(ex_list) if ex_list else 0:.2f}; outputs in {out_dir}")
    json.dump({"used": used, "templates": len(templates), "hit_clips": n_hit, "rule": args.rule},
              open(os.path.join(W, f"tpl_{args.tag}_summary.json"), "w"))


if __name__ == "__main__":
    main()
