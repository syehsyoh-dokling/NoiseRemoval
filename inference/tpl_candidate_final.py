"""Final Track 2 candidate: subtract every template set (pass 1, pass 2, ...) and denoise the remainder.

On clips with at least one template occurrence: x' = x - sum over sets of g*T, then the mean of the given denoisers
on x'. Every other clip: the fallback folder (a plain denoiser output). Output rounded to multiples of 4, then packaged
audit-safely by t2_pack_dir.py (SraVaani on exactly these files).

    python tpl_candidate_final.py --tpl tpl_v40tpl.pkl tpl_v42p2.pkl \
        --weights kaggle_t2/out_v39/s1/final.pt kaggle_t2/out_v39/s2/final.pt --fallback enh_v39avg --tag v42B
"""
import argparse
import glob
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

import common as F

W, SR, log = F.WORK, F.SR, F.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpl", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--fallback", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-pack", action="store_true")
    args = ap.parse_args()
    sets = [pickle.load(open(os.path.join(W, p), "rb")) for p in args.tpl]
    models = []
    for p in args.weights:
        m = F.get_model(); m.load_state_dict(torch.load(os.path.join(W, p), map_location=F.dev)); models.append(m)
    out = os.path.join(W, f"enh_{args.tag}"); os.makedirs(out, exist_ok=True)
    test = sorted(glob.glob(os.path.join(W, "phase2_test", "**", "*.wav"), recursive=True))
    n_sub, n_occ, t0 = 0, 0, time.time()
    for i, p in enumerate(test):
        k = os.path.basename(p)
        hit = [d for d in sets if d["occ"].get(k)]
        if hit:
            x = F.load(p).astype(np.float64)
            for d in hit:
                for st, g, tid in d["occ"][k]:
                    t = d["templates"][tid]; a, b = max(0, st), min(len(x), st + len(t))
                    x[a:b] -= g * t[a - st:b - st]; n_occ += 1
            xs = x.astype(np.float32)
            y = np.mean([F.enhance(m, xs) for m in models], axis=0)
            pk = float(np.abs(y).max())
            if pk > 0.99:
                y = y / pk * 0.99
            q = np.clip(np.round(y * 32768.0 / 4) * 4, -32768, 32764).astype(np.int16)
            n_sub += 1
        else:
            q, _ = sf.read(os.path.join(W, args.fallback, k), dtype="int16")
        sf.write(os.path.join(out, k), q, SR, subtype="PCM_16")
        if i % 1000 == 0:
            log(f"  {i}/{len(test)} | clips with templates {n_sub} | {time.time() - t0:.0f}s")
    log(f"{n_sub} clips: {n_occ} template occurrences subtracted then denoised by {len(models)} model(s); "
        f"others = {args.fallback}")
    del models; torch.cuda.empty_cache()
    if not args.no_pack:
        sys.exit(subprocess.call([sys.executable, "-u", os.path.join(W, "t2_pack_dir.py"), "--dir", f"enh_{args.tag}",
                                  "--tag", args.tag], cwd=W))


if __name__ == "__main__":
    main()
