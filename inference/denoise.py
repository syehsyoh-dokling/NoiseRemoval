"""Denoiser-only output: each model enhances every test clip, its output is peak-limited and rounded to a multiple of 4,
then the models' int16 outputs are averaged and rounded to a multiple of 4 again (the v39avg submission, LB 12.75).

This folder is both the fallback for clips without template occurrences and the reference whose residual (x - y)
seeds blind template discovery.

    python denoise.py --weights ../checkpoints/v39_s1.pt ../checkpoints/v39_s2.pt --tag base
"""
import argparse
import glob
import os
import time

import numpy as np
import soundfile as sf
import torch

import common as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    models = [F.load_model(os.path.join(F.WORK, p)) for p in args.weights]
    out = os.path.join(F.WORK, f"enh_{args.tag}"); os.makedirs(out, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(F.WORK, "phase2_test", "**", "*.wav"), recursive=True))
    F.log(f"{len(wavs)} test clips, {len(models)} model(s) on {F.dev}")
    t0 = time.time()
    for i, p in enumerate(wavs):
        x = F.load(p)
        outs = []
        for m in models:                              # per model: exactly the single-model packaging path
            y = F.enhance(m, x)
            pk = float(np.abs(y).max())
            if pk > 0.99:
                y = y / pk * 0.99
            outs.append(np.clip(np.round(y * 32768.0 / 4) * 4, -32768, 32767).astype(np.int16))
        y = np.mean([o.astype(np.float64) for o in outs], axis=0)
        y = np.clip(np.round(y / 4) * 4, -32768, 32764).astype(np.int16)
        sf.write(os.path.join(out, os.path.basename(p)), y, F.SR, subtype="PCM_16")
        if i % 500 == 0:
            F.log(f"  {i}/{len(wavs)} {time.time() - t0:.0f}s")
    F.log(f"denoised {len(wavs)} clips -> {out} ({time.time() - t0:.0f}s)")
    del models; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
