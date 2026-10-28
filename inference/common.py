"""Shared helpers for Track 2 inference: audio loading, the Demucs dns64 model (vendored code in dn/), enhancement.

The model architecture is facebookresearch/denoiser's dns64 (Demucs, hidden=64, 16 kHz); our fine-tuned weights are
loaded on top, so no pretrained download is needed at inference time.
"""
import os
import time

import numpy as np
import soundfile as sf
import torch

from dn.demucs import Demucs

WORK = os.environ.get("T2_WORK", os.path.dirname(os.path.abspath(__file__)))
SR = 16000
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    return x.mean(1) if x.ndim > 1 else x


def get_model():
    return Demucs(hidden=64, sample_rate=SR).to(dev)


def load_model(path):
    m = get_model()
    m.load_state_dict(torch.load(path, map_location=dev))   # fp16 or fp32 tensors are cast to the fp32 parameters
    return m.eval()


@torch.no_grad()
def enhance(model, x):
    model.eval()
    t = torch.from_numpy(np.asarray(x, np.float32))[None, None].to(dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
        y = model(t)[0, 0].float().cpu().numpy()
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
