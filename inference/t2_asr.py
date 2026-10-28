"""SraVaani-1.0 ASR helpers, restored verbatim from t2_v22.py (deleted by another session on 12-13 Oct).

This is the transcription path of v24 (LB 11.22, audit mismatch 1%): transcribe the final PCM_16 WAVs, batches of 8.
"""
import os

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_asr():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    path = snapshot_download("ARTPARK-IISc/SraVaani-1.0", token=os.environ.get("HF_TOKEN"))
    return AutoModel.from_pretrained(path, trust_remote_code=True).to(dev).eval()


def transcribe(asr, paths, B=8):
    out = []
    for i in range(0, len(paths), B):
        batch = paths[i:i + B]
        try:
            hyps = asr.transcribe(batch, return_hypotheses=True)
            out += [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
        except Exception:
            for p in batch:
                try:
                    h = asr.transcribe([p], return_hypotheses=True)[0]
                    out.append((h.text if hasattr(h, "text") else str(h)).strip())
                except Exception:
                    out.append("")
    return out
