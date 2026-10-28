"""Build the Kaggle dataset folder for v38 (<kaggle-user>/indoml-t2-train).

  val/{natural,synthetic,cleanref}/*.flac + val/validationMetadata.json   (lossless FLAC of the validation WAVs)
  gold_audio/*.flac + gold_metadata.json                                  (Track 1 Gold corpus, unchanged)
  esc_bank.npz                                                            (ESC-50 event snippets, 16 kHz float16)
  v36_evfree_final.pt                                                     (init weights, LB 11.77)
  code/train_k.py + code/dn/{demucs,resample,utils}.py                    (vendored Demucs, MIT/CC-BY-NC per upstream)
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

T2 = os.environ.get("T2_WORK", ".")
OUT = os.path.join(T2, "kaggle_t2_data")
GOLD = os.environ.get("T2_GOLD_DIR", os.path.join(T2, "hf_gold"))
DN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dn")
sys.path.insert(0, T2)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(T2, "validation", "validationMetadata.json"), encoding="utf-8"))
    names = {m["segmentFileName"] for m in meta}
    for src, dst in (("naturalNoisyAudio", "natural"), ("syntheticNoiseAudio", "synthetic"),
                     ("syntheticCleanRefAudio", "cleanref")):
        d = os.path.join(OUT, "val", dst); os.makedirs(d, exist_ok=True)
        n = 0
        for p in sorted(glob.glob(os.path.join(T2, "validation", src, "*.wav"))):
            b = os.path.basename(p)
            if b not in names:
                continue                                     # e.g. the stray "(1).wav" duplicate
            o = os.path.join(d, b[:-4] + ".flac")
            if not os.path.exists(o):
                x, sr = sf.read(p, dtype="int16"); assert sr == 16000
                sf.write(o, x, sr, format="FLAC", subtype="PCM_16")
            n += 1
        print(dst, n, flush=True)
    shutil.copy2(os.path.join(T2, "validation", "validationMetadata.json"), os.path.join(OUT, "val"))

    gd = os.path.join(OUT, "gold_audio"); os.makedirs(gd, exist_ok=True)
    gold = sorted(glob.glob(os.path.join(GOLD, "audio", "*.flac")))
    for p in gold:
        o = os.path.join(gd, os.path.basename(p))
        if not os.path.exists(o):
            shutil.copy2(p, o)
    shutil.copy2(os.path.join(GOLD, "metadata.json"), os.path.join(OUT, "gold_metadata.json"))
    print("gold", len(gold), flush=True)

    import t2_v37
    bank, per_class = t2_v37.esc_bank()
    offs = np.cumsum([0] + [len(b) for b in bank]).astype(np.int64)
    np.savez(os.path.join(OUT, "esc_bank.npz"), audio=np.concatenate(bank).astype(np.float16), offsets=offs)
    print("esc snippets", len(bank), "classes", len(per_class), flush=True)

    shutil.copy2(os.path.join(T2, "v36_evfree_final.pt"), OUT)
    cd = os.path.join(OUT, "code", "dn"); os.makedirs(cd, exist_ok=True)
    open(os.path.join(cd, "__init__.py"), "w").close()
    for f in ("demucs.py", "resample.py", "utils.py"):
        shutil.copy2(os.path.join(DN, f), cd)
    shutil.copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_long.py"),
                 os.path.join(OUT, "code", "train_k.py"))
    json.dump({"title": "indoml-t2-train", "id": os.environ.get("KAGGLE_DATASET", "<kaggle-user>/indoml-t2-train"), "licenses": [{"name": "other"}],
               "description": "Private working copy for Datathon@IndoML 2026 Track 2 (not for redistribution)."},
              open(os.path.join(OUT, "dataset-metadata.json"), "w"), indent=1)
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
    print(f"dataset folder {OUT}: {tot / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
