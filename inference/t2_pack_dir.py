"""Package an existing folder of final (already quantised) Track 2 WAVs: SraVaani on exactly those files, zip, checks.

    python t2_pack_dir.py --dir enh_v40tpl --tag v40tplA
"""
import argparse
import glob
import json
import os
import time
import zipfile

import torch

import t2_asr as V

W = os.environ.get("T2_WORK", os.path.dirname(os.path.abspath(__file__)))
LIMIT = 1_000_000_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    wavs = sorted(glob.glob(os.path.join(W, args.dir, "*.wav")))
    assert len(wavs) == 3178, len(wavs)
    asr = V.load_asr()
    t0 = time.time()
    texts = dict(zip([os.path.basename(p)[:-4] for p in wavs], V.transcribe(asr, wavs)))
    json.dump(texts, open(os.path.join(W, f"transcripts_{args.tag}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    log(f"transcribed {len(texts)} in {time.time() - t0:.0f}s | empty {sum(1 for v in texts.values() if not v.strip())}")
    del asr; torch.cuda.empty_cache()
    ids = sorted(texts)
    jl = "\n".join(json.dumps({"clip_id": k, "text": texts[k]}, ensure_ascii=False) for k in ids) + "\n"
    zp = os.path.join(W, f"datathon_t2_{args.tag}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("transcripts.jsonl", jl.encode("utf-8"))
        for p in wavs:
            z.write(p, os.path.basename(p))
    size = os.path.getsize(zp)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        rows = [json.loads(l) for l in z.read("transcripts.jsonl").decode("utf-8").splitlines() if l.strip()]
        ok = (len(names) == 3179 and {r["clip_id"] for r in rows} == {n[:-4] for n in names if n.endswith(".wav")}
              and not any("/" in n for n in names) and sorted(rows[0]) == ["clip_id", "text"]
              and z.testzip() is None and size < LIMIT)
    log(f"{zp}: {size:,} bytes ({size / 1e9:.3f} GB), {len(names)} entries, format ok: {ok}")


if __name__ == "__main__":
    main()
