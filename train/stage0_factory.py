"""Track 2 Phase-2: in-domain data factory + fine-tuned Demucs, measured at every step (laptop GPU port).

A synthetic clip is exactly the clean reference plus pasted noise (verified: residual outside the pasted
spans ~0.0001), so the organisers' generator can be rebuilt: natural recordings as targets, bank noises
pasted at random places/levels as mixtures. facebookresearch/denoiser's pretrained Demucs (dns64, 16 kHz)
is fine-tuned on that with an SI-SDR loss and measured on 200 held-out synthetic clips never used for
training or for the noise bank.

Stages (each resumable via files in the work dir):
  factory  : build bank + measure identity/pretrained
  train    : fine-tune, periodic held-out measurement, save best  -> demucs_best.pt, train.json
  infer    : enhance the Phase-2 test clips with the chosen blend  -> enh/
  asr      : mandated ASR on enh/                                  -> transcripts.jsonl
  package  : zip with format checks                                -> datathon_t2_v21_factory.zip

    python t2_factory.py --stage all --gpu-frac 0.75
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import soundfile as sf
import torch

WORK = os.environ.get("T2_WORK", os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(WORK, "validation")
SR = 16000
CROP = 4 * SR
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    return x.mean(1) if x.ndim > 1 else x


def si_sdr_np(est, ref):
    est = np.asarray(est, np.float64); ref = np.asarray(ref, np.float64)
    L = min(len(est), len(ref)); est, ref = est[:L], ref[:L]
    a = float(est @ ref) / float(ref @ ref + 1e-12); t = a * ref; e = est - t
    return 10 * np.log10(float(t @ t + 1e-12) / float(e @ e + 1e-12))


# ----------------------------------------------------------------------------- data
def splits():
    meta = json.load(open(os.path.join(VAL, "validationMetadata.json"), encoding="utf-8"))
    syn = [m for m in meta if m.get("syntheticData")]
    nat = [m for m in meta if not m.get("syntheticData")]
    random.Random(0).shuffle(syn)
    return syn[:200], syn[200:], nat          # held-out, bank, targets


def build_bank(bank_meta):
    bank, ratios = [], []
    for m in bank_meta:
        n = m["segmentFileName"]
        mix = load(f"{VAL}/syntheticNoiseAudio/{n}"); ref = load(f"{VAL}/syntheticCleanRefAudio/{n}")
        L = min(len(mix), len(ref)); d = mix[:L] - ref[:L]
        for e in m["NoiseSubCategoryTimeStamp"]:
            if "start" not in e:
                continue
            a, b = int(float(e["start"]) * SR), int(float(e["end"]) * SR)
            if b - a >= 800:
                rn = np.sqrt((d[a:b] ** 2).mean() + 1e-12); rs = np.sqrt((ref[a:b] ** 2).mean() + 1e-12)
                ratios.append(rn / rs)
            a, b = max(0, a - 160), min(L, b + 160)
            if b - a >= 800:
                bank.append(d[a:b].copy())
    return bank, np.array(ratios)


class Factory:
    def __init__(self, nat_meta, bank, ratios, seed=2):
        self.paths = [f"{VAL}/naturalNoisyAudio/{m['segmentFileName']}" for m in nat_meta]
        self.cache, self.bank, self.med = {}, bank, float(np.median(ratios))
        self.rng = random.Random(seed)

    def wave(self, i):
        if i not in self.cache:
            self.cache[i] = load(self.paths[i])
        return self.cache[i]

    def pair(self):
        rng = self.rng
        x = self.wave(rng.randrange(len(self.paths)))
        if len(x) < CROP:
            x = np.pad(x, (0, CROP - len(x)))
        s = rng.randrange(0, len(x) - CROP + 1)
        tgt = x[s:s + CROP].copy(); mix = tgt.copy()
        n_events = 0 if rng.random() < 0.15 else rng.choice([1, 1, 2, 2, 3])
        for _ in range(n_events):
            nz = self.bank[rng.randrange(len(self.bank))]
            if len(nz) > CROP:
                o = rng.randrange(0, len(nz) - CROP + 1); nz = nz[o:o + CROP]
            p = rng.randrange(0, CROP - len(nz) + 1)
            seg = tgt[p:p + len(nz)]
            rs = np.sqrt((seg ** 2).mean() + 1e-8); rn = np.sqrt((nz ** 2).mean() + 1e-8)
            ratio = float(np.exp(rng.gauss(math.log(self.med), 0.5)))
            mix[p:p + len(nz)] += nz * (ratio * rs / rn)
        pk = np.abs(mix).max()
        if pk > 0.99:
            mix, tgt = mix / pk * 0.99, tgt / pk * 0.99
        return mix.astype(np.float32), tgt.astype(np.float32)

    def batch(self, bs):
        xs, ys = zip(*[self.pair() for _ in range(bs)])
        return torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys))[:, None]


# ----------------------------------------------------------------------------- model
def get_model():
    from denoiser import pretrained
    return pretrained.dns64().to(dev)


def si_sdr_loss(est, ref):
    est = est - est.mean(-1, keepdim=True); ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-8)
    t = a * ref; e = est - t
    return -(10 * torch.log10(t.pow(2).sum(-1) / (e.pow(2).sum(-1) + 1e-8) + 1e-8)).mean()


@torch.no_grad()
def enhance(model, x):
    model.eval()
    t = torch.from_numpy(np.asarray(x, np.float32))[None, None].to(dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
        y = model(t)[0, 0].float().cpu().numpy()
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def load_held(held_meta):
    out = []
    for m in held_meta:
        n = m["segmentFileName"]
        out.append((load(f"{VAL}/syntheticNoiseAudio/{n}"), load(f"{VAL}/syntheticCleanRefAudio/{n}")))
    return out


def measure(model, held, alphas=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    ident = float(np.mean([si_sdr_np(mx, rf) for mx, rf in held]))
    outs = [enhance(model, mx) for mx, rf in held]
    res = {a: float(np.mean([si_sdr_np(a * y + (1 - a) * mx, rf) for y, (mx, rf) in zip(outs, held)]))
           for a in alphas}
    return ident, res


# ----------------------------------------------------------------------------- stages
def stage_train(args):
    held_meta, bank_meta, nat = splits()
    bank, ratios = build_bank(bank_meta)
    log(f"held-out {len(held_meta)} | bank {len(bank)} snippets from {len(bank_meta)} clips | targets {len(nat)}")
    log(f"pasted-noise/speech rms ratio p10/50/90 = {np.percentile(ratios, [10, 50, 90]).round(2)}")
    held = load_held(held_meta)
    fac = Factory(nat, bank, ratios)
    model = get_model()
    ident0, res0 = measure(model, held, alphas=(1.0, 0.8, 0.6))
    log(f"BEFORE | identity {ident0:.2f} dB | pretrained dns64 " + ", ".join(f"a={a}: {v:.2f}" for a, v in res0.items()))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    best, hist, t0, run = -1e9, [], time.time(), 0.0
    for step in range(1, args.steps + 1):
        t_step = time.time()
        model.train()
        mx, ty = fac.batch(args.bs); mx, ty = mx.to(dev), ty.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            est = model(mx)
        loss = si_sdr_loss(est.float()[:, 0], ty[:, 0])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if args.gpu_frac < 1.0:
            torch.cuda.synchronize(); time.sleep((time.time() - t_step) * (1 / args.gpu_frac - 1))
        if step % 100 == 0:
            log(f"  step {step}/{args.steps} train SI-SDR {-run:.2f} dB  {(time.time() - t0) / 60:.1f} min "
                f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        if step % args.eval_every == 0 or step == args.steps:
            ident, res = measure(model, held)
            a_best = max(res, key=res.get)
            hist.append({"step": step, "identity": ident, "res": res})
            log(f"== step {step}: held-out identity {ident:.2f} | model a=1.0 {res[1.0]:.2f} | best a={a_best} {res[a_best]:.2f}")
            if res[a_best] > best:
                best = res[a_best]
                torch.save(model.state_dict(), os.path.join(WORK, "demucs_best.pt"))
                json.dump({"best": best, "alpha": a_best, "identity": ident, "step": step, "history": hist},
                          open(os.path.join(WORK, "train.json"), "w"), indent=1)
                log(f"   saved best ({best:.2f} dB, alpha {a_best})")
    log(f"DONE: best held-out SI-SDR {best:.2f} dB vs identity {ident0:.2f} dB")


def stage_infer(args):
    info = json.load(open(os.path.join(WORK, "train.json")))
    alpha = info["alpha"] if args.alpha is None else args.alpha
    model = get_model(); model.load_state_dict(torch.load(os.path.join(WORK, "demucs_best.pt"), map_location=dev))
    test_dir = os.path.join(WORK, "phase2_test")
    if not os.path.isdir(test_dir):
        with zipfile.ZipFile(os.path.join(WORK, "input_data.zip")) as z:
            z.extractall(test_dir)
    wavs = sorted(glob.glob(os.path.join(test_dir, "**", "*.wav"), recursive=True))
    outd = os.path.join(WORK, "enh"); os.makedirs(outd, exist_ok=True)
    log(f"inference on {len(wavs)} clips, alpha {alpha}")
    t0 = time.time()
    for i, p in enumerate(wavs):
        o = os.path.join(outd, os.path.basename(p))
        if os.path.exists(o):
            continue
        x = load(p); y = alpha * enhance(model, x) + (1 - alpha) * x
        pk = float(np.abs(y).max())
        if pk > 0.99:
            y = y / pk * 0.99
        assert len(y) == len(x)
        sf.write(o, y.astype(np.float32), SR, subtype="PCM_16")
        if i % 400 == 0:
            log(f"  {i}/{len(wavs)} {time.time() - t0:.0f}s")
    log(f"enhanced {len(wavs)} clips in {time.time() - t0:.0f}s")


def stage_asr(args):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    path = snapshot_download("ARTPARK-IISc/SraVaani-1.0", token=os.environ.get("HF_TOKEN"))
    asr = AutoModel.from_pretrained(path, trust_remote_code=True).to(dev).eval()
    out_wavs = sorted(glob.glob(os.path.join(WORK, "enh", "*.wav")))
    tpath = os.path.join(WORK, "transcripts.json")
    texts = json.load(open(tpath, encoding="utf-8")) if os.path.exists(tpath) else {}
    todo = [p for p in out_wavs if os.path.basename(p)[:-4] not in texts]
    log(f"ASR on {len(todo)} clips ({len(texts)} already done)")
    B, t0 = 8, time.time()
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        try:
            hyps = asr.transcribe(batch, return_hypotheses=True)
            got = [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
        except Exception:
            got = []
            for p in batch:
                try:
                    h = asr.transcribe([p], return_hypotheses=True)[0]
                    got.append((h.text if hasattr(h, "text") else str(h)).strip())
                except Exception:
                    got.append("")
        for p, t in zip(batch, got):
            texts[os.path.basename(p)[:-4]] = t
        if i % (B * 25) == 0:
            json.dump(texts, open(tpath, "w", encoding="utf-8"), ensure_ascii=False)
            log(f"  {i}/{len(todo)} {time.time() - t0:.0f}s")
    json.dump(texts, open(tpath, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"transcribed {len(texts)} | empty {sum(1 for v in texts.values() if not v.strip())}")


def stage_package(args):
    texts = json.load(open(os.path.join(WORK, "transcripts.json"), encoding="utf-8"))
    out_wavs = sorted(glob.glob(os.path.join(WORK, "enh", "*.wav")))
    ids = sorted(os.path.basename(p)[:-4] for p in out_wavs)
    jl = os.path.join(WORK, "transcripts.jsonl")
    open(jl, "w", encoding="utf-8").write(
        "\n".join(json.dumps({"clip_id": k, "text": texts.get(k, "")}, ensure_ascii=False) for k in ids) + "\n")
    zp = os.path.join(WORK, "datathon_t2_v21_factory.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.write(jl, "transcripts.jsonl")
        for p in out_wavs:
            z.write(p, os.path.basename(p))
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        rows = [json.loads(l) for l in z.read("transcripts.jsonl").decode("utf-8").splitlines() if l.strip()]
        ok = ({r["clip_id"] for r in rows} == {n[:-4] for n in names if n.endswith(".wav")}
              and not any("/" in n for n in names) and sorted(rows[0]) == ["clip_id", "text"])
    log(f"{zp}: {os.path.getsize(zp) / 2**20:.0f} MiB, {len(names)} entries, format ok: {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["train", "infer", "asr", "package", "all"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gpu-frac", type=float, default=1.0)
    args = ap.parse_args()
    if args.gpu_frac < 1.0 and dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac)
        log(f"GPU share capped at {args.gpu_frac:.0%}")
    torch.manual_seed(0); np.random.seed(0)
    stages = ["train", "infer", "asr", "package"] if args.stage == "all" else [args.stage]
    for s in stages:
        log(f"=== stage {s}")
        globals()[f"stage_{s}"](args)


if __name__ == "__main__":
    main()
