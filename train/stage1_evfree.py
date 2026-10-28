"""Track 2 v36: the v24/v25 recipe with event-free training targets.

Diagnosis (14 Oct): our denoiser leaves clean regions almost untouched (44 dB outside events on 200 val clips) but only
reaches 2.6 dB inside noise events, which is where the whole gap to the leaders sits. The cause is in the targets:
v21-v25 trained on val natural + Gold recordings as "clean" targets, but 99.9% of those clips contain real noise events
(mean coverage Gold 26.8%, val natural 30.1%). The model was taught to remove pasted events and to KEEP real ones.

v36 changes exactly that. Same model (Demucs dns64), same noise bank, levels, batch size, crop and schedule shape as v25,
initialised from v24_full_final.pt (v25 weights are gone). The SI-SDR loss is computed only on target samples that are
free of annotated events (timestamps from the Gold / validation metadata, dilated by 150 ms on each side); the 695
synthetic clean references are added as fully clean targets. Crops are drawn preferring clean coverage.

The in-training sanity check reports whole-clip and inside-event SI-SDR on 200 val synthetic clips (in the bank, so a
health check of the mechanism, never used to pick a checkpoint). The final checkpoint is packaged.

    python t2_v36.py --stage all --gpu-frac 0.75
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import time
import zipfile

import numpy as np
import soundfile as sf
import torch

import stage0_factory as F
import t2_asr as V

WORK, VAL, SR, CROP, dev, log = F.WORK, F.VAL, F.SR, F.CROP, F.dev, F.log
GOLD_DIR = os.environ.get("T2_GOLD_DIR", os.path.join(WORK, "hf_gold"))
TAG = "v36_evfree"
INIT = os.environ.get("T2_INIT", os.path.join(WORK, "v24_full_final.pt"))
ENH = os.path.join(WORK, "enh36")
DILATE = int(0.15 * SR)


def event_spans(m):
    out = []
    for e in m.get("NoiseSubCategoryTimeStamp") or []:
        if isinstance(e, dict) and "start" in e and "end" in e:
            out.append((float(e["start"]), float(e["end"])))
    return out


def clean_mask(n, spans):
    m = np.ones(n, np.bool_)                     # bool, not float32: the cache holds thousands of these
    for s, e in spans:
        a, b = max(0, int(s * SR) - DILATE), min(n, int(e * SR) + DILATE)
        if b > a:
            m[a:b] = False
    return m


class EvFreeFactory:
    """Pairs (mixture, target, loss mask). Target = a natural recording; mask = 1 where the target has no real event."""

    def __init__(self, items, bank, ratios, seed=36, cache_size=1000):   # 3000 + float masks exhausted 11.7 GB RAM
        self.items, self.bank, self.med = items, bank, float(np.median(ratios))   # items: (path, spans)
        self.rng = random.Random(seed)
        self.cache, self.cache_size = collections.OrderedDict(), cache_size

    def wave(self, i):
        if i in self.cache:
            self.cache.move_to_end(i); return self.cache[i]
        path, spans = self.items[i]
        x = F.load(path)
        if len(x) < CROP:
            x = np.pad(x, (0, CROP - len(x)))
        v = (x, clean_mask(len(x), spans))
        self.cache[i] = v
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return v

    def pair(self):
        rng = self.rng
        while True:
            x, cm = self.wave(rng.randrange(len(self.items)))
            best = None
            for _ in range(4):
                s = rng.randrange(0, len(x) - CROP + 1)
                c = float(cm[s:s + CROP].mean())
                if best is None or c > best[1]:
                    best = (s, c)
            if best[1] >= 0.3:
                break
        s = best[0]
        tgt = x[s:s + CROP].copy(); msk = cm[s:s + CROP].astype(np.float32); mix = tgt.copy()
        n_events = 0 if rng.random() < 0.15 else rng.choice([1, 1, 2, 2, 3])
        for _ in range(n_events):
            nz = self.bank[rng.randrange(len(self.bank))]
            if len(nz) > CROP:
                o = rng.randrange(0, len(nz) - CROP + 1); nz = nz[o:o + CROP]
            p = rng.randrange(0, CROP - len(nz) + 1)
            seg = tgt[p:p + len(nz)]
            rs = np.sqrt((seg ** 2).mean() + 1e-8); rn = np.sqrt((nz ** 2).mean() + 1e-8)
            mix[p:p + len(nz)] += nz * (float(np.exp(rng.gauss(math.log(self.med), 0.5))) * rs / rn)
        pk = np.abs(mix).max()
        if pk > 0.99:
            mix, tgt = mix / pk * 0.99, tgt / pk * 0.99
        return mix.astype(np.float32), tgt.astype(np.float32), msk

    def batch(self, bs):
        xs, ys, ms = zip(*[self.pair() for _ in range(bs)])
        f = lambda a: torch.from_numpy(np.stack(a))[:, None]
        return f(xs), f(ys), f(ms)


def masked_si_sdr_loss(est, ref, msk):
    est, ref = est * msk, ref * msk
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-8)
    t = a * ref; e = est - t
    return -(10 * torch.log10(t.pow(2).sum(-1) / (e.pow(2).sum(-1) + 1e-8) + 1e-8)).mean()


def sanity(model, held):
    whole, inside = [], []
    for mx, rf, ev in held:
        y = F.enhance(model, mx)[:len(rf)]
        whole.append(F.si_sdr_np(y, rf))
        if ev.sum() > 1600:
            inside.append(F.si_sdr_np(y[ev], rf[ev]))
    return float(np.mean(whole)), float(np.mean(inside))


def stage_train(args):
    meta = json.load(open(os.path.join(VAL, "validationMetadata.json"), encoding="utf-8"))
    syn = [m for m in meta if m.get("syntheticData")]
    nat = [m for m in meta if not m.get("syntheticData")]
    bank, ratios = F.build_bank(syn)
    items = [(f"{VAL}/naturalNoisyAudio/{m['segmentFileName']}", event_spans(m)) for m in nat]
    gmeta = {os.path.splitext(m["segmentFileName"])[0]: m
             for m in json.load(open(os.path.join(GOLD_DIR, "metadata.json"), encoding="utf-8"))}
    gold = sorted(glob.glob(os.path.join(GOLD_DIR, "audio", "*.flac")))
    n_gold = 0
    for p in gold:
        m = gmeta.get(os.path.splitext(os.path.basename(p))[0])
        if m is not None:
            items.append((p, event_spans(m))); n_gold += 1
    items += [(f"{VAL}/syntheticCleanRefAudio/{m['segmentFileName']}", []) for m in syn]
    log(f"targets {len(items)} ({len(nat)} val natural + {n_gold}/{len(gold)} Gold with timestamps + {len(syn)} clean refs) "
        f"| bank {len(bank)} noises from {len(syn)} clips")
    fac = EvFreeFactory(items, bank, ratios)
    held = []
    for m in syn[:200]:
        mx = F.load(f"{VAL}/syntheticNoiseAudio/{m['segmentFileName']}")
        rf = F.load(f"{VAL}/syntheticCleanRefAudio/{m['segmentFileName']}")
        L = min(len(mx), len(rf)); ev = np.zeros(L, bool)
        for s, e in event_spans(m):
            ev[max(0, int(s * SR)):min(L, int(e * SR))] = True
        held.append((mx[:L], rf[:L], ev))
    mfrac = np.mean([fac.pair()[2].mean() for _ in range(400)])
    log(f"mean clean-mask coverage of training crops {mfrac:.2f}")

    model = F.get_model()
    model.load_state_dict(torch.load(INIT, map_location=dev))
    w, i = sanity(model, held)
    log(f"init {os.path.basename(INIT)} | sanity whole-clip {w:.2f} dB | inside events {i:.2f} dB")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    start = 1
    ck_last = os.path.join(WORK, f"{TAG}_resume.pt")
    if os.path.exists(ck_last):
        r = torch.load(ck_last, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); sched.load_state_dict(r["sched"])
        scaler.load_state_dict(r["scaler"]); start = r["step"] + 1
        fac.rng.setstate(r["rng"])
        log(f"resumed at step {start}")
    t0, run = time.time(), 0.0
    for step in range(start, args.steps + 1):
        t_step = time.time(); model.train()
        mx, ty, mk = fac.batch(args.bs); mx, ty, mk = mx.to(dev), ty.to(dev), mk.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            est = model(mx)
        loss = masked_si_sdr_loss(est.float()[:, 0], ty[:, 0], mk[:, 0])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); sched.step()
        run = loss.item() if step == start else 0.99 * run + 0.01 * loss.item()
        if args.gpu_frac < 1.0:
            torch.cuda.synchronize(); time.sleep((time.time() - t_step) * (1 / args.gpu_frac - 1))
        if step % 500 == 0:
            el = (time.time() - t0) / 60
            log(f"  step {step}/{args.steps} train masked SI-SDR {-run:.2f} dB  {el:.1f} min "
                f"(eta {el / (step - start + 1) * (args.steps - step):.0f} min)")
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "rng": fac.rng.getstate()}, ck_last + ".tmp")
            os.replace(ck_last + ".tmp", ck_last)
            w, i = sanity(model, held)
            log(f"== step {step}: checkpoint saved | sanity whole-clip {w:.2f} dB | inside events {i:.2f} dB")
    torch.save(model.state_dict(), os.path.join(WORK, f"{TAG}_final.pt"))
    log("DONE train")


def stage_infer(args):
    model = F.get_model()
    model.load_state_dict(torch.load(os.path.join(WORK, f"{TAG}_final.pt"), map_location=dev))
    wavs = sorted(glob.glob(os.path.join(WORK, "phase2_test", "**", "*.wav"), recursive=True))
    os.makedirs(ENH, exist_ok=True)
    t0 = time.time()
    for i, p in enumerate(wavs):
        x = F.load(p); y = F.enhance(model, x)
        pk = float(np.abs(y).max())
        if pk > 0.99:
            y = y / pk * 0.99
        assert len(y) == len(x)
        sf.write(os.path.join(ENH, os.path.basename(p)), y.astype(np.float32), SR, subtype="PCM_16")
        if i % 500 == 0:
            log(f"  {i}/{len(wavs)} {time.time() - t0:.0f}s")
    log(f"enhanced {len(wavs)} clips in {time.time() - t0:.0f}s")
    del model; torch.cuda.empty_cache()


def stage_asr(args):
    asr = V.load_asr()
    out_wavs = sorted(glob.glob(os.path.join(ENH, "*.wav")))
    t0 = time.time()
    texts = dict(zip([os.path.basename(p)[:-4] for p in out_wavs], V.transcribe(asr, out_wavs)))
    json.dump(texts, open(os.path.join(WORK, f"transcripts_{TAG}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    log(f"transcribed {len(texts)} in {time.time() - t0:.0f}s | empty {sum(1 for v in texts.values() if not v.strip())}")


def stage_package(args):
    texts = json.load(open(os.path.join(WORK, f"transcripts_{TAG}.json"), encoding="utf-8"))
    out_wavs = sorted(glob.glob(os.path.join(ENH, "*.wav")))
    ids = sorted(os.path.basename(p)[:-4] for p in out_wavs)
    jl = os.path.join(WORK, f"transcripts_{TAG}.jsonl")
    open(jl, "w", encoding="utf-8").write(
        "\n".join(json.dumps({"clip_id": k, "text": texts.get(k, "")}, ensure_ascii=False) for k in ids) + "\n")
    zp = os.path.join(WORK, f"datathon_t2_{TAG}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.write(jl, "transcripts.jsonl")
        for p in out_wavs:
            z.write(p, os.path.basename(p))
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        rows = [json.loads(l) for l in z.read("transcripts.jsonl").decode("utf-8").splitlines() if l.strip()]
        ok = (len(names) == 3179 and {r["clip_id"] for r in rows} == {n[:-4] for n in names if n.endswith(".wav")}
              and not any("/" in n for n in names) and sorted(rows[0]) == ["clip_id", "text"]
              and all(r["text"].strip() for r in rows) and z.testzip() is None)
    log(f"{zp}: {os.path.getsize(zp) / 2**20:.0f} MiB, {len(names)} entries, format ok: {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["train", "infer", "asr", "package", "all"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--gpu-frac", type=float, default=1.0)
    args = ap.parse_args()
    if args.gpu_frac < 1.0 and dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac); log(f"GPU share capped at {args.gpu_frac:.0%}")
    torch.manual_seed(36); np.random.seed(36)
    for s in (["train", "infer", "asr", "package"] if args.stage == "all" else [args.stage]):
        log(f"=== stage {s}")
        globals()[f"stage_{s}"](args)


if __name__ == "__main__":
    main()
