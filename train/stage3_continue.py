"""Track 2 v39 on Kaggle (T4 x2): continue v38a (LB 12.366) for 8.5 h, two seeds in parallel (s1, s2).

v38b (+ESC-50 noise) scored 12.059 < v38a, so v39 keeps the validation-only noise bank. Init = final.pt of variant a
from the kernel output <kaggle-user>/indoml-t2-v38 (attached as a kernel source); data and the vendored Demucs code come
from the dataset <kaggle-user>/indoml-t2-train. Peak LR 3e-5, time-based warm-up + cosine. Outputs /kaggle/working/s1|s2.
Run with no arguments to launch both seeds (one per GPU). The original v38 description follows.

Track 2 v38 training on Kaggle (T4 x2): the v36/v37 recipe, longer and with a bigger batch.

Self-contained (Demucs dns64 code vendored in dn/). One process per GPU:
  variant a = v36 recipe: event-free masked SI-SDR loss, validation noise bank only
  variant b = v37 recipe: same + ESC-50 event snippets in the noise bank
Both start from v36_evfree_final.pt (LB 11.77). The learning rate follows a time-based warm-up + cosine schedule so the
run anneals properly inside the wall-clock budget. Checkpoints (resume + plain weights) go to /kaggle/working/<variant>/.

    python train_k.py --variant a --gpu 0 --hours 10.3
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

_dn = [h for r in os.environ.get("K_ROOTS", "/kaggle/input,/kaggle/temp").split(",")
       for h in glob.glob(f"{r}/**/code/dn/demucs.py", recursive=True)]
sys.path.insert(0, os.path.dirname(os.path.dirname(_dn[0])) if _dn else os.path.dirname(os.path.abspath(__file__)))
from dn.demucs import Demucs  # noqa: E402  (vendored denoiser code from the indoml-t2-train dataset)

SR, CROP = 16000, 4 * 16000
ROOTS = os.environ.get("K_ROOTS", "/kaggle/input,/kaggle/temp").split(",")     # overridable for a local smoke test
OUT_ROOT = os.environ.get("K_OUT", "/kaggle/working")
DILATE = int(0.15 * SR)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name):
    hits = [h for r in ROOTS for h in glob.glob(f"{r}/**/{name}", recursive=True)]
    assert hits, f"{name} not found under {ROOTS}"
    return hits[0]


def load(p):
    x, sr = sf.read(p, dtype="float32")
    assert sr == SR, (p, sr)
    return x.mean(1) if x.ndim > 1 else x


def si_sdr_np(est, ref):
    est = est - est.mean(); ref = ref - ref.mean()
    a = float(est @ ref) / (float(ref @ ref) + 1e-12); t = a * ref; e = est - t
    return 10 * np.log10(float(t @ t + 1e-12) / float(e @ e + 1e-12))


def event_spans(m):
    return [(float(e["start"]), float(e["end"])) for e in (m.get("NoiseSubCategoryTimeStamp") or [])
            if isinstance(e, dict) and "start" in e and "end" in e]


def clean_mask(n, spans):
    m = np.ones(n, np.bool_)
    for s, e in spans:
        a, b = max(0, int(s * SR) - DILATE), min(n, int(e * SR) + DILATE)
        if b > a:
            m[a:b] = False
    return m


def build_bank(val_dir, syn_meta):
    bank, ratios = [], []
    for m in syn_meta:
        n = m["segmentFileName"].replace(".wav", ".flac")
        mix = load(f"{val_dir}/synthetic/{n}"); ref = load(f"{val_dir}/cleanref/{n}")
        L = min(len(mix), len(ref)); d = mix[:L] - ref[:L]
        for s, e in event_spans(m):
            a, b = int(s * SR), int(e * SR)
            if b - a >= 800:
                ratios.append(np.sqrt((d[a:b] ** 2).mean() + 1e-12) / np.sqrt((ref[a:b] ** 2).mean() + 1e-12))
            a, b = max(0, a - 160), min(L, b + 160)
            if b - a >= 800:
                bank.append(d[a:b].copy())
    return bank, np.array(ratios)


class Factory:
    def __init__(self, items, bank, ratios, seed, cache_size=4000):
        self.items, self.bank, self.med = items, bank, float(np.median(ratios))
        self.rng = random.Random(seed)
        self.cache, self.cache_size = collections.OrderedDict(), cache_size

    def wave(self, i):
        if i in self.cache:
            self.cache.move_to_end(i); return self.cache[i]
        path, spans = self.items[i]
        x = load(path)
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
                s = rng.randrange(0, len(x) - CROP + 1); c = float(cm[s:s + CROP].mean())
                if best is None or c > best[1]:
                    best = (s, c)
            if best[1] >= 0.3:
                break
        s = best[0]
        tgt = x[s:s + CROP].copy(); msk = cm[s:s + CROP].astype(np.float32); mix = tgt.copy()
        n_events = 0 if rng.random() < 0.15 else rng.choice([1, 1, 2, 2, 3])
        for _ in range(n_events):
            nz = np.asarray(self.bank[rng.randrange(len(self.bank))], dtype=np.float32)
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


@torch.no_grad()
def enhance(model, x, dev):
    model.eval()
    t = torch.from_numpy(np.asarray(x, np.float32))[None, None].to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        y = model(t)[0, 0].float().cpu().numpy()
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def sanity(model, held, dev):
    whole, inside = [], []
    for mx, rf, ev in held:
        y = enhance(model, mx, dev)[:len(rf)]
        whole.append(si_sdr_np(y, rf))
        if ev.sum() > 1600:
            inside.append(si_sdr_np(y[ev], rf[ev]))
    return float(np.mean(whole)), float(np.mean(inside))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["s1", "s2"], default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--hours", type=float, default=8.5)          # v39: finish ~18:30 WIB, well before the deadline
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)            # continuation of v38a: lower peak than its 5e-5
    ap.add_argument("--warm-min", type=float, default=10.0)
    args = ap.parse_args()
    if args.variant is None:
        launch(args); return
    torch.cuda.set_device(args.gpu); dev = f"cuda:{args.gpu}"
    seed = 40 if args.variant == "s1" else 41
    torch.manual_seed(seed); np.random.seed(seed)
    out = os.path.join(OUT_ROOT, args.variant); os.makedirs(out, exist_ok=True)

    val_dir = os.path.dirname(find("validationMetadata.json"))
    meta = json.load(open(f"{val_dir}/validationMetadata.json", encoding="utf-8"))
    syn = [m for m in meta if m.get("syntheticData")]; nat = [m for m in meta if not m.get("syntheticData")]
    bank, ratios = build_bank(val_dir, syn)
    n_val = len(bank)
    if False:                                        # v38b showed ESC-50 noise hurts on the test set: val bank only
        z = np.load(find("esc_bank.npz"))
        flat, offs = z["audio"], z["offsets"]
        bank += [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]      # float16 views, cast on use
    items = [(f"{val_dir}/natural/{m['segmentFileName'].replace('.wav', '.flac')}", event_spans(m)) for m in nat]
    gold_dir = os.path.dirname(find("gold_metadata.json"))
    gmeta = {os.path.splitext(m["segmentFileName"])[0]: m
             for m in json.load(open(f"{gold_dir}/gold_metadata.json", encoding="utf-8"))}
    gold_files = [h for r in ROOTS for h in glob.glob(f"{r}/**/gold_audio/*.flac", recursive=True)]
    for p in sorted(set(gold_files)):
        m = gmeta.get(os.path.splitext(os.path.basename(p))[0])
        if m is not None:
            items.append((p, event_spans(m)))
    items += [(f"{val_dir}/cleanref/{m['segmentFileName'].replace('.wav', '.flac')}", []) for m in syn]
    log(f"variant {args.variant} gpu {args.gpu} | targets {len(items)} | bank {n_val} val + {len(bank) - n_val} ESC")
    fac = Factory(items, bank, ratios, seed)
    held = []
    for m in syn[:200]:
        n = m["segmentFileName"].replace(".wav", ".flac")
        mx = load(f"{val_dir}/synthetic/{n}"); rf = load(f"{val_dir}/cleanref/{n}")
        L = min(len(mx), len(rf)); ev = np.zeros(L, bool)
        for s, e in event_spans(m):
            ev[max(0, int(s * SR)):min(L, int(e * SR))] = True
        held.append((mx[:L], rf[:L], ev))

    model = Demucs(hidden=64, sample_rate=16000).to(dev)
    init = [p for r in ROOTS for p in glob.glob(f"{r}/**/a/final.pt", recursive=True) if "indoml-t2-v38" in p]
    assert init, "v38 variant a final.pt not found (attach the stage-2 kernel output indoml-t2-v38)"
    model.load_state_dict(torch.load(init[0], map_location=dev))
    w, i = sanity(model, held, dev)
    log(f"init {init[0]} | sanity whole-clip {w:.2f} dB | inside events {i:.2f} dB")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    budget = args.hours * 3600.0
    t0, run, step, n_bad, last_save, last_sanity = time.time(), None, 0, 0, time.time(), time.time()
    while True:
        el = time.time() - t0
        if el >= budget:
            break
        frac = el / budget
        lr = args.lr * min(1.0, el / (args.warm_min * 60)) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        step += 1; model.train()
        mx, ty, mk = fac.batch(args.bs); mx, ty, mk = mx.to(dev), ty.to(dev), mk.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            est = model(mx)
        loss = masked_si_sdr_loss(est.float()[:, 0], ty[:, 0], mk[:, 0])
        if not torch.isfinite(loss):
            n_bad += 1; log(f"step {step}: non-finite loss skipped ({n_bad})")
            if n_bad > 50:
                raise RuntimeError("too many non-finite losses")
            continue
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
        if step % 1000 == 0:
            log(f"step {step} | {el / 3600:.2f} h | lr {lr:.2e} | train masked SI-SDR {-run:.2f} dB")
        if time.time() - last_save > 1800:
            torch.save(model.state_dict(), f"{out}/latest.pt"); last_save = time.time()
        if time.time() - last_sanity > 7200:
            w, i = sanity(model, held, dev); last_sanity = time.time()
            log(f"== step {step} | {el / 3600:.2f} h | sanity whole-clip {w:.2f} dB | inside events {i:.2f} dB")
    torch.save(model.state_dict(), f"{out}/final.pt")
    w, i = sanity(model, held, dev)
    json.dump({"variant": args.variant, "steps": step, "hours": args.hours, "bs": args.bs, "lr": args.lr,
               "sanity_whole": w, "sanity_inside": i, "bad": n_bad}, open(f"{out}/summary.json", "w"), indent=1)
    log(f"DONE variant {args.variant}: {step} steps | final sanity whole-clip {w:.2f} dB | inside events {i:.2f} dB")


def launch(args):
    """Kernel entry: run seeds s1 and s2 of the v38a recipe in parallel, one per GPU, and wait for both."""
    import subprocess
    n_gpu = torch.cuda.device_count()
    log(f"GPUs: {n_gpu} | {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")
    plan = [("s1", 0), ("s2", 1)] if n_gpu >= 2 else [("s1", 0)]
    procs = []
    for v, g in plan:
        lf = open(os.path.join(OUT_ROOT, f"train_{v}.log"), "w")
        p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--variant", v, "--gpu", str(g),
                              "--hours", str(args.hours), "--bs", str(args.bs), "--lr", str(args.lr)],
                             stdout=lf, stderr=subprocess.STDOUT)
        procs.append((v, p, lf)); log(f"started {v} on GPU {g}, pid {p.pid}")
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(600)
    for v, p, lf in procs:
        lf.close(); log(f"{v} exit code {p.returncode}")


if __name__ == "__main__":
    main()
