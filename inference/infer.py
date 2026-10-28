"""Track 2 end-to-end inference: denoise -> blind template discovery (2 passes) -> subtract + denoise -> ASR -> zip.

    python infer.py --test-dir /path/to/phase2_test_wavs --work ../work
    python infer.py --test-dir ... --work ../work --templates ../artifacts/templates_pass1.pkl ../artifacts/templates_pass2.pkl

Steps (all outputs under --work):
  1. denoise.py              enh_base/      mean of the two fine-tuned Demucs seeds (fallback + discovery reference)
  2. template_fast.py        tpl_p1.pkl     pass 1: repeated noise waveforms found in the residuals x - y
  3. tpl_write_sub.py        mix_p1sub/     x - sum(g*T) for pass 2
  4. tpl_candidate_final.py  enh_p1B/       pass-1 subtraction + denoiser (reference for pass 2)
  5. template_fast.py        tpl_p2.pkl     pass 2 on the pass-1-subtracted mixtures
  6. tpl_candidate_final.py  enh_final/     all template sets subtracted, then denoised; others = enh_base
     -> t2_pack_dir.py       datathon_t2_final.zip  (SraVaani on exactly the written files, deflate-9, < 1e9 bytes)
With --templates, steps 2-5 are skipped and the template sets released with the code are used (exact reproduction
of the leaderboard submission without the ~4 h CPU discovery).
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def run(cmd, work, extra_env=None):
    env = dict(os.environ, T2_WORK=work, PYTHONIOENCODING="utf-8", **(extra_env or {}))
    print(">>", " ".join(cmd), flush=True)
    rc = subprocess.call([sys.executable, "-u"] + cmd, cwd=HERE, env=env)
    if rc:
        raise SystemExit(f"step failed ({rc}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="folder with the 3178 Phase-2 test WAVs (searched recursively)")
    ap.add_argument("--work", default=os.path.join(REPO, "work"))
    ap.add_argument("--weights", nargs="+", default=[os.path.join(REPO, "checkpoints", "v39_s1.pt"),
                                                      os.path.join(REPO, "checkpoints", "v39_s2.pt")])
    ap.add_argument("--templates", nargs="+", default=None, help="reuse released template sets (skip discovery)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    work = os.path.abspath(args.work); os.makedirs(work, exist_ok=True)
    weights = [os.path.abspath(w) for w in args.weights]
    link = os.path.join(work, "phase2_test")
    if not os.path.exists(link):
        try:
            os.symlink(os.path.abspath(args.test_dir), link, target_is_directory=True)
        except OSError:                               # no symlink rights (e.g. Windows without developer mode)
            shutil.copytree(os.path.abspath(args.test_dir), link)

    run(["denoise.py", "--weights", *weights, "--tag", "base"], work)
    if args.templates:
        tpls = [os.path.abspath(t) for t in args.templates]
    else:
        run(["template_fast.py", "--enh", "enh_base", "--tag", "p1", "--rule", "-99", "--workers", str(args.workers)],
            work)
        run(["tpl_write_sub.py", "--tpl", "tpl_p1.pkl", "--out", "mix_p1sub"], work)
        run(["tpl_candidate_final.py", "--tpl", "tpl_p1.pkl", "--weights", *weights, "--fallback", "enh_base",
             "--tag", "p1B", "--no-pack"], work)
        run(["template_fast.py", "--enh", "enh_p1B", "--tag", "p2", "--rule", "-99", "--workers", str(args.workers)],
            work, {"TPL_MIX": "mix_p1sub"})
        tpls = [os.path.join(work, "tpl_p1.pkl"), os.path.join(work, "tpl_p2.pkl")]
    run(["tpl_candidate_final.py", "--tpl", *tpls, "--weights", *weights, "--fallback", "enh_base", "--tag", "final"],
        work)
    print("submission:", os.path.join(work, "datathon_t2_final.zip"), flush=True)


if __name__ == "__main__":
    main()
