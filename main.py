#!/usr/bin/env python3
"""
IndoML 2026 Datathon - Track 2: Noise Event Removal
Official Model Inference Entry Point (main.py)

Team: syehsyoh (Codabench Submission #995100, Rank 2)
Team Leader: Saifuddin
Email: syehsaifuddin@gmail.com
Phone / WhatsApp: +62 822-7217-1011

Usage:
    python main.py --test-dir /path/to/test_wavs
    python main.py /path/to/test_wavs
    python main.py --test_dir /path/to/test_wavs --work ./work
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(ROOT, "inference")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


def parse_args():
    parser = argparse.ArgumentParser(description="IndoML 2026 Track 2 Inference (main.py)")
    parser.add_argument("pos_test_dir", nargs="?", default=None,
                        help="Positional path to directory containing test .wav clips")
    parser.add_argument("--test-dir", "--test_dir", "--input-dir", "--input_dir",
                        dest="test_dir", default=None,
                        help="Directory containing test .wav clips")
    parser.add_argument("--work", "--output-dir", "--output_dir",
                        dest="work", default=os.path.join(ROOT, "work"),
                        help="Working / output directory for artifacts and submission zip")
    parser.add_argument("--weights", nargs="+",
                        default=[os.path.join(CHECKPOINTS_DIR, "v39_s1.pt"),
                                 os.path.join(CHECKPOINTS_DIR, "v39_s2.pt")],
                        help="Paths to fine-tuned model checkpoint weights")
    parser.add_argument("--templates", nargs="+",
                        default=[os.path.join(ARTIFACTS_DIR, "templates_pass1.pkl"),
                                 os.path.join(ARTIFACTS_DIR, "templates_pass2.pkl")],
                        help="Template files for fast reproduction (~30 min). Set to empty to run full discovery (~4 h).")
    parser.add_argument("--full-discovery", action="store_true",
                        help="Force full CPU blind template discovery from scratch instead of using released templates")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of CPU worker processes for template discovery")
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    test_dir = args.test_dir or args.pos_test_dir
    if not test_dir:
        candidates = [os.path.join(ROOT, "phase2_test"),
                      os.path.join(ROOT, "test_data"),
                      os.path.join(ROOT, "data", "test")]
        for c in candidates:
            if os.path.isdir(c):
                test_dir = c
                print(f"[*] Auto-detected test directory: {test_dir}")
                break
        if not test_dir:
            parser.print_help()
            sys.exit("\n[!] Error: Please specify the test audio directory via: python main.py --test-dir /path/to/test_wavs")

    test_dir = os.path.abspath(test_dir)
    work_dir = os.path.abspath(args.work)
    os.makedirs(work_dir, exist_ok=True)

    cmd = [
        sys.executable, "-u", os.path.join(INFERENCE_DIR, "infer.py"),
        "--test-dir", test_dir,
        "--work", work_dir,
        "--weights", *args.weights,
        "--workers", str(args.workers)
    ]

    if not args.full_discovery and args.templates:
        existing_templates = [t for t in args.templates if os.path.exists(t)]
        if existing_templates:
            cmd += ["--templates", *existing_templates]

    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    print(f"[*] Executing Track 2 Pipeline with entry-point main.py")
    print(f"[*] Test directory : {test_dir}")
    print(f"[*] Work directory : {work_dir}")
    print(f">> {' '.join(cmd)}")

    rc = subprocess.call(cmd, cwd=INFERENCE_DIR, env=env)
    if rc != 0:
        sys.exit(f"[!] Pipeline execution failed with return code {rc}")
    print(f"[✓] Pipeline complete! Output submission zip: {os.path.join(work_dir, 'datathon_t2_final.zip')}")


if __name__ == "__main__":
    main()
