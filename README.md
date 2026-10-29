# Datathon@IndoML 2026, Track 2 (Noise Event Removal): blind noise-template subtraction + fine-tuned Demucs

Team **syehsyoh**. Final Phase-2 leaderboard: **rank 2**, Codabench submission **995100**, combined score
**21.2379** (SI-SDR 16.77 dB, ΔWER×100 4.47, audit pass).

The synthetic test clips reuse a small library of noise waveforms. We find those waveforms blindly in the test set
(from the residual of a denoiser), subtract every exact copy with a fitted gain, and run a fine-tuned Demucs denoiser
on whatever remains. Subtraction lifted our score from 12.75 (denoiser only) to 21.24.

```
x ──► Demucs ×2 (mean) ──► y ──► residual r = x − y ──► blind template discovery (pass 1)
                                                              │
x − Σ g·T  ◄──────────────────────────────────────────────────┘
   │ ──► Demucs ×2 ──► residual ──► discovery pass 2 on x − Σ g·T
   ▼
x − Σ g·T (pass 1 + pass 2) ──► Demucs ×2 (mean) ──► round to ×4 ──► SraVaani ASR ──► zip
clips with no template occurrence: y (denoiser only)
```

## Repository layout

```
README.md
requirements_train.txt          requirements_inference.txt
inference/
  infer.py                      end-to-end inference (one command)
  denoise.py                    two-seed Demucs denoiser, int16 output rounded to multiples of 4
  template_fast.py              blind template discovery on denoiser residuals (CPU, multi-process)
  tpl_write_sub.py              x − Σ g·T for the second discovery pass
  tpl_candidate_final.py        subtract all template sets, denoise the remainder, package
  t2_pack_dir.py, t2_asr.py     audit-safe packaging: SraVaani on exactly the written files, deflate-9 zip
  common.py, dn/                model + audio helpers; vendored Demucs code (facebookresearch/denoiser)
train/
  stage0_factory.py             stage 0: data factory + SI-SDR fine-tune of pretrained dns64
  stage1_evfree.py              stage 1: event-free masked SI-SDR targets (v36)
  stage2_long.py                stage 2: 10.3 h long run on one T4 of a 2×T4 kernel, batch 16 (v38a; the other GPU ran the rejected ESC-50 variant b)
  stage3_continue.py            stage 3: 8.5 h continuation, two seeds s1/s2 (v39) -> released checkpoints
  prep_kaggle_dataset.py        builds the data folder the Kaggle stages read
checkpoints/v39_s1.pt, v39_s2.pt   final denoisers (fp16 storage, 67 MB each)
artifacts/templates_pass{1,2}.pkl  template sets discovered on the Phase-2 test set (exact reproduction)
configs/model_config.yaml       model, discovery and packaging parameters
scripts/run_inference.sh        reproduce the leaderboard submission
docs/technical_report.pdf       system description
```

## Environment and hardware

| | Training | Inference |
| --- | --- | --- |
| Python | 3.13 (laptop 3.13.6; Kaggle image 3.13) | 3.13.6 |
| CUDA | laptop: 12.6 (PyTorch 2.14.0+cu126); Kaggle: the default GPU image of October 2026 | 12.6 (PyTorch 2.14.0+cu126) |
| GPU | Stages 0-1: NVIDIA RTX 4060 Laptop 8 GB. Stages 2-3: Kaggle 2× Tesla T4 16 GB | RTX 4060 Laptop 8 GB (any CUDA GPU with ≥ 4 GB) |
| CPU / RAM | Windows laptop; Kaggle standard GPU session | template discovery uses 8 worker processes, ~2 GB RAM |
| OS | Windows 11 (laptop), Linux (Kaggle) | Windows 11 / Linux |

System dependencies: none beyond a CUDA driver; `soundfile` needs libsndfile (bundled in the pip wheel).

Pretrained models:

| Model | Source | Role |
| --- | --- | --- |
| Demucs dns64 | [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) (`denoiser==0.1.5`, `pretrained.dns64`) | initialisation of stage 0 |
| SraVaani-1.0 | [ARTPARK-IISc/SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | mandated ASR for the transcripts |

## Inference

```bash
pip install -r requirements_inference.txt
# full pipeline: denoise, two discovery passes (~4 h on 8 CPU workers), subtract, denoise, ASR, zip
bash scripts/run_inference.sh /path/to/phase2_test_wavs
# exact reproduction in ~30 min with the released template sets
bash scripts/run_inference.sh /path/to/phase2_test_wavs --fast
```

Input: the 3,178 Phase-2 test WAVs (16 kHz mono; searched recursively). Output: `work/datathon_t2_final.zip`
with `transcripts.jsonl` (`{"clip_id", "text"}` per clip) and one enhanced 16 kHz PCM_16 WAV per clip, the format
of the Phase-2 evaluation. Intermediate folders under `work/`: `enh_base/` (denoiser only), `tpl_p1.pkl`,
`tpl_p2.pkl`, `enh_final/`.

Verified: with the released checkpoints and template sets, `denoise.py` and `tpl_candidate_final.py` reproduce the
submitted audio sample for sample (24/24 checked clips identical to the files of submissions 994584 and 995100,
maximum difference 0).

## Training

Data: the released validation set (695 synthetic clips with clean references and noise timestamps, 1,806 natural
clips) and the Gold tier of the Vaani noise-event dataset (10,772 clips with verified timestamps). No test labels
were used. The noise bank is the exact pasted noise of the synthetic validation clips (mixture minus clean
reference, 2,213 snippets).

Recipe (Demucs dns64, negative SI-SDR loss, 4 s crops, AdamW):

| Stage | Script | Init | Targets / loss | Schedule | Result |
| --- | --- | --- | --- | --- | --- |
| 0 | `stage0_factory.py` | pretrained dns64 | natural val + Gold as targets, bank noise pasted at random positions and levels | 30k steps, bs 8, lr 1e-4 | LB 11.22 |
| 1 | `stage1_evfree.py` | stage 0 | loss only on target samples free of annotated events (±150 ms); 695 clean refs added | 30k steps, bs 8, lr 5e-5 | LB 11.77 |
| 2 | `stage2_long.py --variant a` | stage 1 | as stage 1 | 10.3 h on T4, bs 16, warm-up + cosine, lr 5e-5 (81,482 steps) | LB 12.37 |
| 3 | `stage3_continue.py --variant s1/s2` | stage 2 | as stage 1 | 8.5 h per seed, bs 16, lr 3e-5 (78,220 / 78,631 steps) | mean of both: LB 12.75 |

```bash
pip install -r requirements_train.txt
export T2_WORK=/path/to/workdir      # holds validation/ (official layout) and hf_gold/ (audio/ + metadata.json)
python train/stage0_factory.py --stage train --steps 30000
python train/stage1_evfree.py --stage train          # T2_INIT=<stage-0 weights>
python train/prep_kaggle_dataset.py                  # folder for stages 2-3
python train/stage2_long.py --variant a --gpu 0 --hours 10.3
python train/stage3_continue.py                      # launches s1 and s2 on two GPUs
```

Note on stage 0: the exact script of our stage-0 run (internal version v24) was lost in a workspace cleanup.
`stage0_factory.py` is the same recipe at its first version (v21, validation targets, 3k steps); v24 ran it for 30k
steps with the Gold clips added as targets. Stages 1-3 are the exact scripts that produced the released checkpoints.

## License and data

Our code (everything outside `inference/dn/` and `train/dn/`) is released under the MIT License (`LICENSE`).
`inference/dn/` and `train/dn/` are vendored from facebookresearch/denoiser and keep its CC BY-NC 4.0 license; the
released checkpoints are fine-tuned from the dns64 weights and inherit that license. These upstream terms cannot be
relaxed by us: the denoiser itself is only available under CC BY-NC 4.0, so the trained model is non-commercial while
all code we wrote is permissively licensed. Vaani audio is not redistributed here. ESC-50 was tried in one rejected
variant and is not used by the final system.
