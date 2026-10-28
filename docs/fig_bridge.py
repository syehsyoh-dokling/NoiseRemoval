"""Figure 1 of the Track 2 report: leaderboard bridge from the first fine-tuned denoiser to the final submission.

Each bar is labelled with the absolute Phase-2 score of that system (3 decimals, as on Codabench); the green length
is the gain over the previous row, so the labels never have to add up across rounding.
"""
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEPS = [("Pretrained dns64", 7.3784), ("Fine-tuned dns64, 3k steps", 9.8940), ("30k steps, natural + Gold", 11.2232), ("Event-free masked loss", 11.7715),
         ("10.3 h on one T4", 12.366), ("+8.5 h, 2 seeds", 12.7544), ("Templates, pass 1", 20.0315),
         ("Templates, pass 2 (final)", 21.2379)]
RANK1 = 26.4894


def r3(v):
    """Format with the 4 decimals Codabench reports, the same as Table I (so figure and table always agree)."""
    return str(Decimal(str(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


fig, ax = plt.subplots(figsize=(3.5, 1.9), dpi=300)
prev = 0.0
for i, (name, v) in enumerate(STEPS):
    base = i <= 1                                    # pretrained and the 3k run are separate baselines, not a chain
    lo, h = (0.0, v) if base else (prev, v - prev)
    last = i == len(STEPS) - 1
    ax.barh(i, h, left=lo, height=0.6, color="#9e9e9e" if base else "#4caf50", edgecolor="none")
    ax.text(v + 0.3, i, r3(v), va="center", fontsize=6.5, fontweight="bold" if last else "normal")
    if not last and i >= 1:
        ax.plot([v, v], [i + 0.3, i + 0.7], color="#757575", lw=0.5, ls=":")
    prev = v
n = len(STEPS)
ax.axvline(RANK1, color="#1565c0", lw=0.7, ls="--")
ax.text(RANK1 - 0.2, -0.75, f"rank 1: {RANK1:.2f}", color="#1565c0", fontsize=6, ha="right")
ax.set_yticks(range(n), [s for s, _ in STEPS], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0, 28)
ax.set_xlabel("Phase-2 combined score (SI-SDR + $\\Delta$WER$\\times$100)", fontsize=6.5)
ax.tick_params(axis="x", labelsize=6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_bridge.pdf"))
print("ok")
