"""Figure 1 of the Track 2 report: leaderboard bridge from the first fine-tuned denoiser to the final submission."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEPS = [("Fine-tuned dns64", 9.894), ("Longer, all data", 11.2232), ("Event-free targets", 11.7715),
         ("10.3 h on 2xT4", 12.366), ("+8.5 h, 2 seeds", 12.7544), ("Templates, pass 1", 20.0315),
         ("Templates, pass 2", 21.2379)]

fig, ax = plt.subplots(figsize=(3.5, 2.3), dpi=300)
prev = 0.0
for i, (name, v) in enumerate(STEPS):
    lo, h = (0.0, v) if i == 0 else (prev, v - prev)
    ax.barh(i, h, left=lo, height=0.6, color="#9e9e9e" if i == 0 else "#4caf50", edgecolor="none")
    ax.text(v + 0.3, i, f"{v:.2f}" if i == 0 else f"+{v - prev:.2f}", va="center", fontsize=6.5)
    if i < len(STEPS) - 1:
        ax.plot([v, v], [i + 0.3, i + 0.7], color="#757575", lw=0.5, ls=":")
    prev = v
n = len(STEPS)
ax.barh(n, prev, height=0.6, color="#616161")
ax.text(prev + 0.3, n, f"{prev:.2f}", va="center", fontsize=6.5, fontweight="bold")
ax.axvline(26.49, color="#1565c0", lw=0.7, ls="--")
ax.text(26.3, -0.9, "rank 1", color="#1565c0", fontsize=6, ha="right")
ax.set_yticks(range(n + 1), [s for s, _ in STEPS] + ["Final (v42B)"], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0, 28)
ax.set_xlabel("Phase-2 combined score (SI-SDR + $\\Delta$WER$\\times$100)", fontsize=6.5)
ax.tick_params(axis="x", labelsize=6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_bridge.pdf"))
print("ok")
