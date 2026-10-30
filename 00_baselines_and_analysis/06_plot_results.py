"""Plot macro-F1 of the seven models with 95% bootstrap confidence intervals."""
import matplotlib.pyplot as plt
import pandas as pd

from common import FINE_TUNED, RESULTS

scores = pd.read_csv(RESULTS / "final_scores_with_CI.csv")
summary = pd.read_csv(RESULTS / "baselines_summary.csv")
seed_mean = {(row.task, FINE_TUNED[row.model]): row.f1_mean for row in summary.itertuples()}
COLORS = {"LLM": "#1f77b4", "Fine-tuned": "#d95f02"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, task, title in zip(axes, ["sentiment", "emotion"], ["(a) Sentiment analysis", "(b) Emotion detection"]):
    d = scores[scores["task"] == task].copy()
    d["order"] = [seed_mean.get((task, m), f) for m, f in zip(d["model"], d["f1"])]
    d = d.sort_values("order")
    for position, (_, row) in enumerate(d.iterrows()):
        color = COLORS[row["type"]]
        ax.errorbar(row["f1"], position, xerr=[[row["f1"] - row["f1_lo"]], [row["f1_hi"] - row["f1"]]],
                    fmt="o", color=color, capsize=3)
        if row["type"] == "Fine-tuned":
            ax.plot(seed_mean[(task, row["model"])], position, "D", mfc="white", mec=color, ms=7)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["model"])
    ax.set_xlabel("Macro-F1 (%)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

handles = [
    plt.Line2D([0], [0], marker="o", color=COLORS["LLM"], linestyle="", label="LLM (zero-shot, one run)"),
    plt.Line2D([0], [0], marker="o", color=COLORS["Fine-tuned"], linestyle="", label="Fine-tuned, evaluated run (95% CI)"),
    plt.Line2D([0], [0], marker="D", color=COLORS["Fine-tuned"], mfc="white", linestyle="", label="Fine-tuned, mean of 5 runs"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(RESULTS / "figure_all_models_macroF1.png", dpi=300)
print("saved", RESULTS / "figure_all_models_macroF1.png")
