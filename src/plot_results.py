import matplotlib
matplotlib.use("Agg")  # saves to file, opens no window
import matplotlib.pyplot as plt

models = ["Quantum\n(IQP)", "Logistic\nRegression", "BERT\n(pretrained)"]
val_acc = [0.57, 0.76, 0.92]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(models, val_acc, color=["#7F77DD", "#1D9E75", "#D85A30"])
ax.axhline(0.5, color="gray", linestyle="--", label="Chance (50%)")
ax.set_ylabel("Validation accuracy")
ax.set_ylim(0, 1)
ax.set_title("SST-2 validation accuracy (100 sentences)")
ax.legend()
for bar, v in zip(bars, val_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Saved results.png")