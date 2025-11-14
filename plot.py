import matplotlib.pyplot as plt

# Exp1 results
baseline_train_loss = [5.3412, 3.5758, 2.7420, 2.1661]
baseline_val_loss   = [3.9494, 2.8789, 2.2043, 1.7451]

learnable_train_loss = [5.4042, 3.3787, 2.3282, 1.7368]
learnable_val_loss   = [3.9490, 2.5177, 1.7497, 1.3724]

epochs = range(1, 5)  # 4 epochs

plt.figure(figsize=(8, 5))
plt.plot(epochs, baseline_train_loss, 'o-', label='Sinusoidal Train')
plt.plot(epochs, baseline_val_loss, 'o--', label='Sinusoidal Val')
plt.plot(epochs, learnable_train_loss, 's-', label='Learnable Train')
plt.plot(epochs, learnable_val_loss, 's--', label='Learnable Val')

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Exp1: Training and Validation Loss Curves")
plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure to results folder
plt.savefig("results/exp1_loss_curves.png", dpi=300)
plt.show()


# Exp1 results
baseline_train_loss = [5.3412, 3.5758, 2.7420, 2.1661]
baseline_val_loss   = [3.9494, 2.8789, 2.2043, 1.7451]
baseline_val_bleu   = [0.5035, 0.6229, 0.7000, 0.7515]

learnable_train_loss = [5.4042, 3.3787, 2.3282, 1.7368]
learnable_val_loss   = [3.9490, 2.5177, 1.7497, 1.3724]
learnable_val_bleu   = [0.4855, 0.6539, 0.7358, 0.7804]

epochs = range(1, 5)  # 4 epochs

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot loss curves on left y-axis
ax1.plot(epochs, baseline_train_loss, 'o-', label='Sinusoidal Train Loss', color='blue')
ax1.plot(epochs, baseline_val_loss, 'o--', label='Sinusoidal Val Loss', color='blue', alpha=0.7)
ax1.plot(epochs, learnable_train_loss, 's-', label='Learnable Train Loss', color='red')
ax1.plot(epochs, learnable_val_loss, 's--', label='Learnable Val Loss', color='red', alpha=0.7)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_xticks(epochs)
ax1.grid(alpha=0.3)

# Create second y-axis for BLEU
ax2 = ax1.twinx()
ax2.plot(epochs, baseline_val_bleu, 'o:', label='Sinusoidal Val BLEU', color='blue')
ax2.plot(epochs, learnable_val_bleu, 's:', label='Learnable Val BLEU', color='red')
ax2.set_ylabel("Validation BLEU Score")

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=9)

plt.title("Exp1: Loss and BLEU Progression")
plt.tight_layout()
plt.savefig("results/exp1_loss_bleu_curves.png", dpi=300)
plt.show()
