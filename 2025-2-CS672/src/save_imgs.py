import numpy as np
import matplotlib.pyplot as plt

# Load
ent_base = np.load("results/entropy_baseline.npy").mean(axis=0)
ent_all = np.load("results/entropy_all.npy").mean(axis=0)
ent_high = np.load("results/entropy_high.npy").mean(axis=0)
ent_low = np.load("results/entropy_low.npy").mean(axis=0)

plt.figure(figsize=(12, 6))

# Plot Baseline
plt.plot(ent_base, label="Baseline (No RL)", color='black', linewidth=2, linestyle='--')

# Plot RL Methods
plt.plot(ent_all, label="RL-All", color='orange', alpha=0.7)
plt.plot(ent_high, label="RL-High (Ours)", color='red', linewidth=2)
plt.plot(ent_low, label="RL-Low", color='blue', alpha=0.5)

plt.title("Entropy Profile by Method")
plt.xlabel("Timestep")
plt.ylabel("Base Policy Uncertainty")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/entropy_comparison_final.png")
print("Plot saved.")