import json
import matplotlib.pyplot as plt

log_path_embeddModel = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0/checkpoint-5000/trainer_state.json"
log_path_phi2_ft1 = "./Finetuning Checkpoints 2.0/final-checkpoint/checkpoint-12450/trainer_state.json"

with open(log_path_embeddModel, "r") as f:
    logs = json.load(f)

steps = []
loss_values = []

for log in logs["log_history"]:
    if "loss" in log:
        steps.append(log["step"])
        loss_values.append(log["loss"])

# Plot the smooth loss curve
plt.plot(steps, loss_values, linestyle="-", linewidth=2, alpha=0.8)  # Smooth line without markers
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True, linestyle="--", alpha=0.6)  # Light dashed grid for readability
plt.show()