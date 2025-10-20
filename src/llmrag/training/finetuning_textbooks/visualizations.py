import json
import matplotlib.pyplot as plt

log_path_embeddModel = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka/checkpoint-10128/trainer_state.json"
log_path_phi2_ft2 = "./finetuning_textbooks/finetuning_checkpoints/final_checkpoint/checkpoint-10000/trainer_state.json"

with open(log_path_phi2_ft2, "r") as f:
    logs = json.load(f)

steps = []
loss_values = []
eval_loss_values = []

for log in logs["log_history"]:
    if "loss" in log:
        steps.append(log["step"])
        loss_values.append(log["loss"])
    elif "eval_loss" in log:
        eval_loss_values.append(log["eval_loss"])

# Plot the smooth loss curve
plt.plot(steps, loss_values, linestyle="-", linewidth=2, alpha=0.8, label="Training Loss")  # Smooth line without markers
plt.plot(steps, eval_loss_values, linestyle="-", color='r', linewidth=2, alpha=0.8, label="Validation Loss")  # Smooth line without markers
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.grid(True, linestyle="--", alpha=0.6)  # Light dashed grid for readability
plt.legend()
plt.show()