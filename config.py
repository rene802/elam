import os

base_dir = "results"
model_dir = os.path.join(base_dir, "models")
plot_dir = os.path.join(base_dir, "plots")
root_dir = os.path.join(base_dir, "root")
log_dir = os.path.join(base_dir, "logs")

for d in [model_dir, plot_dir, root_dir, log_dir]:
    os.makedirs(d, exist_ok=True)
