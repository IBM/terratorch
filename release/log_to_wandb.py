import sys
import wandb

wandb.init(project="terratorch")
log_lines = []

for line in sys.stdin:
    stripped = line.strip()
    print(stripped) 
    log_lines.append([stripped])

table = wandb.Table(data=log_lines, columns=["Output"])
wandb.log({"IntegrationTestOutput": table})
