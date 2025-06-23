import sys
import wandb

wandb.init(project="terratorch")

for line in sys.stdin:
        wandb.log({"stdout": line.strip()})
