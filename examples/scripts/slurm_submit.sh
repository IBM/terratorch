#!/bin/sh
# slurm_submit.sh - Batch script for Slurm environment
# Usage:
#   $ slurm_submit.sh <script-name> <num-nodes> <num-gpus>
#   where
#     script-name: Name of script to run in parallel (e.g. sen1floods11_vit_fit.sh)
#     num-nodes:   Number of nodes (1, 2, ...)
#     num-gpus     Number of GPUs in a node (1, 2, 3, or 4)
script_name=$1
num_nodes=$2
num_gpus=$3
logfile="log/slurm-%j.out"
if [ $# -ne 3 ]; then
    echo "Usage: %0 <script-name> <num-nodes> <num-gpus>"
    exit 1
fi
# Used to set CUDA_VISIBLE_DEVICES so that all GPUs on a node can be accessible from all tasks on the node
devices=("" "0" "0,1" "0,1,2" "0,1,2,3")

sbatch -o ${logfile} -e ${logfile} <<EOF
#!/bin/sh
#SBATCH --account=geofm4eo
#SBATCH --partition=booster
#SBATCH --job-name "terratorch"
#SBATCH --nodes=${num_nodes}
#SBATCH --ntasks-per-node=${num_gpus}
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

export CUDA_VISIBLE_DEVICES=${devices[$num_gpus]}
srun -o ${logfile} -e ${logfile} ${script_name}
EOF
