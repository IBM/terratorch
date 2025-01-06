#!/bin/sh

# We assue following environ variables are available from Slurm:
#   SLURM_JOB_NUM_NODES   - Number of nodes
#   SLURM_NTASKS_PER_NODE - Nuber of GPUs (tasks) per node

cat <<EOF
echo "$0: ${SLURM_PROCID} Debug info ----------"
SLURM_JOB_ID="${SLURM_JOB_ID}"
SLURM_JOB_NAME="${SLURM_JOB_NAME}"
SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION}"
SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES}"
SLURM_JOB_ID="${SLURM_JOB_ID}"
SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST}"
SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT}"
HOST=`hostname`
SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE}"
SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
SLURM_PROCID="${SLURM_PROCID}" # global_rank
SLURM_LOCALID="${SLURM_LOCALID}" # local_rank
SLURM_NODEID="${SLURM_NODEID}" # node_rank
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
SLURM_NTASKS="${SLURM_NTASKS}"
echo "=========="
EOF

#export PYTHONFAULTHANDLER=1
#export JSONARGPARSE_DEBUG=true
#export NCCL_DEBUG=WARN

CONF=sen1floods11_vit
TASK=fit
ROOT_DIR=${CONF}_${TASK}_${SLURM_JOB_NUM_NODES}x${SLURM_NTASKS_PER_NODE}
YAML_PATH=../confs/${CONF}.yaml
# Download Sen1Floods11 data as described in https://github.com/IBM/terratorch/blob/main/examples/confs/README.md
# and set the directory below.
SEN1FLOODS11_ROOT=YOUR_DATA_REPOSITORY

time terratorch ${TASK} --config ${YAML_PATH} \
     --trainer.num_nodes ${SLURM_JOB_NUM_NODES} \
     --trainer.devices ${SLURM_NTASKS_PER_NODE} \
     --trainer.default_root_dir ${ROOT_DIR} \
     --data.init_args.train_data_root ${SEN1FLOODS11_ROOT}/v1.1/data/flood_events/HandLabeled/S2Hand/ \
     --data.init_args.train_label_data_root ${SEN1FLOODS11_ROOT}/v1.1/data/flood_events/HandLabeled/LabelHand \
     --data.init_args.val_data_root ${SEN1FLOODS11_ROOT}/v1.1/data/flood_events/HandLabeled/S2Hand/ \
     --data.init_args.val_label_data_root ${SEN1FLOODS11_ROOT}/v1.1/data/flood_events/HandLabeled/LabelHand \
     --data.init_args.test_data_root ${SEN1FLOODS11_ROOT}/v1.1/data/flood_events/HandLabeled/S2Hand/ \
     --data.init_args.test_label_data_root ${SEN1FLOODS11_ROOT}/v1.1/data/flood_events/HandLabeled/LabelHand \
     --data.init_args.train_split ${SEN1FLOODS11_ROOT}/v1.1/splits/flood_handlabeled/flood_train_data.txt \
     --data.init_args.test_split ${SEN1FLOODS11_ROOT}/v1.1/splits/flood_handlabeled/flood_test_data.txt \
     --data.init_args.val_split ${SEN1FLOODS11_ROOT}/v1.1/splits/flood_handlabeled/flood_valid_data.txt
