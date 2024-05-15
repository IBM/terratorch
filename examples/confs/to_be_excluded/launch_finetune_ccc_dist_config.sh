#!/bin/sh
# Ease error debugging on CCC
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# Configure CUDA environment
export CUDA_HOME=/opt/share/cuda-12.1
export CUDA_BIN_PATH=$CUDA_HOME/bin
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDA_HOME/lib
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

#ROOT_DIR=/dccstor/jlsa931/GEOFM
CASE_NAME=swin_bs_16_gpu_yaml
LOG_DIR=your_path_here/torchgeo_uhi
YAML_PATH=your_path_here/uhi.yaml
#YAML_PATH=$2

MEM_LIMIT=48G
N_CPUS=8
N_GPUS=2
N_NODES=1
TASK=fit
Q=pyrite
GPU_TYPE=a100_40gb

JBSUB_STR="jbsub -e ${LOG_DIR}/${CASE_NAME}/${CASE_NAME}.stderr -o ${LOG_DIR}/${CASE_NAME}/${CASE_NAME}.stdout -q ${Q} -m ${MEM_LIMIT} -c ${N_NODES}x${N_CPUS}+${N_GPUS} -r ${GPU_TYPE}"

${JBSUB_STR} blaunch.sh terratorch ${TASK} --config ${YAML_PATH}
    #--trainer.gpus $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w) \
    #--trainer.num_nodes $(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w) \
    #| tee $HOSTNAME.$LSF_PM_XPROCID.$(date +%F_%R).log