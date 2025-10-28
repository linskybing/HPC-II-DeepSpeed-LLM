#! /bin/bash

source /etc/profile.d/modules.sh
module use /home/sky/modulefiles
module purge
module load rocm ucx ucc openmpi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sky/pretrain-deepspeed


export LD_LIBRARY_PATH=/home/sky/miniconda3/envs/deepspeed/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/home/sky/miniconda3/envs/deepspeed/lib/libomp.so:/lib/x86_64-linux-gnu/libtinfo.so.6

export ROOT=/home/sky/LLM # [TODO] Please change this line to your directory path.
export LOGS=$ROOT/logs
export CONFIG=$ROOT/configs
export RUN=$ROOT/run

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True"

export HIP_VISIBLE_DEVICES="1,2"
mpirun -np 2 bash -c 'python $RUN/amd/pretrain.py \
    --deepspeed_config $CONFIG/amd/zero_3_offload.json \
    --batch_size 1 \
    --seq_len 350 \
    --total_steps 100' \
    > "$LOGS/amd/zero_3R_offload_amd.log" 2>&1
