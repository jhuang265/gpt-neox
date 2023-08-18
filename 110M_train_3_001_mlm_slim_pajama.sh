#!/bin/bash
#BSUB -nnodes 170
#BSUB -W 1:00
#BSUB -q batch
#BSUB -o /gpfs/alpine/csc499/scratch/btherien/training_logs/gpt_neox_out.%J
#BSUB -e /gpfs/alpine/csc499/scratch/btherien/training_logs/gpt_neox_err.%J
#BSUB -J MLM_SP_3e-4_001_110M_143
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499
#BSUB -N btherien@uwaterloo.ca
#BSUB -B btherien@uwaterloo.ca

# clean up nodes
jsrun pkill python

# Set up the environment
source /gpfs/alpine/csc499/scratch/btherien/setup.sh
/gpfs/alpine/csc499/scratch/btherien/miniconda3/bin/activate


export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/btherien/latest_install/cache

# Move to the gpt-neox install
export TRAIN_PATH=/gpfs/alpine/csc499/scratch/btherien/jerry_gpt_neox/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/btherien/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/btherien/hostfiles/$LSB_JOBID-hosts


python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs \
pythia_410m_llama_setup_resume.yml \
llama/110M.yml \
datasets_ben/val/slimp.yml \
datasets_ben/train/slim_pajama_606B.yml \
load_ben/none.yml \
schedules/adam_cosine_lr3e-4_3e-5_wu-001.yml
