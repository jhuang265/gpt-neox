#!/bin/bash
#BSUB -nnodes 46
#BSUB -W 2:00
#BSUB -q debug
#BSUB -o /gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/logs/test-modified_roberta_base_2048_mlm-%J.out
#BSUB -e /gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/logs/test-modified_roberta_base_2048_mlm-%J.err
#BSUB -J test-modified_roberta_2048_base_mlm
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499
#BSUB -N jerry.huang@mila.quebec
#BSUB -B jerry.huang@mila.quebec

# Set up the environment
source ~/.bashrc
source /gpfs/alpine/csc499/scratch/jerry.huang/setup.sh

# Activate conda environment
# source /gpfs/alpine/csc499/scratch/jerry.huang/miniconda3/etc/profile.d/conda.sh
conda activate gpt-neox
which python

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/jerry.huang/latest_install/cache

# Move to the gpt-neox install
export TRAIN_PATH=/gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/jerry.huang/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/jerry.huang/hostfiles/$LSB_JOBID-hosts

# Write a file just to ensure we can track all jobs
echo -e "$LSB_JOBID" >> $TRAIN_PATH/info/$LSB_JOBNAME.info

# Run
python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs_mlm \
	setup/setup_modified_roberta_2048_base_resume.yml \
	modified_roberta_2048/modified_roberta_2048_base.yml \
	datasets_ben/val/pile_slimp.yml \
	datasets_ben/train/slim_pajama_606B.yml \
	load_ben/none.yml \
