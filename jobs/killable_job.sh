#!/bin/bash
#BSUB -nnodes 91
#BSUB -W 24:00
#BSUB -q killable
#BSUB -o short_job_%J.out
#BSUB -e short_job_%J.err
#BSUB -J gpt_neox
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499

# Set up the environment
source /gpfs/alpine/csc499/scratch/$(whoami)/setup.sh
# The default cache location is read-only on Summit. Redirect it to somewhere in your scratch dir
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/$(whoami)/cache

# Move to the gpt-neox install
TRAIN_PATH=/gpfs/alpine/csc499/scratch/$(whoami)/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/$(whoami)/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/$(whoami)/hostfiles/$LSB_JOBID-hosts


python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
	         --conf_dir $TRAIN_PATH/configs_jerry redpajama.yaml summit-1-3B_bsz-16.yml summit_setup.yml
