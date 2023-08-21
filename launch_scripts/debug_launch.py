import time
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--job-id", type=int, required=True)
parser.add_argument("--no-sleep", action='store_true',default=False)
args = parser.parse_args()


# sleep 60 minutes to allow training to start
if not args.no_sleep:
  time.sleep(10 * 60)

tmp = "\"/gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/checkpoints/roberta_base/\"".format(args.job_id)

file_contents = "{\n\"load\":" + tmp + "\n}"

with open("/gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/configs/load_ben/debug_mlm.yml",'w') as f:
  f.write(file_contents)

#Change job script
job_script_contents="""#!/bin/bash
#BSUB -nnodes 29
#BSUB -W 2:00
#BSUB -q batch
#BSUB -o /gpfs/alpine/csc499/scratch/jerry.huang/logs/debug_roberta_base_mlm_out.%J
#BSUB -e /gpfs/alpine/csc499/scratch/jerry.huang/logs/debug_roberta_base_mlm_err.%J
#BSUB -J debug_roberta_base_mlm
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

# Kill previous job and setup next job pickup
bkill {}
python /gpfs/alpine/csc499/scratch/jerry.huang/future_launch.py --job-id $LSB_JOBID &

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/jerry.huang/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/jerry.huang/hostfiles/$LSB_JOBID-hosts

python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs_mlm \
setup/setup_roberta_base_resume.yml \
roberta/roberta_base.yml \
datasets_ben/val/pile_slimp.yml \
datasets_ben/train/slim_pajama.yml \
load_ben/debug.yml \
""".format(args.job_id, args.job_id)

job_script_path = "/gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/jobs/debug_roberta_base_mlm.sh"
with open(job_script_path,'w') as f:
  f.write(job_script_contents)

# sleep 4 hours before submitting a new job
if not args.no_sleep:
  time.sleep(30 * 60)

os.system("bsub {}".format(job_script_path))
