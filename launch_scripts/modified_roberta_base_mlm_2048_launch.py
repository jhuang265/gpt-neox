import time
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--job-id", type=int, required=True)
parser.add_argument("--no-sleep", action='store_true',default=False)
args = parser.parse_args()


# sleep 60 minutes to allow training to start
if not args.no_sleep:
  time.sleep(60 * 60)

tmp = "\"/gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/checkpoints/modified_roberta_base_2048_mlm/\"".format(args.job_id)

file_contents = "{\n\"load\":" + tmp + "\n}"

with open("/gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/configs_mlm/load_ben/modified_roberta_base_2048_mlm.yml",'w') as f:
  f.write(file_contents)

#Change job script
job_script_contents="""#!/bin/bash
#BSUB -nnodes 46
#BSUB -W 6:00
#BSUB -q batch
#BSUB -o /gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/logs/modified_roberta_base_2048_mlm-%J.out
#BSUB -e /gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/logs/modified_roberta_base_2048_mlm-%J.err
#BSUB -J modified_roberta_base_2048_mlm
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
python /gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/launch_scripts/modified_roberta_base_2048_mlm_launch.py --job-id $LSB_JOBID &
PYTHON_PID=$!
echo "Hidden ID: $PYTHON_PID"

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/jerry.huang/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/jerry.huang/hostfiles/$LSB_JOBID-hosts

# Write a file just to ensure we can track all jobs
echo -e "$LSB_JOBID" >> $TRAIN_PATH/info/$LSB_JOBNAME.info

# Run
python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs_mlm \
setup/modified_setup_roberta_base_resume.yml \
modified_roberta/modified_roberta_base_2048.yml \
datasets_ben/val/pile_slimp.yml \
datasets_ben/train/slim_pajama_606B.yml \
load_ben/modified_roberta_base_2048_mlm.yml \
""".format(args.job_id, args.job_id)

job_script_path = "/gpfs/alpine/csc499/scratch/jerry.huang/gpt-neox/jobs/modified_roberta_base_2048_mlm-recurring.sh"
with open(job_script_path,'w') as f:
  f.write(job_script_contents)

# sleep 4 hours before submitting a new job
if not args.no_sleep:
  time.sleep(4 * 60 * 60)

os.system("bsub {}".format(job_script_path))
