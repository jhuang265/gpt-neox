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

tmp = "\"/gpfs/alpine/csc499/scratch/btherien/gpt-neox/checkpoints/continued_slim_pajama/JOB-{}_pythia-deduped-410M-iters-131296_warmup-0.0_max-lr-3e-05_min-lr-3e-05_pretrain_slim_pajama_resume\"".format(args.job_id)

file_contents = "{\n\"load\":" + tmp + "\n}"

with open("/gpfs/alpine/csc499/scratch/btherien/gpt-neox/configs/load_ben/3e-5const_0_410M_143_CPT.yml",'w') as f:
  f.write(file_contents)


#Change job script
job_script_contents="""#!/bin/bash
#BSUB -nnodes 46
#BSUB -W 6:00
#BSUB -q batch
#BSUB -o /gpfs/alpine/csc499/scratch/btherien/training_logs/gpt_neox_out.%J
#BSUB -e /gpfs/alpine/csc499/scratch/btherien/training_logs/gpt_neox_err.%J
#BSUB -J 3e-5const_0_410M_143_CPT
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499
#BSUB -N btherien@uwaterloo.ca
#BSUB -B btherien@uwaterloo.ca

# Set up the environment
source /gpfs/alpine/csc499/scratch/btherien/setup.sh
/gpfs/alpine/csc499/scratch/btherien/miniconda3/bin/activate 

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/btherien/latest_install/cache

# Move to the gpt-neox install
export TRAIN_PATH=/gpfs/alpine/csc499/scratch/btherien/gpt-neox
cd $TRAIN_PATH

# Kill previous job and setup next job pickup
bkill {}
python /gpfs/alpine/csc499/scratch/btherien/future_launch.py --job-id $LSB_JOBID &

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/btherien/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/btherien/hostfiles/$LSB_JOBID-hosts

python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs \
pythia_410m_llama_setup_resume.yml \
llama/410M.yml \
datasets_ben/val/pile_slimp.yml \
datasets_ben/train/slim_pajama.yml \
load_ben/3e-5const_0_410M_143_CPT.yml \
schedules/adam_constant_lr3e-5_3e-5_wu-0.yml
""".format(args.job_id, args.job_id)

job_script_path = "/gpfs/alpine/csc499/scratch/btherien/slim_pajama_experiments/resume/hp_410M_train_3_0_constant.sh"
with open(job_script_path,'w') as f:
  f.write(job_script_contents)



# sleep 4 hours before submitting a new job
if not args.no_sleep:
  time.sleep(4 * 60 * 60)
os.system("bsub {}".format(job_script_path))
