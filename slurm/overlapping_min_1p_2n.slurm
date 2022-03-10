#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=5:00:00
#SBATCH --job-name=pce
#SBATCH --output=slurm_%A-%a.out
#SBATCH --mem=50G
#SBATCH --cpus-per-task=50
#SBATCH --array=1-20%20

# load python module
module load python/3.7.3
# module load ruse

# create a temporary directory for this job and save the name
seed_dir=${SLURM_JOB_ID}_`printf "%03d" ${SLURM_ARRAY_TASK_ID}`
tempdir=/flash/FroeseU/fede/${seed_dir}
mkdir ${tempdir}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/pce-simulation
source .venv/bin/activate

# ruse
python3 -m pce.main \
--seed ${SLURM_ARRAY_TASK_ID} \
--dir $tempdir \
--num_pop 1 \
--pop_size 96 \
--num_neurons 2 \
--perf_func OVERLAPPING_STEPS \
--agg_func MIN \
--num_steps 2000 \
--max_gen 2000 \
--cores 48

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir/* deigo:/bucket/FroeseU/fede/pce

# Clean up by removing our temporary directory
rm -r $tempdir