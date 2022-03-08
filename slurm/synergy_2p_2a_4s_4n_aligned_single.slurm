#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=5:00:00
#SBATCH --job-name=formation
#SBATCH --output=slurm_%A-%a.out
#SBATCH --mem=50G
#SBATCH --cpus-per-task=48

# load python module
module load python/3.7.3
# module load ruse

seed=1

# create a temporary directory for this job and save the name
seed_dir=${SLURM_JOB_ID}_`printf "%03d" ${seed}`
tempdir=/flash/FroeseU/fede/${seed_dir}
mkdir ${tempdir}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/formation
source .venv/bin/activate

# ruse
python3 -m formation.main \
--seed ${seed} \
--dir $tempdir \
--synergy_max \
--gen_zfill \
--num_pop 2 \
--pop_size 48 \
--num_agents 2 \
--num_sensors 4 \
--num_neurons 4 \
--max_gen 5000 \
--noshuffle \
--cores 48

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir/* deigo:/bucket/FroeseU/fede/formation/

# Clean up by removing our temporary directory
rm -r $tempdir