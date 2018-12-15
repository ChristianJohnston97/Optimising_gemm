#!/bin/bash
#SBATCH -p par7.q
#SBATCH -n 1
source /etc/profile.d/modules.sh
module purge
module load slurm/current
module load intel/xe_2018.2






