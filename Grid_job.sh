#!/bin/bash -x

#SBATCH --job-name="Grid Search"
#SBATCH --output=./out.%j
#SBATCH --error=./err.%j

#SBATCH --account=jjsc37
#SBATCH --partition=dc-cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mail-user=maria.dinca0501@gmail.com
#SBATCH --mail-type=FAIL


set -e

source /p/project/cjjsc37/software/Isle/isle-jureca-cpu/activate.sh
ml PyTorch

# Load all modules and activate virtualenv.
# This is not necessary, if the venv has already been loaded in the intercative shell.
echo -e "Start `date +"%F %T"` | $SLURM_JOB_ID $SLURM_JOB_NAME | `hostname` | `pwd` \n"
srun python -u /p/home/jusers/dinca1/jureca/ComplexValuedNetworks/NewGridSearch.py
echo -e "End `date +"%F %T"` | $SLURM_JOB_ID $SLURM_JOB_NAME | `hostname` | `pwd` \n"