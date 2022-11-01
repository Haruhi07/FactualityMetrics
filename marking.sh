#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_short
#SBATCH --job-name=eval_0
#SBATCH --time=6:0:0
#SBATCH --mem=8000M

cd "${SLURM_SUBMIT_DIR}"

module load lang/python/anaconda/3.10.4-2021-11-fencis
source venv/bin/activate
python3 -u marker.py --summary results/random/predictions.json --reference results/references.json --source results/source.json --rouge --factcc --output repoch.json
