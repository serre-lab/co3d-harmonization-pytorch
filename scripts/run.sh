#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -n 32
#SBATCH --mem=80G
#SBATCH -J Jay-Harmonization
#SBATCH -o ../logs/log-Jay-Harmonization-%j.out

cd /gpfs/data/tserre/jgopal/co3d-harmonization-pytorch/

source venv/bin/activate

wandb login 01f3b5777f198b3606ba1407874f5c7c4b4ce59b

python3 co3d_harmonization2.py


