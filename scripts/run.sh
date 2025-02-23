#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -n 32
#SBATCH --mem=64G
#SBATCH -J Jay-Harmonization
#SBATCH -o ../logs/log-Jay-Harmonization-%j.out

cd /gpfs/data/tserre/jgopal/co3d-harmonization-pytorch/

source venv/bin/activate

python3 co3d_harmonization2.py


