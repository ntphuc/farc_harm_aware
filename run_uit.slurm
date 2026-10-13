#!/bin/bash
#SBATCH --job-name=farc_brats2021
#SBATCH --output=logs/brats2021_%j.out
#SBATCH --error=logs/brats2021_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

bash RUN_BRATS2021_FARC.sh
