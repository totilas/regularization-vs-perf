#!/bin/bash
#SBATCH --job-name=perforidge
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecyffers@ista.ac.at
#SBATCH --partition=defaultp
#SBATCH --qos=normal
#SBATCH --ntasks=288
#SBATCH --nodes=1
#SBATCH --mem=288G
#SBATCH --time=3-30:00:00
#SBATCH --output=out/slurm-%A_%a.out
#SBATCH --error=out/slurm-%A_%a.err

date
uv run proportional/perforidge.py --config proportional/runs.toml
date
