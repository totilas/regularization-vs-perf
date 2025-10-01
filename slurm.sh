#!/bin/bash
#SBATCH --job-name=perforidge
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecyffers@ista.ac.at
#SBATCH --partition=defaultp
#SBATCH --qos=normal
#SBATCH --ntasks=384
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=3-10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# print the start time
date
uv run proportional/perforidge.py --config proportional/runs.toml
# print the end time
date
