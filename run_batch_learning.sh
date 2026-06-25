#!/bin/bash
#SBATCH --job-name=peloton_batch
#SBATCH --partition=rome        # CPU partition; verifica con `sinfo` quali sono attive per il tuo progetto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00          # PLACEHOLDER — vedi nota sotto, va calibrato
#SBATCH --output=logs/batch_%j.log
#SBATCH --error=logs/batch_%j.err

cd /home/fperlini1/peloton/   # <-- aggiusta al tuo path reale

mkdir -p logs

uv run python batch_learning.py \
  --replications 150 \
  --generations 100 \
  --output-dir /home/fperlini1/snellius_data/batch_learning
