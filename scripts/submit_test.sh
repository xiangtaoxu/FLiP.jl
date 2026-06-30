#!/bin/bash
#SBATCH --job-name=flip_ARN-RF0-test
#SBATCH --nodelist=cbsuxu10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=flip_%j.out
#SBATCH --error=flip_%j.err

set -euo pipefail

# --- Edit these two if they differ on cbsuxu10 -----------------------------
# FLiP.jl repo root on cbsuxu10 (home there is /home/xx286):
FLIP_DIR="$HOME/projects/FLiP.jl"
# ---------------------------------------------------------------------------

CONFIG="${FLIP_DIR}/analyses/flip_config_ARN-RF0-20221115.toml"

cd "$FLIP_DIR"

julia --project=. -t "${SLURM_CPUS_PER_TASK}" \
    -e 'using FLiP; run_pipeline("'"${CONFIG}"'")'

echo "Finished $(date)"
