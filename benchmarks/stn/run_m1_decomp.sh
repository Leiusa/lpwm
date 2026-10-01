#!/bin/bash
# Milestone-1 visual quality check: GT/recon/foreground/background/mask grids for A vs D, all six seeds, one
# short RTX 3090 allocation. Evaluation only (loads saved checkpoints, no training, no gradient updates).
#SBATCH --job-name=m1-decomp
#SBATCH --partition=all
#SBATCH --constraint=rtx3090
#SBATCH --exclude=grogu-4-13,grogu-4-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/grogu/user/junhong3/lpwm-work/experiments/fused-integration/m1/slurm/%x-%j.out

set -euo pipefail
E=/grogu/user/junhong3/lpwm-work/experiments/fused-integration
REPO=$E/tree-m1
TOOLS=$E/m1/tools
CFG=/grogu/user/junhong3/lpwm-work/data/static_bair128_vgg.json
OUTDIR=$E/m1/decomp
mkdir -p "$OUTDIR"

echo "node: $(hostname) | job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

source /grogu/user/junhong3/lpwm-work/venvs/lpwm-stn/bin/activate
export M1_EXPECTED_GPU="NVIDIA GeForce RTX 3090"

RUNS=(seed0-3972541 seed1-3972542 seed2-3972543 seed3-3972544 seed4-3972555 seed5-3972556)
for i in 0 1 2 3 4 5; do
    run=${RUNS[$i]}
    python "$TOOLS/m1_decomp_grid.py" --repo-root "$REPO" --config "$CFG" \
        --eval-json "$E/m1/runs/$run/eval_m1.json" --seed "$i" \
        --out "$OUTDIR/seed${i}_decomp.png"
done
echo "done"
