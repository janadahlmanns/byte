#!/bin/bash -l
# ============================================================
# The VRAM sweep (plan Step 20c, tests/width_probe.py --vram-gib) on TinyGPU.
# Submit from tinyx, in the clone:
#   sbatch.tinygpu hpc/tinygpu_vram_job.sh                          # A100 40 GB (default)
#   sbatch.tinygpu hpc/tinygpu_vram_job.sh --n 11,20,30,40,50       # other n (passed on)
# The probe emulates every usual card size up to this card's memory (2, 3, 4, 6, 8, 10,
# 12, 16, 20, 24, 32 GiB and the card itself); sizes where all runs already fit are not
# rerun. Same population and seeds as the uni PC run (260 x 300), so the two compare.
# Output: <job name>-<job id>.out (the probe's full printout) and
# data/runtime_reports/<time>_<GPU>_job<id>_vram/vram.csv (+ nvidia-smi.txt).
# ============================================================
#SBATCH --job-name=byte-vram
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

SIF=$WORK/byte.sif
CACHE=$WORK/byte-cache          # compiled kernels, kept between jobs (keyed per GPU type)
[ -f "$SIF" ] || { echo "ERROR: $SIF not found (build it: hpc/byte.def)" >&2; exit 1; }
N_LIST=11,20,30,40,50           # the uni PC run's n (2026-11-20)
if [ $# -ge 2 ] && [ "$1" = "--n" ]; then N_LIST=$2; shift 2; fi
[ $# -eq 0 ] || { echo "ERROR: unknown arguments: $* (only --n LIST)" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$CACHE/inductor" "$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton
export OMP_NUM_THREADS=$(nproc) TORCHINDUCTOR_COMPILE_THREADS=$(nproc)
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80 PYTHONUNBUFFERED=1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '-')
OUT=data/runtime_reports/$(date +%Y-%m-%d_%H-%M-%S)_${GPU}_job${SLURM_JOB_ID}_vram
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION, $(nproc) cores; n = $N_LIST; output $OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/nvidia-smi.txt"
git rev-parse --short=10 HEAD > "$OUT/code_version.txt" 2>/dev/null || echo "no git" > "$OUT/code_version.txt"
apptainer exec --nv "$SIF" python -m tests.width_probe --vram-gib --n "$N_LIST" --csv "$OUT/vram.csv"
echo "written: $OUT/vram.csv"
