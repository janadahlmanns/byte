#!/bin/bash -l
# ============================================================
# One run_selected queue on TinyGPU (NHR@FAU) -- submit from tinyx, in the clone:
#   sbatch.tinygpu hpc/tinygpu_job.sh scaling_cuda                  # a preset
#   sbatch.tinygpu hpc/tinygpu_job.sh --configs ea_from_lookup_tensor_cuda
# Another GPU type: override on the command line, e.g.
#   sbatch.tinygpu --partition=a100 --gres=gpu:a100:1 hpc/tinygpu_job.sh scaling_cuda
# Output: <job name>-<job id>.out in the submit folder; the report in
# data/runtime_reports/<time>_<GPU>_job<id>/ as on every other machine.
#
# One job per GPU (plan Step 24): a job array over a preset, task i runs entry i:
#   N=$(apptainer exec $WORK/byte.sif python -m simulate.run_selected --preset scaling_cuda --count)
#   sbatch.tinygpu --array=0-$((N-1))%8 --output=%x-%A_%a.out \
#       --partition=a100 --gres=gpu:a100:1 hpc/tinygpu_job.sh scaling_cuda
# (%8: at most 8 tasks at once). Re-run failed tasks only: --array=3,7. Each task writes
# data/runtime_reports/<time>_<GPU>_job<array id>_<task>/; all of one array:
#   python -m analysis.plot_scaling data/runtime_reports/*_job<array id>_* --out <folder>
# ============================================================
#SBATCH --job-name=byte
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

SIF=$WORK/byte.sif
CACHE=$WORK/byte-cache          # compiled kernels, kept between jobs (keyed per GPU type)
[ -f "$SIF" ] || { echo "ERROR: $SIF not found (build it: hpc/byte.def)" >&2; exit 1; }
[ $# -ge 1 ] || { echo "ERROR: give a preset name, or --configs a,b" >&2; exit 1; }
if [ "$1" = "--configs" ]; then
    [ -z "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "ERROR: a job array needs a preset, not --configs" >&2; exit 1; }
    ARGS=("$@")
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARGS=(--preset "$1" --task "$SLURM_ARRAY_TASK_ID")
else
    ARGS=(--preset "$1")
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$CACHE/inductor" "$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton
# Only the cores Slurm gave this job: torch's threads and the compile workers.
export OMP_NUM_THREADS=$(nproc) TORCHINDUCTOR_COMPILE_THREADS=$(nproc)
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80

echo "job $SLURM_JOB_ID${SLURM_ARRAY_TASK_ID:+ (array $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID)} on $(hostname), partition $SLURM_JOB_PARTITION, $(nproc) cores"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
apptainer exec --nv "$SIF" python -m simulate.run_selected "${ARGS[@]}"
