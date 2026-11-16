#!/bin/bash -l
# ============================================================
# run_selected presets on TinyFat (NHR@FAU), CPU only (plan Step 30), in ONE job --
# submit from tinyx, in the clone:
#   sbatch.tinyfat hpc/tinyfat_job.sh cpu_baseline_tinyfat cpu_baseline_cores@32 cpu_baseline_cores@16
# Each argument is a preset, optionally @N: then that preset runs on the first N of the
# job's cores only (taskset; torch's threads set to N), so one 64-core job also gives
# the 16- and 32-core points. All arguments are checked before anything runs; the
# presets run one after another, each with its own report folder. A failing preset
# does not stop the later ones; the job then exits non-zero.
# Partition `work`: 2 x AMD EPYC 7502, 64 cores / 128 threads per node, cores allocated
# individually (doc.nhr.fau.de/clusters/tinyfat). SMT is on, so the job checks that it
# got PHYSICAL cores: nproc must equal --cpus-per-task and no two allowed CPUs may be
# hyperthreads of one core. Otherwise it stops before running anything.
# Output: <job name>-<job id>.out in the submit folder; the report in
# data/runtime_reports/<time>_<CPU model>_job<id>/ (no GPU: named after the CPU).
# ============================================================
#SBATCH --job-name=byte-cpu
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

SIF=${BYTE_SIF:-$WORK/byte.sif}
CACHE=${BYTE_CACHE:-$WORK/byte-cache}       # compiled kernels, kept between jobs
[ -f "$SIF" ] || { echo "ERROR: $SIF not found (build it: hpc/byte.def)" >&2; exit 1; }
[ $# -ge 1 ] || { echo "ERROR: give one or more presets (preset or preset@cores)" >&2; exit 1; }

# --- the allocation: physical cores only ------------------------------------------
WANT=${SLURM_CPUS_PER_TASK:?ERROR: SLURM_CPUS_PER_TASK not set (submit with sbatch.tinyfat)}
HAVE=$(nproc)
[ "$HAVE" = "$WANT" ] || { echo "ERROR: nproc is $HAVE, but $WANT cores were requested" >&2; exit 1; }
ALLOWED=$(taskset -cp $$ | sed 's/.*: *//')
expand() {  # "0-3,8,10-11" -> one CPU number per line
    tr ',' '\n' <<<"$1" | while IFS=- read -r a b; do seq "$a" "${b:-$a}"; done
}
# CPU,CORE,SOCKET for every CPU of the node; count the distinct (socket, core) pairs
# among the allowed CPUs.
CORES=$(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' \
        | awk -F, -v allowed="$(expand "$ALLOWED" | tr '\n' ' ')" \
          'BEGIN { n = split(allowed, a, " "); for (i = 1; i <= n; i++) ok[a[i]] = 1 }
           ($1 in ok) { seen[$3 "," $2] = 1 } END { print length(seen) }')
[ "$CORES" = "$HAVE" ] || {
    echo "ERROR: the $HAVE allowed CPUs ($ALLOWED) are only $CORES physical cores (hyperthreads)" >&2; exit 1; }
echo "allocation: $HAVE physical cores ($ALLOWED)"

# --- the segments: preset[@cores], all checked first --------------------------------
PRESETS=(); NCORES=()
for seg in "$@"; do
    case "$seg" in
        --*) echo "ERROR: $seg: this script takes preset names (preset or preset@cores)" >&2; exit 1 ;;
        *@*) p=${seg%@*}; n=${seg##*@} ;;
        *)   p=$seg; n=$HAVE ;;
    esac
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: $seg: the core count must be a positive integer" >&2; exit 1; }
    [ "$n" -le "$HAVE" ] || { echo "ERROR: $seg: $n cores, but the job has $HAVE" >&2; exit 1; }
    [ -f "configs/presets/$p.yaml" ] || [ -f "$SLURM_SUBMIT_DIR/configs/presets/$p.yaml" ] \
        || { echo "ERROR: $seg: no preset configs/presets/$p.yaml" >&2; exit 1; }
    PRESETS+=("$p"); NCORES+=("$n")
done

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$CACHE/inductor"
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80

echo "job $SLURM_JOB_ID on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, $HAVE cores"
lscpu | grep -E 'Model name|Thread\(s\) per core|Socket\(s\)' || true
FAILED=()
for i in "${!PRESETS[@]}"; do
    p=${PRESETS[$i]}; n=${NCORES[$i]}
    # The first n of the job's cores (all of them when n = $HAVE): torch's threads, the
    # compile workers and the scalar code's worker count all follow the affinity.
    LIST=$(expand "$ALLOWED" | head -n "$n" | paste -sd, -)
    echo
    echo "=== preset $p on $n cores ($LIST) ==="
    if OMP_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n \
       taskset -c "$LIST" apptainer exec "$SIF" python -m simulate.run_selected --preset "$p"; then
        echo "=== preset $p on $n cores: done ==="
    else
        echo "=== preset $p on $n cores: FAILED (exit $?) ===" >&2
        FAILED+=("$p@$n")
    fi
done
[ ${#FAILED[@]} -eq 0 ] || { echo "ERROR: failed: ${FAILED[*]}" >&2; exit 1; }
