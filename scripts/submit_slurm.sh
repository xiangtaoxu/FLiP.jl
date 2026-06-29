#!/usr/bin/env bash
#
# submit_slurm.sh — generate (and optionally submit) a SLURM batch job that runs
# the FLiP.jl pipeline for a given TOML config.
#
# Usage:
#   scripts/submit_slurm.sh [options] <config.toml>
#
# By default the generated batch script is printed to stdout (a dry run, so you
# can review or redirect it to a file). Pass -s to submit it with sbatch.
#
#   scripts/submit_slurm.sh analyses/flip_config_ALT-HL0-20250430.toml      # preview
#   scripts/submit_slurm.sh -s analyses/flip_config_ALT-HL0-20250430.toml   # submit
#   scripts/submit_slurm.sh -m 210G -t 24:00:00 -w cbsuxu09 -s cfg.toml     # override
#
# --cpus-per-task / the Julia thread count is taken from `pipeline.n_thread` in
# the config (unless -c is given); the job name is derived from
# `pipeline.output_prefix`. Everything else has a default that -flags override.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SUBMIT=0
CPUS=""
MEM="128G"
TIME="12:00:00"
NODELIST=""
PARTITION=""
JOBNAME=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLIP_DIR="$(dirname -- "$SCRIPT_DIR")"

usage() {
    cat >&2 <<EOF
Usage: ${0##*/} [options] <config.toml>

Generate a SLURM batch script that runs the FLiP.jl pipeline on <config.toml>.
By default the script is printed to stdout; pass -s to submit it via sbatch.

Options:
  -s            Submit the job with sbatch (default: just print the script)
  -c N          CPUs per task / Julia threads
                  (default: pipeline.n_thread from the config, else 6)
  -m MEM        Memory request           (default: ${MEM})
  -t TIME       Wall-time limit          (default: ${TIME})
  -w NODELIST   Pin to node(s)           (default: none)
  -p PARTITION  Partition / queue        (default: none)
  -J NAME       Job name                 (default: flip_<output_prefix|config name>)
  -D DIR        FLiP.jl repo root        (default: ${FLIP_DIR})
  -h            Show this help
EOF
}

# ---------------------------------------------------------------------------
# Minimal TOML reader: print the value of KEY within [SECTION] (first match).
# Handles inline "# comments" and surrounding double quotes. Good enough for the
# flat scalar keys we need (pipeline.n_thread, pipeline.output_prefix).
# ---------------------------------------------------------------------------
toml_get() {
    local section="$1" key="$2" file="$3"
    awk -v section="$section" -v key="$key" '
        /^[[:space:]]*\[/ {
            s = $0
            sub(/^[[:space:]]*\[/, "", s)
            sub(/\].*$/, "", s)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
            insec = (s == section)
            next
        }
        insec && /^[[:space:]]*[A-Za-z0-9_]+[[:space:]]*=/ {
            k = $0
            sub(/[[:space:]]*=.*$/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            if (k != key) next
            v = $0
            sub(/^[^=]*=[[:space:]]*/, "", v)
            if (v ~ /^"/) { sub(/^"/, "", v); sub(/".*$/, "", v) }
            else { sub(/[[:space:]]*#.*$/, "", v); gsub(/^[[:space:]]+|[[:space:]]+$/, "", v) }
            print v
            exit
        }
    ' "$file"
}

abspath() {
    local p="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath -- "$p"
    elif [[ -d "$p" ]]; then
        (cd -- "$p" && pwd)
    else
        (cd -- "$(dirname -- "$p")" && printf '%s/%s\n' "$(pwd)" "$(basename -- "$p")")
    fi
}

# ---------------------------------------------------------------------------
# Parse options
# ---------------------------------------------------------------------------
while getopts ":sc:m:t:w:p:J:D:h" opt; do
    case "$opt" in
        s) SUBMIT=1 ;;
        c) CPUS="$OPTARG" ;;
        m) MEM="$OPTARG" ;;
        t) TIME="$OPTARG" ;;
        w) NODELIST="$OPTARG" ;;
        p) PARTITION="$OPTARG" ;;
        J) JOBNAME="$OPTARG" ;;
        D) FLIP_DIR="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Error: unknown option -$OPTARG" >&2; usage; exit 2 ;;
        :)  echo "Error: option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -ne 1 ]]; then
    echo "Error: exactly one <config.toml> argument is required." >&2
    usage
    exit 2
fi

CONFIG_IN="$1"
if [[ ! -f "$CONFIG_IN" ]]; then
    echo "Error: config file not found: $CONFIG_IN" >&2
    exit 1
fi
CONFIG="$(abspath "$CONFIG_IN")"
FLIP_DIR="$(abspath "$FLIP_DIR")"

# ---------------------------------------------------------------------------
# Derive resources from the config (unless overridden on the command line)
# ---------------------------------------------------------------------------
if [[ -z "$CPUS" ]]; then
    n_thread="$(toml_get pipeline n_thread "$CONFIG" || true)"
    if [[ "$n_thread" =~ ^[1-9][0-9]*$ ]]; then
        CPUS="$n_thread"
    else
        CPUS=6   # n_thread missing, serial (0/1), or negative (relative-to-cores)
    fi
fi

if [[ -z "$JOBNAME" ]]; then
    prefix="$(toml_get pipeline output_prefix "$CONFIG" || true)"
    prefix="${prefix%[-_]}"   # drop a single trailing separator
    if [[ -z "$prefix" ]]; then
        base="${CONFIG##*/}"; base="${base%.toml}"
        prefix="${base#flip_config_}"
    fi
    JOBNAME="flip_${prefix}"
fi

# ---------------------------------------------------------------------------
# Emit the batch script. Values known now (job name, resources, paths) are
# expanded here; runtime references ($SLURM_*, $(date), $(hostname)) are kept
# literal so they expand on the compute node.
# ---------------------------------------------------------------------------
emit_script() {
    printf '#!/bin/bash\n'
    printf '#SBATCH --job-name=%s\n' "$JOBNAME"
    [[ -n "$NODELIST"  ]] && printf '#SBATCH --nodelist=%s\n'  "$NODELIST"
    [[ -n "$PARTITION" ]] && printf '#SBATCH --partition=%s\n' "$PARTITION"
    printf '#SBATCH --nodes=1\n'
    printf '#SBATCH --ntasks=1\n'
    printf '#SBATCH --cpus-per-task=%s\n' "$CPUS"
    printf '#SBATCH --mem=%s\n' "$MEM"
    printf '#SBATCH --time=%s\n' "$TIME"
    printf '#SBATCH --output=%%x_%%j.out\n'
    printf '#SBATCH --error=%%x_%%j.err\n'

    cat <<EOF

set -euo pipefail

FLIP_DIR="${FLIP_DIR}"
CONFIG="${CONFIG}"

cd "\$FLIP_DIR"

echo "Job \${SLURM_JOB_ID} on \$(hostname): \${SLURM_CPUS_PER_TASK} threads, started \$(date)"

julia --project=. -t "\${SLURM_CPUS_PER_TASK}" \\
    -e "using FLiP; run_pipeline(\"\${CONFIG}\")"

echo "Finished \$(date)"
EOF
}

# ---------------------------------------------------------------------------
# Print or submit
# ---------------------------------------------------------------------------
if [[ "$SUBMIT" -eq 1 ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "Error: -s given but 'sbatch' was not found in PATH." >&2
        exit 1
    fi
    echo "Submitting '${JOBNAME}'  (cpus=${CPUS}, mem=${MEM}, time=${TIME})" >&2
    emit_script | sbatch
else
    emit_script
    echo "# Dry run: above is the batch script for '${JOBNAME}'. Re-run with -s to submit." >&2
fi
