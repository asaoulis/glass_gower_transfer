#!/usr/bin/env bash
# Full local P5/P6 chain for the VDQ misspec tables + figures (AMENDMENT 2026-10-19).
# Run AFTER the orchestrator's fetch has landed (P4.3 checks first). Idempotent; rows caches are
# rebuilt (--rebuild) because the variate set changed. Core budget: <= 40 cores (taskset).
#   bash run_all.sh [field|band|both] [--skip-phase6] [--skip-selfcheck]
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/home/alex/work/glass_gower_transfer
PY=/data/alex/glass/env/bin/python
ART="$REPO/.claude/runs/eval-and-viz"
R=/data/alex/unblinding/real
CK=/data/alex/ml-checkpoints
WHICH="${1:-both}"; shift || true
SKIP_P6=0; SKIP_SC=0
for a in "$@"; do case "$a" in --skip-phase6) SKIP_P6=1;; --skip-selfcheck) SKIP_SC=1;; esac; done
export PYTHONPATH="$REPO" TQDM_DISABLE=1
cd "$REPO"

ALL7="gower_vdq_nla_m gower_vdq_gb1p0 gower_vdq_gb1p3 gower_vdq_gbk2 gower_vdq_nla gower_vdq_nla_z gower_vdqk2_nla"
NO_K2NLA="gower_vdq_nla_m gower_vdq_gb1p0 gower_vdq_gb1p3 gower_vdq_gbk2 gower_vdq_nla gower_vdq_nla_z"
EXCL="gower_vdq_nla:a_ia,gower_vdq_nla_z:a_ia,gower_vdqk2_nla:a_ia"

pack() {   # key exp repeats summ_fmt obs cores
  local key=$1 exp=$2 reps=$3 sfmt=$4 obs=$5 cores=$6
  local base=$CK/$exp out=$R/work/tier2
  echo "=== [$key] $exp repeats=$reps"
  # P5.1 null self-check + positive control (nla)
  if [ $SKIP_SC = 0 ]; then
    mkdir -p "$out/${key}_vdq_selfcheck"
    (cd "$out/${key}_vdq_selfcheck" && taskset -c "$cores" $PY "$ART/production-model-misspec/artifacts/gower_null_selfcheck.py" \
      --root "$base/summaries" --repeats $reps --match-fmt "$sfmt" --control-variate gower_vdq_nla) \
      > "$out/${key}_vdq_selfcheck/selfcheck.log" 2>&1 || echo "[$key] selfcheck FAILED (see log)"
  fi
  # P5.3 obs-identical mean-p (gate inside; exits 2 on gate failure)
  taskset -c "$cores" $PY "$HERE/meanp_knn.py" --pack "$key"
  # P5.2 phase6 (mock-side cond_knn combiner)
  if [ $SKIP_P6 = 0 ]; then
    taskset -c "$cores" $PY "$ART/real-data-unblinding/artifacts/multiencoder_ood.py" --root "$base/summaries" \
      --moments "$base/misspec" --out "$out/${key}_vdq_phase6" --repeats $reps --match-fmt "$sfmt" \
      --moments-match-fmt "ncosmo300_{r}" --variates $ALL7 --in-dist gower_vdq_nla_m --exclude "$EXCL" \
      --control gower_vdq_nla --hard gower_vdq_gb1p0 gower_vdq_gb1p3 gower_vdq_gbk2 --recommend meanp \
      > "$out/${key}_vdq_phase6.log" 2>&1
  fi
  # rows cache (shared by tables.py and tier2_tables.py) -- rebuilt for the 7-variate set
  taskset -c "$cores" $PY -c "import sys; sys.path.insert(0, '$HERE'); import common as K, vdq_config as C; \
r = K.rows(C.PACKS['$key'], C.VARIATES, rebuild=True); print('[rows]', '$key', len(r['variate']))"
  # P5.6 Tier-2 bias tables: pooled (all 7), pooled without the double misspec (D2), per variate
  T2="taskset -c $cores $PY scripts/tier2_tables.py --base-dir $base --experiment $exp --repeats $reps \
      --in-dist gower_vdq_nla_m --meanp-source obs_knn --meanp-npz $out/${key}_vdq_meanp_knn/meanp_knn.npz \
      --phase6-dir '' --rows-cache $R/inputs/tier2_rows/${key}_vdq_rows.npz --obs-score $R/inputs/cluster_fetch/obs_kids_A2_sc8a1/$obs"
  eval $T2 --variates $ALL7 --out-dir "$out/${key}_vdq_full" > "$out/${key}_vdq_full.log" 2>&1
  eval $T2 --variates $NO_K2NLA --out-dir "$out/${key}_vdq_full/pooled_without_vdqk2_nla" >> "$out/${key}_vdq_full.log" 2>&1
  for v in $ALL7; do
    [ "$v" = gower_vdq_nla_m ] && continue
    eval $T2 --variates gower_vdq_nla_m $v --out-dir "$out/${key}_vdq_full/per_variate/$v" >> "$out/${key}_vdq_full.log" 2>&1
  done
  # P5.4 / P5.5 / P5.7
  taskset -c "$cores" $PY "$HERE/tables.py" --pack "$key" --n-boot 1000 --floor-draws 200 > "$R/work/misspec_vdq/tables_${key}.log" 2>&1
  echo "=== [$key] done"
}

mkdir -p $R/work/misspec_vdq
case "$WHICH" in
  field) pack field gower_npe_finetune_nla_m_vdq_z8_ens1 "0 2 3 4 5" "ncosmo300_{r}" obs_score_A2vdq.json 0-19 ;;
  band)  pack band gower_npe_finetune_band_nla_m_vdq_ens9 "0 1 2 3 4" "ncosmo300_{r}_m0" obs_score_A2vdqband.json 20-39 ;;
  both)
    pack field gower_npe_finetune_nla_m_vdq_z8_ens1 "0 2 3 4 5" "ncosmo300_{r}" obs_score_A2vdq.json 0-19 &
    pack band gower_npe_finetune_band_nla_m_vdq_ens9 "0 1 2 3 4" "ncosmo300_{r}_m0" obs_score_A2vdqband.json 20-39 &
    wait ;;
esac
# P6 + P5.8
taskset -c 0-19 $PY "$HERE/figures.py"
taskset -c 0-19 $PY "$HERE/tables_md.py"
echo "[run_all] done"
