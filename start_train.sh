#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-help}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

usage() {
  cat <<'EOF'
Usage:
  ./start_train.sh smoke
  ./start_train.sh quick
  ./start_train.sh night
  ./start_train.sh benchmark
  ./start_train.sh resume

Common env overrides:
  PYTHON_BIN=python3
  RUN_NAME=my_run
  OUTPUT_DIR=ai/train_v2/runs
  SEED=42
  WORKERS=8
  VERIFY_MASK=false
  PLACEMENT_MODE=append_only
  AF_DTYPE=float32        # float32 | float16
  OPPONENT_MIX=self:1.0   # e.g. self:0.4,random:0.1,greedy_face:0.3,v4_lite:0.2
  LEARNER_SIDE=random     # random | p1 | p2
  STARTING_PLAYER=random  # random | p1 | p2 | learner | opponent
  FOCUS_SCENARIOS_JSON='[{"key":"demo","deck":[1,27,29,31,34,40,41,42,46],"level":1}]'
  FOCUS_DECK_RATE=1.0     # 0 disables focus decks, 1 forces them every episode
  EXPORT_ONNX=0           # quick/night default 0, smoke default 1
  EVAL_GAMES=1
  EVAL_MAX_STEPS=20
  EXTRA_ARGS="--updates 3 --max-steps 50"

Resume:
  RESUME_CHECKPOINT=ai/train_v2/runs/.../checkpoints/update_0001.npz ./start_train.sh resume

Benchmark env:
  WORKERS_LIST=1,2,4,8
  EPISODES=8
  MAX_STEPS=100
  UPDATES=1

Examples:
  ./start_train.sh smoke
  AF_DTYPE=float16 ./start_train.sh quick
  WORKERS=8 EXPORT_ONNX=0 ./start_train.sh night
  WORKERS_LIST=1,2,4,8 EPISODES=8 MAX_STEPS=100 ./start_train.sh benchmark
EOF
}

require_checkpoint_for_resume() {
  if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
    echo "ERROR: RESUME_CHECKPOINT is required for resume mode." >&2
    echo "Example:" >&2
    echo "  RESUME_CHECKPOINT=ai/train_v2/runs/.../checkpoints/update_0001.npz ./start_train.sh resume" >&2
    exit 2
  fi
}

run_experiment() {
  local preset="$1"
  local default_workers="$2"
  local default_export="$3"
  local default_eval_games="$4"
  local default_eval_steps="$5"

  local run_name="${RUN_NAME:-${preset}_$(timestamp)}"
  local output_dir="${OUTPUT_DIR:-ai/train_v2/runs}"
  local seed="${SEED:-42}"
  local workers="${WORKERS:-$default_workers}"
  local verify_mask="${VERIFY_MASK:-false}"
  local placement_mode="${PLACEMENT_MODE:-append_only}"
  local af_dtype="${AF_DTYPE:-float32}"
  local opponent_mix="${OPPONENT_MIX:-self:1.0}"
  local learner_side="${LEARNER_SIDE:-random}"
  local starting_player="${STARTING_PLAYER:-random}"
  local focus_scenarios_json="${FOCUS_SCENARIOS_JSON:-}"
  local focus_deck_rate="${FOCUS_DECK_RATE:-}"
  local export_onnx="${EXPORT_ONNX:-$default_export}"
  local eval_games="${EVAL_GAMES:-$default_eval_games}"
  local eval_max_steps="${EVAL_MAX_STEPS:-$default_eval_steps}"

  local cmd=(
    "$PYTHON_BIN" -m ai.train_v2.experiment
    --name "$run_name"
    --output-dir "$output_dir"
    --preset "$preset"
    --seed "$seed"
    --rollout-workers "$workers"
    --verify-mask "$verify_mask"
    --placement-mode "$placement_mode"
    --action-features-dtype "$af_dtype"
    --opponent-mix "$opponent_mix"
    --learner-side "$learner_side"
    --starting-player "$starting_player"
    --eval-games "$eval_games"
    --eval-max-steps "$eval_max_steps"
  )

  if [[ -n "$focus_scenarios_json" ]]; then
    cmd+=(--focus-scenarios-json "$focus_scenarios_json")
  fi
  if [[ -n "$focus_deck_rate" ]]; then
    cmd+=(--focus-deck-rate "$focus_deck_rate")
  fi

  if [[ "$export_onnx" != "1" && "$export_onnx" != "true" ]]; then
    cmd+=(--no-export-onnx)
  fi

  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    cmd+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=( $EXTRA_ARGS )
    cmd+=("${extra[@]}")
  fi

  echo "Starting TrainV2 experiment:"
  printf '  %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

run_benchmark() {
  local workers_list="${WORKERS_LIST:-1,2,4,8}"
  local episodes="${EPISODES:-8}"
  local max_steps="${MAX_STEPS:-100}"
  local updates="${UPDATES:-1}"
  local seed="${SEED:-42}"
  local verify_mask="${VERIFY_MASK:-false}"
  local placement_mode="${PLACEMENT_MODE:-append_only}"
  local af_dtype="${AF_DTYPE:-float32}"

  local cmd=(
    "$PYTHON_BIN" -m ai.train_v2.benchmark_rollout
    --preset smoke
    --workers "$workers_list"
    --episodes "$episodes"
    --max-steps "$max_steps"
    --updates "$updates"
    --seed "$seed"
    --verify-mask "$verify_mask"
    --placement-mode "$placement_mode"
    --action-features-dtype "$af_dtype"
  )

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=( $EXTRA_ARGS )
    cmd+=("${extra[@]}")
  fi

  echo "Starting TrainV2 rollout benchmark:"
  printf '  %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

case "$MODE" in
  smoke)
    run_experiment smoke "${WORKERS:-1}" "${EXPORT_ONNX:-1}" "${EVAL_GAMES:-1}" "${EVAL_MAX_STEPS:-20}"
    ;;
  quick)
    run_experiment m4_quick "${WORKERS:-4}" "${EXPORT_ONNX:-0}" "${EVAL_GAMES:-1}" "${EVAL_MAX_STEPS:-100}"
    ;;
  night)
    run_experiment m4_night "${WORKERS:-8}" "${EXPORT_ONNX:-0}" "${EVAL_GAMES:-4}" "${EVAL_MAX_STEPS:-200}"
    ;;
  resume)
    require_checkpoint_for_resume
    run_experiment m4_night "${WORKERS:-8}" "${EXPORT_ONNX:-0}" "${EVAL_GAMES:-4}" "${EVAL_MAX_STEPS:-200}"
    ;;
  benchmark)
    run_benchmark
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
