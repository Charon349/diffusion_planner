###################################
# User Configuration Section
###################################
SCENARIO_FILTER="${SCENARIO_FILTER:-train}" # train, val14, val14-collision, test14-random, test14-hard
NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/mnt/workspace/shared/pnc/data/nuplan/dataset}"
NUPLAN_MAP_PATH="${NUPLAN_MAP_PATH:-$NUPLAN_DATA_ROOT/maps}"

if [ -z "${NUPLAN_DATA_PATH:-}" ]; then
  case "$SCENARIO_FILTER" in
    test14-random|test14-hard)
      NUPLAN_DATA_PATH="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test"
      ;;
    *)
      NUPLAN_DATA_PATH="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval"
      ;;
  esac
fi

has_db_files() {
  local path="$1"
  [ -e "$path" ] && find "$path" -maxdepth 3 -type f -name '*.db' -print -quit 2>/dev/null | grep -q .
}

if ! has_db_files "$NUPLAN_DATA_PATH"; then
  case "$SCENARIO_FILTER" in
    test14-random|test14-hard)
      CANDIDATE_PATHS=(
        "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test"
        "$NUPLAN_DATA_ROOT/nuplan-v1.1/test"
        "$NUPLAN_DATA_ROOT"
      )
      ;;
    *)
      CANDIDATE_PATHS=(
        "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval"
        "$NUPLAN_DATA_ROOT/nuplan-v1.1/trainval"
        "$NUPLAN_DATA_ROOT"
      )
      ;;
  esac

  for candidate in "${CANDIDATE_PATHS[@]}"; do
    if has_db_files "$candidate"; then
      echo "NUPLAN_DATA_PATH had no .db files; using detected DB path: $candidate"
      NUPLAN_DATA_PATH="$candidate"
      break
    fi
  done
fi

SAVE_PATH="${SAVE_PATH:-/mnt/workspace/shared/pnc/data/nuplan/data_for_diffusion_${SCENARIO_FILTER}}"
SAVE_LIST="${SAVE_LIST:-./diffusion_planner_${SCENARIO_FILTER}.json}"
TOTAL_SCENARIOS="${TOTAL_SCENARIOS:-}"
###################################

if [[ "$NUPLAN_DATA_PATH" == REPLACE_WITH_* || "$NUPLAN_MAP_PATH" == REPLACE_WITH_* || "$SAVE_PATH" == REPLACE_WITH_* ]]; then
  echo "Please set real paths for NUPLAN_DATA_PATH/NUPLAN_MAP_PATH/SAVE_PATH." >&2
  exit 1
fi

if ! has_db_files "$NUPLAN_DATA_PATH"; then
  echo "No nuPlan .db files found under NUPLAN_DATA_PATH=$NUPLAN_DATA_PATH" >&2
  echo "Check the raw DB location with:" >&2
  echo "  find \"$NUPLAN_DATA_ROOT\" -maxdepth 6 -type f -name '*.db' | head" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "$TOTAL_SCENARIOS" ]; then
  EXTRA_ARGS+=(--total_scenarios "$TOTAL_SCENARIOS")
fi

echo "SCENARIO_FILTER=$SCENARIO_FILTER"
echo "NUPLAN_DATA_PATH=$NUPLAN_DATA_PATH"
echo "NUPLAN_MAP_PATH=$NUPLAN_MAP_PATH"
echo "SAVE_PATH=$SAVE_PATH"
echo "SAVE_LIST=$SAVE_LIST"

python data_process.py \
  --data_path "$NUPLAN_DATA_PATH" \
  --map_path "$NUPLAN_MAP_PATH" \
  --save_path "$SAVE_PATH" \
  --save_list "$SAVE_LIST" \
  --scenario_filter "$SCENARIO_FILTER" \
  "${EXTRA_ARGS[@]}"
