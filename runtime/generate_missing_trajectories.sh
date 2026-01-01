#!/bin/bash
# Generate missing trajectories for an existing trajectory folder
# Usage: ./generate_missing_trajectories.sh [options]
#
# You can either set the configuration variables below OR pass command line arguments.
# Command line arguments will override the configuration variables.
#
# Example:
#   # Edit CONFIG section below, then just run:
#   ./generate_missing_trajectories.sh
#
#   # Or use command line arguments:
#   ./generate_missing_trajectories.sh trajectories/goaloriented/gpt-4o-mini openai/gpt-4o-mini --goal-oriented

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

###############################################################################
# CONFIGURATION - Edit these variables to set defaults
###############################################################################

# Model configuration (the trajectory folder will be auto-derived from model name)
CONFIG_MODEL="openai/gpt-5.2"           # Model to use (e.g., openai/gpt-4o-mini, deepseek/deepseek-chat)
CONFIG_USER_MODEL="openai/gpt-5.2"                             # User model for goal-oriented (leave empty to use same as CONFIG_MODEL)
CONFIG_PERSONA="curious_researcher"              # Persona for goal-oriented agent

# Agent type
CONFIG_GOAL_ORIENTED=true                        # true for goal-oriented, false for ReAct

# Execution settings
CONFIG_MAX_CONCURRENT=10                         # Number of parallel workers
CONFIG_MAX_ITERATIONS=60                         # Max reasoning iterations per query
CONFIG_LOOP_UNTIL_COMPLETE=true                  # Keep retrying until all complete
CONFIG_MAX_RETRIES=10                            # Max retry rounds when looping

# Pass configuration (leave empty for all passes 1,2,3)
CONFIG_SPECIFIC_PASS=""                          # e.g., "1" for pass@1 only, "" for all

# Query file (leave empty for default based on agent type)
CONFIG_QUERY_FILE=""

# Trajectory directory - AUTO-DERIVED from model name
# Format: trajectories/goaloriented/<model_short_name> or trajectories/<model_short_name>
# Override by setting CONFIG_TRAJECTORY_DIR_OVERRIDE
CONFIG_TRAJECTORY_DIR_OVERRIDE=""               # Leave empty to auto-derive, or set manually

###############################################################################
# END CONFIGURATION
###############################################################################

# Auto-derive trajectory directory from model name
_derive_trajectory_dir() {
    local model="$1"
    local goal_oriented="$2"

    # Extract model short name (e.g., "deepseek/deepseek-chat" -> "deepseek-chat")
    local model_short=$(echo "$model" | sed 's|.*/||')

    if [ "$goal_oriented" = true ]; then
        echo "trajectories/goaloriented/${model_short}"
    else
        echo "trajectories/${model_short}"
    fi
}

# Set CONFIG_TRAJECTORY_DIR based on override or auto-derive
if [ -n "$CONFIG_TRAJECTORY_DIR_OVERRIDE" ]; then
    CONFIG_TRAJECTORY_DIR="$CONFIG_TRAJECTORY_DIR_OVERRIDE"
else
    CONFIG_TRAJECTORY_DIR="$(_derive_trajectory_dir "$CONFIG_MODEL" "$CONFIG_GOAL_ORIENTED")"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [trajectory_dir] [model] [options]"
    echo ""
    echo "Arguments (optional if CONFIG variables are set at top of script):"
    echo "  trajectory_dir    Path to trajectory directory (e.g., trajectories/goaloriented/gpt-4o-mini)"
    echo "  model             Model name (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4o-mini)"
    echo ""
    echo "Options:"
    echo "  --goal-oriented        Use goal-oriented agent (run_goaloriented_agent.py)"
    echo "  --react                Use ReAct agent (run_react_agent.py)"
    echo "  --query-file FILE      Custom query file (default: depends on agent type)"
    echo "  --user-model MODEL     User model for goal-oriented agent (default: same as model)"
    echo "  --persona NAME         Persona for goal-oriented agent (default: curious_researcher)"
    echo "  --dry-run              Show what would be generated without actually running"
    echo "  --max-concurrent N     Run N trajectories in parallel (default: 5)"
    echo "  --max-iterations N     Max reasoning iterations per query (default: 60)"
    echo "  --loop                 Keep retrying until all trajectories are complete"
    echo "  --no-loop              Disable loop mode"
    echo "  --max-retries N        Maximum retry rounds when using --loop (default: 10)"
    echo "  --pass N               Only generate missing for pass N (default: all passes 1,2,3)"
    echo ""
    echo "Current CONFIG defaults:"
    echo "  MODEL:           $CONFIG_MODEL"
    echo "  TRAJECTORY_DIR:  $CONFIG_TRAJECTORY_DIR"
    echo "  GOAL_ORIENTED:   $CONFIG_GOAL_ORIENTED"
    echo "  MAX_CONCURRENT:  $CONFIG_MAX_CONCURRENT"
    echo "  LOOP:            $CONFIG_LOOP_UNTIL_COMPLETE"
    echo ""
    echo "Example:"
    echo "  $0                           # Use CONFIG variables"
    echo "  $0 --dry-run                 # Use CONFIG variables with dry-run"
    echo "  $0 trajectories/goaloriented/gpt-4o-mini openai/gpt-4o-mini --goal-oriented"
    exit 1
}

# Initialize from CONFIG variables
TRAJECTORY_DIR="$CONFIG_TRAJECTORY_DIR"
MODEL="$CONFIG_MODEL"
DRY_RUN=false
MAX_CONCURRENT="$CONFIG_MAX_CONCURRENT"
MAX_ITERATIONS="$CONFIG_MAX_ITERATIONS"
GOAL_ORIENTED="$CONFIG_GOAL_ORIENTED"
USER_MODEL="$CONFIG_USER_MODEL"
PERSONA="$CONFIG_PERSONA"
SPECIFIC_PASS="$CONFIG_SPECIFIC_PASS"
LOOP_UNTIL_COMPLETE="$CONFIG_LOOP_UNTIL_COMPLETE"
MAX_RETRIES="$CONFIG_MAX_RETRIES"

# Default query files
REACT_QUERY_FILE="${PROJECT_ROOT}/mcp_generate/queries_verification.json"
GOALORIENTED_QUERY_FILE="${PROJECT_ROOT}/mcp_generate/requests/multitool_50.json"
QUERY_FILE="$CONFIG_QUERY_FILE"

# Check if first argument looks like a path (not an option)
if [ $# -ge 1 ] && [[ ! "$1" =~ ^-- ]]; then
    TRAJECTORY_DIR="$1"
    shift
fi

# Check if second argument looks like a model (not an option)
if [ $# -ge 1 ] && [[ ! "$1" =~ ^-- ]]; then
    MODEL="$1"
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --goal-oriented)
            GOAL_ORIENTED=true
            shift
            ;;
        --react)
            GOAL_ORIENTED=false
            shift
            ;;
        --query-file)
            QUERY_FILE="$2"
            shift 2
            ;;
        --user-model)
            USER_MODEL="$2"
            shift 2
            ;;
        --persona)
            PERSONA="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --loop)
            LOOP_UNTIL_COMPLETE=true
            shift
            ;;
        --no-loop)
            LOOP_UNTIL_COMPLETE=false
            shift
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --pass)
            SPECIFIC_PASS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL" ]; then
    echo -e "${RED}Error: MODEL is not set. Either set CONFIG_MODEL or pass as argument.${NC}"
    usage
fi

if [ -z "$TRAJECTORY_DIR" ]; then
    echo -e "${RED}Error: TRAJECTORY_DIR is not set. Either set CONFIG_TRAJECTORY_DIR or pass as argument.${NC}"
    usage
fi

# Set default query file based on agent type
if [ -z "$QUERY_FILE" ]; then
    if [ "$GOAL_ORIENTED" = true ]; then
        QUERY_FILE="$GOALORIENTED_QUERY_FILE"
    else
        QUERY_FILE="$REACT_QUERY_FILE"
    fi
fi

# Set default user model
if [ -z "$USER_MODEL" ]; then
    USER_MODEL="$MODEL"
fi

# Validate trajectory directory - create if it doesn't exist
if [ ! -d "$TRAJECTORY_DIR" ]; then
    echo -e "${YELLOW}Directory '$TRAJECTORY_DIR' does not exist. Creating it...${NC}"
    mkdir -p "$TRAJECTORY_DIR"
fi

# Print current configuration
echo ""
echo -e "${BLUE}========== CONFIGURATION ==========${NC}"
echo -e "  Model:           ${GREEN}$MODEL${NC}"
echo -e "  User Model:      ${GREEN}$USER_MODEL${NC}"
echo -e "  Trajectory Dir:  ${GREEN}$TRAJECTORY_DIR${NC}"
echo -e "  Agent Type:      ${GREEN}$([ "$GOAL_ORIENTED" = true ] && echo "Goal-Oriented" || echo "ReAct")${NC}"
echo -e "  Max Concurrent:  ${GREEN}$MAX_CONCURRENT${NC}"
echo -e "  Max Iterations:  ${GREEN}$MAX_ITERATIONS${NC}"
echo -e "  Loop Mode:       ${GREEN}$LOOP_UNTIL_COMPLETE${NC}"
echo -e "  Query File:      ${GREEN}$QUERY_FILE${NC}"
[ -n "$SPECIFIC_PASS" ] && echo -e "  Specific Pass:   ${GREEN}$SPECIFIC_PASS${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Get all UUIDs from the query file
echo -e "${YELLOW}Loading queries from $QUERY_FILE...${NC}"
ALL_UUIDS=$(python3 -c "
import json
with open('$QUERY_FILE') as f:
    data = json.load(f)
# Handle both formats: flat list or {items: [...]}
items = data.get('items', data) if isinstance(data, dict) else data
for q in items:
    print(q['uuid'])
")
TOTAL_QUERIES=$(echo "$ALL_UUIDS" | wc -l | tr -d ' ')
echo -e "${GREEN}Found $TOTAL_QUERIES queries${NC}"
echo -e "${BLUE}Agent type: $([ "$GOAL_ORIENTED" = true ] && echo "Goal-Oriented" || echo "ReAct")${NC}"

# Function to get existing UUIDs for a pass
get_existing_uuids() {
    local pass_num=$1
    local pass_dir="${TRAJECTORY_DIR}/pass@${pass_num}"

    if [ -d "$pass_dir" ]; then
        for f in "$pass_dir"/trajectory_*.json; do
            if [ -f "$f" ]; then
                # Extract UUID from filename: trajectory_UUID_TIMESTAMP.json
                basename "$f" | sed 's/trajectory_//' | cut -d'_' -f1
            fi
        done | sort -u
    fi
}

# Function to find missing UUIDs
find_missing_uuids() {
    local pass_num=$1
    local existing=$(get_existing_uuids $pass_num)

    for uuid in $ALL_UUIDS; do
        if ! echo "$existing" | grep -q "^${uuid}$"; then
            echo "$uuid"
        fi
    done
}

# Function to get query index by UUID
get_query_index() {
    local target_uuid=$1
    python3 -c "
import json
with open('$QUERY_FILE') as f:
    data = json.load(f)
items = data.get('items', data) if isinstance(data, dict) else data
for i, q in enumerate(items):
    if q['uuid'] == '$target_uuid':
        print(i)
        break
"
}

# Function to generate a single trajectory (called by parallel workers)
generate_single() {
    local task=$1
    local task_num=$2
    local total=$3
    local uuid="${task%%:*}"
    local pass_num="${task##*:}"
    local result_file="$TEMP_DIR/result_${uuid}_${pass_num}"

    echo -e "${BLUE}[${task_num}/${total}]${NC} ${YELLOW}Starting pass@${pass_num} for ${uuid}...${NC}"

    # Get the query index for this UUID
    local query_index=$(get_query_index "$uuid")

    if [ "$GOAL_ORIENTED" = true ]; then
        # Goal-oriented agent
        if python "${SCRIPT_DIR}/run_goaloriented_agent.py" \
            --seeds "$QUERY_FILE" \
            --model "$MODEL" \
            --user-model "$USER_MODEL" \
            --persona "$PERSONA" \
            --query-index "$query_index" \
            --pass-number "$pass_num" \
            --max-iterations "$MAX_ITERATIONS" \
            --save-trajectory 2>&1; then
            echo "success" > "$result_file"
            echo -e "${BLUE}[${task_num}/${total}]${NC} ${GREEN}✓ Completed pass@${pass_num} for ${uuid}${NC}"
        else
            echo "failed" > "$result_file"
            echo -e "${BLUE}[${task_num}/${total}]${NC} ${RED}✗ Failed pass@${pass_num} for ${uuid}${NC}"
        fi
    else
        # ReAct agent
        if python "${SCRIPT_DIR}/run_react_agent.py" \
            --query-file "$QUERY_FILE" \
            --query-uuid "$uuid" \
            --model "$MODEL" \
            --pass-number "$pass_num" \
            --max-iterations "$MAX_ITERATIONS" \
            --save-trajectory \
            --output-dir "$TRAJECTORY_DIR" 2>&1; then
            echo "success" > "$result_file"
            echo -e "${BLUE}[${task_num}/${total}]${NC} ${GREEN}✓ Completed pass@${pass_num} for ${uuid}${NC}"
        else
            echo "failed" > "$result_file"
            echo -e "${BLUE}[${task_num}/${total}]${NC} ${RED}✗ Failed pass@${pass_num} for ${uuid}${NC}"
        fi
    fi
}

export -f generate_single get_query_index
export SCRIPT_DIR QUERY_FILE MODEL TRAJECTORY_DIR MAX_ITERATIONS GOAL_ORIENTED USER_MODEL PERSONA
export RED GREEN YELLOW BLUE NC

# Main function to run one round of generation
run_generation_round() {
    local round_num=$1

    echo ""
    if [ "$LOOP_UNTIL_COMPLETE" = true ]; then
        echo -e "${BLUE}========== ROUND ${round_num}/${MAX_RETRIES} ==========${NC}"
    fi
    echo -e "${YELLOW}Scanning for missing trajectories...${NC}"
    echo "=============================================="

    MISSING_TOTAL=0
    MISSING_PASS1=()
    MISSING_PASS2=()
    MISSING_PASS3=()

    # Determine which passes to check
    if [ -n "$SPECIFIC_PASS" ]; then
        PASSES_TO_CHECK=($SPECIFIC_PASS)
    else
        PASSES_TO_CHECK=(1 2 3)
    fi

    # Check each pass
    for pass in "${PASSES_TO_CHECK[@]}"; do
        existing_count=$(get_existing_uuids $pass | wc -l | tr -d ' ')
        missing_uuids=$(find_missing_uuids $pass)
        missing_count=$(echo "$missing_uuids" | grep -c . 2>/dev/null | tr -d '[:space:]' || echo 0)
        [ -z "$missing_count" ] && missing_count=0

        echo -e "pass@${pass}: ${existing_count}/${TOTAL_QUERIES} (${RED}${missing_count} missing${NC})"

        if [ "$missing_count" -gt 0 ]; then
            MISSING_TOTAL=$((MISSING_TOTAL + missing_count))
            if [ $pass -eq 1 ]; then
                MISSING_PASS1=($missing_uuids)
            elif [ $pass -eq 2 ]; then
                MISSING_PASS2=($missing_uuids)
            else
                MISSING_PASS3=($missing_uuids)
            fi
        fi
    done

    echo "=============================================="
    echo -e "${YELLOW}Total missing: ${MISSING_TOTAL}${NC}"
    echo ""

    if [ $MISSING_TOTAL -eq 0 ]; then
        echo -e "${GREEN}All trajectories are complete! Nothing to generate.${NC}"
        return 0  # Success - all complete
    fi

    # Show what will be generated
    echo -e "${YELLOW}Will generate the following trajectories:${NC}"
    echo ""

    for uuid in "${MISSING_PASS1[@]}"; do
        [ -n "$uuid" ] && echo "  pass@1: $uuid"
    done
    for uuid in "${MISSING_PASS2[@]}"; do
        [ -n "$uuid" ] && echo "  pass@2: $uuid"
    done
    for uuid in "${MISSING_PASS3[@]}"; do
        [ -n "$uuid" ] && echo "  pass@3: $uuid"
    done

    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would generate $MISSING_TOTAL trajectories (max-concurrent: $MAX_CONCURRENT)${NC}"
        return 0
    fi

    # Ask for confirmation only on first round (or if not looping)
    if [ "$round_num" -eq 1 ]; then
        if [ "$LOOP_UNTIL_COMPLETE" = true ]; then
            read -p "Generate missing trajectories with $MAX_CONCURRENT concurrent workers (will retry up to $MAX_RETRIES times)? [y/N] " confirm
        else
            read -p "Generate $MISSING_TOTAL missing trajectories with $MAX_CONCURRENT concurrent workers? [y/N] " confirm
        fi
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi

    echo ""
    echo -e "${GREEN}Starting generation with ${MAX_CONCURRENT} concurrent workers...${NC}"
    echo ""

    # Create temp directory for tracking results
    TEMP_DIR=$(mktemp -d)
    export TEMP_DIR

    # Build list of all tasks (uuid:pass_num)
    TASK_FILE="$TEMP_DIR/tasks.txt"
    for uuid in "${MISSING_PASS1[@]}"; do
        [ -n "$uuid" ] && echo "${uuid}:1" >> "$TASK_FILE"
    done
    for uuid in "${MISSING_PASS2[@]}"; do
        [ -n "$uuid" ] && echo "${uuid}:2" >> "$TASK_FILE"
    done
    for uuid in "${MISSING_PASS3[@]}"; do
        [ -n "$uuid" ] && echo "${uuid}:3" >> "$TASK_FILE"
    done

    TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

    # Run tasks in parallel using xargs
    # Each line in TASK_FILE is "uuid:pass_num"
    task_num=0
    cat "$TASK_FILE" | while read task; do
        task_num=$((task_num + 1))
        echo "$task $task_num $TOTAL_TASKS"
    done | xargs -P "$MAX_CONCURRENT" -L 1 bash -c 'generate_single $0 $1 $2'

    # Count results (use tr to remove any whitespace/newlines)
    GENERATED=$(find "$TEMP_DIR" -name "result_*" -exec cat {} \; 2>/dev/null | grep -c "success" | tr -d '[:space:]')
    GENERATED=${GENERATED:-0}
    FAILED=$(find "$TEMP_DIR" -name "result_*" -exec cat {} \; 2>/dev/null | grep -c "failed" | tr -d '[:space:]')
    FAILED=${FAILED:-0}

    # Cleanup temp directory
    rm -rf "$TEMP_DIR"

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Round ${round_num} complete!${NC}"
    echo -e "  Generated: ${GREEN}${GENERATED}${NC}"
    echo -e "  Failed: ${RED}${FAILED}${NC}"
    echo -e "  Concurrent workers: ${MAX_CONCURRENT}"
    echo "=============================================="

    # Return 1 if there are still failures (to continue looping)
    if [ "$FAILED" -gt 0 ]; then
        return 1
    fi
    return 0
}

# Main execution
if [ "$LOOP_UNTIL_COMPLETE" = true ]; then
    echo -e "${BLUE}Running in loop mode (max $MAX_RETRIES retries)${NC}"

    for round in $(seq 1 $MAX_RETRIES); do
        if run_generation_round $round; then
            echo ""
            echo -e "${GREEN}=============================================="
            echo -e "ALL TRAJECTORIES COMPLETE!"
            echo -e "==============================================${NC}"
            exit 0
        fi

        if [ $round -lt $MAX_RETRIES ]; then
            echo ""
            echo -e "${YELLOW}Some trajectories failed. Retrying in 5 seconds...${NC}"
            sleep 5
        fi
    done

    echo ""
    echo -e "${RED}=============================================="
    echo -e "MAX RETRIES ($MAX_RETRIES) REACHED"
    echo -e "Some trajectories may still be missing."
    echo -e "==============================================${NC}"
    exit 1
else
    run_generation_round 1
fi
