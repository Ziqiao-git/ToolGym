#!/bin/bash
# Generate missing trajectories for an existing trajectory folder
# Usage: ./generate_missing_trajectories.sh <trajectory_dir> <model> [options]
#
# Example:
#   ./generate_missing_trajectories.sh trajectories/claude-3.5 anthropic/claude-3.5-sonnet
#   ./generate_missing_trajectories.sh trajectories/gpt-4omini openai/gpt-4o-mini --dry-run
#   ./generate_missing_trajectories.sh trajectories/deepseek-v3.2 deepseek/deepseek-v3.2 --max-concurrent 5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QUERY_FILE="${PROJECT_ROOT}/mcp_generate/queries_verification.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <trajectory_dir> <model> [options]"
    echo ""
    echo "Arguments:"
    echo "  trajectory_dir    Path to trajectory directory (e.g., trajectories/claude-3.5)"
    echo "  model             Model name (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4o-mini)"
    echo ""
    echo "Options:"
    echo "  --dry-run              Show what would be generated without actually running"
    echo "  --max-concurrent N     Run N trajectories in parallel (default: 5)"
    echo "  --max-iterations N     Max reasoning iterations per query (default: 10)"
    echo ""
    echo "Example:"
    echo "  $0 trajectories/claude-3.5 anthropic/claude-3.5-sonnet"
    echo "  $0 trajectories/deepseek-v3.2 deepseek/deepseek-v3.2 --max-concurrent 3"
    exit 1
}

# Parse arguments
if [ $# -lt 2 ]; then
    usage
fi

TRAJECTORY_DIR="$1"
MODEL="$2"
DRY_RUN=false
MAX_CONCURRENT=5
MAX_ITERATIONS=10  # Match batch_generate_trajectories.py default

# Parse optional arguments
shift 2
while [ $# -gt 0 ]; do
    case "$1" in
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
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate trajectory directory
if [ ! -d "$TRAJECTORY_DIR" ]; then
    echo -e "${RED}Error: Directory '$TRAJECTORY_DIR' does not exist${NC}"
    exit 1
fi

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

echo ""
echo -e "${YELLOW}Scanning for missing trajectories...${NC}"
echo "=============================================="

MISSING_TOTAL=0
declare -a MISSING_PASS1 MISSING_PASS2 MISSING_PASS3

# Check each pass
for pass in 1 2 3; do
    existing_count=$(get_existing_uuids $pass | wc -l | tr -d ' ')
    missing_uuids=$(find_missing_uuids $pass)
    missing_count=$(echo "$missing_uuids" | grep -c . 2>/dev/null || echo 0)

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
    exit 0
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
    exit 0
fi

# Ask for confirmation
read -p "Generate $MISSING_TOTAL missing trajectories with $MAX_CONCURRENT concurrent workers? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting generation with ${MAX_CONCURRENT} concurrent workers...${NC}"
echo ""

# Create temp directory for tracking results
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

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

# Function to generate a single trajectory (called by parallel workers)
generate_single() {
    local task=$1
    local task_num=$2
    local total=$3
    local uuid="${task%%:*}"
    local pass_num="${task##*:}"
    local result_file="$TEMP_DIR/result_${uuid}_${pass_num}"

    echo -e "${BLUE}[${task_num}/${total}]${NC} ${YELLOW}Starting pass@${pass_num} for ${uuid}...${NC}"

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
}

export -f generate_single
export SCRIPT_DIR QUERY_FILE MODEL TRAJECTORY_DIR TEMP_DIR MAX_ITERATIONS
export RED GREEN YELLOW BLUE NC

# Run tasks in parallel using xargs
# Each line in TASK_FILE is "uuid:pass_num"
task_num=0
cat "$TASK_FILE" | while read task; do
    task_num=$((task_num + 1))
    echo "$task $task_num $TOTAL_TASKS"
done | xargs -P "$MAX_CONCURRENT" -L 1 bash -c 'generate_single $0 $1 $2'

# Count results
GENERATED=$(find "$TEMP_DIR" -name "result_*" -exec cat {} \; | grep -c "success" 2>/dev/null || echo 0)
FAILED=$(find "$TEMP_DIR" -name "result_*" -exec cat {} \; | grep -c "failed" 2>/dev/null || echo 0)

echo ""
echo "=============================================="
echo -e "${GREEN}Generation complete!${NC}"
echo -e "  Generated: ${GREEN}${GENERATED}${NC}"
echo -e "  Failed: ${RED}${FAILED}${NC}"
echo -e "  Concurrent workers: ${MAX_CONCURRENT}"
echo "=============================================="
