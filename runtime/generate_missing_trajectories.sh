#!/bin/bash
# Generate missing trajectories for an existing trajectory folder
# Usage: ./generate_missing_trajectories.sh <trajectory_dir> <model> [--dry-run]
#
# Example:
#   ./generate_missing_trajectories.sh trajectories/claude-3.5 anthropic/claude-3.5-sonnet
#   ./generate_missing_trajectories.sh trajectories/gpt-4omini openai/gpt-4o-mini --dry-run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QUERY_FILE="${PROJECT_ROOT}/mcp_generate/queries_verification.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <trajectory_dir> <model> [--dry-run]"
    echo ""
    echo "Arguments:"
    echo "  trajectory_dir    Path to trajectory directory (e.g., trajectories/claude-3.5)"
    echo "  model             Model name (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4o-mini)"
    echo ""
    echo "Options:"
    echo "  --dry-run         Show what would be generated without actually running"
    echo ""
    echo "Example:"
    echo "  $0 trajectories/claude-3.5 anthropic/claude-3.5-sonnet"
    exit 1
}

# Parse arguments
if [ $# -lt 2 ]; then
    usage
fi

TRAJECTORY_DIR="$1"
MODEL="$2"
DRY_RUN=false

if [ "$3" == "--dry-run" ]; then
    DRY_RUN=true
fi

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
for q in data:
    print(q['uuid'])
")
TOTAL_QUERIES=$(echo "$ALL_UUIDS" | wc -l | tr -d ' ')
echo -e "${GREEN}Found $TOTAL_QUERIES queries${NC}"

# Function to get existing UUIDs for a pass
get_existing_uuids() {
    local pass_num=$1
    find "$TRAJECTORY_DIR" -path "*pass@${pass_num}*" -name "trajectory_*.json" 2>/dev/null | \
        xargs -I {} basename {} | \
        sed 's/trajectory_//' | \
        cut -d'_' -f1-5 | \
        sort -u
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
    echo -e "${YELLOW}[DRY RUN] Would generate $MISSING_TOTAL trajectories${NC}"
    exit 0
fi

# Ask for confirmation
read -p "Generate $MISSING_TOTAL missing trajectories? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting generation...${NC}"
echo ""

# Generate missing trajectories
GENERATED=0
FAILED=0

generate_trajectory() {
    local uuid=$1
    local pass_num=$2

    echo -e "${YELLOW}Generating pass@${pass_num} for ${uuid}...${NC}"

    if python3 "${SCRIPT_DIR}/run_react_agent.py" \
        --query-file "$QUERY_FILE" \
        --query-uuid "$uuid" \
        --model "$MODEL" \
        --pass-number "$pass_num" \
        --save-trajectory \
        --output-dir "$TRAJECTORY_DIR"; then
        echo -e "${GREEN}✓ Completed pass@${pass_num} for ${uuid}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed pass@${pass_num} for ${uuid}${NC}"
        return 1
    fi
}

# Generate pass@1 missing
for uuid in "${MISSING_PASS1[@]}"; do
    [ -z "$uuid" ] && continue
    if generate_trajectory "$uuid" 1; then
        GENERATED=$((GENERATED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# Generate pass@2 missing
for uuid in "${MISSING_PASS2[@]}"; do
    [ -z "$uuid" ] && continue
    if generate_trajectory "$uuid" 2; then
        GENERATED=$((GENERATED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# Generate pass@3 missing
for uuid in "${MISSING_PASS3[@]}"; do
    [ -z "$uuid" ] && continue
    if generate_trajectory "$uuid" 3; then
        GENERATED=$((GENERATED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo -e "${GREEN}Generation complete!${NC}"
echo -e "  Generated: ${GREEN}${GENERATED}${NC}"
echo -e "  Failed: ${RED}${FAILED}${NC}"
echo "=============================================="
