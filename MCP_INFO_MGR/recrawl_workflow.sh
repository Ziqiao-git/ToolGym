#!/bin/bash
# Complete workflow to recrawl Smithery MCP servers and rebuild indexes
# This script automates all steps from data collection to FAISS index building

set -e  # Exit on error

echo "========================================================================"
echo "MCP SERVER RECRAWL WORKFLOW"
echo "========================================================================"
echo ""

# Configuration
PROJECT_ROOT="/Users/xiziqiao/Documents/MCP-Research/MCP-R"
cd "$PROJECT_ROOT"

# Step 1: Crawl Smithery Registry
echo "Step 1/4: Crawling Smithery registry..."
echo "------------------------------------------------------------------------"
python MCP_INFO_MGR/crwal.py \
  --ndjson mcp_data/raw/smithery_servers.ndjson \
  --dedupe

echo ""
echo "✓ Raw data saved to mcp_data/raw/smithery_servers.ndjson"
echo ""

# Step 2: Filter for Remote Servers Only
echo "Step 2/4: Filtering for remote MCP servers..."
echo "------------------------------------------------------------------------"
python MCP_INFO_MGR/filter_remote_servers.py \
  --input mcp_data/raw/smithery_servers.ndjson \
  --output mcp_data/raw/smithery_remote_servers.ndjson \
  --stats

echo ""
echo "✓ Remote servers saved to mcp_data/raw/smithery_remote_servers.ndjson"
echo ""

# Step 3: Check if tool descriptions need updating
echo "Step 3/4: Checking tool descriptions..."
echo "------------------------------------------------------------------------"
TOOL_DESC_FILE="MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson"

if [ -f "$TOOL_DESC_FILE" ]; then
    TOOL_COUNT=$(wc -l < "$TOOL_DESC_FILE" | tr -d ' ')
    TOOL_AGE=$(find "$TOOL_DESC_FILE" -mtime +7 2>/dev/null && echo "old" || echo "recent")

    echo "Existing tool descriptions found:"
    echo "  File: $TOOL_DESC_FILE"
    echo "  Tools: $TOOL_COUNT"
    echo "  Last modified: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$TOOL_DESC_FILE" 2>/dev/null || stat -c "%y" "$TOOL_DESC_FILE" 2>/dev/null | cut -d' ' -f1-2)"
    echo ""

    if [ "$TOOL_AGE" = "old" ]; then
        echo "⚠️  Warning: Tool descriptions are more than 7 days old"
    fi

    echo "Using existing tool descriptions. To refresh, run the tool fetching pipeline manually."
    echo "(See COMMANDS.md for detailed instructions on fetching fresh tool descriptions)"
else
    echo "❌ No tool descriptions found at $TOOL_DESC_FILE"
    echo ""
    echo "You need to fetch tool descriptions first. Run these commands:"
    echo ""
    echo "  # Generate server configs"
    echo "  python MCP_INFO_MGR/generate_smithery_configs.py \\"
    echo "    --input mcp_data/raw/smithery_remote_servers.ndjson \\"
    echo "    --output MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json"
    echo ""
    echo "  # Fetch tool descriptions"
    echo "  python MCP_INFO_MGR/fetch_tool_descriptions.py \\"
    echo "    --input mcp_data/raw/smithery_remote_servers.ndjson \\"
    echo "    --config MCP_INFO_MGR/mcp_data/usable/remote_server_configs.json \\"
    echo "    --output MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson"
    echo ""
    exit 1
fi

echo ""

# Step 4: Summary Statistics
echo "Step 4/4: Summary Statistics"
echo "========================================================================"

# Count files
TOTAL_RAW=$(wc -l < mcp_data/raw/smithery_servers.ndjson | tr -d ' ')
TOTAL_REMOTE=$(wc -l < mcp_data/raw/smithery_remote_servers.ndjson | tr -d ' ')

if [ -f "MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson" ]; then
    TOTAL_TOOLS=$(wc -l < MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson | tr -d ' ')
else
    TOTAL_TOOLS="N/A"
fi

echo "Raw Smithery servers:      $TOTAL_RAW"
echo "Remote servers (filtered): $TOTAL_REMOTE"
echo "Tool descriptions:         $TOTAL_TOOLS"
echo ""

echo ""
echo "========================================================================"
echo "RECRAWL COMPLETE!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Test tools and filter working ones (see COMMANDS.md)"
echo ""
echo "  2. Build FAISS index after filtering:"
echo "     python -m MCP_INFO_MGR.semantic_search.build_search_index \\"
echo "       --input MCP_INFO_MGR/mcp_data/indexed/tool_descriptions.ndjson \\"
echo "       --output MCP_INFO_MGR/semantic_search/"
echo ""
echo "  3. Generate new queries based on updated tools:"
echo "     python mcp_generate/query_generate_goaloriented.py --num-queries 20"
echo ""
echo "  4. Run goal-oriented conversations:"
echo "     python runtime/run_goaloriented_agent.py \\"
echo "       --seeds mcp_generate/requests/goaloriented_seeds.json \\"
echo "       --save-trajectory"
echo ""
