"""
Trajectory-Focused Visualization Script for MCP-R
Generates charts analyzing tool usage patterns from trajectories
For Smithery founder presentation
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import json
from pathlib import Path
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'success': '#2ecc71',
    'failure': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#9b59b6',
    'warning': '#f39c12',
    'meta': '#e67e22',
    'actual': '#27ae60'
}

# Paths
TRAJ_DIR = Path('/Users/xiziqiao/Documents/MCP-Research/MCP-R/trajectories')
OUTPUT_DIR = Path('/Users/xiziqiao/Documents/MCP-Research/MCP-R/evaluation/stats')


def load_all_trajectory_data():
    """Load tool usage statistics from all trajectory files"""
    trajectory_files = list(TRAJ_DIR.glob('*.json'))

    all_trajectories = []

    for traj_file in trajectory_files:
        with open(traj_file, 'r') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        meta_mcp_count = 0
        actual_tool_count = 0

        # Handle two different trajectory formats
        if 'reasoning_trace' in data:
            # Format 1: reasoning_trace format
            reasoning_trace = data.get('reasoning_trace', [])
            for step in reasoning_trace:
                if step.get('type') == 'action':
                    action_text = step.get('content', '')
                    if 'meta-mcp' in action_text:
                        meta_mcp_count += 1
                    else:
                        actual_tool_count += 1

        elif 'execution' in data:
            # Format 2: execution format with tool_calls
            tool_calls = data.get('execution', {}).get('tool_calls', [])
            for tool_call in tool_calls:
                server = tool_call.get('server', '')
                if server == 'meta-mcp':
                    meta_mcp_count += 1
                else:
                    actual_tool_count += 1

        total_actions = meta_mcp_count + actual_tool_count

        all_trajectories.append({
            'file': traj_file.name,
            'meta_mcp': meta_mcp_count,
            'actual_tools': actual_tool_count,
            'total': total_actions,
            'query': metadata.get('query', 'Unknown'),
            'timestamp': metadata.get('timestamp', 'Unknown')
        })

    return all_trajectories


def create_tool_usage_overview():
    """Chart 1: Tool Usage Overview"""
    fig, ax = plt.subplots(figsize=(10, 7))

    trajs = load_all_trajectory_data()
    total_trajs = len(trajs)

    total_meta = sum(t['meta_mcp'] for t in trajs)
    total_actual = sum(t['actual_tools'] for t in trajs)
    total_all = total_meta + total_actual

    avg_meta = total_meta / total_trajs
    avg_actual = total_actual / total_trajs
    avg_total = total_all / total_trajs

    # Create grouped bar chart
    categories = ['Meta-MCP\nSearches', 'Actual\nTool Calls', 'Total\nAPI Calls']
    averages = [avg_meta, avg_actual, avg_total]
    totals = [total_meta, total_actual, total_all]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, averages, width, label='Avg per Task',
                   color=[colors['meta'], colors['actual'], colors['secondary']],
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, avg, total in zip(bars1, averages, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{avg:.2f}\n({total} total)',
                ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_ylabel('Average Count per Trajectory', fontsize=14, weight='bold')
    ax.set_title(f'Tool Usage Pattern Across {total_trajs} Trajectories',
                 fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0, max(averages) * 1.3)
    ax.grid(True, axis='y', alpha=0.3)

    # Add summary text
    summary = f"Pattern: ~{avg_meta:.1f} Meta-MCP + {avg_actual:.1f} Actual = {avg_total:.1f} total"
    ax.text(0.5, 0.95, summary, transform=ax.transAxes,
            ha='center', va='top', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_1_tool_usage_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_1_tool_usage_overview.png")
    plt.close()


def create_tool_distribution():
    """Chart 2: Distribution of Tool Usage"""
    fig, ax = plt.subplots(figsize=(12, 7))

    trajs = load_all_trajectory_data()
    tool_totals = [t['total'] for t in trajs]
    total_dist = Counter(tool_totals)

    counts = sorted(total_dist.keys())
    frequencies = [total_dist[c] for c in counts]

    bars = ax.bar(counts, frequencies, color=colors['primary'],
                   edgecolor='black', linewidth=1.5, width=0.6)

    for bar, count in zip(bars, frequencies):
        height = bar.get_height()
        if height > 0:
            pct = count / len(trajs) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(count)}\n({pct:.1f}%)',
                    ha='center', va='bottom',
                    fontsize=12, weight='bold')

    ax.set_xlabel('Total Tools Used', fontsize=15, weight='bold')
    ax.set_ylabel('Number of Trajectories', fontsize=15, weight='bold')
    ax.set_title(f'Distribution of Tool Usage Across {len(trajs)} Trajectories',
                 fontsize=17, weight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)

    # Add median/mode
    median = np.median(tool_totals)
    mode = max(set(tool_totals), key=tool_totals.count)
    ax.axvline(median, color='red', linestyle='--', linewidth=2.5,
               label=f'Median: {median:.1f}', alpha=0.7)
    ax.axvline(mode, color='orange', linestyle='--', linewidth=2.5,
               label=f'Mode: {mode}', alpha=0.7)
    ax.legend(fontsize=13, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_2_tool_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_2_tool_distribution.png")
    plt.close()


def create_meta_vs_actual_breakdown():
    """Chart 3: Meta-MCP vs Actual Tools Breakdown"""
    fig, ax = plt.subplots(figsize=(12, 7))

    trajs = load_all_trajectory_data()

    # Group by total count
    grouped_data = {}
    for t in trajs:
        total = t['total']
        if total not in grouped_data:
            grouped_data[total] = {'meta': [], 'actual': []}
        grouped_data[total]['meta'].append(t['meta_mcp'])
        grouped_data[total]['actual'].append(t['actual_tools'])

    totals_sorted = sorted(grouped_data.keys())
    avg_meta_by_total = [np.mean(grouped_data[t]['meta']) for t in totals_sorted]
    avg_actual_by_total = [np.mean(grouped_data[t]['actual']) for t in totals_sorted]
    count_by_total = [len(grouped_data[t]['meta']) for t in totals_sorted]

    x = np.arange(len(totals_sorted))
    width = 0.6

    bars1 = ax.bar(x, avg_meta_by_total, width, label='Meta-MCP Search',
                   color=colors['meta'], edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, avg_actual_by_total, width, bottom=avg_meta_by_total,
                   label='Actual Tool Calls', color=colors['actual'],
                   edgecolor='black', linewidth=1.5)

    # Add count labels on top
    for i, (total, count) in enumerate(zip(totals_sorted, count_by_total)):
        height = avg_meta_by_total[i] + avg_actual_by_total[i]
        ax.text(i, height + 0.1, f'n={count}',
                ha='center', va='bottom', fontsize=10, style='italic')

    ax.set_xlabel('Total Tools Used', fontsize=14, weight='bold')
    ax.set_ylabel('Average Count', fontsize=14, weight='bold')
    ax.set_title('Meta-MCP vs Actual Tools Breakdown by Total Usage',
                 fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(totals_sorted, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_3_meta_vs_actual.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_3_meta_vs_actual.png")
    plt.close()


def create_api_statistics():
    """Chart 4a: API Call Statistics Summary"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis('off')

    trajs = load_all_trajectory_data()
    total_trajs = len(trajs)

    total_meta = sum(t['meta_mcp'] for t in trajs)
    total_actual = sum(t['actual_tools'] for t in trajs)
    total_all = total_meta + total_actual

    tool_totals = [t['total'] for t in trajs]

    metrics_text = f"""
    API CALL STATISTICS
    {'='*50}

    Total Trajectories:          {total_trajs:,}
    Total API Calls:             {total_all:,}

    Meta-MCP Searches:           {total_meta:,}
    Actual Tool Calls:           {total_actual:,}

    {'─'*50}
    AVERAGES PER TRAJECTORY
    {'─'*50}

    Avg Total:                   {total_all/total_trajs:.2f}
    Avg Meta-MCP:                {total_meta/total_trajs:.2f}
    Avg Actual:                  {total_actual/total_trajs:.2f}

    {'─'*50}
    DISTRIBUTION STATISTICS
    {'─'*50}

    Median:                      {np.median(tool_totals):.1f}
    Mode:                        {max(set(tool_totals), key=tool_totals.count)}
    Mean:                        {np.mean(tool_totals):.2f}
    Std Dev:                     {np.std(tool_totals):.2f}
    Min:                         {min(tool_totals)}
    Max:                         {max(tool_totals)}

    {'─'*50}
    COST ESTIMATES (@ $0.01/call)
    {'─'*50}

    Total Cost:                  ${total_all * 0.01:.2f}
    Per Trajectory:              ${total_all/total_trajs * 0.01:.4f}

    {'='*50}
    """

    ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=13, verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1.5))

    plt.title(f'API Call Statistics Summary - {total_trajs} Trajectories',
              fontsize=17, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_4a_api_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_4a_api_statistics.png")
    plt.close()


def create_api_breakdown_pie():
    """Chart 4b: API Call Type Breakdown"""
    fig, ax = plt.subplots(figsize=(10, 8))

    trajs = load_all_trajectory_data()
    total_meta = sum(t['meta_mcp'] for t in trajs)
    total_actual = sum(t['actual_tools'] for t in trajs)
    total_all = total_meta + total_actual

    sizes = [total_meta, total_actual]
    labels = [f'Meta-MCP Searches\n{total_meta:,} calls\n({total_meta/total_all*100:.1f}%)',
              f'Actual Tool Calls\n{total_actual:,} calls\n({total_actual/total_all*100:.1f}%)']
    colors_pie = [colors['meta'], colors['actual']]

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct='', startangle=90,
                                       textprops={'fontsize': 13, 'weight': 'bold'},
                                       explode=(0.05, 0.05))

    # Add center annotation
    ax.text(0, 0, f'{total_all:,}\nTotal\nAPI Calls',
            ha='center', va='center', fontsize=16, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title('API Call Type Breakdown', fontsize=17, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_4b_api_breakdown.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_4b_api_breakdown.png")
    plt.close()


def create_trajectory_distribution():
    """Chart 4c: Trajectory Distribution by Tool Usage"""
    fig, ax = plt.subplots(figsize=(12, 7))

    trajs = load_all_trajectory_data()
    total_trajs = len(trajs)
    tool_totals = [t['total'] for t in trajs]

    dist = Counter(tool_totals)
    counts = sorted(dist.keys())
    freqs = [dist[c] for c in counts]

    bars = ax.bar(counts, freqs, color=colors['secondary'],
                  edgecolor='black', linewidth=1.5, width=0.7)

    for bar, count in zip(bars, freqs):
        if count > 0:
            height = bar.get_height()
            pct = count / total_trajs * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(count)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_xlabel('Total Tools Used', fontsize=15, weight='bold')
    ax.set_ylabel('Number of Trajectories', fontsize=15, weight='bold')
    ax.set_title(f'Trajectory Distribution by Tool Usage - {total_trajs} Trajectories',
                 fontsize=17, weight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)

    # Add median/mode lines
    median = np.median(tool_totals)
    mode = max(set(tool_totals), key=tool_totals.count)
    ax.axvline(median, color='red', linestyle='--', linewidth=2.5,
               label=f'Median: {median:.1f}', alpha=0.7)
    ax.axvline(mode, color='orange', linestyle='--', linewidth=2.5,
               label=f'Mode: {mode}', alpha=0.7)
    ax.legend(fontsize=13, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_4c_trajectory_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_4c_trajectory_distribution.png")
    plt.close()


def create_cumulative_api_growth():
    """Chart 5a: Cumulative API Load Growth"""
    fig, ax = plt.subplots(figsize=(12, 7))

    trajs = load_all_trajectory_data()
    total_trajs = len(trajs)

    # Cumulative API calls
    tool_totals = sorted([t['total'] for t in trajs])
    cumulative = np.cumsum(tool_totals)

    ax.plot(range(1, len(cumulative) + 1), cumulative,
            linewidth=3, color=colors['primary'], marker='o', markersize=5)
    ax.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.3, color=colors['primary'])

    ax.set_xlabel('Number of Trajectories', fontsize=15, weight='bold')
    ax.set_ylabel('Cumulative API Calls', fontsize=15, weight='bold')
    ax.set_title(f'Cumulative API Load Growth - {total_trajs} Trajectories',
                 fontsize=17, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    # Add annotation
    total_calls = cumulative[-1]
    ax.annotate(f'Total: {int(total_calls):,} API calls',
                xy=(len(cumulative), total_calls),
                xytext=(len(cumulative)*0.6, total_calls*0.5),
                fontsize=13, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_5a_cumulative_api_growth.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_5a_cumulative_api_growth.png")
    plt.close()


def create_api_percentile_analysis():
    """Chart 5b: API Usage by Percentile"""
    fig, ax = plt.subplots(figsize=(12, 7))

    trajs = load_all_trajectory_data()
    total_trajs = len(trajs)

    # API call efficiency by percentile
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = [np.percentile([t['total'] for t in trajs], p) for p in percentiles]

    bars = ax.bar(range(len(percentiles)), values,
                  color=colors['success'], edgecolor='black', linewidth=1.5, width=0.6)

    for bar, val, pct in zip(bars, values, percentiles):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
               f'{val:.1f}', ha='center', va='bottom',
               fontsize=12, weight='bold')

    ax.set_xticks(range(len(percentiles)))
    ax.set_xticklabels([f'{p}th' for p in percentiles], fontsize=13)
    ax.set_xlabel('Percentile', fontsize=15, weight='bold')
    ax.set_ylabel('Tools Used', fontsize=15, weight='bold')
    ax.set_title(f'API Usage by Percentile - {total_trajs} Trajectories',
                 fontsize=17, weight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)

    # Add median line
    median_val = values[percentiles.index(50)]
    ax.axhline(median_val, color='red', linestyle='--', linewidth=2,
               label=f'Median (50th): {median_val:.1f}', alpha=0.7)
    ax.legend(fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'traj_5b_api_percentile.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: traj_5b_api_percentile.png")
    plt.close()


def main():
    """Generate all trajectory-focused charts"""
    print("\n" + "="*60)
    print("MCP-R Trajectory Analysis - Chart Generator")
    print("="*60 + "\n")

    print("Loading trajectory data...\n")
    trajs = load_all_trajectory_data()
    print(f"✓ Loaded {len(trajs)} trajectories\n")

    print("Generating charts...\n")

    create_tool_usage_overview()
    create_tool_distribution()
    create_meta_vs_actual_breakdown()
    create_api_statistics()
    create_api_breakdown_pie()
    create_trajectory_distribution()
    create_cumulative_api_growth()
    create_api_percentile_analysis()

    print("\n" + "="*60)
    print("✓ All trajectory charts generated successfully!")
    print("="*60)
    print(f"\nOutput location: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. traj_1_tool_usage_overview.png")
    print("  2. traj_2_tool_distribution.png")
    print("  3. traj_3_meta_vs_actual.png")
    print("  4a. traj_4a_api_statistics.png")
    print("  4b. traj_4b_api_breakdown.png")
    print("  4c. traj_4c_trajectory_distribution.png")
    print("  5a. traj_5a_cumulative_api_growth.png")
    print("  5b. traj_5b_api_percentile.png")
    print("\n")


if __name__ == "__main__":
    main()
