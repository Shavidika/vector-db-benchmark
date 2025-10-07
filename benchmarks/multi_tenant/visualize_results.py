"""
Generate visualization charts for multi-tenant benchmark results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read results
results_dir = Path(__file__).parent / "../../results"
csv_file = results_dir / "multitenant_benchmark_summary.csv"

df = pd.read_csv(csv_file)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Multi-Tenant Vector Database Benchmark Results', fontsize=16, fontweight='bold')

# 1. Insertion Time Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(df['Database'], df['Total Insertion Time (s)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax1.set_title('Total Insertion Time (10 tenants x 500 products)', fontweight='bold')
ax1.set_ylabel('Time (seconds)')
ax1.set_xlabel('Database')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s',
             ha='center', va='bottom', fontweight='bold')

# 2. Query Latency Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(df['Database'], df['Avg Query Latency (ms)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax2.set_title('Average Query Latency', fontweight='bold')
ax2.set_ylabel('Latency (ms)')
ax2.set_xlabel('Database')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}ms',
             ha='center', va='bottom', fontweight='bold')

# 3. Memory Usage
ax3 = axes[0, 2]
bars3 = ax3.bar(df['Database'], df['Avg Memory (MB)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax3.set_title('Average Memory Usage', fontweight='bold')
ax3.set_ylabel('Memory (MB)')
ax3.set_xlabel('Database')
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}MB',
             ha='center', va='bottom', fontweight='bold')

# 4. Latency Min/Max/Avg
ax4 = axes[1, 0]
x = np.arange(len(df['Database']))
width = 0.25
ax4.bar(x - width, df['Min Latency (ms)'], width, label='Min', color='#95E1D3')
ax4.bar(x, df['Avg Query Latency (ms)'], width, label='Avg', color='#4ECDC4')
ax4.bar(x + width, df['Max Latency (ms)'], width, label='Max', color='#38ada9')
ax4.set_title('Query Latency Distribution', fontweight='bold')
ax4.set_ylabel('Latency (ms)')
ax4.set_xlabel('Database')
ax4.set_xticks(x)
ax4.set_xticklabels(df['Database'])
ax4.legend()

# 5. Cross-Tenant Leakage (should be all zeros!)
ax5 = axes[1, 1]
bars5 = ax5.bar(df['Database'], df['Cross-Tenant Leakage'], color=['#51CF66', '#51CF66', '#51CF66'])
ax5.set_title('Cross-Tenant Leakage (Lower is Better)', fontweight='bold')
ax5.set_ylabel('Leakage Count')
ax5.set_xlabel('Database')
ax5.set_ylim(0, max(1, df['Cross-Tenant Leakage'].max() + 1))
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)} ✓',
             ha='center', va='bottom', fontweight='bold', color='green', fontsize=14)

# 6. Avg Insertion Time per Tenant
ax6 = axes[1, 2]
bars6 = ax6.bar(df['Database'], df['Avg Insertion Time/Tenant (s)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax6.set_title('Avg Insertion Time per Tenant', fontweight='bold')
ax6.set_ylabel('Time (seconds)')
ax6.set_xlabel('Database')
for bar in bars6:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save chart
chart_file = results_dir / "multitenant_benchmark_charts.png"
plt.savefig(chart_file, dpi=300, bbox_inches='tight')
print(f"✓ Charts saved to {chart_file}")

plt.show()
