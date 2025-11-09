import pandas as pd
import numpy as np
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Read the two CSV files
file1 = r"src\test\.training_results\train-20251107_120805-eb4fea9f.csv"
file2 = r"src\test\.training_results\train-20251107_121642-eb4fea9f.csv"

# Skip the first two rows (config and empty line)
df1 = pd.read_csv(file1, skiprows=2)
df2 = pd.read_csv(file2, skiprows=2)

print("=" * 80)
print("Training Results Comparison")
print("=" * 80)
print(f"\nFile 1: {file1}")
print(f"File 2: {file2}")
print(f"\nTotal rounds: {len(df1)}")

# Calculate differences for each column
print("\n" + "=" * 80)
print("Numerical Differences Statistics")
print("=" * 80)

metrics = ['accuracy', 'average_loss', 'precision', 'recall', 'f1_score']

for metric in metrics:
    diff = np.abs(df1[metric] - df2[metric])

    print(f"\n{metric}:")
    print(f"  Max difference: {diff.max():.10f}")
    print(f"  Mean difference: {diff.mean():.10f}")
    print(f"  Std deviation: {diff.std():.10f}")
    print(f"  Identical rounds: {(diff == 0).sum()} / {len(diff)}")

# Show some examples of differences
print("\n" + "=" * 80)
print("Accuracy differences for first 10 rounds:")
print("=" * 80)
print(f"{'Round':<6} {'File1':<10} {'File2':<10} {'Difference':<15}")
print("-" * 45)
for i in range(min(10, len(df1))):
    acc1 = df1['accuracy'].iloc[i]
    acc2 = df2['accuracy'].iloc[i]
    diff = abs(acc1 - acc2)
    print(f"{i+1:<6} {acc1:<10.4f} {acc2:<10.4f} {diff:<15.10f}")

print("\n" + "=" * 80)
print("Average loss differences for first 10 rounds:")
print("=" * 80)
print(f"{'Round':<6} {'File1':<15} {'File2':<15} {'Difference':<15}")
print("-" * 55)
for i in range(min(10, len(df1))):
    loss1 = df1['average_loss'].iloc[i]
    loss2 = df2['average_loss'].iloc[i]
    diff = abs(loss1 - loss2)
    print(f"{i+1:<6} {loss1:<15.6f} {loss2:<15.6f} {diff:<15.10f}")

# Calculate relative differences
print("\n" + "=" * 80)
print("Relative Difference Percentage (mean)")
print("=" * 80)
for metric in metrics:
    relative_diff = np.abs((df1[metric] - df2[metric]) / df1[metric]) * 100
    print(f"{metric}: {relative_diff.mean():.6f}%")

# Check configuration differences
print("\n" + "=" * 80)
print("Configuration Differences:")
print("=" * 80)
with open(file1, 'r') as f:
    config1 = f.readline().strip()
    print(f"File 1 config: {config1[7:]}")  # Skip "Config," prefix

with open(file2, 'r') as f:
    config2 = f.readline().strip()
    print(f"File 2 config: {config2[7:]}")  # Skip "Config," prefix

# Extract aggregation method
import json
config1_dict = json.loads(config1[7:])
config2_dict = json.loads(config2[7:])
print(f"\nAggregation method difference:")
print(f"  File 1: {config1_dict['aggregation']}")
print(f"  File 2: {config2_dict['aggregation']}")
