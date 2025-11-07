#!/usr/bin/env python3
"""Batch modify config file paths: replace ../../../yamls/ with ../../../../yamls/"""

import os
import glob
import sys

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Config file directory
base_path = r"src\test\fl_lora_sample\convergence_experiment"

# Find all .yaml files
yaml_files = []
for dataset in ["lora_fmnist", "lora_kmnist", "lora_mnist", "lora_qmnist"]:
    pattern = os.path.join(base_path, dataset, "**", "*.yaml")
    yaml_files.extend(glob.glob(pattern, recursive=True))

print(f"Found {len(yaml_files)} YAML files")

modified_count = 0
for yaml_file in yaml_files:
    try:
        # Read file
        with open(yaml_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if modification needed
        if "../../../yamls/" in content:
            # Replace all paths
            new_content = content.replace("../../../yamls/", "../../../../yamls/")

            # Write back
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            modified_count += 1
            print(f"[OK] Modified: {os.path.basename(yaml_file)}")
        else:
            # Check if already correct
            if "../../../../yamls/" in content:
                print(f"[SKIP] Already correct: {os.path.basename(yaml_file)}")
            else:
                print(f"[WARN] Path not found: {os.path.basename(yaml_file)}")

    except Exception as e:
        print(f"[ERROR] Failed to process {yaml_file}: {e}")

print(f"\nComplete! Modified {modified_count} files")
