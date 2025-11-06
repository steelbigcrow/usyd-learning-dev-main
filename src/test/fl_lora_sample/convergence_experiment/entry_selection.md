# Entry File Selection Guide

This document explains which entry file should be used with each configuration file in the convergence experiment directories.

## Overview

The convergence experiment contains 72 configuration files organized across 4 datasets (mnist, fmnist, kmnist, qmnist), 3 methods (RBLA, SP, ZP), 2 rank distributions (Homogeneous, Heterogeneous), and 3 data distribution patterns (one_label, long_tail, feature_shift).

## Configuration File Structure

All configuration files follow this naming convention:
```
{method}_{dataset}_{data_distribution}_{rank_type}_round{rounds}_epoch{epochs}.yaml
```

Example: `rbla_mnist_one_label_homogeneous_round250_epoch1.yaml`

## Entry File Mapping

The entry file to use depends on the **data distribution pattern** in the configuration file name:

### 1. One Label Distribution (one_label)

**Entry File:** `lora_sample_entry.py`

**Configuration Files:** All files containing `_one_label_` in the filename
- Examples:
  - `rbla_mnist_one_label_homogeneous_round250_epoch1.yaml`
  - `sp_fmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - `zp_kmnist_one_label_homogeneous_round250_epoch1.yaml`

**Data Partitioner:** Uses `NoniidDataGenerator` to create highly non-IID data where each client receives samples from only one label.

**Training Rounds:** 250 rounds, 1 epoch per round

### 2. Long Tail Distribution (long_tail)

**Entry File:** `lora_sample_entry_skewed_longtail_noniid.py`

**Configuration Files:** All files containing `_long_tail_` in the filename
- Examples:
  - `rbla_mnist_long_tail_homogeneous_round60_epoch1.yaml`
  - `sp_fmnist_long_tail_heterogeneous_round60_epoch1.yaml`
  - `zp_kmnist_long_tail_homogeneous_round60_epoch1.yaml`

**Data Partitioner:** Uses `SkewedLongtailPartitioner` to create skewed long-tail non-IID data distribution where clients have varying amounts of data across different classes.

**Training Rounds:** 60 rounds, 1 epoch per round

### 3. Feature Shift Distribution (feature_shift)

**Entry File:** `lora_sample_entry_feature_shifted.py`

**Configuration Files:** All files containing `_feature_shift_` in the filename
- Examples:
  - `rbla_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
  - `sp_fmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
  - `zp_kmnist_feature_shift_homogeneous_round60_epoch1.yaml`

**Data Partitioner:** Uses `FeatureShiftedPartitioner` to create feature-shifted data distribution where each label's samples are evenly distributed across all clients.

**Training Rounds:** 60 rounds, 1 epoch per round

## Directory Structure

```
convergence_experiment/
├── lora_mnist/
│   ├── RBLA/
│   │   ├── Homogeneous Rank/
│   │   │   ├── rbla_mnist_one_label_homogeneous_round250_epoch1.yaml → lora_sample_entry.py
│   │   │   ├── rbla_mnist_long_tail_homogeneous_round60_epoch1.yaml → lora_sample_entry_skewed_longtail_noniid.py
│   │   │   └── rbla_mnist_feature_shift_homogeneous_round60_epoch1.yaml → lora_sample_entry_feature_shifted.py
│   │   └── Heterogeneous Rank/
│   │       ├── rbla_mnist_one_label_heterogeneous_round250_epoch1.yaml → lora_sample_entry.py
│   │       ├── rbla_mnist_long_tail_heterogeneous_round60_epoch1.yaml → lora_sample_entry_skewed_longtail_noniid.py
│   │       └── rbla_mnist_feature_shift_heterogeneous_round60_epoch1.yaml → lora_sample_entry_feature_shifted.py
│   ├── SP/ (same structure as RBLA)
│   └── ZP/ (same structure as RBLA)
├── lora_fmnist/ (same structure as lora_mnist)
├── lora_kmnist/ (same structure as lora_mnist)
└── lora_qmnist/ (same structure as lora_mnist)
```

## Quick Reference Table

| Data Distribution Pattern | Entry File | Partitioner | Rounds | Epoch |
|---------------------------|------------|-------------|--------|-------|
| `*_one_label_*` | `lora_sample_entry.py` | NoniidDataGenerator | 250 | 1 |
| `*_long_tail_*` | `lora_sample_entry_skewed_longtail_noniid.py` | SkewedLongtailPartitioner | 60 | 1 |
| `*_feature_shift_*` | `lora_sample_entry_feature_shifted.py` | FeatureShiftedPartitioner | 60 | 1 |

## Usage Example

To run an experiment with a specific configuration:

```python
# For one_label distribution
from lora_sample_entry import SampleAppEntry
entry = SampleAppEntry()
entry.load_config("convergence_experiment/lora_mnist/RBLA/Homogeneous Rank/rbla_mnist_one_label_homogeneous_round250_epoch1.yaml")
entry.run()

# For long_tail distribution
from lora_sample_entry_skewed_longtail_noniid import SampleAppEntry
entry = SampleAppEntry()
entry.load_config("convergence_experiment/lora_mnist/RBLA/Homogeneous Rank/rbla_mnist_long_tail_homogeneous_round60_epoch1.yaml")
entry.run()

# For feature_shift distribution
from lora_sample_entry_feature_shifted import SampleAppEntry
entry = SampleAppEntry()
entry.load_config("convergence_experiment/lora_mnist/RBLA/Homogeneous Rank/rbla_mnist_feature_shift_homogeneous_round60_epoch1.yaml")
entry.run()
```

## Configuration Parameters

All configuration files use the following common parameters:

### Model Configuration
- **Model:** `nn_model_mnist_mlp_lora` (Simple LoRA MLP for MNIST-like datasets)
- **Optimizer:** SGD
- **Loss Function:** Cross Entropy
- **Trainer:** Standard Trainer

### LoRA Configuration
- **Rank Distribution:**
  - Homogeneous: All clients use the same rank ratio (rank_homogeneous.yaml)
  - Heterogeneous: Clients use different rank ratios (rank_heterogeneous.yaml)

### Method Configuration
- **RBLA:** Rank-Based LoRA Aggregation
- **SP:** Subspace Projection
- **ZP:** Zero Padding

### Training Configuration
- **Client Selection:** Random
- **FL Nodes:** 10 clients
- **Batch Size:** 64 (defined in data distribution configs)

## Notes

1. **Entry File Location:** All entry files are located in `src/test/fl_lora_sample/`
   - `lora_sample_entry.py`
   - `lora_sample_entry_skewed_longtail_noniid.py`
   - `lora_sample_entry_feature_shifted.py`

2. **Data Distribution Files:** Referenced configurations are located in `src/test/fl_lora_sample/yamls/data_distribution/`
   - `{dataset}_one_label.yaml`
   - `{dataset}_long_tail.yaml`
   - `{dataset}_feature_shift.yaml`

3. **Rank Distribution Files:** Located in `src/test/fl_lora_sample/yamls/rank_distribution/`
   - `rank_homogeneous.yaml`: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
   - `rank_heterogeneous.yaml`: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

4. **Automatic Entry Selection:** If implementing automatic entry file selection, use the filename pattern to determine which entry file to load based on the data distribution pattern in the configuration filename.

## Total Configuration Files

- **Total:** 72 configuration files
- **Per Dataset:** 18 files (mnist, fmnist, kmnist, qmnist)
- **Per Method:** 24 files (RBLA, SP, ZP)
- **Per Rank Type:** 36 files (Homogeneous, Heterogeneous)
- **Per Data Distribution:** 24 files (one_label, long_tail, feature_shift)

---

**Last Updated:** 2025-11-06
