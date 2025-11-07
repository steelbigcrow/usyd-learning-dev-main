# Training Guide（LoRA 联邦训练说明）

本指南覆盖以下三部分：
- fl_lora_sample/convergence_experiment 下 4 个 LoRA 数据集子目录（lora_fmnist、lora_kmnist、lora_mnist、lora_qmnist）的文件夹结构与配置文件列表。
- 3 个训练脚本与三种数据分布的对应关系，并说明如何通过修改训练脚本中的 `g_app.load_app_config()` 来切换所用配置文件。
- 训练结果在 `src/test/.training_results` 中的组织规则与“结果文件重命名与归档”的操作方法。

---

## 一、配置目录与文件（convergence_experiment）

根目录：`src/test/fl_lora_sample/convergence_experiment`

LoRA 数据集子目录共有 4 个：
- `lora_fmnist`
- `lora_kmnist`
- `lora_mnist`
- `lora_qmnist`

每个数据集目录下统一结构：
- 三种聚合策略：`RBLA/`、`SP/`、`ZP/`
- 两种秩分配目录：`Heterogeneous Rank/`、`Homogeneous Rank/`
- 每个“策略/秩”目录下，提供三种数据分布的 YAML 配置：`feature_shift`、`long_tail`、`one_label`

下方按数据集列出各目录内的配置文件（文件名仅数据集名不同，其余命名模式一致）：

### 1）lora_fmnist
- RBLA
  - Heterogeneous Rank
    - `rbla_fmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `rbla_fmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `rbla_fmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `rbla_fmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `rbla_fmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `rbla_fmnist_one_label_homogeneous_round250_epoch1.yaml`
- SP
  - Heterogeneous Rank
    - `sp_fmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `sp_fmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `sp_fmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `sp_fmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `sp_fmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `sp_fmnist_one_label_homogeneous_round250_epoch1.yaml`
- ZP
  - Heterogeneous Rank
    - `zp_fmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `zp_fmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `zp_fmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `zp_fmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `zp_fmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `zp_fmnist_one_label_homogeneous_round250_epoch1.yaml`

### 2）lora_kmnist
- RBLA
  - Heterogeneous Rank
    - `rbla_kmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `rbla_kmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `rbla_kmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `rbla_kmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `rbla_kmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `rbla_kmnist_one_label_homogeneous_round250_epoch1.yaml`
- SP
  - Heterogeneous Rank
    - `sp_kmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `sp_kmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `sp_kmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `sp_kmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `sp_kmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `sp_kmnist_one_label_homogeneous_round250_epoch1.yaml`
- ZP
  - Heterogeneous Rank
    - `zp_kmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `zp_kmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `zp_kmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `zp_kmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `zp_kmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `zp_kmnist_one_label_homogeneous_round250_epoch1.yaml`

### 3）lora_mnist
- RBLA
  - Heterogeneous Rank
    - `rbla_mnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `rbla_mnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `rbla_mnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `rbla_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `rbla_mnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `rbla_mnist_one_label_homogeneous_round250_epoch1.yaml`
- SP
  - Heterogeneous Rank
    - `sp_mnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `sp_mnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `sp_mnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `sp_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `sp_mnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `sp_mnist_one_label_homogeneous_round250_epoch1.yaml`
- ZP
  - Heterogeneous Rank
    - `zp_mnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `zp_mnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `zp_mnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `zp_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `zp_mnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `zp_mnist_one_label_homogeneous_round250_epoch1.yaml`

### 4）lora_qmnist
- RBLA
  - Heterogeneous Rank
    - `rbla_qmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `rbla_qmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `rbla_qmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `rbla_qmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `rbla_qmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `rbla_qmnist_one_label_homogeneous_round250_epoch1.yaml`
- SP
  - Heterogeneous Rank
    - `sp_qmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `sp_qmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `sp_qmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `sp_qmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `sp_qmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `sp_qmnist_one_label_homogeneous_round250_epoch1.yaml`
- ZP
  - Heterogeneous Rank
    - `zp_qmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
    - `zp_qmnist_long_tail_heterogeneous_round60_epoch1.yaml`
    - `zp_qmnist_one_label_heterogeneous_round250_epoch1.yaml`
  - Homogeneous Rank
    - `zp_qmnist_feature_shift_homogeneous_round60_epoch1.yaml`
    - `zp_qmnist_long_tail_homogeneous_round60_epoch1.yaml`
    - `zp_qmnist_one_label_homogeneous_round250_epoch1.yaml`

命名规则总结：
- `<strategy>_<dataset>_<distribution>_<rank>_round<轮数>_epoch<epoch数>.yaml`
  - strategy ∈ {rbla, sp, zp}
  - dataset ∈ {fmnist, kmnist, mnist, qmnist}
  - distribution ∈ {feature_shift, long_tail, one_label}
  - rank ∈ {heterogeneous, homogeneous}

---

## 二、训练脚本与配置切换

三种数据分布分别对应 3 个训练脚本（均位于 `src/test`）：
- 特征偏移（feature_shift）：`src/test/lora_sample_main_feature_shifted.py`
- 长尾非 IID（skewed long-tail noniid）：`src/test/lora_sample_main_skewed_longtail_noniid.py`
- 单标签非 IID（one_label noniid）：`src/test/lora_sample_main.py`

切换配置文件的方式：在训练脚本内，找到 `g_app.load_app_config()` 所在行，将字符串路径改为想要使用的 YAML 配置文件的相对路径。

示例（每个脚本的默认指向与编辑位置）：
- `src/test/lora_sample_main_feature_shifted.py:26`
  - 例：`./fl_lora_sample/convergence_experiment/lora_mnist/ZP/Homogeneous Rank/zp_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
- `src/test/lora_sample_main_skewed_longtail_noniid.py:24`
  - 例：`./fl_lora_sample/convergence_experiment/lora_mnist/ZP/Homogeneous Rank/zp_mnist_long_tail_homogeneous_round60_epoch1.yaml`
- `src/test/lora_sample_main.py:24`
  - 例：`./fl_lora_sample/convergence_experiment/lora_mnist/ZP/Homogeneous Rank/zp_mnist_one_label_homogeneous_round250_epoch1.yaml`

注意事项：
- 路径请保持相对路径（以脚本所在的 `src/test` 为基准），不要使用绝对路径。
- 三个脚本的主体逻辑相同，只是数据划分器不同；手动修改 `g_app.load_app_config()` 即可切换所用配置文件与数据集/策略/秩设置。

执行训练（统一使用同一 Python 解释器）：

```
"C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe" src\test\lora_sample_main_feature_shifted.py
"C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe" src\test\lora_sample_main_skewed_longtail_noniid.py
"C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe" src\test\lora_sample_main.py
```

（上述三条命令分别对应三种数据分布。若要切换到不同数据集/策略/秩，请按前述方法修改脚本中的 YAML 路径后再运行。）

---

## 三、训练结果的存放与重命名规则

训练日志与结果默认写入：`src/test/.training_results/`

该目录下预先建立了与配置目录完全对应的子目录层级，用于归档对应的训练结果：
- `src/test/.training_results/lora_fmnist/RBLA/Heterogeneous Rank/`（以及其余数据集/策略/秩的对应路径）

训练完成后，会在 `src/test/.training_results/` 生成带时间戳与哈希的结果文件，示例：
- `train-20241107_120000-1a2b3c4d.csv`

为了与所用配置文件一一对应，请进行如下整理：
1. 将结果文件重命名为“与配置文件同名但保留 .csv 扩展名”的文件名。
   - 例如，若使用的配置是：
     - `./fl_lora_sample/convergence_experiment/lora_mnist/ZP/Homogeneous Rank/zp_mnist_one_label_homogeneous_round250_epoch1.yaml`
   - 则将结果文件重命名为：
     - `zp_mnist_one_label_homogeneous_round250_epoch1.csv`
2. 按配置文件所在的目录层级，将重命名后的结果文件移动到对应的子文件夹中：
   - 目标目录：`src/test/.training_results/lora_mnist/ZP/Homogeneous Rank/`

补充说明：
- 若目标子目录不存在，可自行按上述层级创建（通常仓库已预建）。
- 通过“结果文件名 == 配置文件名（仅扩展名不同）”的方式，能方便地建立配置与结果的明确映射关系。
- 监控输出（.monitor）会自动按同批次文件名建立子目录，无需手动干预。

---

## 四、训练过程一致性检查警示（RBLA ≈ ZP）

当“数据集 + 数据分布”相同且采用同构 Rank（Homogeneous Rank）时，RBLA 与 ZP 两种方法的训练结果在数值上应当非常接近（例如最终准确率、损失曲线）。若出现明显偏差，请优先排查：
- YAML 是否一致（除策略差异外，其余训练/模型超参相同）。
- 随机种子、设备与数据加载顺序是否稳定。
- 训练日志是否都已正确“重命名为配置同名”并放入对应子目录，避免误比对。

对比方式（二选一，或同时）：
- 手动观察：打开以下两处的 CSV 结果进行肉眼对比（指标列如 acc、loss 等）：
  - `src/test/.training_results/<dataset>/RBLA/Homogeneous Rank/<rbla_*.csv>`
  - `src/test/.training_results/<dataset>/ZP/Homogeneous Rank/<zp_*.csv>`
- 脚本对比：可在 `src/test/` 下创建独立脚本（例如 `compare_results.py`，可手动修改阈值/路径），读取两份 CSV，计算关键指标的差值（如最终准确率差值 < 1%）。

---

## 五、分层训练计划（24 阶段）

说明：本计划与配置文件的文件夹结构完全一致（4 个数据集 × 3 种策略 × 2 种 Rank = 24 个阶段）。每个阶段包含 3 种数据分布任务（feature_shift、long_tail、one_label）。完成后，请在本文件中将对应的复选框由 `[ ]` 改为 `[x]` 进行标记。

提醒：如果某个阶段的训练文件已经存在，那么就不需要重新跑这个阶段的训练了。

阶段 01：`lora_fmnist / RBLA / Heterogeneous Rank`
- [x] feature_shift -> `rbla_fmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [x] long_tail    -> `rbla_fmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [x] one_label    -> `rbla_fmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [x] 阶段完成

阶段 02：`lora_fmnist / RBLA / Homogeneous Rank`
- [x] feature_shift -> `rbla_fmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [x] long_tail    -> `rbla_fmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [x] one_label    -> `rbla_fmnist_one_label_homogeneous_round250_epoch1.yaml`
- [x] 阶段完成

阶段 03：`lora_fmnist / SP / Heterogeneous Rank`
- [ ] feature_shift -> `sp_fmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_fmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_fmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 04：`lora_fmnist / SP / Homogeneous Rank`
- [x] feature_shift -> `sp_fmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [x] long_tail    -> `sp_fmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_fmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 05：`lora_fmnist / ZP / Heterogeneous Rank`
- [x] feature_shift -> `zp_fmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [x] long_tail    -> `zp_fmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_fmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 06：`lora_fmnist / ZP / Homogeneous Rank`
- [x] feature_shift -> `zp_fmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [x] long_tail    -> `zp_fmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_fmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 07：`lora_kmnist / RBLA / Heterogeneous Rank`
- [ ] feature_shift -> `rbla_kmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_kmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `rbla_kmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 08：`lora_kmnist / RBLA / Homogeneous Rank`
- [ ] feature_shift -> `rbla_kmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_kmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `rbla_kmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 09：`lora_kmnist / SP / Heterogeneous Rank`
- [ ] feature_shift -> `sp_kmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_kmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_kmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 10：`lora_kmnist / SP / Homogeneous Rank`
- [ ] feature_shift -> `sp_kmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_kmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_kmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 11：`lora_kmnist / ZP / Heterogeneous Rank`
- [ ] feature_shift -> `zp_kmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_kmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_kmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 12：`lora_kmnist / ZP / Homogeneous Rank`
- [ ] feature_shift -> `zp_kmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_kmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_kmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 13：`lora_mnist / RBLA / Heterogeneous Rank`
- [ ] feature_shift -> `rbla_mnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_mnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `rbla_mnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 14：`lora_mnist / RBLA / Homogeneous Rank`
- [ ] feature_shift -> `rbla_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_mnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `rbla_mnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 15：`lora_mnist / SP / Heterogeneous Rank`
- [ ] feature_shift -> `sp_mnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_mnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_mnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 16：`lora_mnist / SP / Homogeneous Rank`
- [ ] feature_shift -> `sp_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_mnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_mnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 17：`lora_mnist / ZP / Heterogeneous Rank`
- [ ] feature_shift -> `zp_mnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_mnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_mnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 18：`lora_mnist / ZP / Homogeneous Rank`
- [ ] feature_shift -> `zp_mnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_mnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_mnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 19：`lora_qmnist / RBLA / Heterogeneous Rank`
- [ ] feature_shift -> `rbla_qmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_qmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `rbla_qmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 20：`lora_qmnist / RBLA / Homogeneous Rank`
- [ ] feature_shift -> `rbla_qmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_qmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `rbla_qmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 21：`lora_qmnist / SP / Heterogeneous Rank`
- [ ] feature_shift -> `sp_qmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_qmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_qmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 22：`lora_qmnist / SP / Homogeneous Rank`
- [ ] feature_shift -> `sp_qmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_qmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `sp_qmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 23：`lora_qmnist / ZP / Heterogeneous Rank`
- [ ] feature_shift -> `zp_qmnist_feature_shift_heterogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_qmnist_long_tail_heterogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_qmnist_one_label_heterogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 24：`lora_qmnist / ZP / Homogeneous Rank`
- [ ] feature_shift -> `zp_qmnist_feature_shift_homogeneous_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_qmnist_long_tail_homogeneous_round60_epoch1.yaml`
- [ ] one_label    -> `zp_qmnist_one_label_homogeneous_round250_epoch1.yaml`
- [ ] 阶段完成

## 常见问题（FAQ）
- 配置路径是否必须相对？是。请保持与仓库结构一致，避免绝对路径。
- 三个脚本能否互换配置？可以。核心差异在于数据划分方式；其余训练流程一致。
