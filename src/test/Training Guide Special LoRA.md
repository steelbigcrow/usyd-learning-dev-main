# Training Guide Special LoRA（Special LoRA 联邦训练说明）

本说明聚焦 `special_lora_dataset` 的联邦训练，覆盖以下内容：
- `src/test/fl_lora_sample/convergence_experiment` 下 `special_lora_*` 目录的结构与 36 个 YAML 配置文件说明。
- Special LoRA 配置的关键差异（统一使用 `nn_simple_special_lora_mlp`、默认同构 rank、数据分布轮次设置等）。
- 三个训练脚本如何加载 Special LoRA 配置，以及运行命令示例。
- 训练结果在 `src/test/.training_results/special_lora_*` 中的整理方法。
- 12 个阶段（4 数据集 × 3 聚合策略）的训练计划与勾选表。

---

## 一、配置目录与文件（special_lora_dataset）

根目录：`src/test/fl_lora_sample/convergence_experiment`

Special LoRA 数据集子目录：
- `special_lora_fmnist`
- `special_lora_kmnist`
- `special_lora_mnist`
- `special_lora_qmnist`

与标准 LoRA 数据集不同，Special LoRA 目录下**只有聚合策略三级目录**（`RBLA/`、`SP/`、`ZP/`），不再区分 “Heterogeneous Rank / Homogeneous Rank”。因为 Special LoRA 默认采用同构 rank，目录结构更加扁平。每个策略目录中总是包含三种数据分布配置：
- `feature_shift` → 1 epoch + 60 round
- `long_tail` → 1 epoch + 60 round
- `noniid`（对应 one-label 非 IID）→ 1 epoch + 250 round

命名规则统一为：
```
<strategy>_<dataset>_<distribution>_round<轮数>_epoch1.yaml
```
其中 `<distribution>` ∈ {`feature_shift`, `long_tail`, `noniid`}。

下列列表覆盖全部 36 个配置文件：

### 1）special_lora_fmnist
- RBLA
  - `rbla_fmnist_feature_shift_round60_epoch1.yaml`
  - `rbla_fmnist_long_tail_round60_epoch1.yaml`
  - `rbla_fmnist_noniid_round250_epoch1.yaml`
- SP
  - `sp_fmnist_feature_shift_round60_epoch1.yaml`
  - `sp_fmnist_long_tail_round60_epoch1.yaml`
  - `sp_fmnist_noniid_round250_epoch1.yaml`
- ZP
  - `zp_fmnist_feature_shift_round60_epoch1.yaml`
  - `zp_fmnist_long_tail_round60_epoch1.yaml`
  - `zp_fmnist_noniid_round250_epoch1.yaml`

### 2）special_lora_kmnist
- RBLA
  - `rbla_kmnist_feature_shift_round60_epoch1.yaml`
  - `rbla_kmnist_long_tail_round60_epoch1.yaml`
  - `rbla_kmnist_noniid_round250_epoch1.yaml`
- SP
  - `sp_kmnist_feature_shift_round60_epoch1.yaml`
  - `sp_kmnist_long_tail_round60_epoch1.yaml`
  - `sp_kmnist_noniid_round250_epoch1.yaml`
- ZP
  - `zp_kmnist_feature_shift_round60_epoch1.yaml`
  - `zp_kmnist_long_tail_round60_epoch1.yaml`
  - `zp_kmnist_noniid_round250_epoch1.yaml`

### 3）special_lora_mnist
- RBLA
  - `rbla_mnist_feature_shift_round60_epoch1.yaml`
  - `rbla_mnist_long_tail_round60_epoch1.yaml`
  - `rbla_mnist_noniid_round250_epoch1.yaml`
- SP
  - `sp_mnist_feature_shift_round60_epoch1.yaml`
  - `sp_mnist_long_tail_round60_epoch1.yaml`
  - `sp_mnist_noniid_round250_epoch1.yaml`
- ZP
  - `zp_mnist_feature_shift_round60_epoch1.yaml`
  - `zp_mnist_long_tail_round60_epoch1.yaml`
  - `zp_mnist_noniid_round250_epoch1.yaml`

### 4）special_lora_qmnist
- RBLA
  - `rbla_qmnist_feature_shift_round60_epoch1.yaml`
  - `rbla_qmnist_long_tail_round60_epoch1.yaml`
  - `rbla_qmnist_noniid_round250_epoch1.yaml`
- SP
  - `sp_qmnist_feature_shift_round60_epoch1.yaml`
  - `sp_qmnist_long_tail_round60_epoch1.yaml`
  - `sp_qmnist_noniid_round250_epoch1.yaml`
- ZP
  - `zp_qmnist_feature_shift_round60_epoch1.yaml`
  - `zp_qmnist_long_tail_round60_epoch1.yaml`
  - `zp_qmnist_noniid_round250_epoch1.yaml`

---

## 二、Special LoRA 配置的关键特点

1. **统一模型**  
   全部 YAML 在 `yaml_folder_nn_model_files` 中绑定 `nn_model_special_lora_mlp.yaml: nn_simple_special_lora_mlp`，对应实现 `NNModel_SimpleSpecialLoRAMLP`（`src/usyd_learning/ml_models/lora/_nn_model_simple_special_lora_mlp.py`）。无需再切换到 `nn_model_mnist_mlp_lora`。

2. **Rank 配置**  
   Special LoRA 固定使用 `rank_homogeneous`，因此文件名与目录都不出现 “heterogeneous/homogeneous” 字样。若需要测试异构 rank，需要回到标准 LoRA 数据集。

3. **轮次与数据分布**  
   - `feature_shift`、`long_tail`：`general_round60` + `training_epoch1`
   - `noniid`：`general_round250` + `training_epoch1`（底层数据分布仍然引用 `*_one_label.yaml`）

4. **路径写法**  
   YAML 中的 `yaml_folder_*_path` 统一使用相对路径 `../../../yamls/...`。如需新增配置，请保持相同层级，避免写绝对路径。

---

## 三、训练脚本与配置切换

三个脚本仍位于 `src/test`，但默认都指向 Special LoRA（以 `special_lora_fmnist/RBLA` 为例）。如需切换至其它数据集/策略/分布，只需修改 `g_app.load_app_config()` 的 YAML 路径。

| 数据分布 | 脚本 | 默认配置（行号） |
| --- | --- | --- |
| feature_shift | `src/test/lora_sample_main_feature_shifted.py` | 第 26 行：`./fl_lora_sample/convergence_experiment/special_lora_fmnist/RBLA/rbla_fmnist_feature_shift_round60_epoch1.yaml` |
| long_tail | `src/test/lora_sample_main_skewed_longtail_noniid.py` | 第 24 行：`./fl_lora_sample/convergence_experiment/special_lora_fmnist/RBLA/rbla_fmnist_long_tail_round60_epoch1.yaml` |
| noniid（one-label） | `src/test/lora_sample_main.py` | 第 24 行：`./fl_lora_sample/convergence_experiment/special_lora_fmnist/RBLA/rbla_fmnist_noniid_round250_epoch1.yaml` |

修改步骤：
1. 打开对应脚本，定位 `g_app.load_app_config()`。
2. 将字符串替换为目标 YAML 的相对路径（相对 `src/test`）。示例：`"./fl_lora_sample/convergence_experiment/special_lora_mnist/ZP/zp_mnist_long_tail_round60_epoch1.yaml"`。
3. 保存后运行脚本即可触发指定配置。

运行命令（统一解释器）：
```
"C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe" src\test\lora_sample_main_feature_shifted.py
"C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe" src\test\lora_sample_main_skewed_longtail_noniid.py
"C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe" src\test\lora_sample_main.py
```
注意：
- 仍然必须使用相对路径来引用 YAML。
- 三个脚本逻辑一致，只是导入的分布划分器不同；因此切换数据集或策略只需替换 YAML。

---

## 四、训练结果的存放与整理

Special LoRA 的结果目录位于：`src/test/.training_results/special_lora_*`
- 示例：`src/test/.training_results/special_lora_mnist/RBLA/`
- 目录层级与配置目录一致（数据集/策略）。

整理步骤：
1. 训练完成后，会在 `.training_results` 下生成诸如 `train-20241107_120000-xxxx.csv` 的文件。
2. 将文件重命名为与配置文件同名（仅扩展名改为 `.csv`）。  
   例：`rbla_mnist_long_tail_round60_epoch1.yaml` → `rbla_mnist_long_tail_round60_epoch1.csv`
3. 按目录层级移动到对应子目录：`src/test/.training_results/special_lora_mnist/RBLA/`

Tips：
- 若目标子目录不存在，可参照配置目录手动创建。
- 同批次产生的 `.monitor` 目录会自动创建，无需额外处理。
- 为便于比对，可使用 `compare_results.py` 或自定义脚本读取两个 CSV（例如 RBLA vs ZP）并比较关键指标。

---

## 五、Special LoRA 分阶段训练计划（12 阶段）

每个阶段 = 1 个数据集 × 1 个聚合策略，包含三种数据分布任务。执行完毕后可将 `[ ]` 改为 `[x]` 以记录进度；若该阶段已有结果文件，视为完成，无需重复训练。

阶段 01：`special_lora_fmnist / RBLA`
- [ ] feature_shift -> `rbla_fmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_fmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `rbla_fmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 02：`special_lora_fmnist / SP`
- [ ] feature_shift -> `sp_fmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_fmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `sp_fmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 03：`special_lora_fmnist / ZP`
- [ ] feature_shift -> `zp_fmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_fmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `zp_fmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 04：`special_lora_kmnist / RBLA`
- [ ] feature_shift -> `rbla_kmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_kmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `rbla_kmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 05：`special_lora_kmnist / SP`
- [ ] feature_shift -> `sp_kmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_kmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `sp_kmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 06：`special_lora_kmnist / ZP`
- [ ] feature_shift -> `zp_kmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_kmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `zp_kmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 07：`special_lora_mnist / RBLA`
- [ ] feature_shift -> `rbla_mnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_mnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `rbla_mnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 08：`special_lora_mnist / SP`
- [ ] feature_shift -> `sp_mnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_mnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `sp_mnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 09：`special_lora_mnist / ZP`
- [ ] feature_shift -> `zp_mnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_mnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `zp_mnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 10：`special_lora_qmnist / RBLA`
- [ ] feature_shift -> `rbla_qmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `rbla_qmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `rbla_qmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 11：`special_lora_qmnist / SP`
- [ ] feature_shift -> `sp_qmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `sp_qmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `sp_qmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

阶段 12：`special_lora_qmnist / ZP`
- [ ] feature_shift -> `zp_qmnist_feature_shift_round60_epoch1.yaml`
- [ ] long_tail    -> `zp_qmnist_long_tail_round60_epoch1.yaml`
- [ ] noniid       -> `zp_qmnist_noniid_round250_epoch1.yaml`
- [ ] 阶段完成

---

## 常见问题（FAQ）
- **是否可以混用 standard LoRA 与 Special LoRA 配置？**  
  可以，但请确保脚本加载的 YAML 与结果归档路径一致，避免混淆。
- **noniid 配置为什么叫 “one-label”？**  
  YAML 内部仍引用 `*_one_label.yaml` 数据分布文件，文件名沿用 `noniid` 以凸显实验含义；轮次也固定为 250 round。
- **如何快速对比不同策略？**  
  在 `.training_results/special_lora_<dataset>/<strategy>/` 中收集同一分布的 CSV，然后使用 `compare_results.py` 或 Pandas Notebook 计算最终指标差异即可。

