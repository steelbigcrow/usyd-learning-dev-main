# Data Distribution 推荐修改方案（单一事实源）

本文档描述将数据分布（Non‑IID）改造成“YAML + DataDistribution 统一事实源，分区实现统一消费”的推荐方案。目标是在不新增入口导入、不过度复制分布定义的前提下，让任意数据集与分布 YAML 能被自动解析并用于两套切分引擎（经典 noniid 与 skewed long‑tail）。

## 背景与现状

- 分布定义目前在 YAML 中维护（`data_distribution.use` + `custom_define`）。
- DataDistribution 在节点准备阶段自动解析 YAML 并注册/返回矩阵：
  - 解析：`usyd_learning/ml_data_process/data_distribution.py`
  - 调用：`usyd_learning/fed_node/fed_node_vars.py` 的 `prepare_data_distribution()`
- 经典 noniid 入口（`test/fl_lora_sample/lora_sample_entry.py`）直接把 `use` 名称传给生成器，若名称不在生成器内置表（如 `kmnist_lt`），会失败。
- skewed long‑tail 入口已直接消费 `server_var.data_distribution` 的“密集矩阵”，无需关心名称。

## 设计原则

1) 单一事实源：仅在 YAML + DataDistribution 维护“分布矩阵”。
2) 分区引擎解耦：
   - 经典 noniid 生成器通过显式传入矩阵（`distribution="custom" + data_volum_list=...`）工作。
   - skewed long‑tail 分区器继续从矩阵派生稀疏规格，不做修改。
3) 入口不显式导入“具体分布模块”，避免重复与漂移。

## 必须修改（1 处）

文件：`src/test/fl_lora_sample/lora_sample_entry.py`

- 将经典 noniid 路径从“传名称”改为“传矩阵”：

现状（示例）：

```
allocated_noniid_data = NoniidDataGenerator(train_loader.data_loader).generate_noniid_data(
    distribution=self.server_yaml["data_distribution"]["use"])
```

推荐修改：

```
dist_matrix = server_var.data_distribution
allocated_noniid_data = NoniidDataGenerator(train_loader.data_loader).generate_noniid_data(
    distribution="custom", data_volum_list=dist_matrix)
```

说明：`server_var.data_distribution` 来自 DataDistribution 的自动解析与注入，无需额外导入或配置字段。

## 可选增强（鲁棒性，0–1 处）

文件：`src/usyd_learning/fl_algorithms/noniid/noniid_data_generator.py`

- 在 `distribution_generator(...)` 末尾为“未知名称”增加兜底：当传入的 `distribution` 不在内置表时，调用 DataDistribution 获取矩阵；若也未配置，则再抛错。这样即便入口仍传 `use` 名称，也能通过 YAML 解析获得矩阵。

伪代码示例：

```
try:
    from ...ml_data_process import DataDistribution
    return DataDistribution.use(distribution, data_volum_list)
except Exception:
    raise ValueError("Invalid distribution type ...")
```

此增强可选，但能减少入口对“custom 模式”的强依赖，提高兼容性。

## 无需修改的部分

- skewed long‑tail 入口：`src/test/fl_lora_sample/lora_sample_entry_skewed_longtail_noniid.py`
  - 已从 `server_var.data_distribution` 读取密集矩阵并转换为稀疏规格后分区；保持不变。
- DataDistribution 解析链路：保持不变。
- `src/unittest_ml/unittest_data_distribution.py`：仍然有价值，用于验证标准名与自定义 YAML 的解析结果。

## 配置要求与用法

1) 在分布 YAML（例如 `src/test/fl_lora_sample/yamls/data_distribution/*.yaml`）中：
   - 在 `custom_define` 下定义你的分布矩阵（例如 `kmnist_lt`）。
   - 将 `use` 指向该名称（例如 `use: kmnist_lt`）。

2) 在场景组合 YAML（例如 `script_test-rbla.yaml`）中：
   - 确保将对应分布 YAML 列入 `yaml_folder_data_distribution_files`，并在 `yaml_combination.server_yaml`/`client_yaml` 中引用其别名。

完成上述后，入口将自动从 DataDistribution 获得矩阵，无需导入任何“分布模块”。

## 影响面与工作量

- 必改文件：1 个（经典 noniid 入口）。
- 可选增强：+1 个（noniid 生成器增加兜底）。
- 其余文件、测试与分区器保持不变。

## 验证建议

1) 单元脚本：运行 `src/unittest_ml/unittest_data_distribution.py`，确认分布解析输出符合预期。
2) 小场景实验：
   - 使用 `mnist_one_label` 等标准分布与一个自定义分布（如 `kmnist_lt`）各跑一次；
   - 观察客户端样本量与类别直方图是否符合矩阵设定。

## 常见问答

- 问：是否还需要在入口显式导入“kmnist_lt 模块”？
  - 答：不需要。DataDistribution 已将 YAML 解析为矩阵，入口直接消费矩阵即可。

- 问：两套分区引擎如何选择？
  - 答：保持现有两个入口文件；经典 noniid 入口改为“传矩阵”，skewed 入口已“传矩阵→稀疏规格”。如需通过配置切换，可另增可选的路由字段（非必须）。

- 问：`kmnist_lt` 现在是否可用？
  - 答：是。只要在分布 YAML 的 `custom_define` 下定义 `kmnist_lt` 并设置 `use: kmnist_lt`，入口将自动使用该矩阵进行切分。

