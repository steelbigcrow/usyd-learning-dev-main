# Unit Test Implementation（规划）

本文档汇总 **AdaLoRA、Zero-Pad（ZP）聚合、SVD 聚合、feature shifted 数据分布、skewed long-tail noniid 数据分布** 五个模块在仓库中的核心源码，并规划如何在 `src/unit_test` 中 **整合现有用例、决定保留/修改/移除的文件，并补充缺失的测试**。编写或运行测试前，请在仓库根目录导入 `startup_init.py`（或直接运行现有 `unittest_*.py` 模块），并使用仓库要求的 Python 解释器：  
`C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe`（在 WSL/Bash 中可写作 `/mnt/c/Users/22270/Desktop/study/Sydney\ University/courses/Code\ Space/Capstone/.venv/Scripts/python.exe`）。

---

## 一、相关源码定位

| 模块 | 关键文件 | 作用摘要 |
| --- | --- | --- |
| AdaLoRA | `src/usyd_learning/ml_algorithms/adalora/peft_adalora.py` | LoRA 目标层自动识别与 `peft` 包装。 |
|  | `src/usyd_learning/ml_algorithms/adalora/adalora_rbla_bridge.py` | PEFT ↔ plain-LoRA 转换、rank mask/rr 统一。 |
|  | `src/usyd_learning/model_trainer/trainer/_model_trainer_adalora.py` | AdaLoRA 专用 Trainer，封装设备与模型管理。 |
|  | `src/usyd_learning/ml_models/adalora/_nn_model_adalora_mlp.py`、`src/usyd_learning/ml_models/lora/_nn_model_simple_fixed_adalora_mlp.py` | 示例 MLP 模型（含可插拔 LoRA 层）。 |
|  | `src/usyd_learning/monitoring/instrumentation/adalora.py` | 训练 Rank 监控、广播 pad/slice 监控补丁。 |
| Zero-Pad 聚合 | `src/usyd_learning/fl_algorithms/aggregation/methods/_fed_aggregator_zeropad.py` | ZP 聚合核心（零填充+加权平均）。 |
|  | `src/usyd_learning/fed_strategy/server_strategy_impl/_zp_server.py` | 服务器端策略：PEFT→plain 转换、聚合、回写/广播。 |
|  | `src/usyd_learning/fed_strategy/client_strategy_impl/_zp_client.py`、`src/usyd_learning/fed_strategy/runner_strategy_impl/_zp_runner_strategy.py` | 客户端训练及 Runner 封装流程。 |
| SVD 聚合 | `src/usyd_learning/fl_algorithms/aggregation/methods/_fed_aggregator_svd.py` | SVD 拆分算法、rank mask 聚合。 |
|  | `src/usyd_learning/fed_strategy/server_strategy_impl/_svd_server.py`、`src/usyd_learning/fed_strategy/client_strategy_impl/_svd_client.py`、`src/usyd_learning/fed_strategy/runner_strategy_impl/_svd_runner_strategy.py` | 与 ZP 类似的 server/client 流程。 |
| Feature Shifted 分布 | `src/usyd_learning/fl_algorithms/feature_shifted/feature_shifted_partitioner.py` | 基于标签计数的（均衡/显式）划分。 |
|  | `src/test/fl_lora_sample/lora_sample_entry_feature_shifted.py`、`src/test/lora_sample_main_feature_shifted.py` | 运行入口，演示如何把 partitioner 输出绑定到联邦节点。 |
| Skewed long-tail noniid 分布 | `src/usyd_learning/fl_algorithms/skewed_longtail_noniid/skewed_longtail_partitioner.py` | 非 IID 长尾划分，严格按 YAML 矩阵分配。 |
|  | `src/test/fl_lora_sample/lora_sample_entry_skewed_longtail_noniid.py`、`src/test/lora_sample_main_skewed_longtail_noniid.py` | 长尾场景入口，贯穿 FedRunner 与数据分配。 |

> 现有 `src/unit_test` 中已包含部分相关单测（例如 `unittest_adalora_zeropad_bridge.py`、`unittest_aggregator_zeropad.py`、`unittest_aggregator_svd.py`）。下文在规划新增用例的同时，也会说明哪些旧文件需要合并、重命名或下线，以保持目录结构一致、避免重复覆盖。

---

## 二、单元测试规划（位于 `src/unit_test`）

### 1. AdaLoRA

**现有用例整合**

- `unittest_adalora_zeropad_bridge.py` / `unittest_rbla_adalora_bridge.py`：目前聚焦 LoRA bridge 与 RBLA 的兼容性，可合并为单一的 `unittest_adalora_bridge.py`，按照功能拆成 **PEFT ↔ plain、rank mask 合并** 两大测试类，避免重复 fixture。
- `unittest_adalora_rankhint.py`：覆盖 `_choose_rr` 的部分逻辑；计划在扩展后的 bridge 新用例中直接吸收对应断言，随后移除此文件。
- `unittest_adalora_rbla_multilayer_broadcast.py`：若只验证 `LoRAUtils.broadcast_lora_state_dict` 的通用逻辑，可迁移到 `unittest_lora_broadcast_utils.py`，使 AdaLoRA 专项测试聚焦桥接、trainer、instrumentation。

**新增文件建议**

1. `unittest_adalora_peft_wrap.py`
   - 构造最小 MLP（使用 `torch.nn.Linear`），通过 `unittest.mock` 注入假的 `peft.AdaLoraConfig` / `get_peft_model`，验证：
     - `_auto_linear_module_names` 能按拓扑顺序发现叶节点；
     - `wrap_with_adalora` 在 `target_modules=None` 时自动选择所有线性层；
     - `extra_kwargs` 会被过滤掉 `AdaLoraConfig` 不支持的参数；
     - 未找到目标层时抛出 `ValueError`。
   - 可使用 `tests.utils.simple_linear()`（若不存在则在文件内定义）构造模型，断言 `get_peft_model` 被调用时的参数。

2. `unittest_adalora_peft_bridge_extended.py`
   - 针对 `peft_to_plain_lora_shrunk`/`plain_lora_to_peft`/`select_template_with_max_rank` 做更细粒度校验：
     - 当 `lora_E` 缺失时 fallback 到 `_choose_rr` 计算；
     - rank mask 与 `rank_rr` 的写入；多客户端（不同 rank）情况下 `select_template_with_max_rank` 选择含最高 `lora_A` rank 的模板；
     - `plain_lora_to_peft` 统一 mask 并更新 `ranknum` 的逻辑。
   - 模拟输入可复用 `torch.randn` 生成的张量，确保 dtype/device 覆盖 CPU & CUDA 分支（若 CI 无 GPU，则用 `torch.device("cpu")` 并注释）。

3. `unittest_adalora_instrumentation.py`
   - 利用 `monkeypatch`（或 `unittest.mock`）在不真实调用训练循环的情况下验证 `patch_train_step_rank_snapshot` 和 `patch_broadcast_pad_slice_counters`：
     - 构造假的 `ModelTrainer_Standard`、`LoRAUtils`，触发打补丁后的一次 `train_step`；
     - 确认当 `hub.config.monitoring.adalora.enabled` 为 `"false"` 时不会写 rank；
     - 当 `peft_to_plain_lora_shrunk` 返回 `rank_rr` 时，写入的 `r_eff` 与 mask 一致；
     - `broadcast_lora_state_dict` 被包装后能对 pad/slice 场景累计计数（验证 hub stub 的计数器字段）。

**依赖/准备**

- 继承现有 `startup_init.py` 以解决相对导入。
- 数据样例可直接使用 `torch.ones`；如需配置文件，可参考 `test_data/node_config_template_*.yaml` 构造最小 `node_var`.

### 2. Zero-Pad 聚合（ZP）

**现有用例整合**

- `unittest_aggregator_zeropad.py` 与 `unittest_aggregator_zeropad_adalora.py` 存在交叉：前者验证纯 LoRA 张量，后者测试 AdaLoRA rank mask。建议合并为 `unittest_aggregator_zp.py`，以参数化方式覆盖普通与 AdaLoRA 输入，并清理被合并的旧文件名。
- `unittest_adalora_zeropad_bridge.py` 中与 Zero-Pad 强相关的桥接测试（PEFT→plain→PEFT）迁移到新的 `unittest_zp_aggregation_pipeline.py`，避免散落在 AdaLoRA 专题文件中。

**新增文件建议**

1. `unittest_zp_aggregation_pipeline.py`
   - 以 `FakeServerNode`（含 `node_var.aggregation_method = FedAggregator_ZeroPad(...)`）模拟 `ZpServerStrategy.aggregation`：
     - 准备两个客户端的 PEFT 状态字典（可借助 AdaLoRA bridge 生成 plain LoRA），验证：
       - 客户端更新被 `peft_to_plain_lora_shrunk` 转成 plain；
       - 调用 aggregator 后 `plain_lora_to_peft` 回写；
       - `node_var.aggregated_weight`、`node_var.model_weight` 均引用聚合结果。
   - 增加一个场景：一个客户端完全缺失某层 LoRA A/B，确保 `_find_peft_lora_prefixes` 跳过不完整 prefix。

2. `unittest_aggregator_zeropad_shape_guard.py`
   - 在现有 `unittest_aggregator_zeropad.py` 基础上扩展覆盖：
     - `pad_tensors_to_max_shape` 针对 3D 权重（feature-shifted卷积层）；
     - `aggregate_state_dicts` 对非 LoRA 张量 shape 不一致时自动 padding 的逻辑；
     - 权重归一化为 1 的断言（浮点误差允许 `1e-6`）。

3. （可选）`unittest_zp_server_apply_weight.py`
   - 聚焦 `ZpServerStrategy.apply_weight`：伪造 `ModelEvaluator` 和 `LoRAUtils`，验证 AdaLoRA alpha 补偿是否调用、`broadcast_lora_state_dict` 接收的 `global_sd` 包含 aggregator 输出的全部键。

### 3. SVD 聚合

**现有用例整合**

- `unittest_aggregator_svd.py` 已存在但覆盖面有限（主要是 SVD 分解正确性），需要扩展 rank mask 与 PEFT 输入场景并更名为 `unittest_aggregator_svd_full.py` 以示范围；旧文件在扩展完成后删除。
- `src/integration_test/adalora_svd` 目录含集成测试，若某些断言与单测重复，可把关键逻辑（例如 `select_template_with_max_rank`）下沉到单测，集成测试仅保留端到端覆盖。

**新增文件建议**

1. `unittest_aggregator_svd_rankmask.py`
   - 针对 `FedAggregator_SVD` 中的 mask 聚合逻辑：构造三个客户端，每个客户端的 `rank_mask` 长度不同，验证输出中：
     - `rank_mask` 被右侧补零到 `r_max`，并按权重求平均；
     - 当输入为 PEFT 结构（没有 plain LoRA 键）时，`peft_to_plain_lora_shrunk` 被触发。

2. `unittest_svd_server_pipeline.py`
   - 与 ZP 测试类似，但需额外断言：
     - `_choose_rr` 产生的 `r_max` 与 `select_template_with_max_rank` 返回的模板保持一致；
     - `LoRAUtils.svd_split` 结果被映射回 PEFT，并在 `apply_weight` 中重新广播；
     - 当 `trainer_type="adalora"` 时，会调用 `LoRAUtils.compensate_for_adalora_scaling`。

3. `unittest_svd_client_broadcast.py`
   - 覆盖 `SvdClientTrainingStrategy._broadcast_lora_state_dict` 的 slice/pad 分支：分别测试
     - `r_global > r_local`（slice）；
     - `r_global < r_local`（pad）；
     - 非 LoRA 参数直接替换。

### 4. Feature Shifted 分布

**现有用例整合**

- 目前 `src/unit_test` 尚无 feature shifted 的单测；但 `src/test/fl_lora_sample/lora_sample_entry_feature_shifted.py` 拥有运行脚本。规划新增文件时应避免与 `src/unittest_ml` 的数据划分测试冲突。
- 若在 `src/unittest_ml` 已有类似 partitioner 校验，可将通用 helper（如生成 TensorDataset 的逻辑）移动到公共测试工具模块，避免重复代码。

**新增文件建议**

1. `unittest_feature_shifted_partitioner.py`
   - 重点验证 `FeatureShiftedPartitioner.make_balanced_counts` 与 `partition_from_counts`：
     - 模拟两类标签、总数无法整除 `num_clients` 的情况，确保余数被分配给最后几个客户端；
     - `partition_from_counts` 对非法矩阵（维度不符、负数、超出可用样本）抛出 `ValueError`；
     - 在 `return_loaders=True` 时，返回 `DataLoader` 列表且 `len(loader.dataset)` 与输入计数一致。
   - 数据可通过 `torchvision.datasets.FakeData` 或手动构造的 `TensorDataset` 来生成。

2. `unittest_feature_shifted_entry_binding.py`
   - 构造最小化的 `SampleAppEntry`（来自 `lora_sample_entry_feature_shifted.py`），通过 `unittest.mock`:
     - 注入自定义 YAML（可参考 `test_data` 模板）以及假的 `DatasetLoaderFactory`，断言 `partitioner.partition_from_counts` 与 `partition_evenly` 的选择逻辑；
     - 验证当 `data_distribution` 缺失时会 fallback 到 `partition_evenly`，并根据客户端数量生成 loader。

### 5. Skewed long-tail noniid 分布

**现有用例整合**

- `src/unittest_ml/unittest_skewed_longtail_partitioner.py`、`unittest_skewed_longtail_spec.py` 已覆盖基础行为。计划在 `src/unit_test` 新增的 entry 级测试中只验证 **联邦入口绑定**，并复用 `unittest_ml` 中的 fixture；若新测试实现后发现旧 spec 文件不再需要，可考虑在 `unittest_ml` 目录下标记废弃或删除。
- 如果未来将长尾 partitioner 的所有测试迁移到 `src/unit_test`（保持同一目录），需同步更新 `README`/文档以告知迁移原因，确保 CI 路径不受影响。

**新增文件建议**

1. `unittest_skewed_longtail_partitioner_ext.py`
   - 在 `src/unit_test` 中增加对 `SkewedLongtailPartitioner.partition_from_counts` 的额外场景测试（区别于 `src/unittest_ml/unittest_skewed_longtail_partitioner.py`）：
     - 检查 `_available_counts` 与 `partition_from_counts` 的上界校验（`counts` 求和超过可用样本时抛异常）；
     - `partition` 返回空数据集（比如某客户端全 0）时仍维持索引对齐。

2. `unittest_skewed_longtail_entry.py`
   - 覆盖 `lora_sample_entry_skewed_longtail_noniid.SampleAppEntry`：
     - 使用 `FakeFedRunner` / `FakeFedNodeVars`，注入固定的 `data_distribution` 矩阵，验证每个客户端得到的 `data_sample_num` 与矩阵行和一致；
     - 确认在 `run()` 中的随机数种子设置（`torch.manual_seed(42)`）可被 patch，从而使 partition 结果可预测。

---

## 三、共用基建与数据

- **导入路径**：所有测试文件顶部执行 `from startup_init import *`（或 `import startup_init  # noqa: F401`）以确保 `src` 被加入 `sys.path`。
- **假数据构造**：建议在每个测试文件内提供 `make_tensor_dataset(num_samples, num_classes)` 之类的 helper，避免依赖真实数据集；若需 YAML，可复制 `test_data/data_distribution_template.yaml` 到 `tempfile` 并按需修改。
- **Mock 工具**：优先使用 `unittest.mock`；当需要 patch 第三方（如 `peft`）时，可在测试内注入 `sys.modules["peft"] = FakeModule`.
- **随机性**：统一调用 `torch.manual_seed(1234)`，与业务代码中的 `42` 区分开，方便断言。

---

## 四、执行与集成

1. 单文件调试：  
   ```
   C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe ^
       -m unittest src.unit_test.unittest_adalora_peft_wrap
   ```
2. 批量冒烟：在仓库根目录运行  
   ```
   C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe ^
       -m unittest discover -s src/unit_test -p "unittest_*.py"
   ```
3. 若新增测试依赖伪造的 `peft` 模块或大尺寸张量，请在文件头说明如何跳过（如使用 `unittest.skipUnless`）以便 CI 稳定。

---

通过上述布局，`src/unit_test` 中即可系统性覆盖 AdaLoRA 核心组件、ZP/SVD 聚合链路，以及 feature shifted 与 skewed long-tail 数据分布的关键行为，确保未来改动能被及时捕捉。
