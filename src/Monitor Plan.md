# 监控模块实施方案（AdaLoRA + 5 种聚合）

本文档给出一个与现有训练流程解耦的监控模块方案，覆盖 AdaLoRA 与全部五种聚合方法（fedavg、svd、rbla、sp、zeropad），通过单行导入启用，尽量不修改现有训练/策略/聚合代码。

## 目标

- 高解耦：以运行时补丁和事件订阅接入，不重写核心逻辑。
- 全覆盖：采集 epoch/batch 训练指标、联邦回合编排、聚合/广播细节、AdaLoRA 动态秩、基础资源信息。
- 可插拔：采集器与输出后端（Sink）解耦；默认轻量，可按需开启高级指标。
- 不依赖环境变量：仅通过“单行导入”启用，配置通过 YAML 文件完成，避免污染运行环境。
- 安全低负载：守护与采样机制；可选依赖缺失时自动降级。

## 范围

- 训练循环（客户端本地训练/观察）：loss、lr、吞吐、epoch/batch 时延。
- 联邦编排：回合、选客、服务端阶段、载荷大小、阶段耗时。
- AdaLoRA：逐层 LoRA 秩动态、mask/alpha 等特征、收缩后的有效秩。
- 聚合器差异化指标：fedavg、svd、rbla、sp、zeropad。
- 输出后端：CSV/Console（默认），可选 TensorBoard。

## 包结构

```
usyd_learning/monitoring/
  __init__.py
  auto_enable.py          # 单行导入即启用
  config.py               # 仅 YAML 配置加载
  hub.py                  # MonitorHub：事件分发至各 Sink
  types.py                # 事件与指标的数据结构
  sinks/
    base.py
    csv_sink.py
    console_sink.py
    tensorboard_sink.py   # 可选
  instrumentation/
    torch_trainer.py      # 训练循环（epoch/batch）
    federated.py          # 联邦回合阶段（server/client）
    adalora.py            # 秩/alpha/mask 快照
    aggregators.py        # 聚合器通用与按方法的插桩
```

## 单行启用

- 在入口脚本（如 `src/test/lora_sample_main.py`）顶部加入：

  ```python
  import usyd_learning.monitoring.auto_enable
  ```

- 不使用任何环境变量开关；所有可选项通过 YAML 配置（见“配置”）控制。

## 插桩覆盖面

### 通用（适用于所有训练器/策略/聚合器）

- 训练循环包裹
  - 包裹 `ModelTrainer.train` 与具体 `train_step`。
  - 每个 epoch 发出 `epoch_start/end`（epoch、avg_loss、lr、batch 数、耗时）。
  - 采样式 batch 事件：`loss`、`lr`、`batch_time`、`throughput`、`device`。

- 联邦编排
  - 包裹服务端/客户端公开方法：
    - Server：`receive_client_updates`、`aggregation`、`apply_weight`、`broadcast`、`evaluate`、`prepare`。
    - Client：`run_local_training`、`set_local_weight`。
  - 回合事件：`round_start/end`（回合序号、入选客户端 ID、总样本数、阶段耗时、载荷大小、键数与总元素数）。

- 资源（可选，若缺依赖自动禁用）：
  - 通过 `psutil`/`pynvml` 在 `epoch_end/round_end` 记录 CPU/GPU 内存。

### AdaLoRA（训练侧 + 服务端侧）

- 训练侧（客户端）：
  - 在 `epoch_end` 遍历 LoRA 参数，记录逐层秩 `r`（`.lora_A` 行或 `.lora_B` 列）、mask 1 的数量（如可得）、`lora_alpha`（如可读）、相对上次快照的 `Δr`。
  - 同步记录 loss/lr/吞吐的分布统计。

- 服务端侧：
  - 从客户端更新中，推导聚合前逐层 `r` 分布（min/mean/max）。
  - 广播过程中，在客户端 `set_local_weight` 记录从全局到本地秩的 Slice/Pad 次数（必要时包含 PEFT→plain 的键映射开销）。

### 聚合器专属指标与钩子

- FedAvg
  - 输入：参与客户端、按样本数归一的权重、总样本数。
  - 逐层速览：首参数均值（保留现有调试输出），可选抽样层的权重差异 L2 范数。
  - 耗时：`aggregate()` 总时长；输入/输出张量数量与字节数。
  - 钩子：`AbstractFedAggregator.aggregate`、`_fed_aggregator_fedavg._do_aggregation`。

- SVD
  - 若输入为 PEFT/AdaLoRA，则先转 plain LoRA（记录转换次数/耗时）。
  - 逐层：推断维度 (m,n)、`r_max`、SVD 时长；可选能量保留率（抽样 ≤N 层以控开销）。
  - 聚合后：全局 A/B `r_max` 直方图；PEFT 回投时长；服务端 `apply_weight` 的评估器适配耗时。
  - 钩子：`_fed_aggregator_svd._do_aggregation`、`LoRAUtils.svd_split`、服务端 `apply_weight` 的映射步骤。

- RBLA
  - 输入：逐层 `r_i` 分布；`pad_tensors_to_max_shape` 触发次数；NaN 掩码元素计数（RBLA 的掩码均值机制）。
  - 逐层：累计补零行/列；聚合后 A/B 的 `r_max`。
  - 广播：`broadcast_lora_state_dict` 中的客户端 Slice/Pad 统计。
  - 钩子：`_fed_aggregator_rbla._do_aggregation`、`pad_tensors_to_max_shape`、`aggregate_lora_tensors`、`broadcast_lora_state_dict`。

- SP（Sum‑Product）
  - 聚合：总耗时；键覆盖度；是否存在 `*.sp_aggregated`。
  - 权重应用模式：
    - `replace_w`：`convert_lora_for_sp_inference` 耗时、最终键计数、评估器更新耗时。
    - `regular`：`svd_split_global_weight` 耗时、各层产出的 A/B 形状与目标秩关系。
  - 广播：载荷大小与次数。
  - 钩子：`_fed_aggregator_sp._do_aggregation`、`LoRAUtils.svd_split_global_weight`、`LoRAUtils.convert_lora_for_sp_inference`、服务端 `apply_weight`。

- ZeroPad（zeropad）
  - 聚合：逐层目标最大形状、被补零的客户端数量、总补零元素与占比。
  - 输出：聚合后逐层张量形状、结果中补零比例。
  - 广播：常规载荷指标；Slice/Pad 跟踪同上。
  - 钩子：`_fed_aggregator_zeropad._do_aggregation`（收集补零统计）、通用聚合包裹。

## 事件与运行时补丁

- 订阅 `FedNodeVars` 现有事件（无需改原代码）以记录准备阶段与配置：
  - `on_prepare_data_loader`、`on_prepare_data_distribution`、`on_prepare_data_handler`、`on_prepare_model`、`on_prepare_optimizer`、`on_prepare_loss_func`、`on_prepare_client_selection`、`on_prepare_trainer`、`on_prepare_aggregation`、`on_prepare_strategy`、`on_prepare_training_logger`、`on_prepare_lora_inference_model`、`on_prepare_tokenizer`。

- 运行时补丁（try/finally 保护）：
  - 训练器：`ModelTrainer.train`、各具体 `train_step`。
  - 客户端：`FedNodeClient.run_local_training`、`set_local_weight`。
  - 服务端策略：`aggregation`、`broadcast`、`receive_client_updates`、`apply_weight`、`evaluate`、`prepare`（适配所有 server strategies）。
  - 聚合器：`AbstractFedAggregator.aggregate`（通用），以及按方法的 `_do_aggregation` 与辅助函数（见上节）。
  - LoRA 工具（可选指标）：`LoRAUtils.svd_split`、`svd_split_global_weight`、`convert_lora_for_sp_inference`。

## 输出后端（Sinks）与落盘

- 默认 CSV（路径：`./.monitor/`）
  - `rounds.csv`：
    - `ts, run_id, round, aggregator, clients, total_samples, selected_ids, agg_ms, broadcast_ms, eval_ms, payload_bytes_total, acc, ...`
  - `aggregation_layers.csv`：
    - `ts, run_id, round, method, layer, m, n, r_max, inputs, pad_rows, pad_cols, nan_mask, svd_ms, energy_ratio(optional)`
  - `client_train.csv`：
    - `ts, run_id, round, client_id, local_epochs, data_samples, epoch, loss, avg_loss, lr, batch_p50, batch_p95, throughput`
  - `adalora_ranks.csv`：
    - `ts, run_id, round, role(server|client), node_id, epoch, layer, r, mask_ones(optional), alpha`

- Console：简洁可读的进度输出，遵循现有 `console` 的级别控制。

- TensorBoard（可选）：loss/lr/throughput 标量与秩分布直方图。

## 配置

- 独立监控 YAML（与训练 YAML 完全解耦）：
  - 原则：监控配置不得写入训练 YAML，二者独立维护、独立路径，互不解析彼此内容。
  - 默认查找顺序（auto_enable 自动加载；若均不存在则按内置默认配置运行或保持最小监控）：
    1) `./monitoring/monitoring.yaml`
    2) `./monitoring.yaml`
    3) `./src/test/fl_lora_sample/yamls/monitoring/standard_monitor.yaml`（示例路径）
  - 如需自定义路径，可在项目中（可选）调用 API：`usyd_learning.monitoring.auto_enable.load_config("<your_path>")`。但为满足“最小导入”，推荐将文件放在上述默认路径之一，无需额外代码。

  ```yaml
  monitoring:
    enabled: true
    sinks: [csv, console]        # 可选：tensorboard
    dir: ./.monitor
    sampling:
      batch_interval: 1          # 每 N 个 batch 记录一次
    resources:
      gpu_mem: false
      cpu_mem: false
    adalora:
      enabled: auto              # 自动检测 Peft/AdaLoRA；true/false/auto
      snapshot: epoch            # epoch|round|N_batches
    aggregation:
      common: {enabled: true}
      fedavg: {enabled: true, layer_delta_sample: 8}
      svd:    {enabled: true, energy_sample_layers: 8}
      rbla:   {enabled: true, track_pad_slice: true}
      sp:     {enabled: true, apply_weight_mode_metrics: true}
      zeropad:{enabled: true}
  ```

## 安全与开销

- 默认仅启用轻量指标；高级指标采用抽样（如 SVD 能量比仅抽样少量层）。
- 所有包裹均使用 `try/finally`；异常不阻断原始训练流程。
- 可选依赖（psutil/pynvml/tensorboard）为“软依赖”，缺失即自动禁用相关指标。

## 里程碑

1. M1 – 核心
   - Hub/config/sinks（csv/console），基础训练与回合阶段包裹，FedAvg 通用指标，单行导入。
2. M2 – 聚合器（5 种）
   - `fedavg`、`svd`、`rbla`、`sp`、`zeropad` 的方法特有指标与耗时；广播 Slice/Pad 跟踪。
3. M3 – AdaLoRA
   - 秩/mask/alpha 快照（客户端/服务端），逐回合的 r 分布与 Δr。
4. M4 – 增强
   - 可选资源监控、TensorBoard sink、示例 YAML（`src/test/fl_lora_sample/yamls/monitoring/standard_monitor.yaml`）。
5. M5 – 测试与文档
   - 单测（启用/关闭一致性、CSV 列验证）、文档与快速上手。

## 验证

- 监控开/关（同一随机种子）下，loss/acc 曲线一致。
- 生成的 CSV 文件存在且列完整；重型指标遵循配置与采样。
- tqdm 输出与 console 友好，不相互干扰。

## 快速开始

1) 设置 `PYTHONPATH` 包含 `src`。
2) 在入口脚本加入一行：

```python
import usyd_learning.monitoring.auto_enable
```

3) 像平常一样运行（如 `src/test/lora_sample_main.py`）。监控产物默认写入 `./.monitor/`。
