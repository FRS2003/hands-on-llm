# 分布式训练对照实验：DDP vs DeepSpeed ZeRO-1/2/3

> 目标：在**同一台双卡机、同一全局 batch、同一份数据**下，亲手测出数据并行（DDP）与 ZeRO 不同分片阶段在「单卡显存占用、训练吞吐、通信开销」上的真实差异，而不是只背概念。
> 结论先行：**DDP 只提速、几乎不省单卡显存；模型变大后要靠 ZeRO 逐级分片，分片越狠越省显存但通信越多、速度越慢。**

## 1. 实验环境

| 项 | 值 |
|---|---|
| GPU | 2 × NVIDIA RTX 3080 Ti（每卡 12 GB），PCIe 互联、拓扑 SYS、**无 NVLink** |
| 框架 | PyTorch 2.6 (cu124)、DeepSpeed 0.16.4、LLaMA-Factory 0.9.3、Transformers 4.52 |
| 模型 | Qwen2.5-0.5B / Qwen2.5-1.5B（全参 SFT，非 LoRA） |
| 统一设置 | bf16、梯度检查点、group_by_length、cutoff_len=512、全局 batch=16 |
| 数据 | 3000 条中文指令（训练）/300（评测），A 组跑 40 步、B 组跑 30 步 |

显存由后台 `nvidia-smi` 每秒采样取每卡峰值；吞吐取 Trainer 输出的 `train_samples_per_second`（纯训练时间，不含启动/存盘）。原始日志见 `logs/`，结果表见 `../distributed_compare.csv`。

## 2. 实验 A：DDP 的加速比（Qwen2.5-0.5B）

固定全局 batch=16：单卡 micro=1×accum16；双卡每卡 micro=1×accum8。优化器统一用 paged_adamw_8bit（12G 单卡全参 0.5B 用标准 AdamW 会贴边 OOM，见踩坑 7）。

| 策略 | 卡数 | 每卡峰值显存 | 训练耗时 | 吞吐(samples/s) | 单步耗时 |
|---|---|---|---|---|---|
| DDP | 1 | 10056 MB | 122.94 s | 5.21 | 3.07 s/it |
| DDP | 2 | 10185 / 9770 MB | 68.98 s | **9.28** | 1.72 s/it |

**读数：**
- **加速比 9.28 / 5.21 ≈ 1.78×，并行效率约 89%**。没到线性 2×，损耗来自两卡间梯度 all-reduce 的通信（PCIe、无 NVLink）。
- **双卡每卡显存（约 10G）和单卡（10G）几乎一样**——这是最关键的一点：DDP 每张卡都持有**完整的模型参数、梯度和优化器状态**，只把数据切分到多卡，所以它靠多卡线性扩大吞吐，却**不降低单卡显存门槛**。模型一旦单卡放不下，DDP 也救不了，需要 ZeRO。

## 3. 实验 B：ZeRO 逐级分片（Qwen2.5-1.5B，固定双卡）

1.5B 全参、标准 AdamW（fp32 优化器状态），分别用普通 DDP 与 ZeRO-1/2/3 启动：

| 策略 | 分片对象 | 每卡峰值显存 | 结果 |
|---|---|---|---|
| DDP | 不分片 | 6678 MB 处即失败 | **OOM**：创建优化器状态时一次性还要 5.75 GB，放不下 |
| ZeRO-1 | 优化器状态 | 11166 / 10984 MB | **OOM**：反向传播时仍需 2.88 GB，仅剩 2.47 GB |
| ZeRO-2 | 优化器状态+梯度 | 11102 / 11042 MB | **OOM**：通信桶处差约 0.4 GB |
| ZeRO-3 | 优化器状态+梯度+**参数** | 11913 / 11912 MB | **跑通**，239.4 s，2.01 samples/s，7.91 s/it |

**读数：**
- 1.5B 的 fp32 AdamW 状态（master 权重 + 一阶/二阶矩）约 1.5B×16B ≈ 18 GB 量级，是显存大头。DDP 每卡都要完整一份，必然 OOM。
- ZeRO-1 把优化器状态切到两卡、ZeRO-2 再把梯度也切开，但**模型参数仍每卡完整保留**，1.5B 在 12G 上仍差一口气（ZeRO-2 只差 ~0.4 GB）。
- **ZeRO-3 连模型参数都分片**，用到时再 all-gather、用完释放，是这套 2×12G 上唯一能跑通 1.5B 全参的方案。
- **代价是通信**：ZeRO-3 前向/反向都要反复 all-gather 参数，单步 7.91 s/it、吞吐只有 2.0 samples/s——**用更多通信换来了能跑通的显存**。这正是 ZeRO 各阶段的核心 trade-off。
- TP（张量并行）/PP（流水线并行）在两张 PCIe 3080Ti、1.5B 小模型场景不适用（TP 对互连带宽要求高、PP 需要更多卡和微批次调度），仅做原理梳理，未实测。

## 4. 复现步骤

```bash
# 1) 环境（注意版本，DeepSpeed 0.19 在 torch2.6 有 Muon 导入 bug，用 0.16.4）
pip install "deepspeed==0.16.4"
# 2) 模型走本地离线路径，数据为固定切分的中文指令集，见 yamls/*.yaml
# 3) 逐组启动（run_dp.sh 内部带每秒显存采样与计时）
bash run_dp.sh yamls/ddp1_05.yaml ddp1 1     # 0.5B 单卡
bash run_dp.sh yamls/ddp2_05.yaml ddp2 2     # 0.5B 双卡 DDP
bash run_dp.sh yamls/zddp_15.yaml z_ddp 2    # 1.5B 双卡 DDP（预期 OOM）
bash run_dp.sh yamls/z1_15.yaml  z1 2        # 1.5B ZeRO-1
bash run_dp.sh yamls/z2_15.yaml  z2 2        # 1.5B ZeRO-2
bash run_dp.sh yamls/z3_15.yaml  z3 2        # 1.5B ZeRO-3
# 一键顺序跑：bash run_all_dist.sh / run_zall.sh
```

## 5. 踩坑记录（都是亲手 debug 出来的）

1. **环境里没有 `torchrun` 可执行文件**：改用 `python -m torch.distributed.run`，等价且一定存在。
2. **`python -m torch.distributed.run ... -m llamafactory.train.tuner` 双 `-m` 静默退出**：tuner 内部用了相对导入（`from ..data import ...`），不能当裸脚本。解法是写一个 3 行 `run_tuner.py` 用**绝对导入**包一层，再把文件路径交给 launcher。
3. **命令行 `--gradient_accumulation_steps`/`--deepspeed` 覆盖被 HfArgumentParser 判为未知 key**：该版本统一改为**写进各自 yaml**，启动时只传一个 yaml 位置参数。
4. **上一组失败进程残留占着显存，下一组假性 OOM**：报错里出现「另一个 PID 已占 8 GB」就是它。每组之间确认 `nvidia-smi --query-compute-apps` 清空，必要时 `pkill -9 -f torch.distributed.run`（注意别用会匹配到当前 shell 的模式）。
5. **ZeRO 通信桶过大导致 OOM**：`allgather/reduce_bucket_size=5e8` 会瞬间申请 1–2 GB 缓冲，12G 卡上直接爆，降到 `2e7` 后稳定。
6. **DeepSpeed 会覆盖客户端优化器**：日志里 `DeepSpeed Basic Optimizer = AdamW` 说明 yaml 写的 `paged_adamw_8bit` 在带 deepspeed 配置时被换成了标准 AdamW——对比实验里要清楚自己实际用的是哪个优化器。
7. **12G 单卡全参 0.5B 用标准 AdamW 会在 optimizer.step 贴边 OOM**：`group_by_length` 降低长度不均的峰值 + `paged_adamw_8bit` 压缩优化器状态后稳定，峰值约 10 GB。

## 6. 一句话面试版

> 在两张 3080Ti（PCIe、无 NVLink）上做过同口径对照：0.5B 全参 SFT 双卡 DDP 加速 1.78×、效率 89%，但每卡显存和单卡持平，说明 DDP 只扩吞吐不省显存；1.5B 时 DDP/ZeRO-1/ZeRO-2 先后 OOM，只有把参数也分片的 ZeRO-3 跑通，但 all-gather 通信让步速明显变慢——据此能讲清三种并行与 ZeRO 各阶段「分什么、省什么、付出什么通信代价」。