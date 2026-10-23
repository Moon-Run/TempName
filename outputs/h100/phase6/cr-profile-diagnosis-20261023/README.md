# CR/C/R/A 原始 profile 诊断（2026-10-23）

读取批次 `logs/h100/phase6/ma2r-c-ablation-2x-50steps-20261023-104402-375258/` 的 32 份训练原始 trace；没有重新训练、没有更改旧报告。重现：

```bash
python3 outputs/h100/phase6/cr-profile-diagnosis-20261023/analyze.py
```

`analysis.json` 保存每份 trace SHA256、每 rank/角色统计、全 step GPU 类别时长和逐 rank 提交队列数据。`rank_queue_table` 可直接查找，避免八 rank 中位数掩盖落后进程。

## 为什么 C、R 分别有效，组合未见进一步提升

1. **已有收益重叠。** C 已包含固定配置提交、独立输出及原来的同步优化；R 相对 A 的配置提交/输出改动无法再给 C 提供一次相同收益。本批 C 与 R CPU 投影中位约 127–133 μs，A 约 225–228 μs；CR 与 C 相同，说明 CR 没有再消除这一段开销。源代码可核对 `ma2r_sources.py` 与 C 的 `gemm_reduce_scatter.cc`、Python frontend。
2. **接收优化的预算较小。** 各 rank 的 decoder 时长求和再取八 rank 中位：C 5.684 ms、CR 2.812 ms、R 3.736 ms。C→CR 差值 2.872 ms，仅为正式约 516 ms step 的约 0.56%。这只是量级估算，不是正式训练收益，也不是系统效应的严格上界。继续仅缩短 CR decoder，当前被 profile 观察到的总量也只有约 2.8 ms。要赢约 1%，需缩短约 5.16 ms/step，即每个 256 次投影平均约 20.2 μs 的暴露关键路径。
3. **各 rank 提交不齐与集体等待可能吸收局部节省。** CR 八 rank中位 decoder→下一 GPU 操作间隙约 1.5/1.7 μs，后继操作通常提前排队。然而 CR rank1 的 GEMM 排队中位仅 0.89/0.97 μs，decoder 后继提前排队比例只有 32.0%/37.5%；rank5 同样接近主机供给极限。rank0 的 GEMM 排队中位反而约 15.6/15.9 ms。CR rank1 的整步 IPC/NCCL 时长分别 27.7/186.8 ms，其他多数 rank 约 90/500 ms。这符合少数 rank 的主机提交较迟、其他 rank 在集体处等待的假设。它不能证明具体哪个调用是正式计时瓶颈，亦不能把通信时长全部算成可消除开销。
4. **profile 干扰很大。** CR/C/R/A 的 GPU 时间跨度约 841/847/859/869 ms，明显长于正式约 516/516/515/535 ms。落后 rank 还会随 profile 臂变化，不能把一次 trace 的等待时长或跨臂差值当成稳定性能量。CPU 和 GPU 重叠，二者不能相加。不同 stream 的 NCCL/GPU 时长也不能相加推导总步时。

## 可检验的下一步

CR2 保留 CR 的全部 GPU 算法、packet/selection、双工作区、发布屏障、fresh output及顺序；只测试固定形状的主机提交路径：缓存不可变 GEMM 选择/元数据及 workspace 尺寸，并让 reduction 直接使用已验证的固定布局，减少重复 registry 查找和 slice/view/select。缓存必须按实例和 workspace 生命周期管理，动态 input/weight/output 指针每次刷新，检查 stream/device/shape/option，保留异常后不可复用与 RAII 恢复。

仍需同场 CR2/CR/C/R 四组、每步相同 256 次目标投影，进行两轮正式完整训练。应同时报告 CR2/CR 的直接干预净收益、CR2/C 与 CR2/R，使用配对 block CI 判断是否达到 1%；不能将 profile 中少掉的 CPU 时间作为已实现收益。

## 统计口径

- 每个角色 128 次投影；CPU 范围来自 `GemmRSFunction`，GPU kernel 通过 `External id` 关联，runtime launch 通过 `correlation` 关联。
- `gemm_launch_queue_us = GEMM GPU 开始 − cudaLaunchKernel CPU 返回`；它包含排队与GPU启动延迟，小于等于 5 μs 被视作近乎即时提交。不是一个精确的阻塞原因分类。
- `next_launch_lateness_us = 下一同 stream GPU 操作的 runtime 返回 − decoder GPU 结束`；正值表示下一操作尚未完全提交。R/A 的下一操作可为自身复用屏障，因此应使用 CR/C 比较消费者供给。
- CPU leaf op 去除父子重复计数，但其他字段如 `aten::reshape` 与 `aten::view` 单独报告时仍可能重叠，不能逐列相加。
- 所有数据来自单独采集的 profile step，仅支持瓶颈假设与执行路径检查。
