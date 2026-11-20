# 性能与基准结果

这个仓库没有 benchmark harness：数字是当时手测完写在 commit message 或某篇归档文档里的。所以下表**每一行都带来源**，来源里没有的信息保持「待补」——不替它补。

关于「待补」：三条 commit 记录的硬件已由作者补录为 RTX 3090（2026-11-20），点数与命令当时没有留日志，只能同机重跑才能补齐（见文末「怎么补」）。文档里那些 bakeoff / attribution 表是**同一台机器上的相对对比**，不是产品基准，跨文档比较 FPS 不成立。

## 结果表

| 日期 | 硬件 | 输入 | 命令 | 指标 | 结果 | baseline | 来源 |
|---|---|---|---|---|---|---|---|
| 2026-08-13 | RTX 3090（作者补录） | 单帧 staged-LM，C++ O4 kernel | 待补 | 吞吐 | **5 FPS** | 待补 | commit `ee33380`「O4 cpp 5FPS」 |
| 2026-08-18 | RTX 3090（作者补录） | 100 点 loss 的 C++ 管线 | 待补 | 吞吐 | **25–30 FPS** | 待补 | commit `58b0257`「CPP 100 points loss 25-30FPS」 |
| 2026-08-31 | RTX 3090（作者补录） | DenseKP 的 TensorRT 后端 | 待补 | 正确性测试 | **通过**（原文 TEST PASSED） | 待补 | commit `905fb6c`「TRT TEST PASSED」 |
| 待补（文档未记） | 待补（文档只写单线程、`-march=native`） | frame 513，cam2 + cam4，T=1，138 点 / 204 support | Release，O4 `optimized_native_stable_p`，warmup 后跑 20 次取 median | 单帧 median 耗时 / FPS | D138：129.37 ms（7.73 FPS）<br>S138：23.81 ms（42.01 FPS） | S138（SAM3D 路线） | [densekp_vs_sam3d_cpp_performance_attribution](../docs/Architecture/archive/densekp_vs_sam3d_cpp_performance_attribution.md) |

最后一行不是「我们的最终性能」，而是**归因实验**：它在同一 fixture 上比较两条拟合 policy，结论是 DenseKP 的 value/Jacobian regression 占 D138 的 74.34%、解释了与 S138 差距的约 91.11%。引用它时别把 7.73 / 42.01 FPS 当成端到端流水线速度。

## 这些数字能声称什么

- 「C++ O4 单帧 5 FPS → 100 点 loss 的 C++ 管线 25–30 FPS」是当时两次手测的**吞吐量级**，能说明 C++ 重写把单帧拟合推进到接近实时；具体帧率依赖硬件与点数，没有同机复测记录。
- 「TRT TEST PASSED」只说明 TensorRT 后端的正确性测试通过，**没有给出速度**。
- 归因实验的 FPS 是单帧 staged-LM 拟合的 steady-state（不含文件读取、模型加载、输出保存，见 [src/README.md](../src/README.md) 第 8 节）。
- 仓库里**没有**端到端（导帧 → 合图）的帧率记录。

## 怎么补

1. **C++ 侧**：按 [staged_lm_cpp_o4_single_frame](../docs/Architecture/run/staged_lm_cpp_o4_single_frame.md) 构建并运行 `staged_lm_pipeline_benchmark`（fixture 需自备，`data_process/` 不进 Git）。
2. **Python 侧**：`python src/script/test_function/run_pipeline.py --list_steps` 看可用 step，`--dry_run` 先确认命令。
3. 补齐时把 **硬件 / 命令 / baseline** 三列一起写，并保留日期；只写一个 FPS 没有意义。

如果你能回忆出当时那三次手测的机器和点数，我按你给的补进去，并在「来源」列标注「作者补录」。
