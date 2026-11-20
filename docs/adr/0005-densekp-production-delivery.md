# ADR-0005：DenseKP 生产交付只保留「模型数据 + 替换 targets」这一条链

- **状态**：已生效（2026-11-20 记录；决定早于此）
- **影响面**：生产交付运行时、`runtime exporter`、C++ model bundle
- **一句话**：运行时不再从零准备拟合输入，而是读一份预先准备好的 C++ model bundle（含当前帧 204 support），只替换 DenseKP 的 targets、valid IDs 和 Stage 1 的 local rows，然后直接进 C++ 三段 LM。

## 背景

早先的准备路径是 Python 侧的 `prepare_staged_lm_cpp_input.py`：读 HMR、Sam3DBody observation 和 SMPL-X 模型，现算 support、初值、fixture 再交给 C++。这条路径要跑 Python SMPL-X、Python LM、reference dump，慢且与 C++ 实现重复。

## 决定

生产链固定为：

```text
双鱼眼原图
  → calibration LUT cache
  → 两张普通 ERP 图
  → DenseKP inference / person match
  → 仅旋转 138 个 ERP 点
  → ERP_ROTATE observation JSON
  → runtime exporter（注入离线 C++ model bundle）
  → C++ Stage1 / Pose / PoseShape LM
  → optimized DenseKP138（ERP_ROTATE）
```

要点：

1. **CPW 是下游消费者**，不属于 DenseKP→LM 的必要计算，因此不在这一条链上。
2. 运行时**不再调用** `prepare_staged_lm_cpp_input.py`，也不运行 Python SMPL-X、Python LM、reference dump 或 trajectory 生成。
3. C++ model bundle 里存的是当前帧**已经准备好**的 204 support 模型数据（含初值与模型量）。
4. runtime exporter 的职责被限定成三件事：替换 DenseKP targets、valid IDs、Stage 1 对应的 DenseKP local rows。
5. 只旋转 138 个点，不做全量重算。

## 依据

以上五条都写在 [densekp_production_delivery_refactor](../Architecture/formulate/densekp_production_delivery_refactor.md) 里，其中「CPW 是下游消费者，不属于 DenseKP→LM 的必要计算」和「运行时不再调用 `prepare_staged_lm_cpp_input.py`」是这条决策的核心边界。

## 代价与已知限制

- **bundle 与帧绑定**：204 support 是预先算好的，换了 support 数量或采样策略就要重做 bundle，不能只换 targets。
- **两套路径并存**：离线 fixture 生成（`prepare_staged_lm_cpp_input.py` 等）仍在仓库里，用于生成与对照；读代码时容易误以为它还是生产入口。
- **可追溯性下降**：运行时不再产生 Python 侧中间产物，出问题只能看 C++ 侧日志与 exporter 的输出。

## 未在文档中写明（待作者补充）

- 为什么 support 固定为 204、138 点这个点集是怎么选的。
- 离线 bundle 的生成与验收流程（谁生成、怎么验、版本怎么对应）。
- CPW 具体指哪一段实现、它的输入来自这条链的哪一步。

## 相关文档

- [densekp_production_delivery_refactor](../Architecture/formulate/densekp_production_delivery_refactor.md)
- [persistent_cpp_fitting_worker](../Architecture/formulate/persistent_cpp_fitting_worker.md)
- [per_person_cpp_runtime_preparation](../Architecture/formulate/per_person_cpp_runtime_preparation.md)
