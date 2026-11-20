# ADR-0003：TensorRT 只做显式选择的后端，不做自动回退

- **状态**：已生效（2026-11-20 记录；决定早于此）
- **影响面**：DenseKP 生产交付运行时（`ORTRunner` 一侧）
- **一句话**：DenseKP 推理有两个后端，`ort` 是默认与数值参考，`trt` 只有在调用方显式点名并给出 engine 路径时才用；运行时**不探测文件猜后端，也绝不静默回退**。

## 背景

DenseKP 的点位回归是生产链路的第一段推理。它有两种实现：ONNX Runtime，以及用 TensorRT 加载的 FP16 engine。两者数值不完全相同，而下游的 staged-LM 拟合对点位很敏感。

## 决定

1. `ort` 保持默认，并且是**数值参考实现**。
2. `trt` 是选项：调用方必须显式选择后端、显式给出 engine 路径。
3. 运行时不通过「文件存在与否」来推断后端，也不会在 TensorRT 失败时悄悄退回 ORT。

## 依据

以上三条都出自 [densekp_trt_production_backend](../Architecture/formulate/densekp_trt_production_backend.md) 的原文表述：

- "`ort` remains the default and numerical reference."
- "`trt` loads the accepted TensorRT FP16 engine when the caller explicitly selects it and gives an engine path."
- "The runtime never selects a backend by probing files and never silently falls back from TensorRT to ORT."

两条后端共享同一条交付路径（crop/resize → BGR→RGB → ImageNet 归一化 → 连续 FP32 NCHW → 后端 seam → `(prediction xy + 0.5) * 256` → 仿射还原 → ERP 还原 / 有效性 / 匹配），差异被限制在一个 seam 上。

## 代价与已知限制

- **配置变多**：每次跑都要明确后端与 engine 路径，不能"有 engine 就用"。
- **失败不降级**：TensorRT 不可用时会直接失败，而不是退回 ORT——对生产是对的（避免静默变更数值），对调试不友好。
- **速度收益没有记录**：仓库里只有 commit `905fb6c`「TRT TEST PASSED」说明正确性测试通过，没有 FP16 与 ORT 的速度对比（见 [benchmarks/RESULTS.md](../../benchmarks/RESULTS.md)）。

## 未在文档中写明（待作者补充）

- 为什么用 FP16 而不是 FP32 / INT8。
- TensorRT 与 ORT 在同一批点上的数值差有多大、验收阈值是多少。
- 选 `trt` 的实际收益（延迟、吞吐、显存）。

## 相关文档

- [densekp_trt_production_backend](../Architecture/formulate/densekp_trt_production_backend.md)
- [densekp_production_delivery_refactor](../Architecture/formulate/densekp_production_delivery_refactor.md)
