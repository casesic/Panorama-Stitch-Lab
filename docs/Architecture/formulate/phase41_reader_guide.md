# Phase 4.1 production sequence reader guide

面向第一次接触本项目的读者。目标：不读任何 debugXXX 就能理解
FULL → GUIDED → identity → TrackState → HMR2 / warm start → C++ fitting。

## 正式入口

```text
src/script/human_fitting/run_densekp_track_guided_sequence.py   # CLI（薄）
src/lib_pano/human_fitting/sequence_orchestrator.py             # 编排（无数学）
```

这是当前 Phase 4.1 production composition root。旧入口
`src/script/human_fitting/run_densekp_cuda_staged_lm.py` 仅保留为
single-person、base-fixture、multi-frame staged-LM 的 legacy/reference runner；
它没有 FULL/GUIDED 调度、identity gate、HMR2 cold start 或多人物 TrackState 生命周期。

入口边界是**准备好的 normal-ERP 图像对**。fisheye→LUT→ERP remap 属于上游交付，
留在 `densekp_delivery` 包内，正式 sequence 不做 remap。

## 一条 timeline 上发生了什么

每帧按顺序：

1. `TrackGuidedScheduler.mode(frame, active_tracks)` 决定
   `FULL_DETECTION` / `TRACK_GUIDED`；活跃 track 为 0 或到达 K 周期帧时走 FULL。
   K 在 `TrackGuidedSequenceConfig.detection_interval`（默认 3，已验证值）。
2. FULL：component_bbox ROI → YOLOX B=1 → DenseKP B=1 → 球面几何一对一匹配，
   输出 v2 语义的 `MatchedPerson` 列表（这是 Phase 2.2 已验证路径）。
3. TRACK_GUIDED：每个 active track 用上一帧观测几何预测 seam-aware bbox →
   小图 DenseKP → **保留全部跨相机 candidate** → 去重 → 全体
   track×candidate 时序 cost matrix → 最大基数全局一对一 assignment →
   track/candidate 双侧 ambiguity margin，只有唯一且合格的身份被接受。
   这就是 "identity gate"（Phase 4.1）。
4. GUIDED 失败（ambiguous / no unique assignment / 无有效 crop）→
   同帧立即 FULL recovery（`run_delivery_with_same_frame_recovery`）。
5. 得到本帧 `MatchedPerson` 后进入拟合编排：
   - 有 track 的 person → temporal association（球面角 cost ≤20°，最大基数 assignment）；
   - GUIDED 帧身份已由 gate 解析，直接按 `(track_id, person)` 绑定，不再重算；
   - 新 person → 新 track；
   - 连续存在的 track → previous optimized LM state warm start；
   - new/reinit track → best-view 选择后跑 HMR2 → SMPL-X 两相机归一化。
6. 每个匹配 person：物化 T=1 runtime fixture（注入 observation + pose/beta anchor）
   → subprocess C++ staged-LM（`--runtime-prepare-stage1` 会用当前 pose/beta 重建
   Stage1 几何，禁止复用旧 fixture geometry）→ 解析回 `OptimizedPersonState`
   （warm start 做 bitwise exact 校验）。
7. `MatchedPersonTrackManager.update()` 写回 TrackState → 下一帧。
   TrackState 始终在内存中连续传播；任何"从 JSON 重建上一帧状态"都只属于
   benchmark/resume 场景，不在主链上。

## 不变量（勿破坏）

- DenseKP `keypoint_value/raw_scores` 不是置信度，不参与 validity/identity/LM。
- 不同 person 不共享 beta；多帧共享 beta 仅限同一 person。
- unmatched detections 不进 HMR2、不进 LM。
- HMR2 只出现在 new/reinit track；连续 track 用 previous C++ LM state。
- assignment 的 tie-break 是确定性的 lexicographic key，不要换成其他 solver。

## Python ↔ C++ 边界

唯一边界文件：`src/lib_pano/human_fitting/per_person_cpp.py`。
协议：fixture 目录文件名 + `*_meta.txt` 键值 + 输出二进制/JSON，见
`fitting_io/densekp_runtime_export.py`。C++ 目录为 `cpp/staged_lm/`。
`CppPersonFitterConfig` 默认使用 CPU evaluator，是当前调用方式和构建兼容性默认值，
不表示 CPU evaluator 在算法或性能上优于 CUDA。

## 第一遍跳过的东西

- 所有 `debug084~debug089`（validation / benchmark / forensic 资产，不是
  current production composition root）；
- `src/lib_pano/human_fitting/staged_lm/**`：Python 参考 LM 实现
  （LEGACY_REFERENCE），当前 production 主链是上面的 C++ 路径；
- CUDA kernel、vis、cache/LUT 细节、assignment 重复实现的合并计划。

## 关键文档

- `formulate/phase22_sequence_tracking_cpp_lm.md`（sequence 回归基线）
- `formulate/phase4_track_guided_sparse_delivery.md`（稀疏 delivery）
- `formulate/phase41_guided_identity_gate.md`（identity gate 与 2214 forensic）

## 交付依赖说明

`densekp_delivery` 包当前位于
`Circular_Video/01_recv/20260708/densekp_delivery`（外部交付目录，尚未纳入版本控制）。
生产代码统一经 `lib_pano.delivery_bootstrap` 导入；建议尽快将其迁入版本控制
（详见 Phase A 审计报告 DENSEKP_DELIVERY_VERSION_CONTROL_RECOMMENDATION）。
