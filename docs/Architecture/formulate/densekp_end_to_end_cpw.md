# DenseKP → C++ staged LM → CPW 闭环

## 数据链

```text
DenseKP delivery（fisheye / ERP / ERP_ROTATE）
  → Python observation adapter
  → ERP_ROTATE observation
  → C++ Stage1 / Pose / PoseShape LM
  → final state 的 SMPL-X support forward
  → Sparse-B DenseKP regression（204 → 138）
  → ERP projection
  → optimized_densekp.json（ERP_ROTATE）
  → debug077 CPW
  → cpw_result.jpg
```

LM 内部状态仍包含 pose、camera 和 beta，但 production 输出只保留 CPW 所需的
138 对 DenseKP 点，不保存 SMPL-X 参数或顶点。`keypoint_value` 不参与 loss、
confidence、IRLS 或其他权重计算。

## 坐标约定

- 输入：兼容 `densekp_delivery` 的 `fisheye`、`erp`、`erp_rotate`。
- LM：统一使用 `ERP_ROTATE`，尺寸为 `5760 × 2880`。
- 输出：`ERP_ROTATE`，每行语义为
  `[x_cam2, y_cam2, x_cam4, y_cam4]`。

ERP 输入通过球面旋转矩阵转换；原始双鱼眼坐标先用对应 Double-Sphere
参数反投影到球面，再转 ERP 和 ERP_ROTATE。这里不使用 crop bbox，也不假设
DenseKP 来自 crop 图。

## 入口和输出

端到端脚本：

```text
src/script/test_function/debug078_densekp_fitted_cpw_composite.py
```

该脚本现在要求 `--base_cpp_fixture` 指向离线准备的 DenseKP-ready C++ model
bundle。逐帧 runtime 不再调用 reference prepare、Python SMPL-X 或 Python LM。

单帧输出：

```text
output/frame_xxxx/
├── densekp_observation.json
├── optimized_densekp.json
└── cpw_result.jpg
```

C++ 单帧和多帧 runner 都支持 `--optimized-densekp-only`。启用后仅写优化后的
DenseKP JSON；多帧模式在各 `frame_N/` 下分别写一个 JSON，并使用求解后的
shared beta 计算对应点。

## 验证结论

- Delivery ERP→ERP_ROTATE 与原 delivery 实现逐值一致。
- 双鱼眼 Double-Sphere 投影/反投影回环误差小于 `1e-3 px`。
- 新增输出前后的 LM parameters 与 trial trajectory bitwise unchanged。
- 单帧输出为 138 对点，坐标有限且位于 ERP_ROTATE 合法范围内。
- 4 帧 shared-beta 输出验证通过，每帧均为 138 对点。
- 513 帧 CPW 成功。相对现有 debug077 结果，全图像素差 P95 为 0，超过
  10 灰度的像素占约 0.84%，差异集中在两组独立优化点导致的局部 warp 区域，
  视觉结果一致。

```text
DENSEKP_END_TO_END_CPW_READY=PASS
```
