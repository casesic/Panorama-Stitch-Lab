# ADR-0001：所有观测统一到 ERP 坐标系

- **状态**：已生效（2026-11-20 记录；决定早于此）
- **影响面**：`src/lib_pano/`、`cpp/staged_lm/` 的全部观测与重投影
- **一句话**：鱼眼原图只作为输入，进入流程后立刻转成 ERP；此后 bbox、crop、DenseKP 138 点、staged-LM 残差全部在 ERP 上定义。

## 背景

原始数据是双相机鱼眼视频。鱼眼图像上的人物是弯曲的，同一个点在两台相机里的局部几何不一致，直接在上面定义检测框、crop 和重投影残差都要为每台相机单独写一套模型。

## 决定

流程第 2 步（`021`）把鱼眼帧转成 ERP；之后的所有中间产物都用 ERP 坐标：

```text
双鱼眼原图
  → calibration LUT cache
  → 两张普通 ERP 图
  → DenseKP inference / person match
  → 仅旋转 138 个 ERP 点
  → ERP_ROTATE observation JSON
  → runtime exporter → C++ Stage1 / Pose / PoseShape LM
```

（链路图出自 [densekp_production_delivery_refactor](../Architecture/formulate/densekp_production_delivery_refactor.md)）

staged-LM 三段残差也统一写成「dual-view ERP reprojection + anchor」，见 [staged_lm/overview](../Architecture/staged_lm/overview.md) 的表格。

## 依据

1. **标定与 LUT**：ERP 由标定得到的 LUT 生成并缓存，转一次、后续所有帧复用（同上链路图的 `calibration LUT cache`）。
2. **残差只有一种形式**：三个阶段都吃「双视角 ERP 重投影」，不需要为每台相机的鱼眼畸变单独建残差项。
3. **crop 相机的量不能直接当 ERP 平移**：HMR2 的 `pred_cam` 是 crop 的 weak-perspective camera，[文档明确它不能直接作为 ERP staged-LM 的 translation](../Architecture/formulate/hmr2_external_bbox_initializer.md)。这说明流程里确实存在「crop 局部坐标系」与「ERP 全局坐标系」两套量，最终拟合只认后者。

## 代价与已知限制

- **接缝**：ERP 在 ±180° 处有接缝，跨接缝的框、crop、点序都要特殊处理（周期横坐标）。这是 ERP 的固有成本。
- **采样密度不均匀**：ERP 两极拉伸，同样的像素数在极区覆盖更大的立体角。当前实验数据都是近似水平的环拍，没有专门评估过极区的拟合质量。
- **多一次重采样**：鱼眼 → ERP 会重采样一次，信息损失不可逆。

## 未在文档中写明（待作者补充）

- 为什么选 equirectangular 而不是直接在鱼眼上反投影，或选 cubemap / 多透视平面。
- 接缝处理的具体约定（跨缝框如何裁剪、点序如何回绕）散在代码里，没有一篇统一的说明。

## 相关文档

- [densekp_production_delivery_refactor](../Architecture/formulate/densekp_production_delivery_refactor.md)
- [staged_lm/overview](../Architecture/staged_lm/overview.md)
- [hmr2_external_bbox_initializer](../Architecture/formulate/hmr2_external_bbox_initializer.md)
