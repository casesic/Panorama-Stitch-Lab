# Panorama Stitch Lab

本项目是一个围绕全景 / 鱼眼 / ERP 图像的人体处理实验工程，包含导帧、鱼眼转 ERP、人工 review、人体检测、bbox 平滑、crop、姿态估计、SAM3 分割、mask fusion、SMPL-X fitting、semantic flow、human composite，以及后续 dense optical flow / 最终合成相关模块。

当前推荐的流程入口是：

```bash
python src/script/test_function/run_pipeline.py --list_steps
```

常用命令：

```bash
python src/script/test_function/run_pipeline.py --config data_process/Output/config.json --steps 020,021,035,040,041,025,026,024,038,043,029,030,031,037 --dry_run
python src/script/test_function/run_pipeline.py --config data_process/Output/config.json --steps 020,021,035,040,041,025,026,024,038,043,029,030,031,037
```

建议长流程运行前先用 `--dry_run` 检查生成的命令。配置示例在：

```text
configs/pipeline_config.example.json
```

## 项目流程定位

当前项目可以理解为四个阶段。

1. **预处理模块**
   负责导帧、鱼眼转 ERP、reviewed frames、bbox 检测、bbox smoothing、crop/rotate、pose、SAM3 segmentation、mask fusion、HMR convert。

2. **重建模块**
   负责 dual ERP fitting、SMPL-X vertices/faces/segments 读取和 fitting 可视化。

3. **Composite 效果验证模块**
   负责 semantic flow 和 human composite，用于观察人体跨相机 warp、补全、blend 的效果。这个阶段更像效果验证模块，不一定是最终结果模块。

4. **最终光流 / 合成模块**
   最终结果更可能来自 dense optical flow、RAFT、warp strategy、blend 和 final synthesis 等模块。semantic flow 可以作为人体区域几何先验或验证手段。

debug 脚本仍然保留为可直接运行的入口，但大部分公共逻辑已经迁移到 `src/lib_pano/` 下。

## 更多文档

- `docs/pipeline_overview.md`：按四阶段说明主流程和 step 输入输出。
- `docs/debug_scripts.md`：按职责整理 debug 脚本清单。
- `docs/data_flow.md`：说明主要数据目录关系和默认路径约定。
