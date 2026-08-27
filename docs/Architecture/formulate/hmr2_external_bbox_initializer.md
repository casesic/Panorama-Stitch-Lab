# HMR2 external-bbox initializer

## 目标与边界

`HMR2Initializer` 复用 DenseKP production YOLO 已检测出的 `xyxy` 人框，直接执行
HMR2 crop 与 batch forward。它不创建 Detectron2/ViTDet detector，不使用 PHALP，也不
负责 tracking、重初始化策略或 C++ staged-LM 接入。

```text
ERP BGR image + existing YOLO boxes [B,4]
  -> official ViTDetDataset crop/normalization
  -> one HMR2 FP32 CUDA forward for B people
  -> B HMR2Initialization values
```

模型在 initializer 构造时加载一次、切换为 `eval()` 并常驻 CUDA；每次 `infer()` 只做
crop、传输、forward 和结果转换。CUDA 不可用时直接报错，不会静默退回 CPU。

## API 与数据语义

DenseKP 使用的 `RTMLibYOLOXDetector` 输出全图像素坐标下的连续 `xyxy`，而
`ViTDetDataset` 也直接接受 `xyxy`，因此无需格式转换。dataset 根据 bbox 计算中心和
scale，扩展到模型要求的纵横比，再生成 `256 x 256` RGB ImageNet-normalized crop。
HMR2 ViT backbone 内部去掉水平方向两侧各 32 像素，实际 backbone 输入为
`[B,3,256,192]`。

FP32 forward 的主要输出为：

- `pred_smpl_params.global_orient`: `[B,1,3,3]` rotation matrix；
- `pred_smpl_params.body_pose`: `[B,23,3,3]` rotation matrix；
- `pred_smpl_params.betas`: `[B,10]`；
- `pred_cam`: `[B,3]` crop weak-perspective camera；
- `pred_cam_t`: `[B,3]` crop perspective translation；
- `pred_vertices`: `[B,6890,3]`，默认生成；
- `pred_keypoints_2d`: HMR2 crop-normalized projected joints。

initializer 忠实保留 rotation matrix，不在该层隐式转换为 axis-angle。输出顺序与输入
bbox 顺序一致。

## NumPy/chumpy 兼容

NumPy 1.26 删除的历史别名只在 `_enable_legacy_numpy_aliases_for_chumpy()` 中、HMR2
import 前补齐。没有修改 NumPy、chumpy、checkpoint 或第三方 4D-Humans 源码。

## 旧 fitting 初始化链审计

现有链路是：

```text
4D-Humans demo
  -> rotation matrix 转 axis-angle
  -> cam_crop_to_full 得到 full-image perspective translation
  -> orient[3], pose[69], beta[10], transl[3], scale[1]
debug024_run_hmr_convert.py
  -> SMPL_Param_Seq cache
fitting_io/hmr_init.py
  -> init_smplx_from_smpl54_seq
  -> SMPL-X fitting initial state
```

当前 C++ staged-LM 每帧使用 12 个双相机外参与 63 个 SMPL-X body-pose 参数，并在
PoseShape 阶段使用 10 个 beta。因而后续接入时：

- beta 可以直接沿用 10 维值；
- global orientation 与 body pose 必须显式执行 rotation-matrix -> axis-angle；
- HMR2 的 SMPL 23-joint body pose 要经过现有 SMPL -> SMPL-X 转换/关节选择，不能直接
  当作 staged-LM 的 21-joint、63 维 body pose；
- `pred_cam` 是 crop weak-perspective camera，不能直接作为 ERP staged-LM translation；
  应复用并验证 `cam_crop_to_full` 及当前 ERP/相机坐标约定。

本阶段只完成接口审计，没有重写或接入上述 conversion。

## 性能解释

benchmark 将官方 crop preprocessing、H2D、HMR2 forward、CPU postprocess 分开计时。
HMR2 forward 可以自然 batch；但官方 `ViTDetDataset` 会针对需要较强下采样的大 bbox，在完整高分辨率
ERP 图上复制并可能执行 Gaussian anti-aliasing，因此 initializer 总耗时不只由 GPU
forward 决定。该现象应单独归因，不能误判为 batch forward 变慢。

对长度为 300 帧的 track，若只初始化 1 次，摊销为 `cold_start_ms / 300`；若最多重初始
化 3 次，则为 `cold_start_ms * 3 / 300`。摊销值只用于系统预算，不代表实际 cold-start
延迟。

## 运行 benchmark

```bash
PYTHONPATH=src:Circular_Video/01_recv/20260708 \
conda run -n camerahmr python \
  src/script/test_function/debug084_benchmark_hmr2_external_bbox.py \
  --image data_process/tmp/hmr2_external_bbox_2214/cam4_erp_02214.jpg \
  --boxes data_process/tmp/hmr2_external_bbox_2214_densekp/cam4 \
  --checkpoint /home/yuyue/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt \
  --output data_process/Vis/hmr2_external_bbox_2214 \
  --batch-sizes 1 2 4 --warmup 5 --repeat 30
```

输出包括 `benchmark.json`、`hmr2_initializations.json` 和 bbox + projected-joints sanity
图。模型加载时间不计入 steady-state timing。

## Frame 2214 实测（RTX 3090，FP32）

数据为 `20260320_riding_dataset3` frame 2214 cam4，同一张 ERP 图上的 4 个 production
YOLO bbox；warmup 5 次、正式重复 30 次。

| B | preprocess p50 (ms) | forward p50/p90 (ms) | total p50/p90 (ms) | ms/person | peak MB |
|--:|--:|--:|--:|--:|--:|
| 1 | 15.59 | 24.21 / 24.80 | 40.58 / 41.15 | 40.58 | 2672.65 |
| 2 | 996.85 | 39.52 / 40.09 | 1037.31 / 1066.16 | 518.66 | 2685.26 |
| 4 | 2063.53 | 69.24 / 69.87 | 2134.28 / 2164.62 | 533.57 | 2705.72 |

纯 forward 相对重复 B=1 的理论 workload speedup 为 B=2 `1.23x`、B=4 `1.40x`。
包含官方 preprocessing 后，实测 batched-vs-repeated-B1 总耗时分别为：

| workload | repeated B=1 p50 (ms) | batched p50 (ms) | speedup |
|--:|--:|--:|--:|
| 2 persons | 1068.38 | 1037.31 | 1.030x |
| 4 persons | 2175.91 | 2134.28 | 1.020x |

因此 GPU forward batching 有明确收益，但当前端到端收益被官方逐 bbox、整张 ERP
anti-aliasing preprocessing 掩盖。若多个新 track 同帧出现，推荐按真实人数直接 batch，
最多 B=4；不应为了凑 batch 延迟单人 cold start。

模型参数量为 672,266,656，checkpoint 为 2,709,521,501 bytes，resident CUDA allocation
约 2661.79 MB。B=1 cold start 为约 40.58 ms；若一个 300 帧 track 初始化 1 次，摊销约
0.14 ms/frame，初始化 3 次约 0.41 ms/frame。多人 total 受 bbox preprocessing 内容影响，
不能用 B=1 数字线性外推。
