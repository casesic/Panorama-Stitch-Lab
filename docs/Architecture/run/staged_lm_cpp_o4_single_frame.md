# C++ staged-LM O4 单帧与多帧运行说明

本文说明如何运行当前 C++ staged-LM preferred policy，包括单帧 pipeline
以及第一版共享 beta 的多帧 pipeline：

```text
optimized_native_stable_p（O4）
```

单帧参数维度依次为：

```text
Stage1       12
Pose         75
PoseShape    85 = 75 + 10 beta latent
```

本文使用已经生成的 frame 513 fixtures：

```text
S100：  cam2 99点，cam4 93点
S1024： cam2 1023点，cam4 927点
```

## 生成其他帧的 C++ 输入

统一入口会从 HMR、Sam3DBody observation 和 SMPL-X 模型生成 Stage1、
Pose/Shape 以及最终 pipeline fixture：

```bash
conda run -n camerahmr python \
  src/script/validation/staged_lm/prepare_staged_lm_cpp_input.py \
  --frame 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --surface_sample_ids src/lib_pano/human3d/assets/smplx_surface_ids_region_fps_100.npy \
  --output_root data_process/Vis/cpp_reference_dump_s100 \
  --device cpu
```

生成的 C++ 输入位于：

```text
data_process/Vis/cpp_reference_dump_s100/
  staged_lm_pipeline/frame_000514/cpp_fixture/
```

随后运行 O4：

```bash
./build/forward_reuse_integration/staged_lm_pipeline_reference_main \
  --fixture data_process/Vis/cpp_reference_dump_s100/staged_lm_pipeline/frame_000704/cpp_fixture \
  --output data_process/Vis/cpp_o4_s100/frame_000704 \
  --kernel-policy optimized_native_stable_p
```

不传 `--surface_sample_ids` 时使用当前 Sam3DBody full observation；也可以将其
换成仓库中的 50、138 或 200 点采样 ID。当前入口一次生成并运行一个 frame。

## 1. 进入仓库

```bash
cd /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch
```

下列命令均假设当前目录为仓库根目录。

## 2. 单次运行 S100 O4

```bash
./build/forward_reuse_integration/staged_lm_pipeline_reference_main \
  --fixture data_process/Vis/cpp_reference_dump_s100/staged_lm_pipeline/frame_000513/cpp_fixture \
  --output data_process/Vis/cpp_o4_s100_single/frame_000513 \
  --kernel-policy optimized_native_stable_p
```

成功时输出：

```text
STAGED_LM_PIPELINE_CPP_WRITTEN
```

结果目录：

```text
data_process/Vis/cpp_o4_s100_single/frame_000513/
```

主要文件：

```text
stage1_summary.txt
pose_summary.txt
pose_shape_summary.txt

stage1_final_parameters.f32
pose_final_parameters.f32
pose_shape_final_parameters.f32
final_physical_beta.f32

stage1_trajectory.csv
pose_trajectory.csv
pose_shape_trajectory.csv

pipeline_timing.txt
kernel_dispatch.txt
```

查看总体时间：

```bash
cat data_process/Vis/cpp_o4_s100_single/frame_000513/pipeline_timing.txt
```

查看三个阶段的 LM 结果：

```bash
cat data_process/Vis/cpp_o4_s100_single/frame_000513/stage1_summary.txt
cat data_process/Vis/cpp_o4_s100_single/frame_000513/pose_summary.txt
cat data_process/Vis/cpp_o4_s100_single/frame_000513/pose_shape_summary.txt
```

查看 trial 轨迹：

```bash
column -s, -t data_process/Vis/cpp_o4_s100_single/frame_000513/stage1_trajectory.csv
column -s, -t data_process/Vis/cpp_o4_s100_single/frame_000513/pose_trajectory.csv
column -s, -t data_process/Vis/cpp_o4_s100_single/frame_000513/pose_shape_trajectory.csv
```

## 3. 确认运行的是 O4

```bash
cat data_process/Vis/cpp_o4_s100_single/frame_000513/kernel_dispatch.txt
```

应看到类似：

```text
kernel_policy=optimized_native_stable_p
pose_offset_scalar_calls=0
lbs_selected_transform_scalar_calls=0
optimized_fallback_count=0
```

O4 正常情况下会使用：

```text
pose-offset derivative GEMM
selected-transform GEMM
value pose-offset GEMV
stable selected_posedirs ownership
```

O4 不启用 experimental forward reuse。

## 4. 正式测量 S100 FPS

推荐先预热3次，再正式重复10次：

```bash
./build/forward_reuse_integration/staged_lm_pipeline_benchmark \
  --fixture data_process/Vis/cpp_reference_dump_s100/staged_lm_pipeline/frame_000513/cpp_fixture \
  --output data_process/Vis/cpp_o4_s100_benchmark_manual/frame_000513 \
  --kernel-policy optimized_native_stable_p \
  --warmup 3 \
  --repeat 10
```

成功时输出：

```text
STAGED_LM_PIPELINE_BENCHMARK_WRITTEN
```

原始测量结果：

```text
data_process/Vis/cpp_o4_s100_benchmark_manual/frame_000513/pipeline_runs.csv
```

查看 CSV：

```bash
column -s, -t \
  data_process/Vis/cpp_o4_s100_benchmark_manual/frame_000513/pipeline_runs.csv
```

计算中位数和 FPS：

```bash
python -c '
import csv
import statistics

path = "data_process/Vis/cpp_o4_s100_benchmark_manual/frame_000513/pipeline_runs.csv"
rows = list(csv.DictReader(open(path)))

compute = [float(row["compute_ms"]) for row in rows]
wall = [float(row["wall_ms"]) for row in rows]

compute_median = statistics.median(compute)
wall_median = statistics.median(wall)

print("compute median:", compute_median, "ms")
print("compute FPS:", 1000.0 / compute_median)
print("wall median:", wall_median, "ms")
print("wall FPS:", 1000.0 / wall_median)
print("compute min/max:", min(compute), max(compute))
'
```

当前机器已有实测约为：

```text
compute median ≈ 17.02 ms
compute FPS    ≈ 58.75
```

CPU负载和频率会造成一定波动，应以多次运行的中位数为准。

## 5. 查看分阶段耗时

```bash
python -c '
import csv
import statistics

path = "data_process/Vis/cpp_o4_s100_benchmark_manual/frame_000513/pipeline_runs.csv"
rows = list(csv.DictReader(open(path)))

columns = [
    "compute_ms",
    "stage1_ms",
    "pose_ms",
    "pose_shape_ms",
    "pose_jacobian_ms",
    "pose_objective_ms",
    "pose_ne_ms",
    "pose_shape_jacobian_ms",
    "pose_shape_objective_ms",
    "pose_shape_ne_ms",
]

for name in columns:
    values = [float(row[name]) for row in rows]
    print(f"{name:30s} {statistics.median(values):9.3f} ms")
'
```

当前 S100 的参考中位数约为：

```text
Stage1       1.81 ms
Pose         9.40 ms
PoseShape    5.79 ms
Total       17.02 ms
```

## 6. 读取最终参数

`.f32` 文件为 little-endian float32：

```bash
python -c '
import numpy as np
from pathlib import Path

root = Path("data_process/Vis/cpp_o4_s100_single/frame_000513")

stage1 = np.fromfile(root / "stage1_final_parameters.f32", dtype="<f4")
pose = np.fromfile(root / "pose_final_parameters.f32", dtype="<f4")
pose_shape = np.fromfile(root / "pose_shape_final_parameters.f32", dtype="<f4")
beta = np.fromfile(root / "final_physical_beta.f32", dtype="<f4")

print("Stage1 [12]:")
print(stage1)

print("\nPose [75]:")
print(pose)

print("\nPoseShape [85]:")
print(pose_shape)

print("\nFinal physical beta [10]:")
print(beta)
'
```

PoseShape参数布局：

```text
[0:3]    cam2 global orientation
[3:6]    cam2 translation
[6:9]    cam4 global orientation
[9:12]   cam4 translation
[12:75]  body pose
[75:85]  beta latent
```

物理 beta 已单独保存为：

```text
final_physical_beta.f32
```

其计算为：

\[
\beta=2.5\tanh(z_\beta)
\]

## 7. 运行 S1024 对照

S1024 使用原始 fixture：

```bash
./build/forward_reuse_integration/staged_lm_pipeline_benchmark \
  --fixture data_process/Vis/cpp_reference_dump/staged_lm_pipeline/frame_000513/cpp_fixture \
  --output data_process/Vis/cpp_o4_s1024_benchmark_manual/frame_000513 \
  --kernel-policy optimized_native_stable_p \
  --warmup 3 \
  --repeat 10
```

当前已有实测：

| Policy | cam2/cam4有效点 | Compute median | FPS |
|---|---:|---:|---:|
| S1024 | 1023 / 927 | 207.41 ms | 4.82 |
| S100 | 99 / 93 | 17.02 ms | 58.75 |

S100相对S1024的单帧 O4 compute speedup约为：

\[
207.41/17.02\approx12.19\times
\]

## 8. Timing 边界

benchmark CSV 中：

```text
compute_ms
wall_ms
```

都是 pipeline 内部拟合计时，不包含：

- 进程启动；
- fixture 文件加载；
- 输出文件序列化；
- benchmark CSV 写出。

若需要测量整个可执行文件从启动到结束的wall time，可使用：

```bash
/usr/bin/time -v \
  ./build/forward_reuse_integration/staged_lm_pipeline_reference_main \
  --fixture data_process/Vis/cpp_reference_dump_s100/staged_lm_pipeline/frame_000513/cpp_fixture \
  --output data_process/Vis/cpp_o4_s100_single_timed/frame_000513 \
  --kernel-policy optimized_native_stable_p
```

完整进程wall time不应与pipeline内部compute FPS混为一谈。

## 9. 多帧支持与当前限制

当前已经支持第一版多帧 staged-LM：Stage1 和 Pose 仍按帧独立执行，PoseShape
将全部帧组成一个联合系统，并共享一个 beta：

```text
[frame_0 75D, frame_1 75D, ..., frame_{T-1} 75D, shared beta 10D]
```

总参数维数为 `75*T+10`。详细运行方法见第 11 节。

当前多帧版本暂不包含 temporal pose smoothness、速度/加速度损失、跨帧 camera
loss、线程并行或 CUDA。除共享 beta 外，不同帧的 pose/camera block 之间没有直接
耦合。

另外，S100下O4与C++ scalar reference保持高度一致，但稀疏观测使最终LM参数对
跨backend浮点差异更加敏感。目前S100的C++与Python最终stage boundary尚未通过
原S1024 pipeline parity门禁，因此58.75 FPS是可信的C++性能结果，但在切换为正式
production observation前仍需完成full-10475 holdout和视觉质量评估。

## 10. 可视化 C++ fitting 结果

`debug030_vis_dual_erp.py` 读取标准的 SMPL-X sequence pickle，而 C++ 输出为裸
`.f32` 参数文件，因此需要先进行一次格式转换。完整流程为：

```text
C++ O4输出
  → debug083转换为cam2/cam4 SMPL-X pickle
  → debug030生成双相机ERP叠图
```

### 10.1 运行 C++ O4

以 frame 514、S100 为例：

```bash
./build/forward_reuse_integration/staged_lm_pipeline_reference_main \
  --fixture data_process/Vis/cpp_reference_dump_s100/staged_lm_pipeline/frame_000514/cpp_fixture \
  --output data_process/Vis/cpp_o4_s100/frame_000514 \
  --kernel-policy optimized_native_stable_p
```

C++ 输出目录必须包含：

```text
pose_shape_final_parameters.f32
final_physical_beta.f32
```

### 10.2 转换为标准 SMPL-X pickle

```bash
conda run -n camerahmr python \
  src/script/test_function/debug083_export_cpp_staged_lm_vis.py \
  --cpp_output data_process/Vis/cpp_o4_s100/frame_000514 \
  --fixture data_process/Vis/cpp_reference_dump_s100/staged_lm_pipeline/frame_000514/cpp_fixture \
  --output_dir data_process/Vis/cpp_o4_s100_vis_input
```

转换脚本从 C++ 最终参数中读取 orient、translation、body pose 和 physical beta，
并从同一次运行的 fixture 恢复优化期间冻结的左右手姿态与 scale。输出为：

```text
data_process/Vis/cpp_o4_s100_vis_input/
├── cam2/seg_514_514_seq.pkl
└── cam4/seg_514_514_seq.pkl
```

### 10.3 生成双相机 ERP 叠图

```bash
conda run -n camerahmr python \
  src/script/test_function/debug030_vis_dual_erp.py \
  --result_dir data_process/Vis/cpp_o4_s100_vis_input \
  --segment 514_514 \
  --image_dir data_process/Output/20260311_riding/image/equirect_rotate \
  --output_dir data_process/Vis/cpp_o4_s100_vis/frame_000514 \
  --mesh \
  --scale 0.25
```

最终图片位于：

```text
data_process/Vis/cpp_o4_s100_vis/frame_000514/vis_dual_00514.jpg
```

如需同时保存视频，可在 `debug030` 命令中增加：

```text
--save_video --fps 30
```

## 11. 运行多帧 C++ staged-LM

### 11.1 编译多帧入口

```bash
/usr/bin/cmake -S cpp/staged_lm \
  -B build/forward_reuse_integration \
  -DSTAGED_LM_EIGEN_VECTORIZE=ON \
  -DSTAGED_LM_MARCH_NATIVE=ON

/usr/bin/cmake --build build/forward_reuse_integration \
  --target multiframe_staged_lm_reference_main \
  -j2
```

### 11.2 准备每帧 fixture

多帧入口复用单帧 fixture。需要先按照“生成其他帧的 C++ 输入”一节，为每一帧
分别生成 `cpp_fixture`。注意 bbox 必须来自对应数据集的：

```text
data_process/Output/<dataset>/image/equirect_crop_rotate
```

例如 513–516 帧应存在：

```text
data_process/Vis/cpp_reference_dump_s100/20260311_riding/staged_lm_pipeline/
├── frame_000513/cpp_fixture
├── frame_000514/cpp_fixture
├── frame_000515/cpp_fixture
└── frame_000516/cpp_fixture
```

### 11.3 运行 4 帧联合 fitting

`--fixtures` 接收按时间顺序排列、以逗号分隔的 fixture 路径。逗号后不要添加
空格：

```bash
./build/forward_reuse_integration/multiframe_staged_lm_reference_main \
  --fixtures data_process/Vis/cpp_reference_dump_s100/20260311_riding/staged_lm_pipeline/frame_000513/cpp_fixture,data_process/Vis/cpp_reference_dump_s100/20260311_riding/staged_lm_pipeline/frame_000514/cpp_fixture,data_process/Vis/cpp_reference_dump_s100/20260311_riding/staged_lm_pipeline/frame_000515/cpp_fixture,data_process/Vis/cpp_reference_dump_s100/20260311_riding/staged_lm_pipeline/frame_000516/cpp_fixture \
  --output data_process/Vis/cpp_o4_s100_multiframe/segment_513_516
```

成功时输出：

```text
MULTI_FRAME_STAGED_LM_READY=PASS frames=4 parameter_dimension=310
```

该入口固定使用当前 preferred O4 计算路径：

```text
optimized_native_stable_p
```

### 11.4 查看多帧结果

```bash
cat data_process/Vis/cpp_o4_s100_multiframe/segment_513_516/multiframe_summary.txt
```

摘要包含：

```text
frame_count
parameter_dimension
shared_beta_block_count
stage1_total_ms
pose_total_ms
pose_shape_total_ms
total_ms
ms_per_frame
```

每帧结果按输入顺序保存：

```text
segment_513_516/
├── frame_0/
│   ├── stage1_final_parameters.f32
│   ├── pose_final_parameters.f32
│   └── pose_shape_frame_parameters.f32
├── frame_1/
├── frame_2/
├── frame_3/
├── shared_beta_latent.f32
├── shared_physical_beta.f32
└── multiframe_summary.txt
```

其中每个 `pose_shape_frame_parameters.f32` 都是该帧最终的 75D camera/pose
参数；共享的 10D beta 不会复制到各帧目录，而是只保存一份。

### 11.5 参数维数示例

```text
T=1: 75*1+10 = 85
T=2: 75*2+10 = 160
T=4: 75*4+10 = 310
T=8: 75*8+10 = 610
```

Stage1、Pose 当前仍顺序逐帧运行。只有 PoseShape 使用全部帧联合优化，因此多帧
输出不能用若干次独立单帧 PoseShape 结果直接替代。
