# Panorama Stitch Lab

<sub>**语言 / Language：** 中文（本页）· [English](#english)</sub>

双相机鱼眼 / ERP 全景下的人体捕捉与跨相机合图实验工程（本科毕设）。链路从原始鱼眼视频一路做到「把 A 相机里的人搬进 B 相机画面」的合成图，中途把最贵的 SMPL-X 拟合部分用 C++ / CUDA 重写成可自测、可基准的 staged 求解器。

规模：417 个 Python 文件（约 72k 行）、48 个 `.cpp` + 25 个 `.h` + 2 个 `.cu`（约 10.7k 行）、95 篇 Markdown。

> 数据、模型权重和视频资产都不在仓库里（`data_process/`、`Circular_Video/`、`models/` 都在 `.gitignore`）。克隆下来能读代码、能跑 `--list_steps` 和 C++ 自测，但**跑完整流水线需要自备数据**。

---

## 这是什么

两条可以分开用的东西，通过 `data_process/` 下的中间产物交接，不互相 import。

**1. Python 主流程** —— `src/lib_pano/` + `src/script/test_function/run_pipeline.py`

14 个 step（`020`–`037`），按四段理解：

| 段 | 做什么 | 代表 step |
|---|---|---|
| 预处理 | 导帧、鱼眼转 ERP、review、bbox 检测与平滑、crop/rotate、姿态、SAM3 分割、mask fusion、HMR convert | `020` `021` `035` `040` `041` `025` `026` `027` `024` `038` |
| 重建 | 双 ERP SMPL-X fitting 与可视化 | `029` `030` |
| Composite 验证 | semantic flow + human composite，看跨相机 warp / 补全 / blend 的效果 | `031` `037` |
| 最终合成（方向） | dense optical flow / RAFT / warp / blend —— 最终结果更可能出自这里 | — |

模块都在 `src/lib_pano/`：`calibration` `camera` `composite` `correspondence` `flow` `fitting_io` `human_fitting` `human_vis` `human3d` `pipeline` `stitch`。

**2. Staged LM 拟合** —— `cpp/staged_lm/` + [src/README.md](src/README.md)

把 SMPL-X 拟合拆成三段并逐层手写解析 Jacobian：

1. Stage 1 刚体对齐（相机 global orientation / translation）
2. Stage 2 pose warmup（beta 固定）
3. Stage 2 joint refinement（逐帧 pose + 跨帧共享 bounded beta）

配套 11 个 C++ self-test（注册在 CTest 里）和若干 benchmark 可执行文件；Python 侧 68 个测试在 `tests/`。

## 怎么跑（最小可复现路径）

前提：Python 3.10+（**仓库不含 requirements，环境自建**）、Eigen ≥ 3.3（C++ 部分）、CUDA 可选。

```bash
# 1) 不需要数据就能跑：列出 14 个 step
python src/script/test_function/run_pipeline.py --list_steps

# 2) 主流程。配置文件是本地 JSON（示例不入库），先 --dry_run 看要执行什么
python src/script/test_function/run_pipeline.py \
  --config <your_config.json> \
  --steps 020,021,040,041,025,026,038,029,030,031,037 --dry_run
```

```bash
# 3) 构建 C++ staged LM（默认开 Eigen 向量化与 -march=native；CUDA 用 -DSTAGED_LM_ENABLE_CUDA=OFF 关）
cmake -S cpp/staged_lm -B cpp/staged_lm/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/staged_lm/build -j

# 4) 自测
ctest --test-dir cpp/staged_lm/build --output-on-failure
PYTHONPATH=src pytest -q src/lib_pano/fitting_io/tests src/lib_pano/human_fitting/tests
pytest -q tests/human_fitting/staged_lm
```

**预期产物**：主流程写到 `data_process/` 下按 step 命名的目录（`image/equirect_crop_rotate`、`hmr_cache`、`Vis/…` 等）；C++ 可执行文件落在 `cpp/staged_lm/build/<target>/` 下。

## 结果

三个数字，都在 [benchmarks/RESULTS.md](benchmarks/RESULTS.md) 里带来源和「待补」标注：

- C++ O4 单帧拟合 **5 FPS** → 100 点 loss 的 C++ 管线 **25–30 FPS**（2026-08-13 / 08-18 手测，硬件与命令当时没记）
- DenseKP 的 TensorRT 后端正确性测试通过（2026-08-31，**没有速度数字**）
- 单帧 staged-LM 归因实验：D138 129.37 ms（7.73 FPS）vs S138 23.81 ms（42.01 FPS）

最后一条是**归因对比**，不是端到端流水线速度；仓库里没有端到端帧率记录。

## 代码在哪

| 路径 | 内容 |
|---|---|
| `src/lib_pano/` | 公共实现（标定、相机、合图、对应、光流、拟合 IO、三维人体、stitch） |
| `src/script/test_function/` | 14 个 step 的脚本与 `run_pipeline.py` 编排入口 |
| `cpp/staged_lm/` | C++ / CUDA staged LM：`include/` `src/` `apps/` `tests/` `tools/` |
| `tests/` | Python 测试（staged LM 几何、雅可比、Schur 等） |
| `tools/` | 鱼眼取帧（`fisheye_extractor/`）与单帧拟合（`SingleFitting/`） |
| `docs/` | 设计与运行文档，入口见 [docs/README.md](docs/README.md) |

## 文档

- [docs/README.md](docs/README.md)：文档索引与阅读顺序
- [docs/adr/](docs/adr/)：5 篇「为什么这么定」的短决策记录（ERP、C++ 重写、TensorRT、staged LM 公式化、DenseKP 生产交付）
- [src/README.md](src/README.md)：staged LM 使用说明（输入输出、命令、性能口径）
- `docs/Architecture/formulate/`：仍然约束当前代码的设计文档
- `docs/Architecture/staged_lm/`：LM 公式推导（joint / pose 两条线）
- `docs/Architecture/archive/`：历史 bakeoff / attribution / profiling，只作追溯
- `docs/changelogs/`：2026-08-03 那次大改的记录

## 分支说明

默认分支是 **`ssh_main`**，这份 README 描述的就是它。

还有一个历史分支 **`legacy-2026-05`**：2026-05 的旧实现，138 个文件，与 `ssh_main` **没有共同祖先**（不是同一个起点的两条线，`git merge-base` 为空）。它保留下来只为追溯，**不要基于它继续开发**，也不要做合并尝试。

## 已知限制

- **数据不在仓库**：`data_process/`、`Circular_Video/`、`models/` 被忽略，任何「跑通」都需要自备；本机绝对路径散落在部分 debug 脚本与文档里。
- **没有依赖清单**：Python 依赖靠当时的 conda 环境，仓库内没有 `requirements.txt`。
- **只验证过双相机 ERP**（cam2 / cam4）：更多相机数、不同标定与连续视频的效果在文档里没有记录。
- **合图链不是最终版**：composite 阶段是效果验证，最终合成方向仍是 dense flow 那一路。

---

## English

A research-grade experiment pipeline for human capture and cross-camera stitching on dual-camera fisheye / ERP panorama (undergraduate thesis project). It goes from raw fisheye video to a composite where a person captured by one camera appears in another camera's frame, and in the middle it rewrites the expensive SMPL-X fitting stage in C++ / CUDA as a staged solver with self-tests and benchmarks.

Size: 417 Python files (~72k lines), 48 `.cpp` + 25 `.h` + 2 `.cu` (~10.7k lines), 95 Markdown files. Data, weights and video assets are **not** in the repo (`data_process/`, `Circular_Video/`, `models/` are gitignored), so cloning gets you readable code, `--list_steps`, and the C++ self-tests — running the full pipeline needs your own data.

**Two parts, usable separately**, connected only by intermediate files under `data_process/`:

1. **Python pipeline** — `src/lib_pano/` + `src/script/test_function/run_pipeline.py`, 14 steps (`020`–`037`) covering frame export, fisheye→ERP, bbox detect/smooth, crop/rotate, pose, SAM3 segmentation, mask fusion, HMR convert, dual-ERP SMPL-X fitting, semantic flow and human compositing.
2. **Staged LM fitting** — `cpp/staged_lm/` + [src/README.md](src/README.md): rigid alignment → pose warmup → joint pose+beta refinement, with analytic Jacobians per stage, 11 C++ self-tests under CTest, and benchmark executables.

```bash
python src/script/test_function/run_pipeline.py --list_steps
python src/script/test_function/run_pipeline.py --config <your_config.json> \
  --steps 020,021,040,041,025,026,038,029,030,031,037 --dry_run
cmake -S cpp/staged_lm -B cpp/staged_lm/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/staged_lm/build -j
ctest --test-dir cpp/staged_lm/build --output-on-failure
PYTHONPATH=src pytest -q src/lib_pano/fitting_io/tests src/lib_pano/human_fitting/tests
```

Headline numbers (sources and open gaps in [benchmarks/RESULTS.md](benchmarks/RESULTS.md)): C++ O4 single-frame fitting 5 FPS → 25–30 FPS for the 100-point-loss C++ pipeline (hand-measured 2026-08-13 / 08-18, hardware and command not recorded); the DenseKP TensorRT backend passes its correctness test (no speed figure); a single-frame attribution run measures D138 at 129.37 ms (7.73 FPS) vs S138 at 23.81 ms (42.01 FPS) — that is a policy comparison, not end-to-end throughput.

Docs start at [docs/README.md](docs/README.md); decisions live in [docs/adr/](docs/adr/).
