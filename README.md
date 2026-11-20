# Panorama Stitch Lab

<sub>[中文](#这是什么) · [English](#english)</sub>

双相机鱼眼 / ERP 全景下的人体捕捉与跨相机合图（本科毕设）。链路从原始鱼眼视频走到「把 A 相机里的人搬进 B 相机画面」，中途把最贵的 SMPL-X 拟合用 C++ / CUDA 重写成可自测、可基准的 staged 求解器。

> 数据、权重和视频资产**不在仓库里**（`data_process/`、`Circular_Video/`、`models/` 都被忽略）。克隆下来能读代码、能跑 `--list_steps` 和 C++ 自测，但跑完整流水线要自备数据。

## 这是什么

两块可以分开用的东西，通过 `data_process/` 下的中间产物交接，不互相 import：

**① Python 主流程** —— `src/lib_pano/` + `src/script/test_function/run_pipeline.py`

14 个 step（`020`–`037`）：导帧、鱼眼转 ERP、bbox 检测与平滑、crop/rotate、姿态、SAM3 分割、mask fusion、HMR convert、双 ERP SMPL-X 拟合、semantic flow、人体合图。

**② Staged LM 拟合** —— `cpp/staged_lm/` + [src/README.md](src/README.md)

把拟合拆成「Stage 1 刚体对齐 → Stage 2 pose → 联合 pose+beta」三段，逐层手写解析 Jacobian，带 11 个 C++ self-test 和 benchmark。

规模：约 417 个 Python 文件（72k 行）+ 48 个 `.cpp` / 25 个 `.h` / 2 个 `.cu`（10.7k 行）。

## 怎么跑

前提：Python 3.10+（**仓库不含 requirements**）、Eigen ≥ 3.3、数据自备、CUDA 可选。

```bash
# 看有哪些 step（不需要数据）
python src/script/test_function/run_pipeline.py --list_steps

# 跑主流程，先 --dry_run 看要执行什么
python src/script/test_function/run_pipeline.py --config <your_config.json> \
  --steps 020,021,040,041,025,026,038,029,030,031,037 --dry_run

# 构建并测试 C++ staged LM
cmake -S cpp/staged_lm -B cpp/staged_lm/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/staged_lm/build -j
ctest --test-dir cpp/staged_lm/build --output-on-failure
PYTHONPATH=src pytest -q src/lib_pano/fitting_io/tests src/lib_pano/human_fitting/tests
```

## 结果

| 是什么 | 数字 | 出处 |
|---|---|---|
| C++ staged-LM 单帧拟合 | 5 FPS → **25–30 FPS**（RTX 3090） | [benchmarks/RESULTS.md](benchmarks/RESULTS.md)，点数与命令「待补」 |
| DenseKP 的 TensorRT 后端 | 正确性测试通过，**没有速度数字** | 同上 |
| 单帧归因对比 | D138 129.37 ms（7.73 FPS）vs S138 23.81 ms（42.01 FPS） | 同上，属策略对比不是端到端 |

## 文档

- [docs/README.md](docs/README.md)：文档索引与三种读法
- [docs/adr/](docs/adr/)：5 篇「为什么这么定」的短决策（ERP、C++ 重写、TensorRT、staged LM、DenseKP 交付）
- [src/README.md](src/README.md)：staged LM 的使用说明与性能口径
- `docs/Architecture/formulate/`：仍约束当前代码的设计；`archive/`：历史 bakeoff 与归因
- [PIPELINE_CONTRACT](https://github.com/casesic/mamma_retrain/blob/main/docs/PIPELINE_CONTRACT.md)：四个仓库怎么交接（只有一份，放在 mamma_retrain）

## 分支说明

默认分支是 **`ssh_main`**，这份 README 描述的就是它。另有一个历史分支 **`legacy-2026-05`**：2026-05 的旧实现，138 个文件，与 `ssh_main` **没有共同祖先**，只作追溯，不要基于它开发，也不要尝试合并。

## 已知限制

- 数据不在仓库，跑通要自备；部分 debug 脚本和旧文档里还留着本机绝对路径。
- 没有依赖清单，Python 依赖靠当时的 conda 环境。
- 只在双相机 ERP（cam2 / cam4）上验证过；更多相机数、不同标定与连续视频没有记录。
- composite 阶段是效果验证，最终合成方向（dense flow）还没定。

## English
{: #english}

Human capture and cross-camera compositing on dual-camera fisheye / ERP panorama (undergraduate thesis). Two parts, usable separately: a 14-step Python pipeline (`src/lib_pano/`, `run_pipeline.py`) and a C++/CUDA staged Levenberg–Marquardt SMPL-X fitter (`cpp/staged_lm/`) with 11 self-tests. ~72k lines of Python and ~10.7k lines of C++/CUDA.

Data, weights and footage are **not** in the repo, so cloning gets you readable code, `--list_steps` and the C++ tests; running the pipeline needs your own data. Headline numbers (sources and gaps in [benchmarks/RESULTS.md](benchmarks/RESULTS.md)): C++ single-frame fitting went from 5 FPS to **25–30 FPS** on an RTX 3090; the DenseKP TensorRT backend passes its correctness test with no speed figure recorded.

Docs start at [docs/README.md](docs/README.md); decisions are in [docs/adr/](docs/adr/). Default branch is `ssh_main`; `legacy-2026-05` holds the 2026-05 implementation and shares no history with it.
