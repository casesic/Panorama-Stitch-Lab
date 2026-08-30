# DenseKP staged-LM reader-focused production cleanup

## 范围

本轮只整理 DenseKP CUDA → C++ CUDA staged-LM 的阅读结构。Stage1、Pose、PoseShape、
LM、IRLS、Jacobian、normal equations、Schur、DenseKP regressor、ERP 数学、CUDA kernel
和 tolerance 均未修改。

## 调用链审计

- Production：端到端 runner、`DenseKPRuntime`、delivery pipeline、runtime fixture exporter、
  multi-frame staged-LM app、三个 stage 与 evaluator。
- Profiling：per-frame timing JSON、aggregate report 和 Markdown summary。
- Reference/parity：C++ reference apps 和固定 fixture；production runner 不执行重复 parity。
- Legacy/debug：NPZ、可视化和 CLI compatibility；`keypoint_value` 只在 legacy NPZ 中保存。
- Confirmed dead：两个空 C++ `experimental/` 目录、CLI 尾部过时命令块、一个未使用参数。

## 整理结果

- 新增公开 `DenseKPConfig`；CLI 只负责 argparse → config，production 不再构造 fake argv
  或调用私有 parser。
- DenseKP profiling writer 从 frame pipeline 迁到 `densekp_delivery/profiling.py`。
- 端到端 hash/profile/Markdown helper 迁到 `densekp_staged_lm_reporting.py`。
- `DenseKPRuntime` 按模型资源、image executor 和 resident geometry 生命周期标注。
- runtime exporter 明确区分 204 support vertices 与最多 138 observation rows。
- 更新 C++ reader guide，并新增完整端到端 reader guide。

行数：

| 文件 | before | after |
|---|---:|---:|
| `run_densekp_cuda_staged_lm.py` | 490 | 443 |
| `densekp_delivery/pipeline.py` | 590 | 532 |

迁出的 reporting/profiling 模块分别为 77/65 行；不是删除功能。

## Regression

- Python compileall/import/config smoke：PASS。
- DenseKP coordinate self-test：PASS。
- C++ Release clean build：PASS。
- C++ staged-LM self-tests：11/11 PASS。
- frame 513：DenseKP、optimized DenseKP、LM trajectory exact identical。
- 1856–1928：73 输入、70 valid；1856/1861/1865 仍因 cam4 无检测而 skip。
- 全部有效帧：DenseKP、optimized DenseKP、LM trajectory exact identical。
- steady-state：214.15 → 212.86 ms/frame；4.670 → 4.698 FPS，无回退。

```text
DENSEKP_STAGED_LM_READER_CLEANUP_READY=PASS
```
