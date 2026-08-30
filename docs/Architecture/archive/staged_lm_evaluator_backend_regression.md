# Staged-LM evaluator backend regression

## 测试配置

- fixture：DenseKP frame 513，双相机；
- 另以正式 S100 frame 514 fixture 做一次独立 exact 回归；
- build：Release、Eigen vectorization ON、`-march=native`、fast-math OFF、
  FP contraction OFF；
- policy：`optimized_native_stable_p`；
- 从同一 HEAD 源码、相同 CMake 选项分别构建 baseline/refactor，按奇偶轮换
  先后顺序，各运行 20 次；
- 全部 11 个 C++ self-test 通过。

## Exact correctness

以下文件在 baseline 与 refactor 间逐字节一致：

- `stage1_trajectory.csv`、`pose_trajectory.csv`、
  `pose_shape_trajectory.csv`；
- 三阶段 initial/final parameter FP32 文件；
- `final_physical_beta.f32`；
- `optimized_densekp.json`；
- `build_policy.txt`。

S100 frame 514 的三阶段 trajectory、三阶段 final parameters、physical beta 和
build policy 同样逐字节一致。

状态、accepted/rejected trial、linearization/objective call count、lambda、objective
均包含在 exact trajectory 和剔除 timing 字段后的 summary 对照中，没有发现第一处
分歧。计时字段因 wall clock 自然波动而不参与 correctness 比较。

```text
TRAJECTORY_EXACT=YES
FINAL_PARAMETERS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

## Performance

Pipeline compute time 中位数：

| 版本 | median compute | 相对 baseline |
|---|---:|---:|
| baseline | 53.195 ms | 1.000× |
| evaluator backend | 48.932 ms | 0.920× |

运行环境存在明显频率抖动，因此只把结果用于排除稳定回退，不把约 8.0% 的中位数
下降解释成接口优化收益。虚函数只位于每次 objective/linearization 的粗粒度边界，
不进入逐点、逐顶点或矩阵内循环。

## 结论

```text
CPU_EVALUATOR_BACKEND_READY=PASS
PERFORMANCE_REGRESSION=NO
```
