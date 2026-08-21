# DenseKP production runtime profiling

## Profiling 开关与输出

DenseKP delivery 增加可选参数：

```text
--profile_runtime
```

单独运行 delivery 时输出：

```text
result/densekp_runtime_profile.json
```

`debug078_densekp_fitted_cpw_composite.py` 可通过
`--densekp_profile_json` 读取该文件，并合并 input exporter、C++ LM 和 CPW
计时，最终写到：

```text
frame_xxxx/densekp_runtime_profile.json
```

计时代码只包围原有调用，没有改变 DenseKP、坐标变换、LM 或 CPW 算法。

## 513 帧 cache-hit 结果

### DenseKP delivery

| 子阶段 | 耗时 |
|---|---:|
| LUT cache load | 1.66 ms |
| valid-mask cache load | 116.21 ms |
| 双鱼眼图片读取 | 110.62 ms |
| ERP remap | 49.87 ms |
| DenseKP ORT session creation | 2096.05 ms |
| detector initialization | 198.54 ms |
| detector inference（两相机合计） | 129.37 ms |
| DenseKP inference（两相机合计） | 506.69 ms |
| DenseKP preprocess | 2.87 ms |
| DenseKP/delivery postprocess | 3.76 ms |
| ERP→ERP_ROTATE 点变换 | 0.23 ms |
| observation JSON export | 1.36 ms |
| delivery 内部总计 | 3229.77 ms |

外部 `conda run` wall time 为约 `6.29 s`。其中约 `3.06 s` 发生在
`run_pipeline()` 之前，主要是新进程、Conda/Python 启动和模块导入，不属于
上述函数内计时。

### C++ LM

| 子阶段 | 耗时 |
|---|---:|
| runtime observation exporter | 8.02 ms |
| C++ LM + optimized DenseKP output | 65.40 ms |

### CPW

| 子阶段 | 耗时 |
|---|---:|
| point/image/mask preparation | 1304.25 ms |
| warp-map construction | 2750.18 ms |
| interpolation | 9.89 ms |
| blending/debug panel | 61.20 ms |
| output save | 87.82 ms |
| CPW total | 4216.94 ms |

## Session 生命周期

当前 `cli.py` 每处理一帧都会调用一次 `run_pipeline()`，每次都会新建一个
`ORTRunner`：

- DenseKP ORT session：每次 invocation 创建 1 次，cam2/cam4 共用；
- detector：每次 invocation 初始化 1 次，cam2/cam4 共用；
- DenseKP ONNX 权重：随 ORT session 每次重新加载；
- detector 权重：随 detector 每次 invocation 重新加载；
- 下一帧若再次启动 CLI，新进程不会复用上述 session/model。

实测 `session_creation_count=1`、`detector_initialization_count=1`，但这是单帧
进程内共享，不是跨帧共享。

## 剩余瓶颈

1. CPW warp-map：`2.75 s`，是 CPW 的主要成本。
2. 每帧 DenseKP ORT session creation：`2.10 s`，占 delivery 内部时间约 65%。
3. CPW point/image/mask preparation：`1.30 s`。
4. DenseKP inference：`0.51 s`。
5. 逐帧独立进程/Conda/import：约 `3.06 s`，不在 `run_pipeline` 内部计时中。

ERP remap、138 点坐标变换、JSON export 和 C++ LM 都不是当前主要瓶颈。
下一轮如果优化，应优先验证跨帧常驻 ORT/detector session，以及 CPW warp map
和 valid-mask/image preparation；本轮没有进行这些优化。

后续已加入 correspondence-dependent CPW warp-map cache。同一组点重复运行时，
warp-map 从约 `2722 ms` 降到约 `4.96 ms` cache load，CPW 总计从约
`4203 ms` 降到约 `1488 ms`。不同帧点对变化时不会错误复用。

## Correctness

- profiling 前后 DenseKP observation JSON bitwise identical；
- optimized DenseKP JSON bitwise identical；
- CPW result JPEG bitwise identical；
- C++ regression 11/11 PASS。

```text
DENSEKP_RUNTIME_PROFILING_READY=PASS
```
