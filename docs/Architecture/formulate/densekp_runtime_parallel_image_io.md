# DenseKP runtime parallel image I/O

## 原路径与细分计时

原 production 路径对 cam2、cam4 依次调用 `cv2.imread`，两个 JPEG 串行完成文件
读取和解码，pipeline 没有额外的 NumPy image copy。

新路径将读取拆成：

```text
Path.read_bytes -> np.frombuffer (zero-copy view) -> cv2.imdecode
```

并通过 `DenseKPRuntime` 生命周期内的两个 worker 并行读取 cam2/cam4。runtime
关闭时销毁线程池。不改变输入格式，也没有引入视频解码器或 image cache。

## 1856–1928 实测

| 指标 | before | after |
|---|---:|---:|
| 双图 read wall | 79.135 ms/frame | 58.079 ms/frame |
| image-read speedup | 1.00x | 1.363x |
| image-read reduction | - | 26.61% |
| steady-state | 251.888 ms/frame | 228.513 ms/frame |
| steady-state FPS | 3.970 | 4.376 |
| 全批次 FPS（不含 startup） | 3.497 | 3.809 |

after 的细分均值：

```text
cam2 task          46.999 ms
cam4 task          51.025 ms
pair wall          58.079 ms
file read sum       2.131 ms
JPEG decode sum    95.875 ms
NumPy copy          0.000 ms
```

并行收益为 26.61%，高于 10% 保留门槛。当前主要成本是 JPEG decode；两个 task
并行后 pair wall 接近较慢相机的 decode 时间。

## 正确性与范围

- cam2 image `max_abs_diff=0`；
- cam4 image `max_abs_diff=0`；
- 同进程复跑 DenseKP output bitwise identical；
- optimized DenseKP bitwise identical；
- C++ LM trajectory exact；
- ERP、坐标转换、DenseKP、C++ LM 和输出格式均未修改。

没有增加可选 frame image cache：真实 production 遍次中每个 image path 只解码
一次，不存在可消除的重复 decode；增加缓存只会提高内存占用和失效管理复杂度。

`DENSEKP_RUNTIME_IMAGE_IO_READY=PASS`
