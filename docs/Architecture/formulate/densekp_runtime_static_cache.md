# DenseKP runtime static cache

## 设计

`DenseKPRuntime` 以完整 calibration specification 为 key，在进程生命周期内持有：

- cam2/cam4 ERP LUT；
- cam2/cam4 valid mask；
- calibration specification；
- LUT 所表达的静态 ERP geometry。

第一帧仍从跨进程 calibration cache 读取；后续帧直接复用内存对象。相机参数、
输入尺寸或 ERP 尺寸变化会产生新的 specification key，不会错误复用旧数据。
runtime `close()` 时释放这些引用。

## 1856–1928 实测

检测保持在 equirect 图像上。73 帧中 70 帧进入 LM，1856、1861、1865 因 cam4
无检测而明确标记并跳过。

| 指标 | before | after |
|---|---:|---:|
| LUT 磁盘读取 | 73 | 1 |
| LUT 内存命中 | 0 | 72 |
| valid mask 磁盘读取 | 73 | 1 |
| valid mask 内存命中 | 0 | 72 |
| LUT load | 1.075 ms/frame | 0.025 ms/frame |
| valid mask load | 106.326 ms/frame | 0.095 ms/frame |
| image read | 138.390 ms/frame | 79.135 ms/frame |
| steady-state | 427.353 ms/frame | 251.888 ms/frame |
| steady-state FPS | 2.340 | 3.970 |
| 全批次 FPS（不含 startup） | 2.181 | 3.497 |

稳态 FPS 提升约 `69.7%`。image read 本轮没有修改；其耗时变化属于不同运行间的
OS page-cache/I/O 波动，不计作静态 cache 的直接收益。

## 正确性

- 独立重载的 LUT：bitwise exact；
- 独立重载的 valid mask：bitwise exact；
- 相同 fisheye 输入生成的 ERP：bitwise exact；
- 优化后同进程完整复跑：DenseKP output、optimized DenseKP 和 LM trajectory exact；
- 坐标转换、DenseKP、LM、loss、Jacobian、solver 和 tolerance 均未修改。

不同 ORT/detector 进程之间本身存在检测框的非确定性，因此不使用两次独立进程的
JSON 文件哈希来归因 static cache；采用同进程复跑和静态资产/ERP bitwise parity
隔离验证 cache 行为。

`DENSEKP_RUNTIME_STATIC_CACHE_READY=PASS`
