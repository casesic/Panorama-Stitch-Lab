## 整体架构图整体分五层,从上到下依次是 **UI 层 → Controller 层 → Cache 层 → IO 层 → 外部资源**。控制流(实线)从用户按键开始向下传递,数据流(虚线)从视频文件向上传递到画面、从画面向下落盘到输出目录。下面分模块讲。

<img src="C:\Users\casesic\AppData\Roaming\Typora\typora-user-images\image-20260409171429706.png" alt="image-20260409171429706" style="zoom:50%;" />

## 一、MainWindow(UI 容器 + 键盘路由)

**职责**:接收所有用户输入(键盘、菜单、按钮),把它们翻译成对 Controller 的调用。本身不做业务逻辑,纯路由。

**关键设计**:
- 全局快捷键集中注册在这一层,不要散落到各子控件,否则焦点切换会丢事件。
- 按住 → 时,Qt 默认会 autoRepeat,要让它生效(不要在 ControlBar 的按钮上拦截键盘事件)。
- 菜单/按钮和快捷键都指向同一个 Controller 方法,不要写两份逻辑。

**对外接口**(伪代码):
```
on_key_right()       → controller.step(+1)
on_key_left()        → controller.step(-1)
on_key_shift_right() → controller.step(+step_size)
on_key_s()           → controller.save_current()
on_key_o()           → open_video_dialog()
on_key_d()           → choose_output_dir_dialog()
```

## 二、VideoCanvas(画面显示)

**职责**:把 Controller 给的一帧画到屏幕上,仅此而已。

**关键设计**:
- 接收 numpy 数组(RGB),内部转 QImage/QPixmap 显示。**转换时必须 `.copy()`**,否则 numpy buffer 被回收会导致花屏或崩溃。
- 显示前按画布尺寸 downscale,**保留原始帧**(给 FrameSaver 用,保存的是原图不是缩放图)。
- 维护一个 `current_frame` 引用,FrameSaver 直接从这里取。
- 可选:支持鼠标滚轮缩放、拖动平移,方便看细节。

**对外接口**:
```
show(frame_ndarray, frame_index)  # Controller 调用
get_current_frame() → ndarray     # FrameSaver 调用
```

## 三、ControlBar & StatusBar(辅助控件)

**ControlBar 职责**:进度滑块、播放/暂停按钮、跳帧步长输入框、打开视频按钮、选择输出目录按钮。

**StatusBar 职责**:显示当前帧号 / 总帧数、当前时间戳、保存状态提示("已保存 00042.png")、缓存命中率(调试用)。

**关键设计**:
- 滑块拖动要**节流**:拖动过程中只更新显示位置,不真的 seek;松手(`sliderReleased`)才调 `controller.goto()`。否则狂拖会让 Decoder 排队卡死。
- 滑块和 Controller 双向绑定,但要避免循环:Controller 通知 UI 时设个 flag,跳过滑块的 valueChanged 回调。

## 四、PlaybackController(核心逻辑)⭐

**职责**:整个应用的大脑。维护"当前帧号"这个唯一状态,把所有操作翻译成"我现在要第 N 帧",然后协调 Cache、Provider、Decoder 拿到这一帧,推给 Canvas。

**最关键的设计 —— 快速路径 vs 慢速路径分离**:

```
goto(target_idx):
    if target_idx == current:
        return                              # 啥也不用做
    
    # 路径 A:命中缓存(最快, ~0ms)
    if target_idx in cache:
        frame = cache.get(target_idx)
    
    # 路径 B:顺序下一帧(很快, ~5ms)
    elif target_idx == current + 1:
        frame = decoder.read_next()
        cache.put(target_idx, frame)
    
    # 路径 C:跳帧(慢, ~50-200ms)
    else:
        frame = decoder.seek_and_read(target_idx)
        cache.put(target_idx, frame)
    
    canvas.show(frame, target_idx)
    current = target_idx
    provider.request_prefetch_around(target_idx)   # 通知预取
```

为什么这么分?**逐帧 next 比 seek 快几十倍**,你按住 → 时几乎全走路径 A 和 B,所以才有"刷过去"的体验。

**其他职责**:
- 维护 step_size(跳帧步长)、output_dir 等设置
- 提供 `save_current()` 给保存功能用
- 处理边界:`target_idx < 0` clamp 到 0,`>= total` clamp 到 total-1

## 五、FrameCache(LRU 缓存)

**职责**:存最近用到的帧,key 是帧号,value 是 numpy 数组。

**关键设计**:
- 容量按内存定:1080p 一帧约 6MB,120 帧约 720MB。可配置。
- **淘汰策略不用标准 LRU,用"距离当前帧最远"策略**:你来回看的帧不会被误删,真正被淘汰的是离当前帧最远的(很可能不再访问)。
- 必须线程安全:Provider 线程写,Controller 线程读。一把锁就够了,操作很短。

**对外接口**:
```
get(idx) → frame or None
put(idx, frame)
contains(idx) → bool
clear()
```

## 六、FrameProvider(后台预取)⭐

**职责**:在后台线程预测你下一步要看的帧,提前解码塞进 Cache。

**关键设计**:
- **单线程**就够,不要开多个 decoder(PyAV 的 decoder 不是线程安全的,而且 seek 串行更稳)。
- 维护一个**预取请求队列**,Controller 每次 goto 后调用 `request_prefetch_around(idx)`,把 idx 周围窗口内未缓存的帧号塞进队列。
- **窗口非对称**:`[idx-5, idx+30]`,前向多预取(因为你大概率往前看)。
- **过期请求要丢弃**:worker 取出任务时,如果发现帧号离 `current` 太远(比如 >60),直接跳过。

```
worker_loop:
    while running:
        target = queue.get()
        if abs(target - controller.current) > 60:
            continue                        # 过期了,丢
        if target in cache:
            continue                        # 已经有了
        frame = decoder.read(target)        # 注意:和主线程共用 decoder 要加锁
        cache.put(target, frame)
```

**注意**:Provider 和 Controller 共用一个 Decoder,所以 Decoder 内部得加锁,或者干脆 Provider 用一个独立的 Decoder 实例(打开同一个文件,各自 seek)。后者更简单,推荐。

## 七、VideoDecoder(解码封装)⭐

**职责**:封装 PyAV(或 OpenCV)的细节,对外只暴露三个方法。

**对外接口**:
```
open(path)
read_next() → frame              # 顺序读,不 seek,最快
seek_and_read(idx) → frame       # 精确跳到指定帧
get_info() → {fps, total, w, h}
```

**关键设计 —— PyAV 精确 seek**:
1. 把帧号转 PTS:`target_pts = int(frame_idx / fps / time_base)`
2. `container.seek(target_pts, backward=True, any_frame=False)` 跳到目标之前最近的关键帧
3. 从关键帧开始 decode,丢弃前面的帧,直到解出 `frame_idx >= target` 的那一帧
4. seek 之后**必须重建 decode 迭代器**,否则可能拿到旧数据

**变帧率(VFR)处理**:打开视频时后台扫描一遍 packet,建一个 `frame_idx → pts` 索引表,后续 seek 直接查表,100% 准确。1 小时视频扫描约 2-5 秒,可以加进度条。

## 八、FrameSaver(异步保存)

**职责**:把指定帧写成 PNG 文件,文件名 `{idx:05d}.png`。

**关键设计**:
- **必须异步**:写盘到线程池,主线程不等。否则你按 S 时会卡一下,影响快速浏览节奏。
- **去重**:同一帧号已经存过就跳过(或覆盖,看你需求)。
- **文件名冲突**:理论上 `{idx:05d}.png` 唯一,但用户可能切换过视频。可以加个开关:"输出到 `{output_dir}/{video_stem}/00042.png`" 按视频名分子目录。
- **完成回调**:写完通知 StatusBar 显示"已保存"。

```
save_async(frame, idx, out_dir):
    filename = f"{idx:05d}.png"
    path = out_dir / filename
    thread_pool.submit(_write, frame, path, on_done=notify_status)

_write(frame, path):
    cv2.imwrite(str(path), cv2.cvtColor(frame, BGR))   # 注意通道顺序
```

## 九、典型数据流串一遍

**场景:你按住 → 连续看 50 帧**

1. Qt 触发 `on_key_right()` × 50 次
2. 每次 MainWindow 调 `controller.step(+1)` → `controller.goto(current+1)`
3. 第 1-5 次:走路径 B(read_next),很快
4. 第 6 次开始:Provider 已经把 6-30 帧预取好了,后续走路径 A(命中缓存),0ms
5. 每次 goto 后 Provider 收到新的 `request_prefetch_around`,继续向前预取
6. Canvas 刷新画面,StatusBar 更新帧号

**场景:看到合适的画面按 S**

1. MainWindow 调 `controller.save_current()`
2. Controller 从 Canvas 拿原始帧 + 当前帧号,调 `saver.save_async()`
3. Saver 提交到线程池立即返回,UI 不卡
4. 写盘完成后回调 StatusBar 显示"已保存 00042.png"

---

这套架构的精髓就是 **"主线程同步查缓存,后台线程异步预取,Decoder 区分顺序读和跳读"**。
