### 畸变
#### 标定畸变的原理
棋盘标定法：有M张图片(每张图片有单独的外参)，每张图片上N个角点，所以独立未知参数9+6M个，提供2MN个方程   
1.已知输入：   
(1) 角点世界坐标$P_w(X_i, Y_i, 0)^T$ 棋盘格上N个角点
(2)每张图片角点的实际像素坐标$ p_{ij} = (u_{ij},v_{ij})^T$    
2.待求参数：   
(1)内参矩阵$K$($f_x,f_y,c_x,c_y$ )     
(2)畸变系数(径向$k_1,k_2,k_3$,切向$p_1,p_2$)    
(3)外参：$R=[r_1,r_2,r_3]$可以表示世界坐标系x,y,z轴在相机坐标系中的向量,   $T= t_1,t_2,t_3$  
3.标定逻辑：
(1)第一阶段：忽略畸变，线性求解内参和外参    
$$
P_C = X \cdot r_1 + Y \cdot r_2 + t = X \cdot \begin{bmatrix} r_{11} \\ r_{21} \\ r_{31} \end{bmatrix} + Y \cdot \begin{bmatrix} r_{12} \\ r_{22} \\ r_{32} \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \\ t_z \end{bmatrix}  \quad Z=0,省略r_3行
$$
 $H=K*[R:t]$为3*3矩阵，有效参数8个通过至少4个方程求解

(2)第二阶段：引入畸变，非线性优化所有参数
$$
\min_{K, \{R_i, t_i\}, k_1, k_2, k_3, p_1, p_2} \sum_{i=1}^{M} \sum_{j=1}^{N} \left[ (u_{ij} - \hat{u}_{ij})^2 + (v_{ij} - \hat{v}_{ij})^2 \right]
$$
$u_{ij}$是实际的像素值，$\hat{u}_{ij}$是用已知参数的估计值

用 $Levenberg-Marquardt$算法迭代优化参数，直到收敛

[相机校准](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)  
世界坐标->相机坐标->归一化坐标(/Zc)->畸变归一化坐标(应用畸变公式)->像素坐标(乘内参)

#### 畸变公式
畸变参数：$center(x,y), a, b, c, d, e, f$
$$
x_{\text{distorted}} = (x_{\text{undist}} - c_x) \cdot R(r^2) + T_x(r^2, x_{\text{undist}}, y_{\text{undist}}) + c_x
$$
$$
y_{\text{distorted}} = (y_{\text{undist}} - c_y) \cdot R(r^2) + T_y(r^2, x_{\text{undist}}, y_{\text{undist}}) + c_y
$$
$c_x,c_y$畸变中心的`归一化`坐标，   
$r^2$无畸变坐标相对于畸变中心的径向距离平方, $r^2 = (x_{undist}-c_x)^2+(y_{undist}-c_y)^2$ ,    
$R(r)^2$径向畸变因子，
$R(r)^2 = 1 + ar^2 + br^4 +cr^6$
$T_x, T_y$切向畸变因子，f=0无额外项
$$
\begin{align*}
T_x &= 2d \cdot (x_{\text{undist}} - c_x) (y_{\text{undist}} - c_y) + e \cdot \left( r^2 + 2(x_{\text{undist}} - c_x)^2 \right) \\
T_y &= d \cdot \left( r^2 + 2(y_{\text{undist}} - c_y)^2 \right) + 2e \cdot (x_{\text{undist}} - c_x) (y_{\text{undist}} - c_y)
\end{align*}
$$


[项目提供的畸变公式](https://panotools.org/dersch/barrel/barrel.html)
$$
r_{src} = ( a * r_{dest}^3 + b * r_{dest}^2 + c * r_{dest} + d ) * r_{dest}
$$
#### 畸变的种类
主要都是径向畸变，分为桶形畸变和枕形畸变，是由于焦距的问题，而切向畸变是由于相机安装的问题一般可以规避   
[畸变和为什么有畸变](https://zhuanlan.zhihu.com/p/1920436280124211549)

### 外参
#### lon,lat,theta

定义世界坐标系，Z轴：指向上方, X轴：指向前方, Y轴：指向左方
lon:绕着Z轴向左旋转为正
lat:绕着Y轴向上旋转为证
theta:绕着X轴旋转

<img src="D:\Backup_Files\picture-storage\photos\三维旋转矩阵.png" alt="三维旋转矩阵" style="zoom:50%;" />

### 鱼眼相机

fov超过140度的为鱼眼镜头，针孔相机是相似性投影，使用达到180度的视角，图像会变为无穷大

<img src="D:\Backup_Files\picture-storage\photos\相似性投影.png" alt="相似性投影" style="zoom:50%;" />

#### 鱼眼相机的成像原理

鱼眼镜头设计为了更大的视场角，选择非相似性成像，引入畸变，出射角一定小于入射角

<img src="D:\Backup_Files\picture-storage\photos\非相似性投影.png" alt="非相似性投影" style="zoom:50%;" />



等距投影模型

$r_d$指图像上点到畸变中心的距离，$f$为鱼眼相机的焦距，$θ$是入射光线与鱼眼相机光轴之间

$r_d = f\theta$

<img src="D:\Backup_Files\picture-storage\photos\等距投影.png" alt="等距投影" style="zoom:50%;" />

等立体角投影

$r_d = 2fsin(\frac{\theta}{2})$

<img src="D:\Backup_Files\picture-storage\photos\等立体角投影.png" alt="等立体角投影" style="zoom:50%;" />

还有正交投影模型和体视投影模型，具体请参考[一文详解分析鱼眼相机投影成像模型和畸变模型](https://mp.weixin.qq.com/s?__biz=MzI4MDQ4NzQ5MQ==&mid=2247483973&idx=1&sn=2ca009ae8ba81c0b66bf3db9c603790b&chksm=ebb6f169dcc1787f11de8f61c538d55dc4023b85a204784a611a309fde31a286e8d491db3c5b&token=829651387&lang=zh_CN#rd)

#### 通用鱼眼相机模型-Kannala-Brandt 模型

鱼眼相机在生产过程中不可能完全精确按照投影模型，为了方便鱼眼相机的标定，根据以上的四种鱼眼相机投影模型，Kannala-Brandt提出了一种鱼眼相机的一般多项式近似模型

$\theta_d = k_0\theta+k_1\theta^3+k_2\theta^5+...$

#### 鱼眼相机成像过程

<img src="D:\Backup_Files\picture-storage\photos\坐标系.png" alt="坐标系" style="zoom:50%;" />

<img src="D:\Backup_Files\picture-storage\photos\鱼眼成像模型.png" alt="鱼眼成像模型" style="zoom:50%;" />

世界坐标系中一点$P_w=(X_w,Y_w,Z_w)$

1.世界坐标系—>相机坐标系$(X_c,Y_c,Z_c)$

$P_c=RP_w+T$

$P_c=(X_c,Y_c,Z_c)$

2.相机坐标系转归一化球面上 $(x_s,y_s,z_s)$

$x_s = X_c/Z_c \quad y_s=Y_c/Z_c \quad z_s=1$

3.计算$\theta$

$r_s=\sqrt{x_s^2+y_s^2}$

$tan\theta=r_s/z_s=r_s$

$\theta =arctan(r_s)$

4.利用KB模型修正$\theta$

因为镜头设计有误差

$\theta_d = \theta(1+k_1\theta^2+k_2\theta^4+k_3\theta^6+k_4\theta^8)$

5.计算该点在图像物理坐标系的中的坐标$(x_i,y_i)$

$r_i = f\theta_d$

$r_i/r_s = x_i/x_s$

$x_i=r_i*x_s/r_s=f\theta_d*x_s/r_s = \theta_d*x_s/r_s \quad f=1 \quad由于f在内参中已经被包含了$

$y_i=\theta_d*y_s/r_s$

6.->图像像素坐标系$(u,v)$

$K$中的$f_x=f/dx \quad f_y=f/dy$

$u=f_xx_i+c_x$

$v=f_yy_i+c_y$



#### 鱼眼图像投射到经纬度图像 

两个都是像素坐标系

经纬度图像的投影模型：水平方向x轴：对应球面的经度，范围 $[0,2\pi]$， 垂直方向y轴：对应球面的纬度，范围$[-\pi/2,\pi/2]$

1.经纬度图像像素坐标系$\to$球面三维向量

(1)像素$(i,j)$转为经纬度$(λ, φ)$

$\lambda = 2\pi - \frac{2\pi \times i}{W_{\text{out}}}$

$φ=\frac{\pi×j}{H_{out}}$ (0为北极，$\pi$为南极)

$(W_{out}, H_{out})$通常为2:1，匹配360*180度球面

(2)经纬度转为三维向量

<img src="D:\Backup_Files\picture-storage\photos\鱼眼成像模型.png" alt="鱼眼成像模型" style="zoom:50%;" />

$x_1 = \sinφ  \cos\lambda$

$y_1 = \sinφ  \sin\lambda$

$z_1 = \cosφ  $

- 当 φ=π/2（赤道），z1=0，x1 和 y1 构成水平平面
- 当 φ=0（北极），x1=y1=0，z1=1

2.球面三维向量旋转以对准鱼眼$(x_2,y_2,z_2)$

鱼眼是以光轴为中心的半球视场，经纬度图像 “以水平赤道为中心的 360° 视场”

<img src="D:\Backup_Files\picture-storage\photos\球面旋转.jpg" alt="球面旋转" style="zoom:50%;" />

绕 y 轴旋转 90°,然后仍需归一化确保$x_2^2+y_2^2+z_2^2=1$

$rot=\begin{bmatrix} cos\theta &0 &sin\theta \\ 0&1&0 \\ -sin\theta &0&con\theta \end{bmatrix}$

3.球面三维向量$\to$鱼眼图像坐标(u,v)

天顶角 $\theta = arccos(z_2) $

水平角 $\phi=arctan(y_2,x_2)$

当 $θ ≤ FOV/2$ 时，该向量对应的场景点才被鱼眼镜头捕获

$r = \text{radius} \times \frac{\theta_2}{\text{FOV}/2}$

$u = c_x + r \cdot \cos\phi_2$

$v = c_y + r \cdot \sin\phi_2$




### src/tools中脚本的解释
video_crop.sh: 根据给定宽高和左上角位移裁剪视频. 
`./src/tools/video_crop.sh input.mp4 640 480 0 0`
video_scale.sh: 按给定宽高缩放视频
`./src/tools/video_scale.sh input.mp4 1280 720`
video_select.sh: 按帧号区间抽取片段
`./src/tools/video_select.sh input.mp4 100 200`
video_snap.sh: 抽一帧成 PNG（单帧截图）
`./src/script/tools/video_snap.sh Circular_Video/00_send/20251016/scene_1/origin_1.mp4 1`
video_split.sh: 把输入的不同视频流与音频流拆成单独文件
`./src/tools/video_split.sh input.mp4`
video2png.sh: 按 start/step/end 的序列批量调用 snap（批量抽帧）。
`./src/tools/video2png.sh input.mp4 0 10 100`

### 代码函数的问题
1. `intrinsic_path = r"src\calibration\intrinsic.json"`r表示raw的意思，eg:r'\n'就表示\和n
2. remap(src, dst, map1, map2, interpolation) 
dst(x,y)=src(map_x(x,y),map_y(x,y)),
map1(x,y)=(x,y), map2(x,y)=empty OR map1(x,y)=x , map2(x,y)=y
3. numpy.ndarry.shape = (h, w)

### 注意
1. 做一些公式上的东西最好还是要查查资料，不要光靠AI
[OPENCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)    
[NUMPY](https://numpy.org/doc/stable/reference/index.html)    
2. http://172.23.66.245:10730/CodeOfPython

### 计划
1. 了解畸变的原理 √
2. 去opencv网站上了解从cv.remap和cv.undistort函数 √
3. 写map1,map2,remap代替undistor函数  √
4. 查看提供的warp_image里的像素误差 ×
5. 畸变参数理解有问题，换新的数据  ×
6. 新的数据fov计算有问题 ×
7. 查看kalibr ×
8. 外参可视化 z 轴朝前，x 轴朝右，y 轴朝下
9. 鱼眼投射到三维可视化
10. 明白问题在哪了,初始的视频给定的h和w和标定的是相反的
11. 事实上我可以从任意地方开出来 让lon=0对准[-pi, pi]之间的值
12. 给定的外参的理解不可能呈现preview的顺序，所以交换R[1]和R[2] 还是需要确认一下数据
12. 重叠处要选择哪块的像素
13. 相机光心投射到preview图要对的上
14. 归一化可能错误不应该用fx，要和xi有关
15. 去畸变投射经纬图要倒转图像(3840*2880) 
16. 左右移动图片PS得到合理
17. 无穷远处对齐了，近处的对齐(找简单的优化方法为了对比我们的)
18. 可以找两张图像对应的点看它们的变换
19. 主要就是R,T外参的问题, 如果要用到T得拿到真正的X,Y,Z世界坐标，而不是投射到球面上的
20. 投射到经纬图上cam4(229)和cam6(161)在Y轴上和cam2(225,157)相差了4个像素 有问题要推导公式 已知Xws,Yws,Zws,求 Xcs,Ycs,Zcs 但Xw = R Xc +T
21. 分别看正着和倒着经纬图上差几个像素，然后可以手动修一下,或者直接用特征点匹配来解决
22. 重写pano_stitch支持倒着和正着拼接
23. 需要知道d,相机距离地面的高度

24. 人像的问题，mask不掉人手上的手机，人像对不齐
### 临时数据
这里的都是经纬图被映射到右边的
cam24和cam26对齐
x1_rotate, y1_rotate, z1_rotate = -0.0253406929675796854, 2.1000942047035394, 0
x2_rotate, y2_rotate, z2_rotate = -0.017997000207730319, -2.0832903962207115,6.285002668279742

cam24和cam46对齐
x1_rotate, y1_rotate, z1_rotate = -0.0253406929675796854, 2.1000942047035394, 0
x2_rotate, y2_rotate, z2_rotate = -0.055997000207730319, -2.0632903962207115,6.285002668279742

cam26和46
x1_rotate, y1_rotate, z1_rotate = 0.0127, 2.0850942047035394, 0
x2_rotate, y2_rotate, z2_rotate = -0.017997000207730319, -2.0832903962207115,6.285002668279742

正常R
x1_rotate, y1_rotate, z1_rotate = 0.0033406929675796854, 2.0912942047035394, 6.287048634358602
x1_rotate, y1_rotate, z1_rotate = -0.012997000207730319, -2.0972903962207115, 6.285002668279742

x<0