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


### src/tools中脚本的解释
video_crop.sh: 根据给定宽高和左上角位移裁剪视频. 
`./src/tools/video_crop.sh input.mp4 640 480 0 0`
video_scale.sh: 按给定宽高缩放视频
`./src/tools/video_scale.sh input.mp4 1280 720`
video_select.sh: 按帧号区间抽取片段
`./src/tools/video_select.sh input.mp4 100 200`
video_snap.sh: 抽一帧成 PNG（单帧截图）
`./src/tools/video_snap.sh Circular_Video/00_send/20251016/scene_1/origin_1.mp4 1`
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

### 计划
1. 了解畸变的原理 √
2. 去opencv网站上了解从cv.remap和cv.undistort函数 √
3. 写map1,map2,remap代替undistor函数  √