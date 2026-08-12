# Staged LM 各阶段计算公式与 Jacobian 构成

> 基于当前 `debug109_new` 代码整理。本文只描述当前 production 数学，不引入额外推导或算法修改。  
> 当前三阶段名称为：
>
> 1. **Stage 1 — Rigid**
> 2. **Stage 2 — Pose**
> 3. **Stage 2 — Pose/Shape**
>
> 第三阶段在旧文档中有时仍写作 Joint，但当前代码正式目录和 Stage 名称已经是 `pose_shape` / `STAGE2_POSE_SHAPE`。

---

## 0. 记号

设：

- $t=1,\dots,T$：frame；
- $c\in\{2,4\}$：cam2 / cam4；
- $i=1,\dots,N_{tc}$：该 frame-camera group 中的 selected SMPL-X vertex；
- $W,H$：ERP 图像宽、高；
- $\mathbf y_{tci}\in\mathbb R^2$：观测 2D 点；
- $\mathbf v_{tci}\in\mathbb R^3$：最终送入 ERP projection 的 3D vertex；
- $\boldsymbol\phi_{tc}\in\mathbb R^3$：camera-specific global orient（axis-angle）；
- $\mathbf t_{tc}\in\mathbb R^3$：camera-specific translation；
- $\boldsymbol\theta_t\in\mathbb R^{63}$：21 个 body joint 的 axis-angle；
- $\boldsymbol\beta\in\mathbb R^B$：physical shape，当前 $B=10$；
- $\mathbf z_\beta\in\mathbb R^B$：LM 实际优化的 beta latent；
- $s_{tc}$：冻结的 camera-specific scale；
- $J$：SMPL-X 总 joint 数，当前代码为 55；
- $\mathbf w_i\in\mathbb R^J$：vertex $i$ 的 LBS weights。

下面统一使用**列向量数学记号**。代码内部部分 tensor 使用 row-vector 表示，但数值含义一致。

---

# 1. 三个阶段的参数向量

## 1.1 Stage 1：Rigid

每帧 12 维：

$$
\mathbf x_t^{(1)}
=
\begin{bmatrix}
\boldsymbol\phi_{t2}\\
\mathbf t_{t2}\\
\boldsymbol\phi_{t4}\\
\mathbf t_{t4}
\end{bmatrix}
\in\mathbb R^{12}
$$

全 segment：

$$
\mathbf x^{(1)}
=
[\mathbf x_1^{(1)},\dots,\mathbf x_T^{(1)}]
\in\mathbb R^{12T}
$$

body pose、beta、hands、scale 均冻结。

对应代码：

- `parameter_layout.py`
- `analytic_stage1.py`
- `stage_definitions/rigid.py`

---

## 1.2 Stage 2：Pose

每帧 75 维：

$$
\mathbf x_t^{(2)}
=
\begin{bmatrix}
\boldsymbol\phi_{t2}\\
\mathbf t_{t2}\\
\boldsymbol\phi_{t4}\\
\mathbf t_{t4}\\
\boldsymbol\theta_t
\end{bmatrix}
\in\mathbb R^{75}
$$

其中：

$$
75 = 3+3+3+3+63
$$

全 segment：

$$
\mathbf x^{(2)}\in\mathbb R^{75T}
$$

此阶段：

- body pose $\boldsymbol\theta_t$ 可优化；
- physical beta 固定在 `beta_anchor`；
- hand pose 和 scale 冻结。

---

## 1.3 Stage 2：Pose/Shape

参数：

$$
\mathbf x^{(3)}
=
\begin{bmatrix}
\mathbf x_1^{(2)}\\
\vdots\\
\mathbf x_T^{(2)}\\
\mathbf z_\beta
\end{bmatrix}
\in\mathbb R^{75T+B}
$$

当前：

$$
B=10
$$

因此同时优化：

- 每帧两个 camera 的 orient / translation；
- 每帧共享 body pose；
- 整个 segment 共享一个 shape $\boldsymbol\beta$。

---

# 2. 公共 SO(3) 公式

代码：

`geometry/so3.py`

令 axis-angle：

$$
\boldsymbol\phi\in\mathbb R^3,\qquad
\theta=\|\boldsymbol\phi\|
$$

定义：

$$
[\boldsymbol\phi]_\times
=
\begin{bmatrix}
0&-\phi_z&\phi_y\\
\phi_z&0&-\phi_x\\
-\phi_y&\phi_x&0
\end{bmatrix}
$$

Rodrigues：

$$
R(\boldsymbol\phi)
=
I
+
\frac{\sin\theta}{\theta}[\boldsymbol\phi]_\times
+
\frac{1-\cos\theta}{\theta^2}[\boldsymbol\phi]_\times^2
$$

代码对 $\theta\to0$ 使用 Taylor 展开。

SO(3) right Jacobian：

$$
J_r(\boldsymbol\phi)
=
I
-
\frac{1-\cos\theta}{\theta^2}
[\boldsymbol\phi]_\times
+
\frac{\theta-\sin\theta}{\theta^3}
[\boldsymbol\phi]_\times^2
$$

对一个局部点 $\mathbf p$：

$$
\boxed{
\frac{\partial (R\mathbf p)}
{\partial\boldsymbol\phi}
=
-R[\mathbf p]_\times J_r(\boldsymbol\phi)
}
$$

这是 Stage 1 rigid orient Jacobian 的直接核心公式。

对旋转矩阵本身，第 $k$ 个 axis-angle 参数：

$$
\boxed{
\frac{\partial R}{\partial\phi_k}
=
R\,[J_r(\boldsymbol\phi)\mathbf e_k]_\times
}
$$

这条形式用于 Pose / Pose-Shape 中传播每个 joint rotation matrix 的导数。

---

# 3. 公共 ERP projection 与 Jacobian

代码：

`geometry/erp.py`

首先有一个固定坐标旋转：

$$
\tilde{\mathbf v}
=
Q\mathbf v
=
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
$$

其中 $Q$ 对应代码中的 `ROTATION_MATRIX`。

定义：

$$
h^2=x^2+z^2,\qquad
h=\sqrt{x^2+z^2}
$$

$$
r^2=x^2+y^2+z^2,\qquad
r=\sqrt{r^2}
$$

经纬角：

$$
\lambda=\operatorname{atan2}(x,z)
$$

$$
\varphi=\arcsin\left(-\frac{y}{r}\right)
$$

ERP pixel：

$$
\boxed{
u=
W\left(
\frac{\lambda}{2\pi}
+\frac12
\right)
}
$$

$$
\boxed{
v=
H\left(
\frac12-\frac{\varphi}{\pi}
\right)
}
$$

因此：

$$
\pi(\mathbf v)
=
\begin{bmatrix}
u\\v
\end{bmatrix}
$$

## 3.1 ERP 对 camera point 的 Jacobian

$$
\frac{\partial u}{\partial[x,y,z]}
=
\frac{W}{2\pi}
\begin{bmatrix}
\dfrac{z}{x^2+z^2}
&
0
&
-\dfrac{x}{x^2+z^2}
\end{bmatrix}
$$

$$
\frac{\partial v}{\partial[x,y,z]}
=
\frac{H}{\pi}
\begin{bmatrix}
-\dfrac{yx}{h\,r^2}
&
\dfrac{h}{r^2}
&
-\dfrac{yz}{h\,r^2}
\end{bmatrix}
$$

记：

$$
J_{\rm ERP}(\tilde{\mathbf v})
=
\frac{\partial(u,v)}
{\partial(x,y,z)}
\in\mathbb R^{2\times3}
$$

考虑固定旋转 $Q$：

$$
\boxed{
J_\pi(\mathbf v)
=
J_{\rm ERP}(Q\mathbf v)\,Q
}
$$

这就是后续所有 Stage 最外层 projection Jacobian。

> 代码中还包含 `clamp_min`、latitude clamp 和无效 latitude 区域置零等数值保护；上式写的是有效区域内的解析公式。

---

# 4. 公共 reprojection residual 与 IRLS

每个点的 raw residual：

$$
\boxed{
\mathbf e_{tci}
=
\pi(\mathbf v_{tci})-\mathbf y_{tci}
\in\mathbb R^2
}
$$

整个 Jacobian 最终都是：

$$
\boxed{
J_{tci}
=
J_\pi(\mathbf v_{tci})
\frac{\partial\mathbf v_{tci}}
{\partial\mathbf x}
}
$$

因此真正的难点就是：

$$
\frac{\partial\mathbf v}{\partial\mathbf x}
$$

在三个 Stage 中分别是什么。

---

## 4.1 Charbonnier true objective

代码：

`irls.py`

每个 group $g=(t,c)$：

$$
\alpha_g
=
\frac{1}
{N_{\text{frames},c}\,N_g}
$$

当前 `camera_weight = 1`。

Charbonnier：

$$
\rho_i
=
\sqrt{
\|\mathbf e_i\|^2+\epsilon^2
}
-\epsilon
$$

true reprojection objective：

$$
\boxed{
F_{\rm reproj}
=
\sum_g
\alpha_g
\sum_{i\in g}
\rho_i
}
$$

IRLS 在当前 linearisation point 冻结：

$$
\boxed{
w_i
=
\frac{\alpha_g}
{\sqrt{
\|\mathbf e_i\|^2+\epsilon^2
}}
}
$$

每个点的 $u,v$ 两行使用相同 weight。

定义 row scale：

$$
q_i=\sqrt{w_i}
$$

则 surrogate residual：

$$
\tilde{\mathbf e}_i
=
q_i\mathbf e_i
$$

surrogate Jacobian：

$$
\boxed{
\tilde J_i
=
q_i J_i
}
$$

---

## 4.2 Raw L2 模式

代码中：

$$
s_{\rm rep}
=
\frac{1}
{\sigma_{\rm px}\sqrt{N_{\rm total}}}
$$

默认：

$$
\sigma_{\rm px}=10
$$

于是：

$$
\tilde{\mathbf e}
=
s_{\rm rep}\mathbf e
$$

$$
\tilde J
=
s_{\rm rep}J
$$

true objective：

$$
F_{\rm reproj}
=
\frac12
\|\tilde{\mathbf e}\|^2
$$

---

# 5. Stage 1 — Rigid

代码：

- `analytic_stage1.py`
- `stage_definitions/rigid.py`

---

## 5.1 Stage 1 的 vertex：$\mathbf v=\ ?$

Stage 1 cache 先在：

- global orient = 0；
- translation = 0；
- body pose = anchor；
- beta = anchor；
- hands = frozen；
- scale = frozen；

的条件下得到 selected vertex。

对一个 cached point：

- $\mathbf l_i$：`local_vertices`
- $\mathbf c$：`rotation_center`

生产公式：

$$
\boxed{
\mathbf v_i
=
R(\boldsymbol\phi)\mathbf l_i
+
\mathbf c
+
\mathbf t
}
$$

对应：

```python
points =
    local @ rotation.T
    + rotation_center
    + translation
```

---

## 5.2 Stage 1 vertex Jacobian

对 orient：

$$
\boxed{
J_{\mathbf v,\phi}
=
\frac{\partial\mathbf v}
{\partial\boldsymbol\phi}
=
-R[\mathbf l_i]_\times J_r(\boldsymbol\phi)
}
$$

shape：

$$
3\times3
$$

对 translation：

$$
\boxed{
J_{\mathbf v,t}
=
I_3
}
$$

因此 camera 的 3D Jacobian：

$$
\frac{\partial\mathbf v}
{\partial[\boldsymbol\phi,\mathbf t]}
=
\begin{bmatrix}
-R[\mathbf l_i]_\times J_r(\boldsymbol\phi)
&
I_3
\end{bmatrix}
\in\mathbb R^{3\times6}
$$

---

## 5.3 Stage 1 2D Jacobian

使用 chain rule：

$$
J_{\phi,i}
=
J_\pi(\mathbf v_i)
\left(
-R[\mathbf l_i]_\times J_r(\boldsymbol\phi)
\right)
$$

$$
J_{t,i}
=
J_\pi(\mathbf v_i)
$$

因此单 camera：

$$
\boxed{
J_i^{\rm cam}
=
\begin{bmatrix}
J_{\phi,i}
&
J_{t,i}
\end{bmatrix}
\in\mathbb R^{2\times6}
}
$$

---

## 5.4 Stage 1 每个 Jacobian column 具体是什么

每帧：

$$
[
\phi_{2x},
\phi_{2y},
\phi_{2z},
t_{2x},
t_{2y},
t_{2z},
\phi_{4x},
\phi_{4y},
\phi_{4z},
t_{4x},
t_{4y},
t_{4z}
]
$$

### cam2 residual row

$$
\boxed{
J_{t,cam2,i}
=
\begin{bmatrix}
J_{\pi}J_{\mathbf v,\phi_2}
&
J_\pi
&
0_{2\times3}
&
0_{2\times3}
\end{bmatrix}
}
$$

即：

| column | 公式 |
|---|---|
| 0:3 cam2 orient | $J_\pi[-R_2[\mathbf l]_\times J_r(\phi_2)]$ |
| 3:6 cam2 translation | $J_\pi$ |
| 6:9 cam4 orient | 0 |
| 9:12 cam4 translation | 0 |

### cam4 residual row

$$
\boxed{
J_{t,cam4,i}
=
\begin{bmatrix}
0
&
0
&
J_{\pi}J_{\mathbf v,\phi_4}
&
J_\pi
\end{bmatrix}
}
$$

---

## 5.5 Stage 1 H / g

加权后：

$$
\tilde{\mathbf r}_t
=
\begin{bmatrix}
\tilde{\mathbf r}_{t2}\\
\tilde{\mathbf r}_{t4}
\end{bmatrix}
$$

$$
\tilde J_t
=
\begin{bmatrix}
\tilde J_{t2}\\
\tilde J_{t4}
\end{bmatrix}
$$

每帧独立：

$$
\boxed{
H_t=\tilde J_t^\top\tilde J_t
\in\mathbb R^{12\times12}
}
$$

$$
\boxed{
\mathbf g_t
=
\tilde J_t^\top\tilde{\mathbf r}_t
\in\mathbb R^{12}
}
$$

不同 frame 没有共享参数，所以全局：

$$
H
=
\operatorname{blkdiag}(H_1,\dots,H_T)
$$

---

# 6. Stage 2 — Pose

代码：

- `pose/cache.py`
- `pose/forward.py`
- `pose/derivatives.py`
- production batch：`pose/batched.py`

此阶段 beta 固定。

---

## 6.1 Frozen shaped vertex

固定 physical beta：

$$
\boldsymbol\beta
=
\boldsymbol\beta_{\rm anchor}
$$

SMPL-X shaped mesh：

$$
\boxed{
\mathbf v_i^{shape}
=
\bar{\mathbf v}_i
+
\sum_b
S_{ib}\beta_b
}
$$

Stage 2 Pose 中该值在 LM 内不变，因此被 cache。

joints 同样固定：

$$
\boxed{
\mathbf J_j
=
\mathcal J(\mathbf V^{shape})_j
}
$$

---

## 6.2 Local rotations

对于一个 frame-camera group，真正有导数的 rotation 参数为：

$$
\mathbf q
=
\begin{bmatrix}
\boldsymbol\phi_{tc}\\
\boldsymbol\theta_t
\end{bmatrix}
\in\mathbb R^{66}
$$

因为：

$$
66=3+63
$$

完整 SMPL-X 55 个 local rotations：

$$
R_j=R_j(\mathbf q)
$$

其中：

- root rotation 来自该 camera 的 $\boldsymbol\phi_{tc}$；
- 21 body joints 来自 $\boldsymbol\theta_t$；
- jaw / eyes 为固定零；
- left/right hand pose 固定。

---

## 6.3 Pose feature

代码：

`compute_selected_pose_offsets()`

SMPL-X pose feature：

$$
\boxed{
\mathbf f
=
\operatorname{vec}
\left(
R_1-I,
R_2-I,
\dots,
R_{J-1}-I
\right)
}
$$

root $R_0$ 不进入 pose feature。

因此：

$$
\frac{\partial\mathbf f}{\partial\mathbf q}
=
\operatorname{vec}
\left(
\frac{\partial R_1}{\partial\mathbf q},
\dots,
\frac{\partial R_{J-1}}{\partial\mathbf q}
\right)
$$

---

## 6.4 Pose blend shape

selected posedirs 记为 $P_{p,i}\in\mathbb R^3$。

$$
\boxed{
\Delta\mathbf v_i^{pose}
=
\sum_p
f_p P_{p,i}
}
$$

于是：

$$
\boxed{
\frac{\partial\Delta\mathbf v_i^{pose}}
{\partial\mathbf q}
=
\sum_p
P_{p,i}
\frac{\partial f_p}
{\partial\mathbf q}
}
$$

posed vertex：

$$
\boxed{
\mathbf v_i^{posed}
=
\mathbf v_i^{shape}
+
\Delta\mathbf v_i^{pose}
}
$$

因为 beta 固定：

$$
\boxed{
\frac{\partial\mathbf v_i^{posed}}
{\partial\mathbf q}
=
\frac{\partial\Delta\mathbf v_i^{pose}}
{\partial\mathbf q}
}
$$

---

# 7. Pose 阶段 FK

对于 joint $j$：

$$
\mathbf d_j
=
\begin{cases}
\mathbf J_0,&j=0\\
\mathbf J_j-\mathbf J_{\operatorname{parent}(j)},&j>0
\end{cases}
$$

local homogeneous transform：

$$
\boxed{
L_j
=
\begin{bmatrix}
R_j & \mathbf d_j\\
0&1
\end{bmatrix}
}
$$

Pose 阶段 joints 固定，所以：

$$
\frac{\partial L_j}{\partial\mathbf q}
=
\begin{bmatrix}
\dfrac{\partial R_j}{\partial\mathbf q}
&
0\\
0&0
\end{bmatrix}
$$

global transform：

$$
G_0=L_0
$$

$$
\boxed{
G_j
=
G_{p(j)}L_j
}
$$

导数 product rule：

$$
\boxed{
\frac{\partial G_j}{\partial\mathbf q}
=
\frac{\partial G_{p(j)}}{\partial\mathbf q}L_j
+
G_{p(j)}
\frac{\partial L_j}{\partial\mathbf q}
}
$$

---

# 8. Pose 阶段 rest correction

代码使用：

$$
C_j
=
\begin{bmatrix}
I_3&-\mathbf J_j\\
0&1
\end{bmatrix}
$$

relative transform：

$$
\boxed{
A_j
=
G_jC_j
}
$$

Stage 2 Pose 中 joints 固定：

$$
\frac{\partial C_j}{\partial\mathbf q}=0
$$

所以：

$$
\boxed{
\frac{\partial A_j}{\partial\mathbf q}
=
\frac{\partial G_j}{\partial\mathbf q}C_j
}
$$

---

# 9. Pose 阶段 LBS：$\mathbf v=\ ?$

每个 selected vertex：

$$
\boxed{
T_i
=
\sum_j
w_{ij}A_j
}
$$

homogeneous posed vertex：

$$
\mathbf h_i
=
\begin{bmatrix}
\mathbf v_i^{posed}\\
1
\end{bmatrix}
$$

skinned vertex：

$$
\boxed{
\mathbf v_i^{skin}
=
[T_i\mathbf h_i]_{xyz}
}
$$

最终 production vertex：

$$
\boxed{
\mathbf v_{tci}
=
s_{tc}\mathbf v_i^{skin}
+
\mathbf t_{tc}
}
$$

注意：

> camera-specific global orient 已经作为 root rotation $R_0$ 进入 FK，因此这里没有再额外乘一次 global rotation。

---

## 9.1 LBS Jacobian

$$
\frac{\partial T_i}{\partial\mathbf q}
=
\sum_j
w_{ij}
\frac{\partial A_j}{\partial\mathbf q}
$$

$$
\frac{\partial\mathbf h_i}{\partial\mathbf q}
=
\begin{bmatrix}
\dfrac{\partial\mathbf v_i^{posed}}{\partial\mathbf q}\\
0
\end{bmatrix}
$$

于是：

$$
\boxed{
\frac{\partial\mathbf v_i^{skin}}
{\partial\mathbf q}
=
\left[
\frac{\partial T_i}{\partial\mathbf q}
\mathbf h_i
+
T_i
\frac{\partial\mathbf h_i}{\partial\mathbf q}
\right]_{xyz}
}
$$

最终 scale：

$$
\boxed{
\frac{\partial\mathbf v_{tci}}
{\partial\mathbf q}
=
s_{tc}
\frac{\partial\mathbf v_i^{skin}}
{\partial\mathbf q}
}
$$

translation：

$$
\boxed{
\frac{\partial\mathbf v_{tci}}
{\partial\mathbf t_{tc}}
=
I_3
}
$$

---

# 10. Pose 阶段 2D Jacobian

定义：

$$
J_q
=
J_\pi(\mathbf v)
\frac{\partial\mathbf v}{\partial\mathbf q}
\in\mathbb R^{2\times66}
$$

其中：

$$
J_q
=
[
J_{\phi_c}\mid J_{\theta}
]
$$

- 前 3 列：camera root orient；
- 后 63 列：body pose。

translation：

$$
J_t=J_\pi(\mathbf v)
$$

---

## 10.1 cam2 的 75 维 local Jacobian

$$
\boxed{
J_{t,cam2,i}
=
[
J_{\phi_2}
\mid
J_{t_2}
\mid
0_{2\times3}
\mid
0_{2\times3}
\mid
J_{\theta}
]
}
$$

column 对应：

| local column | 参数 | 构成 |
|---|---|---|
| 0:3 | cam2 orient | $J_\pi\,\partial\mathbf v/\partial\phi_2$ |
| 3:6 | cam2 translation | $J_\pi$ |
| 6:9 | cam4 orient | 0 |
| 9:12 | cam4 translation | 0 |
| 12:75 | body pose | $J_\pi\,\partial\mathbf v/\partial\theta_t$ |

---

## 10.2 cam4 的 75 维 local Jacobian

$$
\boxed{
J_{t,cam4,i}
=
[
0
\mid
0
\mid
J_{\phi_4}
\mid
J_{t_4}
\mid
J_{\theta}
]
}
$$

Pose production batch 中，真正直接计算的 active block 为：

$$
\boxed{
J_{\rm active}
=
[
J_{\phi_c}
\mid
J_{t_c}
\mid
J_\theta
]
\in\mathbb R^{2N\times69}
}
$$

因为：

$$
69=3+3+63
$$

然后再根据 cam2/cam4 scatter 到 75 维 frame block。

---

# 11. Pose anchor

每帧：

$$
\boxed{
\mathbf r_{\theta,t}
=
s_\theta
(
\boldsymbol\theta_t
-
\boldsymbol\theta_t^{anchor}
)
}
$$

其中：

$$
\boxed{
s_\theta
=
\sqrt{
\frac{
2\lambda_\theta
}{
T\cdot63
}
}
}
$$

默认：

$$
\lambda_\theta=1
$$

anchor Jacobian：

$$
\boxed{
J_{\theta,anchor}
=
s_\theta I_{63}
}
$$

放在 75 维中的 body-pose columns：

$$
[0_{63\times12}\mid s_\theta I_{63}]
$$

因此对 H/g：

$$
\boxed{
H_{\theta\theta}
\mathrel{+}=
s_\theta^2 I
}
$$

$$
\boxed{
g_\theta
\mathrel{+}=
s_\theta^2
(
\theta_t-\theta_t^{anchor}
)
}
$$

---

# 12. Pose 阶段 H / g

每 frame 的两个 camera 都只依赖该 frame 的 75 维参数。

因此：

$$
\boxed{
H_t
=
\sum_{c\in\{2,4\}}
\tilde J_{tc}^{\top}\tilde J_{tc}
+
H_{\theta,anchor}
}
$$

$$
\boxed{
\mathbf g_t
=
\sum_c
\tilde J_{tc}^{\top}\tilde{\mathbf r}_{tc}
+
\mathbf g_{\theta,anchor}
}
$$

全局 H 仍然 block diagonal：

$$
H=
\operatorname{blkdiag}(H_1,\dots,H_T)
$$

---

# 13. Stage 2 — Pose/Shape：beta latent

代码：

`pose_shape/shape.py`

实际 LM 参数不是 physical beta，而是：

$$
\mathbf z_\beta
$$

映射：

$$
\boxed{
\boldsymbol\beta
=
L_\beta
\tanh(\mathbf z_\beta)
}
$$

其中：

$$
L_\beta=\texttt{beta\_abs\_limit}=2.5
$$

逐元素：

$$
\boxed{
\frac{\partial\beta_b}
{\partial z_b}
=
L_\beta
\left(
1-\tanh^2 z_b
\right)
}
$$

定义 diagonal：

$$
\boxed{
D_z
=
\operatorname{diag}
\left(
L_\beta
(1-\tanh^2\mathbf z_\beta)
\right)
}
$$

---

# 14. Pose/Shape dynamic shape：$\mathbf v^{shape}=\ ?$

代码：

- `pose_shape/cache.py`
- `pose_shape/shape.py`
- `pose_shape/forward.py`

selected vertex：

$$
\boxed{
\mathbf v_i^{shape}
=
\mathbf v_i^{base}
+
S_i\boldsymbol\beta
}
$$

其中：

$$
S_i\in\mathbb R^{3\times B}
$$

因此：

$$
\boxed{
\frac{\partial\mathbf v_i^{shape}}
{\partial\boldsymbol\beta}
=
S_i
}
$$

对 latent：

$$
\boxed{
\frac{\partial\mathbf v_i^{shape}}
{\partial\mathbf z_\beta}
=
S_iD_z
}
$$

---

# 15. Pose/Shape dynamic joints

代码预先得到：

$$
D_j
=
\frac{\partial\mathbf J_j}
{\partial\boldsymbol\beta}
\in\mathbb R^{3\times B}
$$

实际由：

$$
\boxed{
D_j
=
J_{\rm regressor}\cdot shapedirs
}
$$

对应代码：

```python
torch.einsum("jv,vcb->jcb", J_regressor, beta_shapedirs)
```

current joints：

$$
\boxed{
\mathbf J_j(\beta)
=
\mathbf J_j^{base}
+
D_j\boldsymbol\beta
}
$$

因此：

$$
\boxed{
\frac{\partial\mathbf J_j}
{\partial\mathbf z_\beta}
=
D_jD_z
}
$$

---

# 16. Pose/Shape posed vertex

Pose offset仍然是：

$$
\Delta\mathbf v_i^{pose}(\mathbf q)
$$

因此：

$$
\boxed{
\mathbf v_i^{posed}
=
\mathbf v_i^{shape}(\beta)
+
\Delta\mathbf v_i^{pose}(\mathbf q)
}
$$

对 pose：

$$
\boxed{
\frac{\partial\mathbf v_i^{posed}}
{\partial\mathbf q}
=
\frac{\partial\Delta\mathbf v_i^{pose}}
{\partial\mathbf q}
}
$$

对 beta latent：

$$
\boxed{
\frac{\partial\mathbf v_i^{posed}}
{\partial\mathbf z_\beta}
=
S_iD_z
}
$$

---

# 17. Pose/Shape combined FK

代码：

`pose_shape/derivatives.py`

令 combined derivative variable：

$$
\boldsymbol\xi
=
[
\mathbf q,\mathbf z_\beta
]
$$

其中：

$$
\dim(\mathbf q)=66
$$

---

## 17.1 Relative joint translation

$$
\mathbf d_j
=
\mathbf J_j-\mathbf J_{p(j)}
$$

对 beta latent：

$$
\boxed{
\frac{\partial\mathbf d_j}
{\partial\mathbf z_\beta}
=
\frac{\partial\mathbf J_j}
{\partial\mathbf z_\beta}
-
\frac{\partial\mathbf J_{p(j)}}
{\partial\mathbf z_\beta}
}
$$

---

## 17.2 Local transform

$$
L_j
=
\begin{bmatrix}
R_j&\mathbf d_j\\
0&1
\end{bmatrix}
$$

对 pose：

$$
\boxed{
\frac{\partial L_j}{\partial\mathbf q}
=
\begin{bmatrix}
\dfrac{\partial R_j}{\partial\mathbf q}&0\\
0&0
\end{bmatrix}
}
$$

对 beta latent：

$$
\boxed{
\frac{\partial L_j}{\partial\mathbf z_\beta}
=
\begin{bmatrix}
0&
\dfrac{\partial\mathbf d_j}{\partial\mathbf z_\beta}\\
0&0
\end{bmatrix}
}
$$

---

## 17.3 Global FK

$$
G_j=G_{p(j)}L_j
$$

对任意 combined parameter：

$$
\boxed{
dG_j
=
(dG_{p(j)})L_j
+
G_{p(j)}dL_j
}
$$

这是代码中的两个 einsum：

```python
d_parent @ local
+
parent_global @ d_local
```

---

# 18. Pose/Shape rest correction 的 beta 项

现在 joints 会变，所以 rest correction 也会变：

$$
C_j
=
\begin{bmatrix}
I&-\mathbf J_j\\
0&1
\end{bmatrix}
$$

Pose：

$$
\frac{\partial C_j}{\partial\mathbf q}=0
$$

Beta latent：

$$
\boxed{
\frac{\partial C_j}{\partial\mathbf z_\beta}
=
\begin{bmatrix}
0&
-\dfrac{\partial\mathbf J_j}{\partial\mathbf z_\beta}\\
0&0
\end{bmatrix}
}
$$

relative transform：

$$
A_j=G_jC_j
$$

因此：

$$
\boxed{
dA_j
=
(dG_j)C_j
+
G_j(dC_j)
}
$$

这比 Pose-only 多出的第二项非常重要。

---

# 19. Pose/Shape LBS Jacobian

仍有：

$$
T_i
=
\sum_jw_{ij}A_j
$$

$$
\mathbf h_i
=
\begin{bmatrix}
\mathbf v_i^{posed}\\
1
\end{bmatrix}
$$

$$
\mathbf v_i^{skin}
=
[T_i\mathbf h_i]_{xyz}
$$

对任意变量 $\xi$：

$$
\boxed{
\frac{\partial\mathbf v_i^{skin}}
{\partial\xi}
=
\left[
\frac{\partial T_i}{\partial\xi}\mathbf h_i
+
T_i
\frac{\partial\mathbf h_i}{\partial\xi}
\right]_{xyz}
}
$$

其中：

$$
\frac{\partial T_i}{\partial\xi}
=
\sum_j
w_{ij}
\frac{\partial A_j}{\partial\xi}
$$

---

## 19.1 Pose block

$$
\frac{\partial\mathbf h_i}{\partial\mathbf q}
=
\begin{bmatrix}
\dfrac{\partial\Delta\mathbf v_i^{pose}}{\partial\mathbf q}\\
0
\end{bmatrix}
$$

所以：

$$
\boxed{
J_{\mathbf v,q}
=
s_{tc}
\frac{\partial\mathbf v_i^{skin}}
{\partial\mathbf q}
}
$$

---

## 19.2 Beta-latent block

$$
\frac{\partial\mathbf h_i}{\partial\mathbf z_\beta}
=
\begin{bmatrix}
S_iD_z\\
0
\end{bmatrix}
$$

同时 beta 还通过：

$$
\mathbf J(\beta)
\rightarrow
FK
\rightarrow
A_j
\rightarrow
T_i
$$

影响 vertex。

因此：

$$
\boxed{
J_{\mathbf v,z}
=
s_{tc}
\left[
\underbrace{
\frac{\partial T_i}{\partial\mathbf z_\beta}\mathbf h_i
}_{\text{beta 改变 joints/FK/LBS transform}}
+
\underbrace{
T_i
\frac{\partial\mathbf h_i}{\partial\mathbf z_\beta}
}_{\text{beta 改变 shaped vertex}}
\right]_{xyz}
}
$$

最终：

$$
\boxed{
\mathbf v_{tci}
=
s_{tc}\mathbf v_i^{skin}
+
\mathbf t_{tc}
}
$$

---

# 20. Pose/Shape 2D Jacobian

projection 后：

$$
\boxed{
J_q^{2D}
=
J_\pi
J_{\mathbf v,q}
\in\mathbb R^{2\times66}
}
$$

$$
\boxed{
J_\beta^{2D}
=
J_\pi
J_{\mathbf v,z}
\in\mathbb R^{2\times B}
}
$$

---

# 21. Pose/Shape 每个 Jacobian column 具体构成

对于一个 cam2 group：

$$
\boxed{
J^{frame}_{cam2}
=
[
J_{\phi_2}
\mid
J_{t_2}
\mid
0
\mid
0
\mid
J_\theta
]
\in\mathbb R^{2N\times75}
}
$$

对于 cam4：

$$
\boxed{
J^{frame}_{cam4}
=
[
0
\mid
0
\mid
J_{\phi_4}
\mid
J_{t_4}
\mid
J_\theta
]
}
$$

共享 beta：

$$
\boxed{
J^{beta}_{tc}
=
J_\pi
\frac{\partial\mathbf v_{tc}}
{\partial\mathbf z_\beta}
\in\mathbb R^{2N\times B}
}
$$

所以一个 group 的 global Jacobian row block 为：

$$
\boxed{
J_g
=
[
0
\cdots
J^{frame}_{tc}
\cdots
0
\mid
J^{beta}_{tc}
]
}
$$

只有：

1. 当前 frame 的 75 维 block 非零；
2. 最后的共享 beta block 非零。

这正是 Arrowhead 结构出现的原因。

---

# 22. Pose/Shape production active 69 维

`pose_shape/batched.py` 在每个 camera group 先直接生成：

$$
\boxed{
J_a
=
[
J_{\phi_c}
\mid
J_{t_c}
\mid
J_\theta
]
\in\mathbb R^{2N\times69}
}
$$

因为：

$$
69=3+3+63
$$

beta 独立为：

$$
\boxed{
J_b
\in\mathbb R^{2N\times B}
}
$$

之后根据 group 是 cam2 还是 cam4，将 69 维 active columns scatter 到 frame 的 75 维位置。

---

# 23. Pose/Shape pose anchor

和 Pose 阶段完全相同：

$$
\boxed{
\mathbf r_{\theta,t}
=
s_\theta
(
\boldsymbol\theta_t
-
\boldsymbol\theta_t^{anchor}
)
}
$$

$$
\boxed{
J_{\theta,anchor}
=
s_\theta I_{63}
}
$$

---

# 24. Pose/Shape shape anchor

代码：

`pose_shape/beta_derivatives.py`

physical beta anchor residual：

$$
\boxed{
\mathbf r_\beta
=
s_\beta
(
\boldsymbol\beta(\mathbf z_\beta)
-
\boldsymbol\beta^{anchor}
)
}
$$

其中：

$$
\boxed{
s_\beta
=
\sqrt{
\frac{
2\lambda_\beta
}{B}
}
}
$$

默认：

$$
\lambda_\beta=10
$$

Jacobian 对 latent：

$$
\boxed{
J_{\beta,anchor}
=
s_\beta D_z
}
$$

因为 $D_z$ 是 diagonal，所以该 anchor Jacobian 也是 diagonal。

H contribution：

$$
\boxed{
H_{\beta\beta}^{anchor}
=
(s_\beta D_z)^\top(s_\beta D_z)
=
\operatorname{diag}
\left[
(s_\beta\,d\beta/dz)^2
\right]
}
$$

gradient：

$$
\boxed{
\mathbf g_\beta^{anchor}
=
(s_\beta D_z)^\top
\mathbf r_\beta
}
$$

---

# 25. Pose/Shape Arrowhead H / g

对 frame $t$ 的 group，记加权后的：

$$
J_f
=
\tilde J^{frame}
\in\mathbb R^{2N\times75}
$$

$$
J_b
=
\tilde J^{beta}
\in\mathbb R^{2N\times B}
$$

$$
\tilde{\mathbf r}
=
\text{weighted reprojection residual}
$$

则：

## 25.1 Frame block

$$
\boxed{
A_t
\mathrel{+}=
J_f^\top J_f
}
$$

shape：

$$
75\times75
$$

pose anchor 继续加入：

$$
A_t[12:75,12:75]
\mathrel{+}=
s_\theta^2I
$$

---

## 25.2 Frame-beta cross block

$$
\boxed{
B_t
\mathrel{+}=
J_f^\top J_b
}
$$

shape：

$$
75\times B
$$

---

## 25.3 Shared beta block

$$
\boxed{
C
\mathrel{+}=
J_b^\top J_b
}
$$

最后再加入 shape anchor：

$$
\boxed{
C
\mathrel{+}=
J_{\beta,anchor}^\top
J_{\beta,anchor}
}
$$

---

## 25.4 Gradient

frame：

$$
\boxed{
\mathbf g_t
\mathrel{+}=
J_f^\top\tilde{\mathbf r}
}
$$

beta：

$$
\boxed{
\mathbf g_\beta
\mathrel{+}=
J_b^\top\tilde{\mathbf r}
}
$$

再分别加入 pose / shape anchor gradient。

---

# 26. 最终 Arrowhead 矩阵

$$
\boxed{
H
=
\begin{bmatrix}
A_1&0&\cdots&0&B_1\\
0&A_2&\cdots&0&B_2\\
\vdots&&\ddots&&\vdots\\
0&0&\cdots&A_T&B_T\\
B_1^\top&B_2^\top&\cdots&B_T^\top&C
\end{bmatrix}
}
$$

$$
\boxed{
\mathbf g
=
\begin{bmatrix}
\mathbf g_1\\
\mathbf g_2\\
\vdots\\
\mathbf g_T\\
\mathbf g_\beta
\end{bmatrix}
}
$$

之所以没有：

$$
H_{frame_i,frame_j},\quad i\neq j
$$

是因为一个 reprojection group 只依赖：

- 自己 frame 的参数；
- 全局 shared beta。

---

# 27. 三个 Stage 的 Jacobian 结构总览

## Stage 1

每帧 12：

$$
[\phi_2,t_2,\phi_4,t_4]
$$

cam2：

$$
[J_{\phi_2},J_{t_2},0,0]
$$

cam4：

$$
[0,0,J_{\phi_4},J_{t_4}]
$$

没有 pose / beta anchor。

---

## Stage 2 Pose

每帧 75：

$$
[\phi_2,t_2,\phi_4,t_4,\theta]
$$

cam2：

$$
[J_{\phi_2},J_{t_2},0,0,J_\theta]
$$

cam4：

$$
[0,0,J_{\phi_4},J_{t_4},J_\theta]
$$

另有：

$$
J_{\theta,anchor}
$$

---

## Stage 2 Pose/Shape

全局：

$$
[x_1^{75},\dots,x_T^{75},z_\beta^B]
$$

group：

$$
[
0,\dots,
J_f^{75},
\dots,0,
J_\beta^B
]
$$

并加入：

$$
J_{\theta,anchor}
$$

$$
J_{\beta,anchor}
$$

因此 H 从 block-diagonal 变成 Arrowhead。

---

# 28. LM 中所有 Stage 最终共同使用的公式

所有 Stage 都先得到 surrogate residual：

$$
\tilde{\mathbf r}
$$

以及对应 Jacobian：

$$
\tilde J
$$

Gauss-Newton：

$$
\boxed{
H
=
\tilde J^\top\tilde J
}
$$

$$
\boxed{
\mathbf g
=
\tilde J^\top\tilde{\mathbf r}
}
$$

LM：

$$
\boxed{
(H+\lambda I)\Delta
=
-\mathbf g
}
$$

candidate：

$$
\boxed{
\mathbf x_{\rm new}
=
\mathbf x+\Delta
}
$$

代码最终 parameter update 会把 $\Delta$ 转为 float32 后加到当前参数。

---

# 29. Pose/Shape Schur solve

对：

$$
A_t^\lambda=A_t+\lambda I_{75}
$$

$$
C^\lambda=C+\lambda I_B
$$

原方程：

$$
A_t^\lambda\Delta_t
+
B_t\Delta_\beta
=
-\mathbf g_t
$$

$$
\sum_tB_t^\top\Delta_t
+
C^\lambda\Delta_\beta
=
-\mathbf g_\beta
$$

定义：

$$
X_{g,t}
=
(A_t^\lambda)^{-1}\mathbf g_t
$$

$$
X_{B,t}
=
(A_t^\lambda)^{-1}B_t
$$

Schur complement：

$$
\boxed{
S
=
C^\lambda
-
\sum_t
B_t^\top X_{B,t}
}
$$

右端：

$$
\boxed{
\mathbf b_S
=
-\mathbf g_\beta
+
\sum_t
B_t^\top X_{g,t}
}
$$

求：

$$
\boxed{
\Delta_\beta
=
S^{-1}\mathbf b_S
}
$$

回代：

$$
\boxed{
\Delta_t
=
-X_{g,t}
-
X_{B,t}\Delta_\beta
}
$$

对应：

`arrowhead_schur.py`

---

# 30. LM trial acceptance

代码：

`lm/trial.py`

预测下降：

$$
\boxed{
\Delta F_{\rm pred}
=
-\mathbf g^\top\Delta
-\frac12\Delta^\top H\Delta
}
$$

真实下降：

$$
\boxed{
\Delta F_{\rm actual}
=
F(\mathbf x)
-
F(\mathbf x+\Delta)
}
$$

$$
\boxed{
\rho
=
\frac{
\Delta F_{\rm actual}
}{
\max(\Delta F_{\rm pred},10^{-12})
}
}
$$

candidate 被接受需要同时：

$$
\boxed{
\Delta F_{\rm pred}>0
}
$$

$$
\boxed{
\Delta F_{\rm actual}>0
}
$$

$$
\boxed{
\rho>0
}
$$

---

# 31. 最核心的一条链

如果只想记住一条公式：

$$
\boxed{
\text{parameter}
\rightarrow
R
\rightarrow
pose\ blend
\rightarrow
FK
\rightarrow
LBS
\rightarrow
\mathbf v
\rightarrow
ERP
\rightarrow
\mathbf r
}
$$

对应 Jacobian：

$$
\boxed{
\frac{\partial\mathbf r}{\partial\mathbf x}
=
\frac{\partial\pi}{\partial\mathbf v}
\frac{\partial\mathbf v}{\partial\mathbf x}
}
$$

Stage 之间真正变化的是：

### Stage 1

$$
\frac{\partial\mathbf v}{\partial\mathbf x}
=
\left[
-R[\mathbf l]_\times J_r
\mid I
\right]
$$

### Pose

$$
\frac{\partial\mathbf v}{\partial\theta}
=
s
\left[
\frac{\partial T}{\partial\theta}h
+
T\frac{\partial h}{\partial\theta}
\right]
$$

### Pose/Shape

Pose 部分同上，而 beta 多出：

$$
\boxed{
\frac{\partial\mathbf v}{\partial z_\beta}
=
s
\left[
\underbrace{
\frac{\partial T}{\partial z_\beta}h
}_{joints/FK/LBS}
+
\underbrace{
T\frac{\partial h}{\partial z_\beta}
}_{shaped\ vertex}
\right]
}
$$

这就是第三阶段比第二阶段复杂的核心。

---

# 32. 公式到当前代码的对应关系

| 数学部分 | 当前代码 |
|---|---|
| axis-angle $R$、$J_r$ | `geometry/so3.py` |
| ERP $\pi(v)$、$J_\pi$ | `geometry/erp.py` |
| Stage1 $v,R,t,J$ | `analytic_stage1.py` |
| Pose fixed shape | `pose/cache.py` |
| Pose value chain | `pose/forward.py` |
| Pose analytic chain | `pose/derivatives.py` |
| Pose production batch H/g | `pose/batched.py` |
| beta latent $\to\beta$ | `pose_shape/shape.py` |
| Pose/Shape value chain | `pose_shape/forward.py` |
| beta-only reference J | `pose_shape/beta_derivatives.py` |
| combined pose/beta J | `pose_shape/derivatives.py` |
| serial Arrowhead A/B/C/g | `pose_shape/arrowhead.py` |
| production batched A/B/C/g | `pose_shape/batched.py` |
| Schur solve | `arrowhead_schur.py` |
| IRLS weights | `irls.py` |
| H/g 与 solve dispatch | `lm/equations.py` |
| LM predicted/actual/rho | `lm/trial.py` |

---

# 33. 一句话总结三个阶段

**Stage 1**

$$
\boxed{
v=R(\phi)l+c+t
}
$$

只求 rigid orient / translation Jacobian。

**Stage 2 Pose**

$$
\boxed{
v
=
s\,
LBS(
v_{shape}(\beta_{fixed})
+
\Delta v_{pose}(\theta),
R(\phi,\theta)
)
+t
}
$$

求 camera orient + translation + body pose Jacobian。

**Stage 2 Pose/Shape**

$$
\boxed{
v
=
s\,
LBS(
v_{shape}(\beta(z))
+
\Delta v_{pose}(\theta),
R(\phi,\theta),
J(\beta(z))
)
+t
}
$$

同时求：

$$
\boxed{
J_{\rm frame}\in\mathbb R^{2N\times75}
}
$$

和：

$$
\boxed{
J_{\beta}\in\mathbb R^{2N\times B}
}
$$

并由它们组成 Arrowhead $H/g$。

