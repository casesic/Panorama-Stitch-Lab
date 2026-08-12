# Staged LM：C++ / GPU 移植用计算图

> 基于当前 `debug109_new` 代码整理。  
> 目标不是重新推导数学，而是把当前 Python production 路径写成接近 C++ 的“赋值语句 + shape + 依赖 + 计算频率”形式，便于：
>
> 1. 先做 C++ CPU reference；
> 2. 对照 Python 做逐中间量验证；
> 3. 再判断哪些步骤适合 BLAS / Eigen，哪些值得做 GPU kernel / fusion。
>
> **本文保持当前代码语义，不代表已经采用了最优实现。**
>
> 当前三阶段：
>
> - Stage 1：`STAGE1_RIGID`
> - Stage 2：`STAGE2_POSE`
> - Stage 3：`STAGE2_POSE_SHAPE`

---

# 0. 统一维度与频率记号

## 0.1 维度

| 符号 | 含义 | 当前值 / shape |
|---|---|---|
| `T` | segment frame 数 | 动态 |
| `G` | 非空 frame-camera group 数 | 通常约 `2T` |
| `N_g` | group `g` 的有效 selected vertex 数 | 动态 |
| `Nmax` | batch padding 后最大点数 | `max(N_g)` |
| `V` | SMPL-X full vertex 数 | model 决定 |
| `J` | SMPL-X joint 数 | `55` |
| `J_var` | 有优化旋转参数的 joint 数 | `22 = root + 21 body` |
| `P` | pose feature 维数 | `(J-1)*9 = 486` |
| `Qrot` | root orient + body pose rotation 参数 | `3 + 63 = 66` |
| `Qactive` | 单 camera 真正 active 的 frame 参数 | `3 orient + 3 transl + 63 pose = 69` |
| `Qframe` | 每 frame 完整 Pose 参数 | `75` |
| `B` | beta 维数 | `10` |
| `Qcombined` | Pose/Shape FK derivative 维数 | `66 + B = 76` |

---

## 0.2 计算频率

本文使用：

| 标记 | 频率 |
|---|---|
| **INIT** | 模型 / segment 初始化一次 |
| **STAGE** | 每个 Stage 构建一次 |
| **LIN** | 每个 LM linearisation point 一次；初始点 + 每个 accepted point |
| **TRIAL** | 每个 LM candidate trial 一次 |
| **GROUP** | 每个 frame-camera group |
| **POINT** | 每个 selected vertex |

当前 LM 中：

- `H/g` 在 **LIN** 时计算；
- 同一个 linearisation 下多个 damping trial 复用同一个 `H/g`；
- true objective 在初始点计算一次，然后每个实际 candidate **TRIAL** 计算一次；
- accepted candidate 的 objective 值直接复用到下一 outer。

因此 C++ 第一版应先复现：

```text
LIN:
    build weights
    build H, g

TRIAL:
    solve (H + lambda I) delta = -g
    evaluate true objective(candidate)
    accept / reject
```

不要把 `H/g` 误写成每个 trial 都重新计算。

---

# 1. 全局数据流

```text
SMPL-X model
  │
  ├─ v_template [V,3]
  ├─ shapedirs [V,3,B]
  ├─ posedirs [P, V*3] → reshape [P,V,3]
  ├─ J_regressor [J,V]
  ├─ parents [J]
  └─ lbs_weights [V,J]
        │
        ▼
Stage2SharedGeometryCache                         [STAGE]
        │
        ├─ selected_posedirs[g] [P,N_g,3]
        ├─ selected_lbs_weights[g] [N_g,J]
        ├─ targets[g] [N_g,2]
        ├─ vertex_ids[g] [N_g]
        ├─ frozen hand pose
        └─ frozen scale
        │
        ├─────────────────────────────┐
        ▼                             ▼
     Pose cache                 Pose/Shape cache
     beta fixed                 beta dynamic
        │                             │
        ▼                             ▼
   FULL_BATCH Pose                FULL_BATCH Pose/Shape
        │                             │
        ▼                             ▼
      H, g                     Arrowhead A/B/C/g
        │                             │
        └─────────── LM solver ──────┘
```

---

# 2. Stage 1：Rigid

对应代码：

```text
analytic_stage1.py
geometry/so3.py
geometry/erp.py
```

参数每 frame：

```text
x_t [12] =
[
    cam2_orient[3],
    cam2_transl[3],
    cam4_orient[3],
    cam4_transl[3]
]
```

---

## 2.1 Stage 1 静态 cache

### Python 对应

`build_stage1_analytic_cache()`

### 伪代码

```cpp
for frame t:
    body_pose = body_pose_anchor[t]

    for camera c in {cam2, cam4}:
        orient0 = [0,0,0]
        transl0 = [0,0,0]

        full_vertices, joints =
            SMPLX_forward(
                orient0,
                body_pose,
                frozen_hand_pose[t,c],
                beta_anchor,
                transl0,
                frozen_scale[t,c]
            )

        rotation_center[g] = joints[root]              // [3]

        local_vertices[g] =
            selected(full_vertices, vertex_ids[g])
            - rotation_center[g]                       // [N_g,3]

        targets[g] = observed_uv[g]                    // [N_g,2]
```

### 计算频率

**STAGE：一次。**

之后 Stage 1 LM 不再运行完整 SMPL-X。

---

## 2.2 Stage 1 当前点的 3D vertex

对 group `g=(t,c)`：

```cpp
orient      = x[frame_block + camera_offset : +3]      // [3]
translation = x[frame_block + camera_offset + 3 : +6]  // [3]

R = axis_angle_to_matrix(orient)                       // [3,3]

points =
    local_vertices[g] * R^T
    + rotation_center[g]
    + translation                                     // [N_g,3]
```

数学：

$$
\mathbf p_i
=
R(\phi)\mathbf l_i
+
\mathbf c
+
\mathbf t
$$

### shape

| 变量 | shape |
|---|---|
| `local_vertices` | `[N_g,3]` |
| `R` | `[3,3]` |
| `rotation_center` | `[3]` |
| `translation` | `[3]` |
| `points` | `[N_g,3]` |

### 频率

**LIN / GROUP**

---

## 2.3 Stage 1 orient Jacobian

```cpp
Jr = so3_right_jacobian(orient)                        // [3,3]

for point i:
    skew_l = skew(local_vertices[i])                   // [3,3]

    d_point_d_orient[i] =
        -R * skew_l * Jr                               // [3,3]
```

数学：

$$
\frac{\partial(Rl_i)}{\partial\phi}
=
-R[l_i]_\times J_r(\phi)
$$

shape：

```text
d_point_d_orient : [N_g,3,3]
```

频率：

**LIN / GROUP / POINT**

---

## 2.4 Stage 1 ERP

```cpp
uv, J_proj =
    project_unified_erp_with_jacobian(points)          // [N_g,2], [N_g,2,3]

residual =
    uv - targets[g]                                    // [N_g,2]
```

对 orient：

```cpp
J_orient =
    J_proj @ d_point_d_orient                          // [N_g,2,3]
```

对 translation：

```cpp
J_transl =
    J_proj                                              // [N_g,2,3]
```

camera local Jacobian：

```text
J_camera = concat(J_orient, J_transl)
         : [N_g,2,6]
         → flatten [2*N_g,6]
```

放入 frame 12 维 block：

### cam2

```text
J_local =
[
    J_orient_cam2, J_transl_cam2,
    0,             0
]                                                       // [2N_g,12]
```

### cam4

```text
J_local =
[
    0,             0,
    J_orient_cam4, J_transl_cam4
]                                                       // [2N_g,12]
```

---

## 2.5 Stage 1 IRLS row scale

在 linearisation point `$x_k$`：

```cpp
norm_i = sqrt(residual_u^2 + residual_v^2)

w_i =
    alpha_g / sqrt(norm_i^2 + eps^2)

sqrt_w_i = sqrt(w_i)
```

然后：

```cpp
r_scaled[row_u] = sqrt_w_i * residual_u
r_scaled[row_v] = sqrt_w_i * residual_v

J_scaled[row_u,:] = sqrt_w_i * J_local[row_u,:]
J_scaled[row_v,:] = sqrt_w_i * J_local[row_v,:]
```

权重在当前 **LIN** 内冻结。

---

## 2.6 Stage 1 H/g

每 frame 收集 cam2/cam4：

```cpp
J_frame = concat_rows(J_cam2, J_cam4)                  // [M_t,12]
r_frame = concat(r_cam2, r_cam4)                       // [M_t]

J64 = float64(J_frame)
r64 = float64(r_frame)

H_t = J64^T * J64                                      // [12,12]
g_t = J64^T * r64                                      // [12]
```

全局：

```text
H = block_diag(H_0, ..., H_{T-1})                      // [12T,12T]
g = concat(g_0, ..., g_{T-1})                          // [12T]
```

---

## 2.7 Stage 1 C++ CPU 第一版建议

最直接的 C++ reference：

```cpp
for each group:
    compute R
    for each point:
        compute 3D point
        compute ERP + J_proj
        compute d_point_d_orient
        assemble one 2x12 Jacobian block
        apply row scale
        accumulate H_frame += J_i^T * J_i
        accumulate g_frame += J_i^T * r_i
```

这里**不必真的存完整 `J_frame`**。

Stage 1 很适合作为：

```text
Python → C++ CPU
```

的第一块验证，因为没有 FK / LBS derivative。

---

# 3. Stage 2：Pose — 静态 cache

对应：

```text
stage2_shared_cache.py
pose/cache.py
pose/batched.py
```

---

## 3.1 Shared selected rows

一次选出：

```cpp
posedirs_by_vertex =
    reshape(model.posedirs, [P,V,3])                   // [486,V,3]

selected_posedirs[g] =
    posedirs_by_vertex[:, vertex_ids[g], :]             // [P,N_g,3]

selected_lbs_weights[g] =
    model.lbs_weights[vertex_ids[g], :]                 // [N_g,55]
```

频率：

**STAGE**

---

## 3.2 Pose fixed-beta shape

Pose 阶段 beta 固定：

```cpp
beta = beta_anchor                                      // [B]

shape_components =
    concat(beta, zero_expression)

v_shaped =
    v_template
    + blend_shapes(shape_components, shape_dirs)        // [V,3]

joints =
    J_regressor * v_shaped                              // [55,3]
```

对每 group：

```cpp
selected_v_shaped[g] =
    v_shaped[vertex_ids[g]]                             // [N_g,3]
```

频率：

**STAGE**

所以 Pose LM 内：

```text
v_shaped  常量
joints    常量
```

---

# 4. Stage 2：Pose — FULL_BATCH production 计算图

production：

```text
pose/batched.py
```

batch padding 后：

```text
G      = group_count
N      = Nmax
```

静态 tensor：

| 变量 | shape |
|---|---|
| `selected_v_shaped` | `[G,N,3]` |
| `selected_posedirs` | `[G,486,N,3]` |
| `selected_lbs_weights` | `[G,N,55]` |
| `left_hand_pose` | `[G,45]` |
| `right_hand_pose` | `[G,45]` |
| `scale` | `[G,1]` |
| `targets` | `[G,N,2]` |
| `valid_mask` | `[G,N]` |
| `parents` | `[55]` |

---

## 4.1 Gather 当前参数

输入：

```text
parameters : [75T]
```

先 reshape：

```cpp
framed =
    reshape(parameters, [T,75])                        // [T,75]
```

每 group 根据 `frame_positions`、`camera_indices` gather：

```cpp
group_orientations  // [G,3]
group_translations  // [G,3]
group_body_poses    // [G,63]
frame_body_poses    // [T,63]
```

频率：

**LIN**

---

## 4.2 Assemble 55 个 axis-angle

```cpp
axis_angles =
concat_joint_axis_angles(
    group_orientations,                                // root     [G,1,3]
    group_body_poses.reshape(G,21,3),                  // body     [G,21,3]
    zero_jaw_eye_eye,                                  //          [G,3,3]
    left_hand_pose.reshape(G,15,3),                    //          [G,15,3]
    right_hand_pose.reshape(G,15,3)                    //          [G,15,3]
)                                                      // [G,55,3]
```

只有前 22 joints 的 axis-angle 是变量：

```text
root 1 + body 21 = 22
```

---

## 4.3 Rotation value

```cpp
R =
    axis_angle_to_matrix(axis_angles)                  // [G,55,3,3]
```

频率：

**LIN**

---

## 4.4 Rotation Jacobian

```cpp
Jr =
    so3_right_jacobian(axis_angles[:,0:22])            // [G,22,3,3]

basis =
    skew(transpose(Jr))                                // [G,22,3,3]

rotation_blocks =
    R[:,0:22] * basis                                  // [G,22,3,3,3]
```

通过静态 `derivative_mapping [22,66,3]` scatter：

```cpp
dR =
    zeros([G,55,3,3,66])

dR[:,0:22,:,:,:] =
    map(rotation_blocks, derivative_mapping)
```

输出：

```text
R  : [G,55,3,3]
dR : [G,55,3,3,66]
```

其中 66 columns：

```text
0:3   = current camera global orient
3:66  = body pose 63
```

---

# 5. Pose：pose feature / pose blend

## 5.1 Pose feature

```cpp
I = eye(3)

feature =
    reshape(R[:,1:55] - I, [G,486])                    // [G,486]

d_feature =
    reshape(dR[:,1:55], [G,486,66])                    // [G,486,66]
```

注意：

- root `R[:,0]` 不进入 pose feature；
- hand rotation 虽然进入 feature value，但其 Jacobian columns 为 0，因为 hand pose 冻结。

---

## 5.2 Selected pose offsets

当前代码：

```cpp
offsets =
    einsum("gp,gpnc->gnc",
           feature,
           selected_posedirs)                          // [G,N,3]

d_offsets =
    einsum("gpq,gpnc->gncq",
           d_feature,
           selected_posedirs)                          // [G,N,3,66]
```

数学：

$$
\Delta v^{pose}_{gni}
=
\sum_p f_{gp}P_{gpni}
$$

$$
\frac{\partial\Delta v^{pose}}
{\partial q}
=
\sum_p
\frac{\partial f_p}{\partial q}
P_p
$$

然后：

```cpp
v_posed =
    selected_v_shaped + offsets                        // [G,N,3]
```

Pose fixed beta，所以：

```cpp
d_v_posed =
    d_offsets                                          // [G,N,3,66]
```

---

# 6. Pose：FK

输入：

```text
R       [G,55,3,3]
dR      [G,55,3,3,66]
joints  [55,3]        // fixed beta
parents [55]
```

---

## 6.1 Relative joint positions

```cpp
joints_g =
    broadcast(joints, [G,55,3])

relative_joints =
    joints_g

relative_joints[:,1:] -=
    joints_g[:, parents[1:]]                           // [G,55,3]
```

---

## 6.2 Local transform

```cpp
L =
    zeros([G,55,4,4])

L[:,:,:3,:3] = R
L[:,:,:3, 3] = relative_joints
L[:,:,3,3]   = 1
```

Derivative：

```cpp
dL =
    zeros([G,55,4,4,66])

dL[:,:,:3,:3,:] = dR
```

因为 Pose 阶段 joints 固定：

```text
d(relative_joint_translation)/dq = 0
```

---

## 6.3 Depth-wise global FK

root：

```cpp
G_global[:,0]  = L[:,0]
dG_global[:,0] = dL[:,0]
```

每一层 depth：

```cpp
for joints j at same depth:
    p = parent[j]

    G_global[j] =
        G_global[p] * L[j]

    dG_global[j] =
        dG_global[p] * L[j]
        +
        G_global[p] * dL[j]
```

shape：

```text
G_global  : [G,55,4,4]
dG_global : [G,55,4,4,66]
```

---

## 6.4 Rest correction

```cpp
Rest =
    identity4 repeated [G,55]

Rest[:,:,:3,3] =
    -joints_g

A =
    G_global * Rest                                    // [G,55,4,4]

dA =
    dG_global * Rest                                   // [G,55,4,4,66]
```

因为 Pose 阶段：

```text
dRest/dq = 0
```

输出的 `A` 就是代码中的 `relative_transforms`。

---

# 7. Pose：selected LBS

输入：

```text
v_posed   [G,N,3]
d_v_posed [G,N,3,66]
weights   [G,N,55]
A         [G,55,4,4]
dA        [G,55,4,4,66]
```

---

## 7.1 Homogeneous vertex

```cpp
h =
    concat(v_posed, 1)                                 // [G,N,4]

dh =
    concat(d_v_posed, 0)                               // [G,N,4,66]
```

---

## 7.2 Blend transform

当前 production 只对 top 3 rows 做 value path：

```cpp
T_top =
    einsum("gnj,gjab->gnab",
           weights,
           A[:,:,:3,:])                                // [G,N,3,4]
```

完整 derivative：

```cpp
dT =
    einsum("gnj,gjabq->gnabq",
           weights,
           dA)                                         // [G,N,4,4,66]
```

---

## 7.3 Skinned vertex

```cpp
v_skin =
    einsum("gnab,gnb->gna",
           T_top,
           h)                                          // [G,N,3]
```

Derivative 两项：

```cpp
dv_skin =
    einsum(weights, dA_top, h)
    +
    einsum(T_top, dh)                                  // [G,N,3,66]
```

即：

$$
dv_{skin}
=
(dT)h
+
T(dh)
$$

---

## 7.4 Scale + translation

```cpp
vertices =
    v_skin * scale
    + group_translation                                // [G,N,3]

d_vertices =
    dv_skin * scale                                    // [G,N,3,66]
```

注意：

- camera orient 已经作为 root rotation 进入 FK；
- translation 不在 `d_vertices[...,66]` 中；
- translation Jacobian 之后直接由 ERP projection Jacobian 插入。

---

# 8. Pose：ERP + active 69 Jacobian

```cpp
uv, J_proj =
    ERP_with_jacobian(vertices.reshape(G*N,3))

uv =
    reshape(uv, [G,N,2])

J_proj =
    reshape(J_proj, [G,N,2,3])
```

对 66 rotation variables：

```cpp
J_uv_rotpose =
    einsum("gnab,gnbq->gnaq",
           J_proj,
           d_vertices)                                 // [G,N,2,66]
```

当前 active 69 维按：

```text
[ orient(3), translation(3), body_pose(63) ]
```

组装：

```cpp
J_active =
    concat(
        J_uv_rotpose[:,:,:,0:3],                       // orient      3
        J_proj,                                         // translation 3
        J_uv_rotpose[:,:,:,3:66]                       // pose        63
    )                                                   // [G,N,2,69]
```

residual：

```cpp
r =
    uv - targets                                       // [G,N,2]
```

padding：

```cpp
r        *= valid_mask
J_active *= valid_mask
```

---

# 9. Pose：加权后 H/g

row scale：

```text
row_scales : [G,N,2]
```

应用：

```cpp
r_w =
    r * row_scales                                     // [G,N,2]

J_w =
    J_active * row_scales[...,None]                    // [G,N,2,69]
```

flatten + FP64：

```cpp
r64 =
    float64(reshape(r_w, [G,2N]))                      // [G,2N]

J64 =
    float64(reshape(J_w, [G,2N,69]))                   // [G,2N,69]
```

每 group：

```cpp
H_active[g] =
    J64[g]^T * J64[g]                                  // [69,69]

g_active[g] =
    J64[g]^T * r64[g]                                  // [69]
```

---

## 9.1 69 → 75 scatter

cam2 active indices：

```text
[0:6, 12:75]
```

cam4：

```text
[6:12, 12:75]
```

因此：

```cpp
H_frame[t][active_indices, active_indices] += H_active[g]
g_frame[t][active_indices]                 += g_active[g]
```

最后：

```text
H_frame : [T,75,75]
g_frame : [T,75]
```

---

## 9.2 Pose anchor

```cpp
pose_delta =
    frame_body_pose[t]
    - body_pose_anchor[t]                              // [63]

r_pose_anchor =
    pose_delta * pose_anchor_scale                     // [63]
```

加入：

```cpp
H_frame[t][12:75,12:75] +=
    pose_anchor_scale^2 * I63

g_frame[t][12:75] +=
    pose_delta * pose_anchor_scale^2
```

最终 Pose H：

```text
H : [75T,75T]
```

仍为 block diagonal。

---

# 10. Stage 3：Pose/Shape — 静态 cache

对应：

```text
pose_shape/cache.py
pose_shape/shape.py
```

与 Pose 最大区别：

```text
beta 不再固定
```

因此不能 cache 最终 `v_shaped` 和 `joints`，只 cache 线性基底。

---

## 10.1 beta=0 的 base shape

```cpp
zero_beta       = zeros([B])
zero_expression = zeros(...)

base_v_shaped =
    v_template
    + blend_shapes([zero_beta, zero_expression],
                   [shapedirs, expr_dirs])              // [V,3]

beta_shapedirs =
    model.shapedirs                                     // [V,3,B]
```

---

## 10.2 base joints 与 beta joint derivative

```cpp
base_joints =
    J_regressor * base_v_shaped                         // [55,3]

d_joints_d_beta =
    einsum("jv,vcb->jcb",
           J_regressor,
           beta_shapedirs)                              // [55,3,B]
```

这里：

```text
d_joints_d_beta
```

是**静态常量**。

---

## 10.3 selected shape rows

每 group：

```cpp
selected_base_v_shaped[g] =
    base_v_shaped[vertex_ids[g]]                       // [N_g,3]

selected_beta_shapedirs[g] =
    beta_shapedirs[vertex_ids[g]]                      // [N_g,3,B]
```

FULL_BATCH：

```text
selected_base_v_shaped  [G,N,3]
selected_beta_shapedirs [G,N,3,B]
```

频率：

**STAGE**

---

# 11. Pose/Shape：latent beta → dynamic shape

输入：

```text
beta_latent z : [B]
beta_abs_limit = 2.5
```

---

## 11.1 Physical beta

```cpp
tanh_z =
    tanh(z)                                             // [B]

beta =
    beta_abs_limit * tanh_z                             // [B]

d_beta_d_z =
    beta_abs_limit * (1 - tanh_z^2)                    // [B]
```

频率：

**LIN**

---

## 11.2 Dynamic joints

```cpp
joints =
    base_joints
    + einsum("jcb,b->jc",
             d_joints_d_beta,
             beta)                                     // [55,3]
```

Derivative：

```cpp
d_joints_d_z =
    d_joints_d_beta
    * d_beta_d_z[None,None,:]                          // [55,3,B]
```

---

## 11.3 Dynamic selected shape

FULL_BATCH：

```cpp
v_shaped =
    selected_base_v_shaped
    +
    einsum("gncb,b->gnc",
           selected_beta_shapedirs,
           beta)                                       // [G,N,3]
```

Derivative：

```cpp
d_v_shaped_d_z =
    selected_beta_shapedirs
    * d_beta_d_z[None,None,None,:]                     // [G,N,3,B]
```

---

# 12. Pose/Shape：rotation / pose blend

rotation 部分完全复用 Pose：

```cpp
R, dR_d_pose =
    pose_batched_rotations_and_jacobians(...)          // [G,55,3,3]
                                                        // [G,55,3,3,66]
```

pose feature：

```cpp
feature =
    reshape(R[:,1:] - I, [G,486])

d_feature =
    reshape(dR_d_pose[:,1:], [G,486,66])
```

pose offsets：

```cpp
pose_offsets =
    einsum("gp,gpnc->gnc",
           feature,
           selected_posedirs)                          // [G,N,3]

d_pose_offsets =
    einsum("gpq,gpnc->gncq",
           d_feature,
           selected_posedirs)                          // [G,N,3,66]
```

posed vertex：

```cpp
v_posed =
    v_shaped
    + pose_offsets                                     // [G,N,3]
```

combined derivative：

```cpp
d_v_posed =
    concat(
        d_pose_offsets,                                // pose [66]
        d_v_shaped_d_z                                 // beta [B]
    )                                                  // [G,N,3,66+B]
```

当前：

```text
66+B = 76
```

---

# 13. Pose/Shape：combined FK

这是第三阶段比 Pose-only 多出的主要计算。

输入：

```text
R             [G,55,3,3]
dR_pose       [G,55,3,3,66]
joints        [55,3]
d_joints_z    [55,3,B]
```

---

## 13.1 Relative joints value

```cpp
Jg =
    broadcast(joints, [G,55,3])

relative_joint =
    Jg

relative_joint[:,1:] -=
    Jg[:,parents[1:]]                                  // [G,55,3]
```

---

## 13.2 Relative joints beta derivative

```cpp
dJg =
    broadcast(d_joints_z, [G,55,3,B])

d_relative_joint =
    dJg

d_relative_joint[:,1:] -=
    dJg[:,parents[1:]]                                 // [G,55,3,B]
```

---

## 13.3 Local transform

```cpp
L =
    zeros([G,55,4,4])

L[:,:,:3,:3] = R
L[:,:,:3,3]  = relative_joint
L[:,:,3,3]   = 1
```

Combined derivative：

```cpp
dL =
    zeros([G,55,4,4,66+B])

dL[:,:,:3,:3,0:66] =
    dR_pose

dL[:,:,:3,3,66:] =
    d_relative_joint
```

---

## 13.4 Global FK

root：

```cpp
G_global[:,0]  = L[:,0]
dG_global[:,0] = dL[:,0]
```

depth-wise：

```cpp
G_global[j] =
    G_global[parent] * L[j]

dG_global[j] =
    dG_global[parent] * L[j]
    +
    G_global[parent] * dL[j]
```

shape：

```text
G_global  [G,55,4,4]
dG_global [G,55,4,4,76]
```

---

## 13.5 Dynamic rest correction

```cpp
Rest =
    identity4 repeated [G,55]

Rest[:,:,:3,3] =
    -Jg
```

因为 joints 依赖 beta：

```cpp
dRest =
    zeros([G,55,4,4,76])

dRest[:,:,:3,3,66:] =
    -dJg
```

relative transform：

```cpp
A =
    G_global * Rest                                    // [G,55,4,4]
```

Derivative：

```cpp
dA =
    dG_global * Rest
    +
    G_global * dRest                                   // [G,55,4,4,76]
```

这就是 Pose/Shape 相对 Pose-only 多出的关键项：

```text
G_global * dRest
```

---

# 14. Pose/Shape：LBS

输入：

```text
weights    [G,N,55]
A          [G,55,4,4]
dA         [G,55,4,4,76]
v_posed    [G,N,3]
d_v_posed  [G,N,3,76]
```

---

## 14.1 Homogeneous

```cpp
h =
    concat(v_posed, 1)                                 // [G,N,4]

dh =
    concat(d_v_posed, 0)                               // [G,N,4,76]
```

---

## 14.2 Selected transform

production 当前：

```cpp
T_top =
    einsum("gnj,gjab->gnab",
           weights,
           A[:,:,:3,:])                                // [G,N,3,4]
```

---

## 14.3 Vertex

```cpp
v_skin =
    einsum("gnab,gnb->gna",
           T_top,
           h)                                          // [G,N,3]
```

Derivative：

```cpp
dv_skin =
    einsum(weights, dA_top, h)
    +
    einsum(T_top, dh)                                  // [G,N,3,76]
```

即：

$$
dv
=
\underbrace{(dT)h}_{FK/LBS transform 变化}
+
\underbrace{T(dh)}_{posed vertex 变化}
$$

其中 beta 的两条路径同时存在：

```text
beta
 ├─> v_shaped ─────────────> h ───────┐
 │                                    ├─> v_skin
 └─> joints -> FK -> Rest -> A -> T ──┘
```

---

## 14.4 Scale + translation

```cpp
vertices =
    v_skin * scale
    + group_translation                                // [G,N,3]

d_vertices =
    dv_skin * scale                                    // [G,N,3,76]
```

切分：

```cpp
d_vertices_pose =
    d_vertices[...,0:66]                               // [G,N,3,66]

d_vertices_beta =
    d_vertices[...,66:]                                // [G,N,3,B]
```

---

# 15. Pose/Shape：ERP Jacobian split

```cpp
uv, J_proj =
    ERP_with_jacobian(vertices)                        // [G,N,2], [G,N,2,3]

J_uv_all =
    J_proj @ d_vertices                                // [G,N,2,76]
```

pose rotation：

```text
J_uv_all[...,0:66]
```

beta：

```text
J_beta = J_uv_all[...,66:]                             // [G,N,2,B]
```

active frame 69：

```cpp
J_active =
    concat(
        J_uv_all[...,0:3],                             // camera orient
        J_proj,                                        // camera translation
        J_uv_all[...,3:66]                             // body pose
    )                                                  // [G,N,2,69]
```

residual：

```cpp
r =
    uv - targets                                       // [G,N,2]
```

---

# 16. Pose/Shape：weighted GEMM

```cpp
r_w  = r        * row_scales                           // [G,N,2]
Ja_w = J_active * row_scales[...,None]                 // [G,N,2,69]
Jb_w = J_beta   * row_scales[...,None]                 // [G,N,2,B]
```

flatten FP64：

```cpp
r64 =
    float64(reshape(r_w, [G,2N]))

Ja64 =
    float64(reshape(Ja_w, [G,2N,69]))

Jb64 =
    float64(reshape(Jb_w, [G,2N,B]))
```

每 group：

```cpp
Haa =
    Ja64^T * Ja64                                      // [G,69,69]

Hab =
    Ja64^T * Jb64                                      // [G,69,B]

Hbb =
    Jb64^T * Jb64                                      // [G,B,B]

ga =
    Ja64^T * r64                                       // [G,69]

gb =
    Jb64^T * r64                                       // [G,B]
```

---

# 17. Pose/Shape：69 → Arrowhead 75/B

每 group `g` 属于 frame `t`。

根据 cam2 / cam4 active index：

```cpp
A[t][idx,idx] += Haa[g]                                // [75,75]

Bcross[t][idx,:] += Hab[g]                             // [75,B]

g_frame[t][idx] += ga[g]                               // [75]

C += Hbb[g]                                            // [B,B]

g_beta += gb[g]                                        // [B]
```

因此：

```text
A       [T,75,75]
Bcross  [T,75,B]
C       [B,B]
g_frame [T,75]
g_beta  [B]
```

---

# 18. Pose/Shape anchors

## 18.1 Pose anchor

每 frame：

```cpp
delta_pose =
    body_pose[t] - body_pose_anchor[t]                 // [63]

r_pose =
    delta_pose * pose_anchor_scale                     // [63]
```

累计：

```cpp
A[t][12:75,12:75] +=
    pose_anchor_scale^2 * I63

g_frame[t][12:75] +=
    delta_pose * pose_anchor_scale^2
```

---

## 18.2 Shape anchor

```cpp
beta, dbdz =
    physical_beta_from_latent(z)

r_shape =
    (beta - beta_anchor)
    * shape_anchor_scale                               // [B]

diag =
    dbdz * shape_anchor_scale                          // [B]
```

累计：

```cpp
C += diag(diag^2)

g_beta +=
    diag * r_shape
```

---

# 19. Pose/Shape Arrowhead system

最终：

$$
H=
\begin{bmatrix}
A_1 & 0 & \cdots & B_1\\
0 & A_2 & \cdots & B_2\\
\vdots & & \ddots & \vdots\\
B_1^T & B_2^T & \cdots & C
\end{bmatrix}
$$

```text
A_t : [75,75]
B_t : [75,B]
C   : [B,B]
```

gradient：

```text
g =
[
    g_frame[0],
    ...
    g_frame[T-1],
    g_beta
]
```

---

# 20. Schur solve

对应：

```text
arrowhead_schur.py
```

LM damping：

```cpp
A_lambda[t] =
    A[t] + lambda * I75

C_lambda =
    C + lambda * I_B
```

先 local solve：

```cpp
Xg[t] =
    solve(A_lambda[t], g_frame[t])                     // [75]

XB[t] =
    solve(A_lambda[t], Bcross[t])                      // [75,B]
```

Schur：

```cpp
S =
    C_lambda

rhs =
    -g_beta

for t:
    S   -= Bcross[t]^T * XB[t]
    rhs += Bcross[t]^T * Xg[t]
```

求 beta：

```cpp
delta_beta =
    solve(S, rhs)                                      // [B]
```

回代：

```cpp
delta_frame[t] =
    -Xg[t]
    -XB[t] * delta_beta                                // [75]
```

---

# 21. 三阶段计算图对比

| 部分 | Stage1 | Pose | Pose/Shape |
|---|---|---|---|
| SMPL-X full forward 每个 LM step | 否 | 否 | 否 |
| rotation | 1 个 camera root | 55 joints value / 22 variable | 同 Pose |
| pose blend | 无 | 有 | 有 |
| dynamic shape | 无 | 无，beta fixed | 有 |
| dynamic joints | 无 | 无 | 有 |
| FK | 无 | 有 | 有，含 beta derivative |
| LBS | 无 | 有 | 有 |
| ERP | 有 | 有 | 有 |
| frame active J | 6 / camera | 69 / camera | 69 / camera |
| shared beta J | 无 | 无 | `B=10` |
| H 结构 | 12x12/frame block | 75x75/frame block | Arrowhead |
| solve | dense | dense | Schur |

---

# 22. 主要中间 tensor 的“是否需要长期保存”

## 22.1 建议作为 C++ static cache

这些可以在 Stage build 时存：

### Pose / PoseShape 共用

```text
vertex_ids
targets
selected_posedirs
selected_lbs_weights
left_hand_pose
right_hand_pose
scale
parents
fk_depth_indices
active_indices_cam2
active_indices_cam4
valid_mask
```

### Pose only

```text
selected_v_shaped
fixed joints
```

### Pose/Shape

```text
selected_base_v_shaped
selected_beta_shapedirs
base_joints
d_joints_d_beta
```

---

## 22.2 每个 LIN 必须重新算

### Stage1

```text
R
Jr
points
J_proj
J_camera
H/g
```

### Pose

```text
R
dR
feature
d_feature
pose_offsets
d_pose_offsets
FK
dFK
LBS
dLBS
vertices
d_vertices
ERP J
J_active
H/g
```

### Pose/Shape

Pose 全部 +

```text
beta
d_beta_d_z
dynamic joints
d_joints_d_z
dynamic v_shaped
d_v_shaped_d_z
combined FK derivative
J_beta
A/B/C/g
```

---

# 23. 粗略计算复杂度：用于判断 GPU 候选

> 下面是**符号量级**，不是实测时间。真正优先级仍要由 C++ profiler / GPU profiler 决定。

---

## 23.1 Stage1

每 group：

```text
rotation + ERP + 2x6 local J + 12x12 H/g accumulation

复杂度 ≈ O(N_g)
```

矩阵维数都很小。

**判断：**

- C++ CPU 很适合；
- GPU 单独做 Stage1 的收益可能受 launch overhead 限制；
- 若以后 GPU 化，更适合把“point transform + ERP + J + H/g reduction”合成一个 kernel。

---

## 23.2 Pose pose-offset derivative

最大 contraction 之一：

```text
d_feature       [G,486,66]
selected_posedir[G,486,N,3]

→ d_offsets     [G,N,3,66]
```

符号量：

$$
O(G\cdot N\cdot P\cdot3\cdot66)
$$

其中：

```text
P = 486
```

这是很明显的 dense contraction。

**CPU 候选：**

- BLAS / Eigen matrix multiplication；
- 避免通用 einsum runtime。

**GPU 候选：**

- reshape 后 GEMM / strided batched GEMM；
- 不一定需要自定义 kernel。

---

## 23.3 Pose / PoseShape FK derivative

Pose：

```text
dG [G,55,4,4,66]
```

Pose/Shape：

```text
dG [G,55,4,4,76]
```

每 joint 是很多固定 `4x4` 小矩阵乘法。

符号量：

$$
O(G\cdot J\cdot Q)
$$

但每个任务矩阵极小。

**CPU：**

- 手写 fixed-size 4x4 / Eigen fixed-size matrix；
- 不建议调用重量级通用 GEMM。

**GPU：**

- 非常适合“按 depth，一 joint/group 一线程块或 warp”的 custom kernel；
- 不适合把每个 4x4 独立交给 cuBLAS。

---

## 23.4 LBS derivative

核心：

```text
sum_j weight[n,j] * dA[j] * h[n]
+
T[n] * dh[n]
```

Pose：

```text
Q=66
```

Pose/Shape：

```text
Q=76
```

量级：

$$
O(G\cdot N\cdot J\cdot Q)
$$

这部分同时有：

```text
vertex dimension N
joint dimension 55
derivative dimension 66/76
```

**这是后续 GPU kernel 最值得重点看的区域之一。**

---

## 23.5 ERP + chain Jacobian

```text
J_proj    [G,N,2,3]
d_vertex  [G,N,3,Q]

→ J_uv    [G,N,2,Q]
```

量级：

$$
O(G\cdot N\cdot Q)
$$

容易 point-wise 并行。

GPU 上可以考虑与：

```text
ERP value
ERP J
J_uv
residual
row scale
```

融合。

---

## 23.6 H/g GEMM

Pose：

```text
J [G,2N,69]
```

每 group：

```text
J^T J : [69,69]
J^T r : [69]
```

量级：

$$
O(G\cdot2N\cdot69^2)
$$

Pose/Shape 额外：

$$
O(G\cdot2N\cdot69B)
$$

和：

$$
O(G\cdot2N\cdot B^2)
$$

**这部分是标准 dense matrix multiply。**

优先考虑：

- CPU：BLAS；
- GPU：cuBLAS / batched GEMM；
- 不建议第一版就手写 GEMM kernel。

---

# 24. 当前最自然的可合并项

这里分成：

```text
C++ CPU 第一版
```

和：

```text
后续 GPU
```

不要混在第一版里一次做完。

---

## 24.1 Stage1 可合并

当前逻辑：

```text
point transform
→ ERP
→ projection J
→ orientation J
→ local J
→ row scale
→ H/g
```

C++ 可以直接写成：

```cpp
for point:
    compute r_i
    compute J_i [2,12]

    H += J_i^T J_i
    g += J_i^T r_i
```

即：

```text
不用 materialize global J
```

这是安全且明显的融合。

---

## 24.2 Pose：feature → offsets

当前：

```text
feature
d_feature
→ offsets
d_offsets
```

可以视为两次 matrix contraction：

```text
offsets   = P^T * feature
d_offsets = P^T * d_feature
```

C++ CPU 第一版建议明确写成矩阵乘，不需要复制 Python einsum API。

后续 GPU：

```text
GEMM / batched GEMM
```

---

## 24.3 Pose：FK value + derivative

当前已经在同一逻辑中传播：

```text
G
dG
```

C++ CPU 应继续保持一次树遍历：

```cpp
for depth:
    value
    derivative
```

不要先完整算 value 再第二遍重算 derivative 的父链。

---

## 24.4 Pose / PoseShape：LBS value + derivative

当前已经共享：

```text
A
dA
h
dh
```

后续 GPU 有较强融合潜力：

```text
weights × A
weights × dA
vertex apply
derivative apply
```

甚至可以避免显式保存：

```text
selected_transforms
d_selected_transforms
```

但**第一版 C++ CPU reference 不建议直接做这个优化**，先逐中间量匹配 Python。

---

## 24.5 ERP + derivative + row scaling

GPU 候选：

```text
vertices
→ uv/J_proj
→ J_uv
→ residual
→ row scaling
```

都是 point-local。

可以考虑单 kernel：

```text
erp_residual_jacobian_kernel
```

输出：

```text
weighted r
weighted J
```

或者进一步直接参与 H/g reduction。

---

## 24.6 H/g 不建议手写 GPU 矩阵乘

当前：

```text
69 x 2N
```

属于相对规整的 GEMM。

第一选择：

```text
cuBLAS
```

而不是：

```text
custom J^T J kernel
```

除非 profiler 后证明：

```text
materialize J + GEMM
```

的显存流量成为主要瓶颈。

---

# 25. 建议的 C++ CPU 移植顺序

不要一开始直接搬完整 pipeline。

---

## Step 1：公共数学

实现并逐项验证：

```text
skew()
axis_angle_to_matrix()
so3_right_jacobian()
rotated_point_jacobian()

erp_projection()
erp_projection_with_jacobian()
```

测试：

```text
Python vs C++
```

输入用固定随机 tensor dump。

---

## Step 2：Stage1

实现：

```text
Stage1 cache input
→ transform
→ ERP/J
→ H/g
```

验证：

```text
points
uv
J_proj
J_orient
J_local
H
g
```

都匹配后，再接 LM。

---

## Step 3：Pose value path

暂时**不要 Jacobian**：

```text
R
→ pose_feature
→ pose_offsets
→ FK
→ LBS
→ vertices
→ ERP
```

检查：

```text
R
pose_offsets
relative_transforms
v_skin
vertices
uv
```

---

## Step 4：Pose Jacobian

逐层加入：

```text
dR
→ d_feature
→ d_offsets
→ dFK
→ dLBS
→ d_vertices
→ J_active
```

每一层 Python dump 对 C++。

---

## Step 5：Pose H/g

先生成 `J_active`：

```text
[2N,69]
```

然后 BLAS：

```text
J^T J
J^T r
```

验证 69 → 75 scatter 和 pose anchor。

---

## Step 6：Pose/Shape value

加入：

```text
latent beta
dynamic v_shaped
dynamic joints
dynamic FK
```

先只验证 value。

---

## Step 7：Pose/Shape beta derivative

加入：

```text
d_beta/dz
d_v_shape/dz
d_joints/dz
combined dFK
combined dLBS
J_beta
```

---

## Step 8：Arrowhead

实现：

```text
A
B
C
g_frame
g_beta
```

然后实现 / 接入 Schur。

---

## Step 9：完整 LM trajectory

最后验证：

```text
status
accepted/rejected
lambda
objective
parameters
```

不要在 kernel 级 parity 还没稳定时先调 LM。

---

# 26. C++ 模块建议

第一版不需要做复杂框架，可直接按数学拆：

```text
staged_lm_cpp/
├── so3.h/.cpp
├── erp.h/.cpp
├── stage1.h/.cpp
├── pose_forward.h/.cpp
├── pose_jacobian.h/.cpp
├── pose_normal_equations.h/.cpp
├── pose_shape_forward.h/.cpp
├── pose_shape_jacobian.h/.cpp
├── arrowhead.h/.cpp
└── lm.h/.cpp
```

数据结构建议只有必要 cache：

```cpp
PoseStaticCache
PoseBatchCache
PoseShapeStaticCache
PoseShapeBatchCache
```

先不要把 Python 中所有 timing / compatibility wrapper 1:1 搬过去。

---

# 27. C++ 第一版建议的数据类型

为了和当前 Python production 对齐：

```text
geometry / residual / Jacobian intermediates:
    float32

H / g / linear solve:
    float64
```

当前 Python 明确在：

```text
J/r → H/g
```

前转换到 FP64。

因此 C++ reference 第一版也应保留这个边界。

不要一开始全 double 或全 float，否则不好做 exact / tight parity 定位。

---

# 28. GPU 算子初步边界

等 C++ CPU 跑通后，建议按下面边界 profile。

---

## Kernel/GEMM A：rotation

```text
axis_angle
→ R
→ Jr
→ dR
```

输入：

```text
[G,22,3] variable joints
+ frozen hand rotations
```

输出：

```text
R  [G,55,3,3]
dR [G,55,3,3,66]
```

---

## GEMM B：pose blend

```text
feature / d_feature
× selected_posedirs
→ offsets / d_offsets
```

优先 GEMM，不急着 custom kernel。

---

## Kernel C：depth-wise FK

```text
R,dR,joints,djoints
→ relative transforms
→ d relative transforms
```

这是 fixed-small-matrix tree kernel 候选。

---

## Kernel D：selected LBS

```text
weights
A/dA
v_posed/dv_posed
→ vertices/dvertices
```

高优先级 custom kernel 候选。

---

## Kernel E：ERP chain

```text
vertex/dvertex
→ uv
→ J_proj
→ J_uv
→ residual
→ weighted rows
```

高并行 point-wise。

---

## GEMM F：normal equations

Pose：

```text
J69^T J69
J69^T r
```

Pose/Shape：

```text
Ja^T Ja
Ja^T Jb
Jb^T Jb
Ja^T r
Jb^T r
```

优先 cuBLAS。

---

## Kernel G：scatter / Arrowhead reduction

```text
69-block
→ frame 75-block
→ A/B/C/g
```

规模小，可和前后步骤结合评估，但不是第一优先级。

---

# 29. CPU / GPU 最值得观察的中间量

做 profiler 时建议至少按下面归因：

```text
1. rotation_value_jacobian
2. pose_feature_offsets
3. combined_fk
4. selected_lbs
5. erp_projection_chain
6. row_scaling
7. normal_equation_gemm
8. frame_scatter
9. schur
10. true_objective
```

不要只看整个：

```text
Pose Stage
Pose/Shape Stage
```

否则很难决定 GPU 应该搬哪一块。

---

# 30. Python → C++ parity checkpoint

建议 Python 端新增一次性 dump 脚本，把固定 frame/group 的中间量保存为 NPZ。

---

## 30.1 Stage1 dump

```text
local_vertices
rotation_center
orient
translation
R
Jr
points
uv
J_proj
d_point_d_orient
J_local
row_scales
H_frame
g_frame
```

---

## 30.2 Pose dump

```text
group_orientations
group_translations
group_body_poses

R
dR

feature
d_feature

pose_offsets
d_pose_offsets
v_posed

relative_transforms
d_relative_transforms

v_skin
d_v_skin

vertices
d_vertices

uv
J_proj
J_active

row_scales
H_active
g_active
H_frame
g_frame
```

---

## 30.3 Pose/Shape dump

Pose 全部 +

```text
beta_latent
beta
d_beta_d_latent

dynamic_joints
d_joints_d_latent

v_shaped
d_v_shaped_d_latent

d_relative_pose
d_relative_beta

d_vertices_pose
d_vertices_beta

J_beta

Haa
Hab
Hbb
ga
gb

A
Bcross
C
g_frame
g_beta
```

这份 dump 会成为 C++ CPU 最重要的 reference。

---

# 31. 最终 C++ CPU 目标接口建议

第一版可以先做到：

```cpp
Stage1NormalEquations compute_stage1_normal_equations(
    const Stage1Cache& cache,
    const VectorXf& parameters,
    const VectorXf& row_scales
);

PoseNormalEquations compute_pose_normal_equations(
    const PoseBatchCache& cache,
    const VectorXf& parameters,
    const Tensor& row_scales,
    const PoseAnchors& anchors
);

PoseShapeNormalEquations compute_pose_shape_normal_equations(
    const PoseShapeBatchCache& cache,
    const VectorXf& parameters,
    const Tensor& row_scales,
    const PoseAnchors& pose_anchors,
    const VectorXf& beta_anchor
);
```

其中输出只需要当前 LM 真正消费的：

### Stage1 / Pose

```text
residual
H
g
```

### Pose/Shape

```text
residual
A
B
C
g_frame
g_beta
```

先不要让第一版 C++ 承担：

```text
JSON
可视化
debug history
Python compatibility
```

---

# 32. 现在最重要的三个判断

从当前计算图来看：

### 1. C++ CPU 是合理的下一步

因为现在所有热路径已经写成：

```text
明确的赋值
小矩阵 FK
dense contraction
selected LBS
ERP
H/g
```

不再依赖 PyTorch autograd 才能定义数学。

### 2. 不建议直接整套改 CUDA

因为不同块性质不同：

```text
pose blend / H-g     → GEMM
FK                   → small-matrix custom kernel
LBS                  → custom kernel 候选
ERP                   → point-wise kernel
Schur                 → 小系统，优先 CPU/C++ 或后判断
```

直接“整个 Stage 一个 CUDA kernel”并不自然。

### 3. C++ CPU 的真正作用不仅是提速

更重要的是获得一条：

```text
Python analytic reference
        ↓
C++ CPU deterministic reference
        ↓
GPU implementation
```

的验证链。

这样 GPU 每次优化都可以检查：

```text
value
J
H/g
LM trajectory
```

而不是只看最终合图是否“差不多”。

---

# 33. 推荐下一步

当前最适合立即做：

```text
Python intermediate dump
    ↓
Stage1 C++ CPU
    ↓
Pose value C++ CPU
    ↓
Pose Jacobian C++ CPU
    ↓
Pose/Shape
```

其中第一项建议先生成：

```text
cpp_reference_dump.npz
```

并固定一个典型 frame / group，例如当前长期用于验证的 frame 513。

这会让后续 C++ 移植从“照代码抄”变成：

```text
每完成一层
→ 和 Python intermediate tensor 对齐
→ 再进入下一层
```

风险会低很多。
