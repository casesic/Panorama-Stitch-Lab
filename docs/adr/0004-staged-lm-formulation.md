# ADR-0004：SMPL-X 拟合拆成三段，而不是一个大 LM

- **状态**：已生效（2026-11-20 记录；决定早于此）
- **影响面**：`cpp/staged_lm/`、`src/lib_pano/human_fitting/`
- **一句话**：把 85 维的联合优化拆成「Stage 1 刚体 → Pose warmup → 联合 pose+beta」三段，每段有自己的参数化、残差、解析 Jacobian 与求解器；跨帧共享一个 bounded beta。

## 背景

逐帧 SMPL-X 拟合同时要定相机外参、人体姿态和体型。放在一个 LM 里做，初值稍微偏一点就整锅不收敛；而且三段量纲差别大（12 维相机 / 75 维 pose / 10 维 beta），阻尼策略很难一套吃下。

## 决定

| 阶段 | 参数维度 | 残差 | Jacobian / 法方程 | 求解器 |
|---|---:|---|---|---|
| Stage 1 rigid | `T × 12`（两台相机 orient/transl） | 双视角 ERP 重投影 | 解析 orient/transl Jacobian；直接组装 `H/g` | dense Cholesky，失败回退 dense solve |
| Pose warmup | `75T`（每帧 12 相机 + 63 pose） | ERP 重投影 + pose anchor | 解析批量 pose Jacobian | dense Cholesky，失败回退 dense solve |
| Joint refinement | `75T + 10`（每帧 85 = 75 + 10 beta） | ERP 重投影 + pose/shape anchor | 解析批量 pose 与共享 beta Jacobian；Arrowhead `A/B/C/g` | Arrowhead Schur，失败回退 dense |

（表出自 [staged_lm/overview](../Architecture/staged_lm/overview.md)）

配套约定：

- 三种阶段定义是**固定**的，由 `stage_definitions/` 构造；`lm/runner.py` 与阶段无关，只吃每次接受的迭代的 `Linearization`，并控制阻尼、接受/拒绝、收敛判定与 dense/Arrowhead 求解分派。
- 跨帧 beta 用 **bounded latent**：Stage1→Pose 复制 rigid 12 维、由 pose anchor 初始化 63 维 body pose；Pose→PoseShape 复制 75 维，并用 `z = atanh(beta_anchor / 2.5)` 初始化 10 维 beta latent。
- Stage1 与 Pose/Shape 各有最小 LM adapter；pipeline 模块只做输入构造与 handoff，不含数学。

## 依据

1. **可测**：三段各自有 value / 解析 Jacobian / `H/g` 的独立 reference 与 self-test（CTest 里 11 个用例），跨后端 stage boundary、trial/accepted/rejected 计数、objective/linearization 调用次数都能对齐，`PIPELINE_FIRST_DIVERGENCE=NONE`。
2. **成本可归因**：[staged_lm_cpp_pipeline_reference](../Architecture/archive/staged_lm_cpp_pipeline_reference.md) 显示 Pose 占 68.24%、PoseShape 占 30.97%、Stage1 只占 0.67%，所以优化火力可以集中在 pose 导数上，不必动 Stage 1。
3. **求解器可换**：Arrowhead Schur 与 dense 两条路可以互相回退，不用改残差。

## 代价与已知限制

- **误差会传导**：Stage 1 的刚体误差会被后两段继承，没有回头联合修外参。
- **阶段间 handoff 要一致**：beta latent 用 `atanh` 参数化，超出 ±2.5 的 anchor 会落在定义域外，需要有边界处理。
- **不是单一目标函数**：三段分别收敛，最终目标值与「一次性联合优化」的全局最优没有可比性。

## 未在文档中写明（待作者补充）

- 为什么 beta 用 `atanh(beta/2.5)` 的有界参数化，而不是直接优化 beta。
- 三段的接受/拒绝阈值与阻尼初始值的选取依据。
- 是否做过「不拆、一次性联合」的对照实验。

## 相关文档

- [staged_lm/overview](../Architecture/staged_lm/overview.md)
- [joint_math_walkthrough](../Architecture/staged_lm/joint_math_walkthrough.md) / [pose_math_walkthrough](../Architecture/staged_lm/pose_math_walkthrough.md)
- [math_to_code](../Architecture/staged_lm/math_to_code.md)
