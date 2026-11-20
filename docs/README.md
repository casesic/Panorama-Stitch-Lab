# 文档索引

这个仓库的文档按「还约束不约束代码」分层。判断标准只有一条：**改代码时会不会被它挡住**。

| 位置 | 是什么 | 什么时候看 |
|---|---|---|
| [adr/](adr/) | 5 篇决策记录（ADR）：为什么统一到 ERP、为什么用 C++ 重写拟合、为什么 TensorRT 只做显式后端、为什么拟合拆成三段、为什么 DenseKP 生产交付这么拆 | 想知道「当初为什么这么定」，5 分钟能读完一篇 |
| [Architecture/formulate/](Architecture/formulate/) | **仍然约束当前代码行为**的设计：Phase 2.2/4/4.1 主链语义、Python↔C++ 边界契约、HMR2 初始化、ROI/matching、delivery 层 | 要改拟合、改交付边界、改 ROI 匹配时 |
| [Architecture/run/](Architecture/run/) | 运行手册：C++ staged-LM 的 O4 单帧/多帧跑法、CUDA→C++ 链路 | 要在机器上把 C++ 拟合跑起来时 |
| [Architecture/staged_lm/](Architecture/staged_lm/) | 公式推导与代码对照：joint / pose 两条线的数学走查、符号迁移、早停审计 | 要核对雅可比、改数学时 |
| [Architecture/archive/](Architecture/archive/) | 历史 bakeoff / attribution / profiling / CUDA parity 记录 | 只作追溯；结论可能已被后续实现取代 |
| [changelogs/](changelogs/) | 按日期的大改记录（目前 1 篇，2026-08-03） | 想知道某次大改动了什么 |

`formulate/` 里的 [README](Architecture/formulate/README.md) 说明了归档规则：**bakeoff 类文档默认写进 `archive/`；只有结论会长期约束实现的文档才放 `formulate/`。**

## 三种读法

**第一次来，想知道这是什么**：根目录 [README](../README.md) → [adr/0001](adr/0001-erp-as-common-frame.md) → [adr/0004](adr/0004-staged-lm-formulation.md)。大约 15 分钟。

**要把 C++ 拟合跑起来**：[Architecture/run/staged_lm_cpp_o4_single_frame](Architecture/run/staged_lm_cpp_o4_single_frame.md) → 根目录 README 的构建命令 → [src/README.md](../src/README.md)（输入输出与性能口径）。

**要改拟合数学**：[Architecture/staged_lm/overview](Architecture/staged_lm/overview.md)（三段维度与求解器）→ `joint_math_walkthrough` / `pose_math_walkthrough` → `math_to_code` 对照 → 改完跑 `pytest -q tests/human_fitting/staged_lm` 与 `ctest --test-dir cpp/staged_lm/build`。

## 写作约定

- 新文档先问一句：它**约束**实现，还是**记录**一次实验？前者进 `formulate/`（或 `run/`），后者进 `archive/`。
- 数字要带口径：机器、fixture、点数、policy、是否 warmup。性能数字不带口径没有可比性。
- 不要就地改写历史文档的结论——结论变了就新增一篇，并在旧文顶部加一行「已被 X 取代」。
