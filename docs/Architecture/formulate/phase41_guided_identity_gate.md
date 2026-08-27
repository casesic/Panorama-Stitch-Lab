# Phase 4.1：Guided identity gate

## 结论

Phase 4.1 将 guided acceptance 从“每个 track 独立取 crop 内第一个 match”改为：

```text
所有 active tracks
→ 每个 guided crop 保留全部 cross-camera matches
→ 球面 DenseKP 几何去重（source track 仅为 provenance）
→ 完整 previous-track × current-candidate cost matrix
→ 最大匹配数优先、总 cost 最小的全局一对一 assignment
→ track-side + candidate-side ambiguity margin
→ 仅唯一且绝对 temporal cost 合格的身份进入 fitting
```

DenseKP score、`keypoint_value` 和 appearance embedding 均未参与。LM、HMR2、bbox
predictor、K=3、DenseKP/YOLOX B=1 均未修改。

## 2214 forensic

真实重跑结果并不支持“两个 guided crop 吸附到同一个 current person”的假设：

```text
previous tracks         = 2
raw guided candidates   = 2
unique candidates       = 2
dedup groups            = [[0], [1]]
```

完整 temporal cost matrix（degree）：

| Previous track | Candidate 0 | Candidate 1 |
| ---: | ---: | ---: |
| Track 0 | 0.4410 | 16.2456 |
| Track 1 | 15.6190 | 0.5124 |

全局结果：

- Track 0 → Candidate 0；track/candidate 两侧 margin 分别约 15.80°/15.18°；
- Track 1 → Candidate 1；track/candidate 两侧 margin 分别约 15.11°/15.73°；
- 两者均为清晰的一对一身份，不是 margin 擦线，也不是 duplicate candidate。

FULL reference 的 cam2=1、cam4=4、matched=1，只说明 FULL cam2 detector 没有提供
Track 1 的跨相机 match。FULL 不是 identity 真值，不能据此将几何上清晰可分的 Track 1
强制判错。因此本次归类为：

```text
TRUE_GUIDED_DETECTOR_MISS_RECOVERY (evidence strongly supports)
```

即 guided path 补回了 FULL detector 漏检的人，而不是“错误自信吸附”。若未来出现一个
candidate 同时接近两条 track，candidate-side margin 会输出 `GUIDED_AMBIGUOUS`，并由
现有 orchestrator 在同一帧只执行一次 FULL recovery。

专项 artifact：

```text
data_process/Vis/densekp_phase41_identity_forensics_2214/
  phase41_2214_identity_forensics.json
```

## 真实序列 2213–2221

使用 crop margin=0.1、K=3。guided correctness 行如下：

| Frame | Track | Identity | Candidate | Temporal cost | Recovery |
| ---: | ---: | --- | ---: | ---: | --- |
| 2214 | 0 | GUIDED_ACCEPTED | 0 | 0.441° | no |
| 2214 | 1 | GUIDED_ACCEPTED | 1 | 0.512° | no |
| 2215 | 0 | GUIDED_ACCEPTED | 0 | 0.433° | no |
| 2216 | 0 | GUIDED_ACCEPTED | 0 | 0.395° | no |
| 2217 | 0 | GUIDED_ACCEPTED | 0 | 0.320° | no |
| 2218 | 0 | GUIDED_ACCEPTED | 0 | 0.347° | no |
| 2219 | 0 | GUIDED_ACCEPTED | 0 | 0.455° | no |
| 2220 | 0 | GUIDED_ACCEPTED | 0 | 0.637° | no |
| 2221 | 0 | GUIDED_ACCEPTED | 0 | 0.607° | no |
| 2221 | 2 | GUIDED_ACCEPTED | 1 | 0.801° | no |

Track 0 连续性保持。主 Track 0 的 C++ 回归为 LM 8/8 success、warm state 8/8 exact。

## Margin bake-off

2214 的最小双侧分离超过 15°，因此下列 debug margin 都得到相同身份结论：

| margin_deg | accepted | ambiguous | recovery | known identity errors |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 2 | 0 | 0 | 0 |
| 0.25 | 2 | 0 | 0 | 0 |
| 0.50 | 2 | 0 | 0 | 0 |
| 1.00 | 2 | 0 | 0 | 0 |
| 2.00 | 2 | 0 | 0 | 0 |

样本很小，0.25°/1.25 ratio 只是当前 debug 默认值，不宣称已由数据充分标定。

## 性能

63-sample steady-state，ERP remap 不计入：

| Metric | Phase 4 | Phase 4.1 |
| --- | ---: | ---: |
| K | 3 | 3 |
| detector calls/frame | 1.33 | 1.33 |
| DenseKP calls/frame | 3.89 | 3.89 |
| mixed delivery | 84.94 ms | 98.59 ms |
| delivery FPS | 11.77 | 10.14 |
| recovery rate | — | 0% |

Identity gate correctness pass 的逐 guided-frame median 为约 0.55 ms；按 K=3 全部 frame
摊销约 0.80 ms/frame，满足 `<1 ms/frame` 预算。多 track frame 的 Python forensic
单次约 2.4–2.7 ms，若人物数量进一步增加可再优化，但当前不影响 100 ms gate。

性能 artifact：

```text
data_process/Vis/densekp_phase41_track_guided_2213_2221/phase4_bakeoff.json
```

## Tests 与 decision flags

```text
47 passed
```

```text
GUIDED_CANDIDATE_RETAIN_ALL=PASS
GUIDED_CANDIDATE_DEDUP=PASS
ERP_SEAM_CANDIDATE_DEDUP=PASS

GLOBAL_TRACK_CANDIDATE_ASSIGNMENT=PASS
GLOBAL_ONE_TO_ONE_IDENTITY=PASS
TRACK_SIDE_AMBIGUITY_GATE=PASS
CANDIDATE_SIDE_AMBIGUITY_GATE=PASS

FRAME2214_IDENTITY_FORENSICS=PASS
FRAME2214_AMBIGUITY_DETECTED=NO
FRAME2214_FORENSIC_CONCLUSION=TRUE_GUIDED_DETECTOR_MISS_RECOVERY

SAME_FRAME_FULL_RECOVERY_IDENTITY=PASS
TRACK0_SEQUENCE_PERSISTENCE=PASS
TRACK1_FALSE_GUIDED_ACCEPT_FIXED=UNRESOLVED_NOT_DEMONSTRATED_FALSE

DENSEKP_CONFIDENCE_USED=NO
IDENTITY_GATE_OVERHEAD_MS=0.80_K3_AMORTIZED

PHASE4_BASELINE_MIXED_DELIVERY_MS=84.94
PHASE41_MIXED_DELIVERY_MS=98.59
PHASE41_MIXED_DELIVERY_FPS=10.14

TRACK_GUIDED_DELIVERY_10FPS_GATE=PASS
GUIDED_IDENTITY_GATE=PASS
TRACK_GUIDED_PRODUCTION_CANDIDATE=YES
ASYNC_10FPS_CANDIDATE=YES

CPP_LM_MATH_CHANGED=NO
ERP_REMAP_INCLUDED_IN_TARGET=NO
```

这里 `TRACK1_FALSE_GUIDED_ACCEPT_FIXED` 不写成 PASS，是因为 forensic 证据表明它并非
已证实的 false accept；正确行为是保留清晰的一对一 guided assignment，而不是根据 FULL
漏检制造假阴性。
