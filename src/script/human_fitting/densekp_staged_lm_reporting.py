"""Profiling, parity hashes, and human-readable reports for the E2E runner."""

from __future__ import annotations

import json
from pathlib import Path
import statistics


def load_numeric_profile(path: Path) -> dict[str, float | str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    numeric = {
        key: float(value) for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if isinstance(values.get("erp_remap_backend"), str):
        numeric["erp_remap_backend"] = values["erp_remap_backend"]
    return numeric


def mean(rows: list[dict[str, float]], key: str) -> float:
    return statistics.fmean(row.get(key, 0.0) for row in rows) if rows else 0.0


def remap_backend_is_cuda(rows: list[dict]) -> bool:
    return bool(rows) and all(
        row.get("erp_remap_backend") == "cuda" for row in rows
    )


def write_summary(path: Path, report: dict) -> None:
    steady = report["steady_state"]
    stages = steady["average_stage_ms"]
    ranked = sorted(stages.items(), key=lambda item: item[1], reverse=True)
    lines = [
        "# DenseKP + CUDA staged-LM 端到端性能归因",
        "",
        f"- 数据：`20260311_riding`，{report['frames'][0]}–{report['frames'][-1]}，"
        f"共 {report['frame_count']} 帧",
        f"- 冷启动帧：{report['warmup']['frame']}（单独统计，不进入稳态均值）",
        f"- 有效/跳过：{report['valid_frame_count']}/{report['skipped_frame_count']}，"
        f"跳过帧：`{report['skipped_frames']}`",
        f"- 稳态：{steady['frame_count']} 帧，{steady['ms_per_frame']:.2f} ms/frame，"
        f"{steady['fps']:.3f} FPS",
        "", "## 稳态平均耗时", "", "| 阶段 | ms/frame | 占比 |",
        "|---|---:|---:|",
    ]
    total = float(steady["ms_per_frame"])
    for name, value in ranked:
        percent = 100.0 * value / total if total else 0.0
        lines.append(f"| {name} | {value:.3f} | {percent:.2f}% |")
    lines.extend(["", "## Top 3 瓶颈", ""])
    for index, (name, value) in enumerate(ranked[:3], 1):
        lines.append(f"{index}. `{name}`：{value:.3f} ms/frame")
    lines.extend([
        "", "## 下一步建议", "",
        "1. 优先处理实测占比最高的 DenseKP delivery 阶段，不改 LM 数学。",
        "2. detector 或 ORT 位居前列时，分别评估复用 bbox 或模型执行方式。",
        "3. ERP remap 成为主要占比后，再评估 GPU remap 或复用上游 ERP。",
        "", "`DENSEKP_END_TO_END_PROFILING_READY=PASS`", "",
    ])
    for marker in (
        "DENSEKP_RUNTIME_IMAGE_IO_READY", "DENSEKP_CUDA_ERP_REMAP_READY",
    ):
        if report.get(marker) == "PASS":
            lines.extend([f"`{marker}=PASS`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
