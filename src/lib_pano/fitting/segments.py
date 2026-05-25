def split_long_segments(
    segments: list[list[int]],
    max_len: int,
) -> list[list[int]]:
    """将超过 max_len 帧的段切成等长子段。"""
    if max_len <= 0:
        return segments
    result: list[list[int]] = []
    for seg in segments:
        for i in range(0, len(seg), max_len):
            result.append(seg[i:i + max_len])
    return result


def split_reconstructable_subsegments(frame_indices: list[int], reconstructable: dict[int, bool]) -> list[list[int]]:
    subsegments: list[list[int]] = []
    current: list[int] = []
    for frame_idx in frame_indices:
        if reconstructable.get(frame_idx, False):
            if current and frame_idx != current[-1] + 1:
                subsegments.append(current)
                current = []
            current.append(frame_idx)
        elif current:
            subsegments.append(current)
            current = []
    if current:
        subsegments.append(current)
    return subsegments
