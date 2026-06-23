"""SMPL-X 序列初始化工具：从 SMPL-54 序列初始化 SMPL-X，以及按索引切片 SMPL-X 序列。"""

from __future__ import annotations

from typing import Sequence

from .smplx_sequence import SMPLX_Seq_Forward_Module

from .utils import build_smpl_init_seq


def init_smplx_from_smpl54_seq(smpl54, index_seq):
    """按索引子集从 SMPL-54 序列初始化对应的 SMPL-X 序列。"""
    sub = build_smpl_init_seq(smpl54, index_seq)
    return SMPLX_Seq_Forward_Module.init_from_smpl54_seq(sub)


def slice_smplx_seq(
    smplx_seq: SMPLX_Seq_Forward_Module.SMPLX_Param_Seq,
    index_seq: Sequence[int],
) -> SMPLX_Seq_Forward_Module.SMPLX_Param_Seq:
    """按 index_seq 抽取帧，复制各姿态/平移参数及共享的 beta/scale，返回新的 SMPL-X 序列。"""
    out = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(
        num_batch=len(index_seq), gender=smplx_seq.gender
    )
    for dst, src in enumerate(index_seq):
        out.orient[dst] = smplx_seq.orient[src]
        out.transl[dst] = smplx_seq.transl[src]
        out.body_pose[dst] = smplx_seq.body_pose[src]
        out.left_hand_pose[dst] = smplx_seq.left_hand_pose[src]
        out.right_hand_pose[dst] = smplx_seq.right_hand_pose[src]
    out.beta = smplx_seq.beta.clone()
    out.scale = smplx_seq.scale.clone()
    return out
