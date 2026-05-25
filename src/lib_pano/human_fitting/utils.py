from __future__ import annotations

from typing import Any, Sequence

from lib_dynamic_hvip.fitting.smpl_seq_forward_module import SMPL_Seq_Forward_Module as SMPLFitter


def build_smpl_init_seq(smpl_batch_seq: Any, index_seq: Sequence[int]):
    smpl_seq_init = SMPLFitter.SMPL_Param_Seq(num_batch=len(index_seq))
    for seq_idx, frame_idx in enumerate(index_seq):
        smpl_seq_init.orient[seq_idx] = smpl_batch_seq.orient[frame_idx]
        smpl_seq_init.pose[seq_idx] = smpl_batch_seq.pose[frame_idx]
        smpl_seq_init.transl[seq_idx] = smpl_batch_seq.transl[frame_idx]
    smpl_seq_init.beta = smpl_batch_seq.beta.clone()
    smpl_seq_init.scale = smpl_batch_seq.scale.clone()
    return smpl_seq_init
