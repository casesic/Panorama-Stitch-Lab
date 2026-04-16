from __future__ import annotations

import torch

from lib_dynamic_hvip.smpl.smpl_lib import get_smplx_batch, forward_smplx_tensor_batch


SMPLX_BODY_22 = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine1": 3,
    "L_Knee": 4,
    "R_Knee": 5,
    "Spine2": 6,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10,
    "R_Foot": 11,
    "Neck": 12,
    "L_Collar": 13,
    "R_Collar": 14,
    "Head": 15,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
    "L_Elbow": 18,
    "R_Elbow": 19,
    "L_Wrist": 20,
    "R_Wrist": 21,
    "Nose": 55,
    "L_Eye": 57,
    "R_Eye": 56,
    "L_Ear": 59,
    "R_Ear": 58,
}

SMPLX_JOINT_MAP = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine1": 3,
    "L_Knee": 4,
    "R_Knee": 5,
    "Spine2": 6,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10,
    "R_Foot": 11,
    "Neck": 12,
    "L_Collar": 13,
    "R_Collar": 14,
    "Head": 15,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
    "L_Elbow": 18,
    "R_Elbow": 19,
    "L_Wrist": 20,
    "R_Wrist": 21,
    "Nose": 55,
    "R_Eye": 56,
    "L_Eye": 57,
    "R_Ear": 58,
    "L_Ear": 59,
    "L_BigToe": 60,
    "L_SmallToe": 61,
    "L_Heel": 62,
    "R_BigToe": 63,
    "R_SmallToe": 64,
    "R_Heel": 65,
}

SMPLX_LHAND_15 = {
    "L_Index1": 25,
    "L_Index2": 26,
    "L_Index3": 27,
    "L_Middle1": 28,
    "L_Middle2": 29,
    "L_Middle3": 30,
    "L_Pinky1": 31,
    "L_Pinky2": 32,
    "L_Pinky3": 33,
    "L_Ring1": 34,
    "L_Ring2": 35,
    "L_Ring3": 36,
    "L_Thumb1": 37,
    "L_Thumb2": 38,
    "L_Thumb3": 39,
}

SMPLX_RHAND_15 = {
    "R_Index1": 40,
    "R_Index2": 41,
    "R_Index3": 42,
    "R_Middle1": 43,
    "R_Middle2": 44,
    "R_Middle3": 45,
    "R_Pinky1": 46,
    "R_Pinky2": 47,
    "R_Pinky3": 48,
    "R_Ring1": 49,
    "R_Ring2": 50,
    "R_Ring3": 51,
    "R_Thumb1": 52,
    "R_Thumb2": 53,
    "R_Thumb3": 54,
}

SMPLX_ALL = {**SMPLX_JOINT_MAP, **SMPLX_LHAND_15, **SMPLX_RHAND_15}

COCO133_LHAND = {
    "L_Hand_wrist": 91,
    "L_Hand_thumb1": 92,
    "L_Hand_thumb2": 93,
    "L_Hand_thumb3": 94,
    "L_Hand_thumb4": 95,
    "L_Hand_index1": 96,
    "L_Hand_index2": 97,
    "L_Hand_index3": 98,
    "L_Hand_index4": 99,
    "L_Hand_middle1": 100,
    "L_Hand_middle2": 101,
    "L_Hand_middle3": 102,
    "L_Hand_middle4": 103,
    "L_Hand_ring1": 104,
    "L_Hand_ring2": 105,
    "L_Hand_ring3": 106,
    "L_Hand_ring4": 107,
    "L_Hand_pinky1": 108,
    "L_Hand_pinky2": 109,
    "L_Hand_pinky3": 110,
    "L_Hand_pinky4": 111,
}

COCO133_RHAND = {
    "R_Hand_wrist": 112,
    "R_Hand_thumb1": 113,
    "R_Hand_thumb2": 114,
    "R_Hand_thumb3": 115,
    "R_Hand_thumb4": 116,
    "R_Hand_index1": 117,
    "R_Hand_index2": 118,
    "R_Hand_index3": 119,
    "R_Hand_index4": 120,
    "R_Hand_middle1": 121,
    "R_Hand_middle2": 122,
    "R_Hand_middle3": 123,
    "R_Hand_middle4": 124,
    "R_Hand_ring1": 125,
    "R_Hand_ring2": 126,
    "R_Hand_ring3": 127,
    "R_Hand_ring4": 128,
    "R_Hand_pinky1": 129,
    "R_Hand_pinky2": 130,
    "R_Hand_pinky3": 131,
    "R_Hand_pinky4": 132,
}


class SMPLX_Seq_Forward_Module:
    class SMPLX_Param:
        def __init__(self, gender: str = "neutral"):
            self.orient = torch.zeros(3)
            self.transl = torch.zeros(3)
            self.body_pose = torch.zeros(63)
            self.left_hand_pose = torch.zeros(45)
            self.right_hand_pose = torch.zeros(45)
            self.beta = torch.zeros(10)
            self.scale = torch.ones(1)
            self.gender = gender

    class SMPLX_Param_Seq:
        def __init__(self, num_batch: int = 1, gender: str = "neutral"):
            self.num_batch = num_batch
            self.orient = torch.zeros(num_batch, 3)
            self.transl = torch.zeros(num_batch, 3)
            self.body_pose = torch.zeros(num_batch, 63)
            self.left_hand_pose = torch.zeros(num_batch, 45)
            self.right_hand_pose = torch.zeros(num_batch, 45)
            self.beta = torch.zeros(10)
            self.scale = torch.ones(1)
            self.gender = gender

        def get_param_at_frame(self, idx: int) -> "SMPLX_Seq_Forward_Module.SMPLX_Param":
            p = SMPLX_Seq_Forward_Module.SMPLX_Param(gender=self.gender)
            p.orient = self.orient[idx]
            p.transl = self.transl[idx]
            p.body_pose = self.body_pose[idx]
            p.left_hand_pose = self.left_hand_pose[idx]
            p.right_hand_pose = self.right_hand_pose[idx]
            p.beta = self.beta
            p.scale = self.scale
            return p

    def __init__(self, gender: str = "neutral"):
        self.gender = gender
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = get_smplx_batch(gender=self.gender)
        return self._model

    def forward_param(self, param: "SMPLX_Seq_Forward_Module.SMPLX_Param"):
        model = self._get_model()
        verts, joints, joints_body = forward_smplx_tensor_batch(
            smplx_model=model,
            global_orient=param.orient.unsqueeze(0),
            body_pose=param.body_pose.unsqueeze(0),
            left_hand_pose=param.left_hand_pose.unsqueeze(0),
            right_hand_pose=param.right_hand_pose.unsqueeze(0),
            betas=param.beta.unsqueeze(0),
            transl=param.transl.unsqueeze(0),
            scale=param.scale.unsqueeze(0),
        )
        return verts[0], joints[0], joints_body[0]

    @staticmethod
    def init_from_smpl54_seq(smpl54_seq) -> "SMPLX_Seq_Forward_Module.SMPLX_Param_Seq":
        n = smpl54_seq.num_batch
        seq = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=n, gender=smpl54_seq.gender)
        seq.orient = smpl54_seq.orient.clone()
        seq.transl = smpl54_seq.transl.clone()
        seq.body_pose = smpl54_seq.pose[:, :63].clone()
        seq.beta = smpl54_seq.beta.clone()
        seq.scale = smpl54_seq.scale.clone()
        seq.left_hand_pose = torch.zeros(n, 45)
        seq.right_hand_pose = torch.zeros(n, 45)
        return seq
