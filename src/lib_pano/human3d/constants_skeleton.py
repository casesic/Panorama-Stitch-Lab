
import numpy as np

eval_scenes_dict={
    'test1':['playground1_00', 'playground1_01'],
    'test2':['stadiumEntrance_00', 'stadiumEntrance_01'],
    'test': ['playground1_00', 'playground1_01', 'stadiumEntrance_00', 'stadiumEntrance_01']
}

class Skeleton_Format():
    def __init__(self, name_type:str):
        assert name_type in ['coco', 'smpl_24', 'smpl_54']
        

SMPL_PARENT = [
    -1,  0,  0,  0,  
    1,  2,  3,  4,  5,  
    6,  7,  8,  9,  
    9,  9, 12, 
    13, 14,
    16, 17, 
    18, 19, 
    20, 21]

SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23
}

SMPL_EXTRA_30 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34, \
    'L_Hand_thumb':35, 'L_Hand_index': 36, 'L_Hand_middle':37, 'L_Hand_ring':38, 'L_Hand_pinky':39, \
    'R_Hand_thumb':40, 'R_Hand_index':41,'R_Hand_middle':42, 'R_Hand_ring':43, 'R_Hand_pinky': 44, \
    'R_Hip': 45, 'L_Hip':46, 'Neck_LSP':47, 'Head_top':48, 'Pelvis':49, 'Thorax_MPII':50, \
    'Spine_H36M':51, 'Jaw_H36M':52, 'Head':53
}

SMPL_ALL_54 = {**SMPL_24, **SMPL_EXTRA_30}

OPENPOSE_18 = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, 'L_Wrist': 7, 'R_Hip':8, 'R_Knee':9, 'R_Ankle':10,
    'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, 'R_Eye':14, 'L_Eye':15, 'R_Ear':16, 'L_Ear':17
}

# COCO
COCO_17 = {
    'Nose':0, 'L_Eye':1, 'R_Eye':2, 'L_Ear':3, 'R_Ear':4, 'L_Shoulder':5, 'R_Shoulder':6, 'L_Elbow':7, 'R_Elbow':8, \
    'L_Wrist': 9, 'R_Wrist':10, 'L_Hip':11, 'R_Hip':12, 'L_Knee':13, 'R_Knee':14, 'L_Ankle':15, 'R_Ankle':16
}

COCO_FOOT_6 = {
    'L_BigToe':17, 'L_SmallToe':18, 'L_Heel':19, 'R_BigToe':20, 'R_SmallToe':21, 'R_Heel':22
}

COCO_BODY_WITH_FOOT_23 = {**COCO_17, **COCO_FOOT_6}

VITPOSE_25 = {
    'Nose': 0, 'Neck': 1, 'R_Shoulder': 2, 'R_Elbow': 3, 'R_Wrist': 4, 'L_Shoulder': 5, 'L_Elbow': 6, 'L_Wrist': 7,
    'Mid_Hip': 8, 'R_Hip': 9, 'R_Knee': 10, 'R_Ankle': 11, 'L_Hip': 12, 'L_Knee': 13, 'L_Ankle': 14,
    'R_Eye': 15, 'L_Eye': 16, 'R_Ear': 17, 'L_Ear': 18, 'L_BigToe': 19, 'L_SmallToe': 20, 'L_Heel': 21, 'R_BigToe': 22,
    'R_SmallToe': 23,'R_Heel': 24,
}

MPI_INF_28 = {
    'spine3_mpi_28': 0, 'spine4_mpi_28': 1, 'spine2_mpi_28': 2, 'spine_mpi_28': 3, 'Pelvis': 4,\
    'Neck':5, 'Head':6, 'Head_top':7, 'left_clavicle_mpi_28':8, 'L_Shoulder':9, 'L_Elbow':10,\
    'L_Wrist':11, 'L_Hand':12, 'right_clavicle_mpi_28':13, 'R_Shoulder':14, 'R_Elbow':15, 'R_Wrist':16,\
    'R_Hand':17, 'L_Hip':18, 'L_Knee':19, 'L_Ankle':20, 'L_SmallToe':21, 'L_BigToe':22,\
    'R_Hip':23, 'R_Knee':24, 'R_Ankle':25, 'R_SmallToe':26, 'R_BigToe':27
}

H36M_32 = {
    'Hip_H36M':0, 'R_Hip':1, 'R_Knee':2, 'R_Ankle':3, 'R_BigToe':4, 'R_SmallToe':5, 'L_Hip':6, 'L_Knee':7, 'L_Ankle':8, 'L_BigToe':9, 'L_SmallToe':10,\
    'Pelvis':11, 'Spine_H36M':12, 'Neck1_H36M':13, 'Jaw_H36M':14, 'Head':15, 'Neck2_H36M':16, 'L_Shoulder':17, 'L_Elbow':18, 'L_Wrist':19, 'L_HandThumb_H36M':20, \
    'L_OuterThigh_H36M':21, 'L_Wrist_End_H36M':22, 'L_Hand_H36M':23, 'Neck':24, 'R_Shoulder':25, 'R_Elbow':26, 'R_Wrist':27,\
    'R_HandThumb_H36M':28, 'R_OuterThigh_H36M':29, 'R_Wrist_End_H36M':30, 'R_Hand_H36M':31
}

name_joint_jta_22 = [
    'head_top',
    'head_center',
    'neck',
    'right_clavicle',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_clavicle',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'spine0',
    'spine1',
    'spine2',
    'spine3',
    'spine4',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
]
assert len(name_joint_jta_22) == 22, len(name_joint_jta_22)
JTA_22 = {name_joint_jta_22[i]:i for i in range(len(name_joint_jta_22))}

def change_dwpose_18_to_coco17(kps:np.ndarray):
    """
    Args:
        kps: (L, 18, 2/3)
    Returns:
        kps: (L, 17, 2/3)
    """

    dwpose_18_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    coco_17_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    kps[:, dwpose_18_idx] = kps[:, coco_17_idx]
    kps = kps[:, :17, :]
    return kps


SKELETON_COCO_17 = np.array([
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [ 5, 11], [ 6, 12], [ 5,  6],
        [ 5,  7], [ 6,  8], [ 7,  9], [ 8, 10], [ 1,  2], [ 0,  1], [ 0,  2], [ 1,  3], [ 2,  4], [ 3,  5], [ 4,  6]
    ], 
    dtype=np.int64
)
SKELETON_FOOT = np.array([
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [ 5, 11], [ 6, 12], [ 5,  6],
        [ 5,  7], [ 6,  8], [ 7,  9], [ 8, 10], [ 1,  2], [ 0,  1], [ 0,  2], [ 1,  3], [ 2,  4], [ 3,  5], [ 4,  6],
        [15, 19], [15, 18], [15, 17], [16, 22], [16, 20], [16, 21]
    ], 
    dtype=np.int64
)

def get_index_transform(source_format, target_format):
    index=[]
    for joint_name in target_format:
        if joint_name in source_format:
            index.append(source_format[joint_name])
    return index

TARGE_SMPL_KP_WORLDPOSE_FROM_54 = np.array([24, 17, 16, 19, 18, 21, 20, 2, 1, 5, 4, 8, 7, 11, 10])
assert TARGE_SMPL_KP_WORLDPOSE_FROM_54.shape == (15,), TARGE_SMPL_KP_WORLDPOSE_FROM_54.shape