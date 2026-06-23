import torch
import numpy as np

COCO_17_KEYPOINTS = {
    'nose': 0, 
    'left_eye': 1, 
    'right_eye': 2, 
    'left_ear': 3, 
    'right_ear': 4,
    'left_shoulder': 5, 
    'right_shoulder': 6, 
    'left_elbow': 7, 
    'right_elbow': 8,
    'left_wrist': 9, 
    'right_wrist': 10, 
    'left_hip': 11,   # No valid
    'right_hip': 12,  # No valid
    'left_knee': 13,  # No valid
    'right_knee': 14, # No valid
    'left_ankle': 15, # No valid
    'right_ankle': 16 # No valid
}

COCO_17_SKELETON_TOPOLOGY = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

SMPL_24_JOINTS = {
    'pelvis': 0,
    'left_hip': 1,      'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,     'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,  'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,'right_shoulder': 17,
    'left_elbow': 18,   'right_elbow': 19,
    'left_wrist': 20,   'right_wrist': 21,
    'left_hand': 22,    'right_hand': 23
}

INDEX_CONVERT_COCO_TO_SMPLX = {
    # 身体 17点，它和 COCO_17_KEYPOINTS (0-16) 是一一对应的关系
    # (注意：128和127分别是经过处理后更贴合检测出的表皮的髋关节回归点，但原版 smplx 默认只有118个关节点，所以127/128会导致CUDA越界报错！我们将其退回真实的 smplx 左右髋骨标号 1 和 2)
    'body': torch.LongTensor([
        55,   # COCO 0: nose
        57,   # COCO 1: left_eye
        56,   # COCO 2: right_eye
        59,   # COCO 3: left_ear
        58,   # COCO 4: right_ear
        16,   # COCO 5: left_shoulder
        17,   # COCO 6: right_shoulder
        18,   # COCO 7: left_elbow
        19,   # COCO 8: right_elbow
        20,   # COCO 9: left_wrist
        21,   # COCO 10: right_wrist
        1,    # COCO 11: left_hip (修改：128 -> 1)
        2,    # COCO 12: right_hip (修改：127 -> 2)
        4,    # COCO 13: left_knee
        5,    # COCO 14: right_knee
        7,    # COCO 15: left_ankle
        8     # COCO 16: right_ankle
    ]),
    
    # 以下如果后续你想加足面、人脸或者手部映射可用：
    'foot': torch.LongTensor(np.arange(60, 66, dtype=int)),
    
    # 51个面部关键点
    'face': torch.LongTensor(np.arange(76, 127, dtype=int)),
    
    # COCO手部/Openpose手部的 21个左手和右手点
    'lhand': torch.LongTensor([20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]),
    'rhand': torch.LongTensor([21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75])
}

mp_to_coco_idx = [
    0,   # Nose
    2,   # Left Eye (近似)
    5,   # Right Eye 
    7,   # Left Ear
    8,   # Right Ear
    11,  # Left Shoulder
    12,  # Right Shoulder
    13,  # Left Elbow
    14,  # Right Elbow
    15,  # Left Wrist
    16,  # Right Wrist
    23,  # Left Hip
    24,  # Right Hip
    25,  # Left Knee
    26,  # Right Knee
    27,  # Left Ankle
    28   # Right Ankle
]

mp_keypoints_format = {
            'coco_body_start': 0, 'coco_body_end': 17,   # 这 17 个点和 YOLO 输出完全一致
            'lhand_start': 17, 'lhand_end': 38,          # 左手的 21 个点
            'rhand_start': 38, 'rhand_end': 59,          # 右手的 21 个点
            'face_start': 59, 'face_end': 527            # 脸部的 468 个点
        }


COCO_133_KEYPOINTS = {
    # 身体 17个（和原来完全一样）
    'Nose': 0, 'L_Eye': 1, 'R_Eye': 2, 'L_Ear': 3, 'R_Ear': 4,
    'L_Shoulder': 5, 'R_Shoulder': 6, 'L_Elbow': 7, 'R_Elbow': 8,
    'L_Wrist': 9, 'R_Wrist': 10, 'L_Hip': 11, 'R_Hip': 12,
    'L_Knee': 13, 'R_Knee': 14, 'L_Ankle': 15, 'R_Ankle': 16,
    # 脚部 6个
    'L_BigToe': 17, 'L_SmallToe': 18, 'L_Heel': 19,
    'R_BigToe': 20, 'R_SmallToe': 21, 'R_Heel': 22,
    # 面部 68个（23~90）
    'Face_0': 23, 'Face_1': 24,  # ... 省略，优化时不使用
    # 左手 21个（91~111）
    'L_Hand_wrist': 91,
    'L_Hand_thumb1': 92,  'L_Hand_thumb2': 93,  'L_Hand_thumb3': 94,  'L_Hand_thumb4': 95,
    'L_Hand_index1': 96,  'L_Hand_index2': 97,  'L_Hand_index3': 98,  'L_Hand_index4': 99,
    'L_Hand_middle1':100, 'L_Hand_middle2':101, 'L_Hand_middle3':102, 'L_Hand_middle4':103,
    'L_Hand_ring1':  104, 'L_Hand_ring2':  105, 'L_Hand_ring3':  106, 'L_Hand_ring4':  107,
    'L_Hand_pinky1': 108, 'L_Hand_pinky2': 109, 'L_Hand_pinky3': 110, 'L_Hand_pinky4': 111,
    # 右手 21个（112~132）
    'R_Hand_wrist': 112,
    'R_Hand_thumb1': 113, 'R_Hand_thumb2': 114, 'R_Hand_thumb3': 115, 'R_Hand_thumb4': 116,
    'R_Hand_index1': 117, 'R_Hand_index2': 118, 'R_Hand_index3': 119, 'R_Hand_index4': 120,
    'R_Hand_middle1':121, 'R_Hand_middle2':122, 'R_Hand_middle3':123, 'R_Hand_middle4':124,
    'R_Hand_ring1':  125, 'R_Hand_ring2':  126, 'R_Hand_ring3':  127, 'R_Hand_ring4':  128,
    'R_Hand_pinky1': 129, 'R_Hand_pinky2': 130, 'R_Hand_pinky3': 131, 'R_Hand_pinky4': 132,
}