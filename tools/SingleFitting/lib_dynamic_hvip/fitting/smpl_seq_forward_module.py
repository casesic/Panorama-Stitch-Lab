from typing import Tuple, Dict, List, Optional, Union

import os
import os.path as osp
import torch.nn.functional as F
import numpy as np
import cv2
import smplx
import trimesh
import math

from icecream import ic

import cv2 as cv
import torch

from human_body_prior.train.vposer_smpl import VPoser
from smplx.body_models import SMPLXOutput, SMPLOutput, SMPLHOutput

from icecream import ic

from lib_dynamic_hvip import smpl
from lib_dynamic_hvip.smpl.constants import (
    PCA_MEAN_HAND,
    INDEX_CONVERT_COCO_TO_SMPLH,
    INDEX_CONVERT_COCO_TO_SMPLX
)
from lib_dynamic_hvip.render.persp_project import project_3D_points_with_RT
from lib_dynamic_hvip.detection.engine.base_engine import Base_Detect_Engine
from lib_dynamic_hvip.fitting.init_smpl.init_scale import get_rough_leg_length_from_joints_3D
from lib_dynamic_hvip.smpl.smpl import SMPL
from lib_dynamic_hvip.utils.utils_np import select_max_length_range_from_bool_array
from lib_dynamic_hvip.smpl.local_setting import PATH_SMPL_MODELS
from lib_dynamic_hvip.smpl.smpl_lib import (
    get_smpl_full, 
    get_vposer, 
    forward_smpl_full_tensor,
    forward_smpl_full_tensor_batch
)
from lib_dynamic_hvip.smpl.smpl_full_54.smpl_model_full import SMPL_Full
from lib_dynamic_hvip.smpl.smpl_full_54.get_full_joints import get_full_joints_by_model_full_tensor
from lib_dynamic_hvip.constants_skeleton import SMPL_ALL_54


class SMPL_Seq_Forward_Module():
    
    Weight_of_Losses_Default = {
        'body_2d_annot': 2e-3,
        'body_2d_det': 1e-3,
        'body_3d': 1e-3,
        'shape': 1e-3
    }
    
    class SMPL_Param():
        def __init__(self, dim_shape:int=10, gender:str='neutral'):
            self.orient = torch.zeros(3)
            self.transl = torch.zeros(3)
            # self.latent = torch.zeros(32)
            self.pose = torch.zeros(69)
            self.scale = torch.ones(1)

            if dim_shape == 10:
                self.beta = torch.zeros(10)
            elif dim_shape == 300:
                self.beta = torch.zeros(300)
            else:
                raise ValueError(dim_shape)
            
            self.gender = gender
            
    class SMPL_Param_Batch():
        def __init__(self, dim_shape:int=10, gender:str='neutral', num_batch:int=1):
            self.num_batch = num_batch
            self.orient = torch.zeros(num_batch, 3)
            self.transl = torch.zeros(num_batch, 3)
            # self.latent = torch.zeros(num_batch, 32)
            self.pose = torch.zeros(num_batch, 69) 
            self.scale = torch.ones(num_batch,)

            if dim_shape == 10:
                self.beta = torch.zeros(num_batch, 10)
            elif dim_shape == 300:
                self.beta = torch.zeros(num_batch, 300)
            else:
                raise ValueError(dim_shape)
            
            self.gender = gender
            
    @staticmethod
    def make_smpl_param_batch_from_list_smpl_param(list_smpl_param:List[SMPL_Param]) -> SMPL_Param_Batch:
        num_batch = len(list_smpl_param)
        smpl_param_batch = SMPL_Seq_Forward_Module.SMPL_Param_Batch(num_batch=num_batch)
        
        GENDER = None
        for index_batch, smpl_param in enumerate(list_smpl_param):
            smpl_param_batch.orient[index_batch] = smpl_param.orient
            smpl_param_batch.transl[index_batch] = smpl_param.transl
            smpl_param_batch.pose[index_batch] = smpl_param.pose
            smpl_param_batch.scale[index_batch] = smpl_param.scale
            smpl_param_batch.beta[index_batch] = smpl_param.beta
            
            if GENDER is None:
                GENDER = smpl_param.gender
            else:
                assert GENDER == smpl_param.gender, (GENDER, smpl_param.gender)
        
        return smpl_param_batch

        

    class SMPL_Param_Seq():
        def __init__(self, dim_shape:int=10, gender:str='neutral', num_batch:int=1):
            self.num_batch = num_batch
            self.orient = torch.zeros(num_batch, 3)
            self.transl = torch.zeros(num_batch, 3)
            # self.latent = torch.zeros(num_batch, 32)
            self.pose = torch.zeros(num_batch, 69)
            self.scale = torch.ones(1,)

            if dim_shape == 10:
                self.beta = torch.zeros(10,)
            elif dim_shape == 300:
                self.beta = torch.zeros(300,)
            else:
                raise ValueError(dim_shape)
            
            self.gender = gender

        def get_smpl_param_at_frame(self, index_frame:int) -> "SMPL_Seq_Forward_Module.SMPL_Param":
            smpl_param = SMPL_Seq_Forward_Module.SMPL_Param(dim_shape=self.beta.shape[0], gender=self.gender)
            smpl_param.orient = self.orient[index_frame]
            smpl_param.transl = self.transl[index_frame]
            smpl_param.pose = self.pose[index_frame]
            smpl_param.scale = self.scale
            smpl_param.beta = self.beta
            return smpl_param

    @staticmethod
    def make_smpl_param_seq_from_avg_smpl_param_batch(smpl_param_batch:SMPL_Param_Batch):
        assert isinstance(smpl_param_batch, SMPL_Seq_Forward_Module.SMPL_Param_Batch), type(smpl_param_batch)
        dim_shape = smpl_param_batch.beta.shape[1]
        gender = smpl_param_batch.gender
        num_batch = smpl_param_batch.num_batch
        smpl_param_seq = SMPL_Seq_Forward_Module.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=num_batch)
        smpl_param_seq.orient = smpl_param_batch.orient.clone()
        smpl_param_seq.transl = smpl_param_batch.transl.clone()
        smpl_param_seq.pose = smpl_param_batch.pose.clone()
        smpl_param_seq.scale = smpl_param_batch.scale.mean(dim=0)
        smpl_param_seq.beta = smpl_param_batch.beta.mean(dim=0)
        return smpl_param_seq

    def __init__(
            self,
            dim_shape:int=10
        ):
        
        #load multi gender model
        if dim_shape == 10:
            version_smpl = 'v100'
        elif dim_shape == 300:
            version_smpl = 'v110'
        else:
            raise ValueError()
        
        self.dim_shape = dim_shape
        
        self.smpl_models = {
            'neutral': get_smpl_full(version=version_smpl, gender='neutral'),
            'male': get_smpl_full(version=version_smpl, gender='male'),
            'female': get_smpl_full(version=version_smpl, gender='female')
        }
        self.vposer = get_vposer()
        
        # self.weight_of_losses = SMPL_Seq_Forward_Module.Weight_of_Losses_Default

    def init_scale_by_3D_joints(self, joints_3d:np.ndarray, joint_format:Dict[str, int]):
        rough_leg_length = get_rough_leg_length_from_joints_3D(joints_3d, joint_format)
        scale = torch.tensor(rough_leg_length / 0.85, dtype=torch.float32)
        return scale

    def init_RT_by_3D_joints(
            self,
            joints_3d:np.ndarray,
            joint_format:Dict[str, int],
            scale:float=1.0
        ):
        '''
        idea:
        first, make an smpl model with zero pose and trans
        then, compute the RT with certain correspondance between smpl joints and 3D joints
        '''

        smpl_para_default = SMPL_Seq_Forward_Module.SMPL_Param(dim_shape=self.dim_shape)
        smpl_para_default.scale = scale
        ____, joints_54_default = self.forward_smpl_param(smpl_para_default)
        
        if 'L_Hip' in joint_format and 'R_Hip' in joint_format:
            list_name_select = ['L_Hip', 'R_Hip', 'L_Shoulder', 'R_Shoulder']
        else:
            list_name_select = ['L_Hip_SMPL', 'R_Hip_SMPL', 'L_Shoulder', 'R_Shoulder']

        points_src = torch.zeros((len(list_name_select), 3))
        points_dst = np.zeros((len(list_name_select), 3))

        for count_select, name_select in enumerate(list_name_select):
            assert name_select in joint_format.keys(), (name_select, joint_format)
            points_src[count_select] = joints_54_default[SMPL_ALL_54[name_select]] 
            points_dst[count_select] = joints_3d[joint_format[name_select]]

        

        RT = trimesh.transformations.superimposition_matrix(
            points_src.detach().cpu().numpy().T,
            points_dst.T
        )

        R_tensor = torch.tensor(RT[0:3, 0:3], dtype=torch.float32)
        J_root_default = joints_54_default[0, :]
        transl_bias = (J_root_default - J_root_default @ R_tensor.T) * scale
        transl = torch.tensor(RT[0:3, 3], dtype=torch.float32) - transl_bias.detach()
        orient_np = cv.Rodrigues(RT[0:3, 0:3])[0]
        orient = torch.from_numpy(orient_np).reshape(3)

        # smpl_para_tmp = SMPL_Fitter.SMPL_Param(dim_shape=self.dim_shape)
        # smpl_para_tmp.orient = orient
        # smpl_para_tmp.transl = transl
        # smpl_para_tmp.scale = scale
        # ____, joints_54_tmp = self.forward_smpl_param(smpl_para_tmp)

        # for count_select, name_select in enumerate(list_name_select):
        #     assert name_select in joint_format.keys(), (name_select, joint_format)
        #     points_src[count_select] = joints_54_tmp[SMPL_ALL_54[name_select]] 

        # ic(points_dst)
        # ic(points_src.detach().cpu().numpy() @ RT[0:3, 0:3].T + RT[0:3, 3])
        # ic(points_src)

        return orient, transl


    def init_param_by_3D_joints(
            self,
            joints_3d:np.ndarray,
            joint_format:Dict[str, int]
        ):
        scale = self.init_scale_by_3D_joints(joints_3d, joint_format)
        orient, transl = self.init_RT_by_3D_joints(joints_3d, joint_format, scale)
        smpl_param_init = SMPL_Seq_Forward_Module.SMPL_Param(dim_shape=self.dim_shape)
        smpl_param_init.orient = orient
        smpl_param_init.transl = transl
        smpl_param_init.scale = scale

        # print('init_param_by_3D_joints')
        # print(smpl_param_init.transl)
        # print(smpl_param_init.scale)
        return smpl_param_init
    
    def forward_smpl_param_batch(self, smpl_param_batch:Union[SMPL_Param_Batch, SMPL_Param_Seq]):
        
        gender = smpl_param_batch.gender
        assert gender in ['neutral', 'male', 'female'], gender 
        
        B = smpl_param_batch.num_batch
        
        # body_pose_batch = self.vposer.decode(smpl_param_batch.latent, output_type='aa')

        # assert body_pose_batch.shape == (B, 1, 21, 3), body_pose_batch.shape

        body_pose_batch = smpl_param_batch.pose
        poses = torch.zeros((B, 72), device=body_pose_batch.device, dtype=body_pose_batch.dtype)
        poses[:, 3:] = body_pose_batch
        poses[:, 0:3] = smpl_param_batch.orient

        if isinstance(smpl_param_batch, SMPL_Seq_Forward_Module.SMPL_Param_Batch):
            betas = smpl_param_batch.beta
            scales = smpl_param_batch.scale
        elif isinstance(smpl_param_batch, SMPL_Seq_Forward_Module.SMPL_Param_Seq):
            betas_one_person = smpl_param_batch.beta
            scales_one_person = smpl_param_batch.scale
            betas = torch.stack([betas_one_person] * B, dim=0)
            scales = torch.stack([scales_one_person] * B, dim=0).reshape((B,))
        else:
            raise TypeError(type(smpl_param_batch))
        
        transl = smpl_param_batch.transl
        
        assert betas.shape == (B, self.dim_shape), betas.shape
        assert scales.shape == (B,), scales.shape
        assert transl.shape == (B, 3), transl.shape

        verts_smpl_batch, joints_54_batch = forward_smpl_full_tensor_batch(
            smpl_model=self.smpl_models[gender],
            poses=poses,
            shapes=betas,
            trans=transl,
            scale=scales,
            ret_joints_54=True
        )
        
        return verts_smpl_batch, joints_54_batch
            

    def forward_smpl_param(self, smpl_param:SMPL_Param):

        gender = smpl_param.gender
        assert gender in ['neutral', 'male', 'female'], gender

        # body_pose = self.vposer.decode(smpl_param.latent, output_type='aa').reshape(-1)
        # assert body_pose.shape == (63,), body_pose.shape
        body_pose = smpl_param.pose
        poses = torch.zeros((72), device=body_pose.device, dtype=body_pose.dtype)
        poses[3:] = body_pose
        poses[0:3] = smpl_param.orient

        verts_smpl, joints_54 = forward_smpl_full_tensor(
            smpl_model=self.smpl_models[gender],
            poses=poses,
            shapes=smpl_param.beta,
            trans=smpl_param.transl,
            scale=smpl_param.scale,
            ret_joints_54=True
        )

        return verts_smpl, joints_54
    # =============================================================================
    # SECTION 1: 在类内部新增 erp_project 静态方法
    # =============================================================================
    @staticmethod
    def erp_project(
        points_3d: torch.Tensor,  
        W: int,                     
        H: int,                     
    ) -> torch.Tensor:              
        """
        Equirectangular Projection
        """
        X = points_3d[..., 0]
        Y = points_3d[..., 1]
        Z = points_3d[..., 2]
    
        r = torch.norm(points_3d, dim=-1).clamp_min(1e-6)
    
        lon = torch.atan2(X, Z)                              # [-π,  π]
        lat = torch.asin((-Y / r).clamp(-1 + 1e-6, 1 - 1e-6))  # [-π/2, π/2]
    
        u = (lon / (2.0 * math.pi) + 0.5) * W
        # v = (lat / math.pi - 0.5) * H
        v = (0.5 - lat / math.pi) * H  
    
        return torch.stack([u, v], dim=-1)
    # =============================================================================
    # SECTION 2: 在类内部新增 compute_loss_keypoints_2d_erp 方法
    # =============================================================================
    def compute_loss_keypoints_2d_erp(
        self,
        image_W: int,
        image_H: int,
        joints_54_current: torch.Tensor,
        keypoints_detected: torch.Tensor,
        keypoints_format: Dict[str, int],
        keypoints_weight: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Equirectangular 2D keypoint reprojection loss.
 
        Drop-in replacement for
        compute_loss_keypoints_2d_from_single_view_seq_by_keypoints_weight,
        with identical return signature: (dict_loss, joints_2d_54_current).
 
        Args:
            image_W / image_H : panorama dimensions in pixels.
            joints_54_current : (B, 54, 3) SMPL joints in camera space.
            keypoints_detected: (B, J, 3)  detected keypoints (u, v, conf).
            keypoints_format  : {joint_name: index_in_detected_array}
            keypoints_weight  : optional per-joint weight dict.
 
        Returns:
            dict_loss               : {"body_2d_det": scalar_tensor}
            joints_2d_54_current    : (B, 54, 2) projected pixel coords
        """
        if keypoints_weight is None:
            keypoints_weight = {}
 
        B = joints_54_current.shape[0]
        J = len(keypoints_format)
        assert joints_54_current.shape  == torch.Size([B, 54, 3]), joints_54_current.shape
        assert keypoints_detected.shape == torch.Size([B, J,  3]), keypoints_detected.shape
 
        # ---- joint index mapping ----
        index_54_to_std, index_kp_to_std, weights_list = [], [], []
        for name_joint, index_joint in keypoints_format.items():
            if name_joint in SMPL_ALL_54:
                index_54_to_std.append(SMPL_ALL_54[name_joint])
                index_kp_to_std.append(index_joint)
                weights_list.append(keypoints_weight.get(name_joint, 1.0))
 
        assert len(index_54_to_std) > 10, \
            f"Too few matched joints ({len(index_54_to_std)}). Check keypoints_format vs SMPL_ALL_54."
 
        dev = joints_54_current.device
        index_54_to_std = torch.tensor(index_54_to_std, dtype=torch.long,    device=dev)
        index_kp_to_std = torch.tensor(index_kp_to_std, dtype=torch.long,    device=dev)
        weights_tensor  = torch.tensor(weights_list,     dtype=torch.float32, device=dev)
 
        # ---- spherical projection → (B, 54, 2) ----
        joints_2d_54_current = SMPL_Seq_Forward_Module.erp_project(
            joints_54_current, image_W, image_H
        )  # (B, 54, 2)
 
        joints_2d_std_current  = joints_2d_54_current[:, index_54_to_std, :]  # (B, J_COM, 2)
        keypoints_detected_std = keypoints_detected[:, index_kp_to_std, :]    # (B, J_COM, 3)
 
        J_COMMON = len(index_54_to_std)
        assert joints_2d_std_current.shape  == torch.Size([B, J_COMMON, 2])
        assert keypoints_detected_std.shape == torch.Size([B, J_COMMON, 3])
 
        # ---- weighted L2 ----
        loss_per_joint = torch.norm(
            joints_2d_std_current - keypoints_detected_std[:, :, :2], dim=2
        )  # (B, J_COMMON)
 
        w_norm        = weights_tensor / (weights_tensor.sum() + 1e-8)
        w_norm        = w_norm.unsqueeze(0).expand(B, -1)
        weighted_loss = loss_per_joint * keypoints_detected_std[:, :, 2] * w_norm
 
        dict_loss = {"body_2d_det": weighted_loss.sum(dim=1).mean()}
        return dict_loss, joints_2d_54_current
    
    def compute_loss_keypoints_2d_from_one_frame_multi_view(
            self,
            joints_54_current:torch.Tensor,
            list_camera_param:List[Dict[str, np.ndarray]],
            list_annotation:Optional[List[Dict[str, np.ndarray]]]=None,
            list_detection:Optional[List[Dict[str, np.ndarray]]]=None
        ) -> Dict[str, torch.Tensor]:
        

        assert not (list_annotation is None and list_detection is None), (list_annotation, list_detection)

        #assert joints_54_current.shape == (54, 3), joints_54_current.shape
        
        N_VIEW = len(list_camera_param)

        tensor_loss_kp2d_annot = torch.zeros((N_VIEW), dtype=torch.float32)
        tensor_loss_kp2d_det = torch.zeros((N_VIEW), dtype=torch.float32)

        for count_view in range(N_VIEW):
            K = torch.from_numpy(list_camera_param[count_view]['K']).to(torch.float32)
            R = torch.from_numpy(list_camera_param[count_view]['RT'][0:3, 0:3]).to(torch.float32)
            T = torch.from_numpy(list_camera_param[count_view]['RT'][0:3, 3]).to(torch.float32)
            
            joints_2d_54_current = (joints_54_current @ R.T + T) @ K.T
            joints_2d_54_current = joints_2d_54_current[:, 0:2] / joints_2d_54_current[:, 2:3]
            
            # from lib_dynamic_hvip.vis.vis_2d import vis_points
            # ic(joints_2d_54_current.shape)
            # path_vis = '%d.jpg'
            # image_bg = np.zeros((3008, 4112, 3), dtype=np.uint8)
            # vis_points(image_bg, joints_2d_54_current.detach().cpu().numpy())
            # vis_points(image_bg, list_annotation[count_view]['joint_2d'].detach().cpu().numpy(), list_colors=[(255, 255, 255)])
            # ic(list_annotation[count_view]['joint_2d'])
            # cv2.imwrite(path_vis % count_view, image_bg)
            # image_bg = np.zeros((1920, 1920

            if list_annotation is not None:
                joint_2d_annot = list_annotation[count_view]['joint_2d']
                format_annot = list_annotation[count_view]['joint_format']
                
                list_loss_all_joints = []
                for name_joint, index_joint in format_annot.items():
                    if name_joint in SMPL_ALL_54.keys():
                        index_joint_smpl = SMPL_ALL_54[name_joint]
                        loss_annot = torch.norm(joints_2d_54_current[index_joint_smpl] - joint_2d_annot[index_joint])
                        list_loss_all_joints.append(loss_annot)
                loss_kp2d_annot = torch.stack(list_loss_all_joints).mean()
                tensor_loss_kp2d_annot[count_view] = loss_kp2d_annot
                
            if list_detection is not None:
                joint_2d_detect = list_detection[count_view]['joint_2d']
                format_detect = list_detection[count_view]['joint_format']
                
                list_loss_all_joints = []
                for name_joint, index_joint in format_detect.items():
                    if name_joint in SMPL_ALL_54.keys():
                        index_joint_smpl = SMPL_ALL_54[name_joint]
                        loss_det = torch.norm(joints_2d_54_current[index_joint_smpl] - joint_2d_detect[index_joint][:2]).mean()
                        list_loss_all_joints.append(loss_det)
                loss_kp2d_det = torch.stack(list_loss_all_joints).mean()
                tensor_loss_kp2d_det[count_view] = loss_kp2d_det
        
        
        dict_losses = {}
        if list_annotation is not None:
            dict_losses['body_2d_annot'] = torch.mean(tensor_loss_kp2d_annot)
        if list_detection is not None:
            dict_losses['body_2d_det'] = torch.mean(tensor_loss_kp2d_det)
            
        #ic(tensor_loss_kp2d_annot, dict_losses['body_2d_annot'])

        return dict_losses
    
    def compute_loss_keypoints_3d_from_one_frame_multi_view(
            self,
            joints_54_current:torch.Tensor,
            list_annotation:Optional[List[Dict[str, np.ndarray]]]=None
        ) -> Dict[str, torch.Tensor]:
        

        assert list_annotation is not None, (list_annotation)

        #assert joints_54_current.shape == (54, 3), joints_54_current.shape
        
        N_VIEW = len(list_annotation)

        tensor_loss_kp3d_annot = torch.zeros((N_VIEW), dtype=torch.float32)

        for count_view in range(N_VIEW):
            if list_annotation is not None:
                joint_3d_annot = list_annotation[count_view]['joint_3d']
                format_annot = list_annotation[count_view]['joint_format']
                
                list_loss_all_joints = []
                for name_joint, index_joint in format_annot.items():
                    if name_joint in SMPL_ALL_54.keys():
                        index_joint_smpl = SMPL_ALL_54[name_joint]
                        loss_annot = torch.norm(joints_54_current[index_joint_smpl] - joint_3d_annot[index_joint])
                        list_loss_all_joints.append(loss_annot)
                loss_kp3d_annot = torch.stack(list_loss_all_joints).mean()
                tensor_loss_kp3d_annot[count_view] = loss_kp3d_annot
        
        dict_losses = {}
        if list_annotation is not None:
            dict_losses['body_3d'] = torch.mean(tensor_loss_kp3d_annot)

        return dict_losses

    def compute_loss_keypoints_2d_from_multi_view(
            self,
            joints_54_current:torch.Tensor,
            list_camera_param:List[Dict[str, np.ndarray]],
            list_annotation:Optional[List[Dict[str, np.ndarray]]]=None,
            list_detection:Optional[List[Dict[str, np.ndarray]]]=None
        ) -> Dict[str, torch.Tensor]:
        '''
        can be both used for single view and multi view
        '''    

        assert not (list_annotation is None and list_detection is None), (list_annotation, list_detection)

        if len(joints_54_current.shape) == 2:
            assert joints_54_current.shape == (54, 3), joints_54_current.shape
            N_FRAME = 1
            joints_54_current = joints_54_current.reshape((1, 54, 3))
        elif len(joints_54_current.shape) == 3:
            N_FRAME = joints_54_current.shape[0]
            assert joints_54_current.shape == (N_FRAME, 54, 3), joints_54_current.shape
        else:
            raise ValueError(joints_54_current.shape)
        
        N_VIEW = len(list_camera_param)

        tensor_loss_kp2d_annot = torch.zeros((N_VIEW), dtype=torch.float32)
        tensor_loss_kp2d_det = torch.zeros((N_VIEW), dtype=torch.float32)

        for count_view in range(N_VIEW):
            K = torch.from_numpy(list_camera_param[count_view]['K']).to(torch.float32)
            R = torch.from_numpy(list_camera_param[count_view]['RT'][0:3, 0:3]).to(torch.float32)
            T = torch.from_numpy(list_camera_param[count_view]['RT'][0:3, 3]).to(torch.float32)
            
            assert joints_54_current.shape == (N_FRAME, 54, 3), joints_54_current.shape
            joints_2d_54_current = (joints_54_current @ R.T + T.reshape(1, 1, 3)) @ K.T
            joints_2d_54_current = joints_2d_54_current[:, :, 0:2] / joints_2d_54_current[:, :, 2:3]
            assert joints_2d_54_current.shape == (N_FRAME, 54, 2), joints_2d_54_current.shape
            
            # from lib_dynamic_hvip.vis.vis_2d import vis_points
            # ic(joints_2d_54_current.shape)
            # path_vis = '%d.jpg'
            # image_bg = np.zeros((3008, 4112, 3), dtype=np.uint8)
            # vis_points(image_bg, joints_2d_54_current.detach().cpu().numpy())
            # vis_points(image_bg, list_annotation[count_view]['joint_2d'].detach().cpu().numpy(), list_colors=[(255, 255, 255)])
            # ic(list_annotation[count_view]['joint_2d'])
            # cv2.imwrite(path_vis % count_view, image_bg)
            # image_bg = np.zeros((1920, 1920

            if list_annotation is not None:
                joint_2d_annot = list_annotation[count_view]['joint_2d']
                format_annot = list_annotation[count_view]['joint_format']
                J_2D_ANNOT = len(format_annot)
                if N_FRAME == 1:
                    if joint_2d_annot.shape == (J_2D_ANNOT, 2):
                        joint_2d_annot = joint_2d_annot.reshape(1, J_2D_ANNOT, 2)
                assert joint_2d_annot.shape == (N_FRAME, J_2D_ANNOT, 2), joint_2d_annot.shape
                joint_2d_annot_tensor = torch.from_numpy(joint_2d_annot).to(torch.float32)
                    
                
                list_loss_all_joints = []
                for name_joint, index_joint in format_annot.items():
                    if name_joint in SMPL_ALL_54.keys():
                        index_joint_smpl = SMPL_ALL_54[name_joint]
                        loss_annot_one_view_one_joint_all_frame:torch.Tensor = torch.norm(
                            joints_2d_54_current[:, index_joint_smpl, :] - joint_2d_annot_tensor[:, index_joint, :],
                            dim=1
                        )
                        assert loss_annot_one_view_one_joint_all_frame.shape == (N_FRAME,), loss_annot_one_view_one_joint_all_frame.shape
                        loss_annot_one_view_one_joint = loss_annot_one_view_one_joint_all_frame.mean()
                        list_loss_all_joints.append(loss_annot_one_view_one_joint)
                #打印里面的loss
                # list_loss_all_joints_print = [loss.detach().cpu().item() for loss in list_loss_all_joints]
                # print('list_loss_all_joints_print', list_loss_all_joints_print)
                loss_kp2d_annot = torch.stack(list_loss_all_joints).mean()
                tensor_loss_kp2d_annot[count_view] = loss_kp2d_annot
                
            if list_detection is not None:
                joint_2d_detect = list_detection[count_view]['joint_2d']
                format_detect = list_detection[count_view]['joint_format']
                
                J_2D_DET = len(format_detect)
                if N_FRAME == 1:
                    if joint_2d_detect.shape == (J_2D_DET, 2):
                        joint_2d_detect = joint_2d_detect.reshape(1, J_2D_DET, 2)
                assert joint_2d_detect.shape == (N_FRAME, J_2D_DET, 2), joint_2d_detect.shape
                
                list_loss_all_joints = []
                for name_joint, index_joint in format_detect.items():
                    if name_joint in SMPL_ALL_54.keys():
                        index_joint_smpl = SMPL_ALL_54[name_joint]
                        loss_det_one_view_one_joint_all_frame = torch.norm(
                            joints_2d_54_current[index_joint_smpl] - joint_2d_detect[index_joint],
                            dim=1
                        )
                        assert loss_det_one_view_one_joint_all_frame.shape == (N_FRAME,), loss_det_one_view_one_joint_all_frame.shape
                        loss_det_one_view_one_joint = loss_det_one_view_one_joint_all_frame.mean()
                        list_loss_all_joints.append(loss_det_one_view_one_joint)
                loss_kp2d_det = torch.stack(list_loss_all_joints).mean()
                tensor_loss_kp2d_det[count_view] = loss_kp2d_det
        
        
        dict_losses = {}
        if list_annotation is not None:
            dict_losses['body_2d_annot'] = torch.mean(tensor_loss_kp2d_annot)
        if list_detection is not None:
            dict_losses['body_2d_det'] = torch.mean(tensor_loss_kp2d_det)
            
        #ic(tensor_loss_kp2d_annot, dict_losses['body_2d_annot'])

        return dict_losses
    
    def compute_loss_keypoints_2d_from_single_view_seq(
            self,
            camera_K:torch.Tensor,
            joints_54_current:torch.Tensor,
            keypoints_detected:torch.Tensor,
            keypoints_format:Dict[str, int]
        ):
        
        B = joints_54_current.shape[0]
        J = len(keypoints_format.keys())
        assert camera_K.shape == torch.Size([3, 3]), camera_K.shape
        assert joints_54_current.shape == torch.Size([B, 54, 3]), joints_54_current.shape
        assert keypoints_detected.shape == torch.Size([B, J, 3]), keypoints_detected.shape

        #compute index coverting
        index_54_to_std = []
        index_kp_to_std = []
        for name_joint, index_joint in keypoints_format.items():
            if name_joint in SMPL_ALL_54.keys():
                index_joint_smpl = SMPL_ALL_54[name_joint]
                index_54_to_std.append(index_joint_smpl)
                index_kp_to_std.append(index_joint)
        index_54_to_std = torch.tensor(index_54_to_std, dtype=torch.long)
        index_kp_to_std = torch.tensor(index_kp_to_std, dtype=torch.long)

        assert index_54_to_std.shape == index_54_to_std.shape, (index_54_to_std.shape, index_54_to_std.shape)
        assert index_54_to_std.shape[0] > 10, index_54_to_std.shape[0]

        joints_2d_54_current = joints_54_current @ camera_K.T
        joints_2d_54_current = joints_2d_54_current[:, :, 0:2] / joints_2d_54_current[:, :, 2:3]
        joints_2d_std_current = joints_2d_54_current[:, index_54_to_std, :]

        keypoints_detected_std = keypoints_detected[:, index_kp_to_std, :]

        J_COMMON = len(index_54_to_std)
        assert joints_2d_std_current.shape == torch.Size([B, J_COMMON, 2]), joints_2d_std_current.shape
        assert keypoints_detected_std.shape == torch.Size([B, J_COMMON, 3]), keypoints_detected_std.shape
        

        #use confidence as the weight of the loss
        loss_kp2d = torch.norm(joints_2d_std_current - keypoints_detected_std[:, :, 0:2], dim=2) * keypoints_detected_std[:, :, 2]

        dict_loss = {'body_2d_det': torch.mean(loss_kp2d)}
        return dict_loss

        #joint_54_current should be in the camera space

    def compute_loss_keypoints_3d_sequence(
            self,
            joints_3d_src:torch.Tensor,
            joints_3d_tgt:torch.Tensor,
            joint_format_target:Dict[str, int]
        ) -> Dict[str, torch.Tensor]:

        N_FRAME = joints_3d_src.shape[0]
        J_SRC = joints_3d_src.shape[1]
        J_TGT = joints_3d_tgt.shape[1]
        assert len(joint_format_target.keys()) == J_TGT
        assert J_SRC == 54, J_SRC
        assert J_TGT > 15, J_TGT

        assert joints_3d_src.shape == (N_FRAME, J_SRC, 3), joints_3d_src.shape
        assert joints_3d_tgt.shape == (N_FRAME, J_TGT, 3), joints_3d_tgt.shape
        

        joint_format_src = SMPL_ALL_54
        joint_format_tgt = joint_format_target
        
        list_index_match:List[Tuple[int, int]] = []
        for name_joint, index_joint in joint_format_src.items():
            if name_joint in joint_format_tgt:
                list_index_match.append((index_joint, joint_format_tgt[name_joint]))

                
        N_MATCH_J = len(list_index_match)
        
        loss_all = torch.zeros(N_FRAME, N_MATCH_J)
        
        for index_match, (index_j_src, index_j_tgt) in enumerate(list_index_match):
            distance_all_frame_one_joint = torch.norm(joints_3d_src[:, index_j_src, :] - joints_3d_tgt[:, index_j_tgt, :], dim=1)
            assert distance_all_frame_one_joint.shape == (N_FRAME,), distance_all_frame_one_joint.shape
            if index_j_src == 1 or index_j_src == 2:  # R_Hip_SMPL or L_Hip_SMPL
                distance_all_frame_one_joint = F.relu(distance_all_frame_one_joint - 0.1)
            loss_all[:, index_match] = distance_all_frame_one_joint
            
        loss_all_frame = torch.mean(loss_all, dim=1)
        assert loss_all_frame.shape == (N_FRAME,), loss_all_frame.shape

        dict_loss = {'body_3d': torch.sum(loss_all_frame)}
        return dict_loss

    def conpute_loss_velocity(
            self,
            joints_24_current:torch.Tensor,
            FPS:float=30.0
        ) -> Dict[str, torch.Tensor]:

        def get_joints_vel_weight(velocity: torch.Tensor) -> torch.Tensor:
            """
            输入:
                velocity: (...,) 任意形状的速度张量(单位: m/s)
            输出:
                weight: (...,) 相同形状的权重张量
            """
            weight = torch.zeros_like(velocity)
            low_thresh = 1.5
            high_thresh = 7.5

            # 1.5 < v <= 7.5: 线性增长部分（归一化为 0-1）
            mask_mid = (velocity > low_thresh) & (velocity <= high_thresh)
            weight[mask_mid] = (velocity[mask_mid] - low_thresh) / (high_thresh - low_thresh)  # 从0递增到1

            # v > 7.5: 快速增长部分
            mask_high = velocity > high_thresh
            weight[mask_high] = torch.exp((velocity[mask_high] - high_thresh) / 2.0) + 1.0  # 快速增大

            return weight

        L = joints_24_current.shape[0]
        assert joints_24_current.shape == (L, 24, 3), joints_24_current.shape
        joints_24_seq = joints_24_current  # shape (L, 24, 3)

        # Compute velocities
        joints_24_vel = (joints_24_seq[1:] - joints_24_seq[:-1]) * FPS # shape (L-1, 24, 3)
   
        # Compute losses
        joints_24_vel_abs = torch.norm(joints_24_vel, dim=-1) # shape (L-1, 24)
        joints_24_vel_weight = get_joints_vel_weight(joints_24_vel_abs) # shape (L-1, 24)
        joints_24_vel_loss = torch.mean(joints_24_vel_abs * joints_24_vel_weight) # shape (L-1,)

        loss_dict = {
            'joints_24_vel': joints_24_vel_loss,
        }
        return loss_dict

    def compute_loss_smpl_verts(self,
                            smpl_verts_src: torch.Tensor,
                            smpl_verts_tgt_np: List[np.ndarray]
                            ) -> Dict[str, torch.Tensor]:
        if len(smpl_verts_tgt_np) == 0:
            return {"smpl_verts": torch.tensor(0.0, device=smpl_verts_src.device)}
        
        smpl_verts_tgt = torch.tensor(np.array(smpl_verts_tgt_np), 
                                    dtype=torch.float32,
                                    device=smpl_verts_src.device)
        N_FRAME = len(smpl_verts_tgt_np)
        assert smpl_verts_src.shape == (N_FRAME, 6890, 3), \
            f"{(N_FRAME, 6890, 3)}, {smpl_verts_src.shape}"
        assert smpl_verts_tgt.shape == (N_FRAME, 6890, 3), \
            f"{(N_FRAME, 6890, 3)}, {smpl_verts_tgt.shape}"
        
        vertex_diff = smpl_verts_src - smpl_verts_tgt
        per_vertex_dist = torch.norm(vertex_diff, p=2, dim=-1)
        per_frame_loss = torch.mean(per_vertex_dist, dim=-1)
        total_loss = torch.mean(per_frame_loss)
        
        return {"smpl_verts": total_loss}
                    

    def compute_loss_prior(
            self,
            smpl_param:Union[SMPL_Param, SMPL_Param_Batch, SMPL_Param_Seq]
        ) -> Dict[str, torch.Tensor]:
        '''
        compute the prior loss of the smpl param
        '''
        losses = {
            'pose': smpl_param.pose.norm(),
            'shape': smpl_param.beta.norm()
        }
        return losses
        
    def sum_losses_with_weight(
            self, 
            dict_losses:Dict[str, torch.Tensor], 
            weight:Dict[str, float]
        ) -> torch.Tensor:
        loss = 0
        for key in dict_losses.keys():
            loss += dict_losses[key] * weight[key]
        return loss

    def get_smpl_mesh_from_param(self, smpl_param:SMPL_Param):
        verts_smpl, __ = self.forward_smpl_param(smpl_param)
        mesh_smpl = trimesh.Trimesh(vertices=verts_smpl.detach().cpu().numpy(), faces=self.smpl_models['neutral'].faces)
        return mesh_smpl
    

    def compute_loss_keypoints_2d_from_single_view_seq_by_keypoints_weight(
            self,
            camera_K: torch.Tensor,
            joints_54_current: torch.Tensor,
            keypoints_detected: torch.Tensor,
            keypoints_format: Dict[str, int],
            keypoints_weight: Optional[Dict[str, float]] = None
        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        if keypoints_weight is None:
            keypoints_weight = {}   
        B = joints_54_current.shape[0]
        J = len(keypoints_format.keys())
        assert camera_K.shape == torch.Size([3, 3]), camera_K.shape
        assert joints_54_current.shape == torch.Size([B, 54, 3]), joints_54_current.shape
        assert keypoints_detected.shape == torch.Size([B, J, 3]), keypoints_detected.shape

        # ---- compute index converting ----
        index_54_to_std = []
        index_kp_to_std = []
        weights_list = []
        joints_used = []  # 新增，存储参与计算的关节点名字
        # SMPL_ALL_54['L_Hip'] = SMPL_ALL_54['L_Hip_SMPL']
        # SMPL_ALL_54['R_Hip'] = SMPL_ALL_54['R_Hip_SMPL']
        for name_joint, index_joint in keypoints_format.items():
            if name_joint in SMPL_ALL_54.keys():
                index_joint_smpl = SMPL_ALL_54[name_joint]

                index_54_to_std.append(index_joint_smpl)
                index_kp_to_std.append(index_joint)

                # custom weight if exists, else 1.0
                w = keypoints_weight.get(name_joint, 1.0)
                weights_list.append(w)

                joints_used.append(name_joint)  # 记录参与计算的关节点
        # print("Joints used in 2D keypoint loss:", joints_used)  # 打印参与计算的关节点名称
        # print(len(joints_used), "joints used out of", len(keypoints_format), "available keypoints.")
        # exit()
        index_54_to_std = torch.tensor(index_54_to_std, dtype=torch.long)
        index_kp_to_std = torch.tensor(index_kp_to_std, dtype=torch.long)
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(joints_54_current.device)

        assert index_54_to_std.shape[0] == index_kp_to_std.shape[0], "Mapping mismatch"
        assert index_54_to_std.shape[0] > 10, index_54_to_std.shape[0]

        # ---- project joints to 2D ----
        joints_2d_54_current = joints_54_current @ camera_K.T
        joints_2d_54_current = joints_2d_54_current[:, :, 0:2] / joints_2d_54_current[:, :, 2:3]
        joints_2d_std_current = joints_2d_54_current[:, index_54_to_std, :]

        keypoints_detected_std = keypoints_detected[:, index_kp_to_std, :]

        J_COMMON = index_54_to_std.shape[0]
        assert joints_2d_std_current.shape == torch.Size([B, J_COMMON, 2]), joints_2d_std_current.shape
        assert keypoints_detected_std.shape == torch.Size([B, J_COMMON, 3]), keypoints_detected_std.shape

        # ---- compute per-joint L2 loss ----
        loss_per_joint = torch.norm(
            joints_2d_std_current - keypoints_detected_std[:, :, :2],
            dim=2
        )  # shape: (B, J_COMMON)

        # ---- normalize keypoints_weights ----
        w_norm = weights_tensor / (weights_tensor.sum() + 1e-8)  # shape: (J_COMMON,)
        w_norm = w_norm.unsqueeze(0).expand(B, -1) # broadcast to (B, J_COMMON)

        # ---- combine: L2 * confidence * normalized weight ----
        weighted_loss = loss_per_joint * keypoints_detected_std[:, :, 2] * w_norm

        dict_loss = {
            'body_2d_det': weighted_loss.sum(dim=1).mean(),  # 先对每帧求和，再求 batch mean
        }
        return dict_loss, joints_2d_54_current

    def fit_from_multi_frame_seq_by_vr_v1(
        self,
        detection_results_list: List[Base_Detect_Engine.Single_Image_Results],
        camera: np.ndarray,
        nose_position: np.ndarray,
        plane_cam:Tuple[np.ndarray, float],
        smpl_param_init: Optional[SMPL_Param_Seq] = None,
        person_height:Optional[torch.Tensor]=None,
    ) -> SMPL_Param_Seq:
        """
        多帧（单目 + VR）联合拟合。

        camera: 3x3 np.ndarray

        """
        assert isinstance(detection_results_list, list) and len(detection_results_list) > 0
        N = len(detection_results_list)

        # ---------------------------------------------------------
        # 1. IMU VR → camera 坐标变换
        # ---------------------------------------------------------

        def joints_camera_to_world(
            joints_cam: torch.Tensor,  # (N, J, 3)
            R_cam: np.ndarray,         # (3,3) world -> camera
            tvec_cam: np.ndarray       # (3,) or (3,1)
        ) -> torch.Tensor:
            """
            将相机坐标系下的 joints 变换到世界坐标系
            """
            device = joints_cam.device
            dtype = joints_cam.dtype

            R = torch.from_numpy(R_cam).to(device=device, dtype=dtype)   # (3,3)
            t = torch.from_numpy(tvec_cam).to(device=device, dtype=dtype).view(1, 1, 3)  # (1,1,3)

            # X_world = R^T @ (X_cam - t)
            joints_world = torch.matmul(
                joints_cam - t,
                R.T
            )

            return joints_world


        from lib_dynamic_hvip.utils.utils_vr import batch_get_vr_bias
        from lib_dynamic_hvip.dataset.vr_collect.lib_vr_collect.utils import get_vr_bais_by_t
        from lib_dynamic_hvip.utils.utils_smpl_filter import SMPL_RotationMatrix_Filter_P3D
        from lib_dynamic_hvip.network.single_recon.gvhmr.utils_gvhmr import compute_static_conf_loss
        smpl_filter = SMPL_RotationMatrix_Filter_P3D()

        n_cam, d_cam = plane_cam
        n_cam_tensor = torch.as_tensor(n_cam, dtype=torch.float32)  # (3,)
        d_cam_tensor = torch.as_tensor(d_cam, dtype=torch.float32)  # scalar
     
        N = nose_position.shape[0]
    
        positions_nose = torch.from_numpy(nose_position[:, 0, :]).float()


        print('nose_camera', positions_nose[N // 2])
        # ---------------------------------------------------------
        # 2. Keypoints stack
        # ---------------------------------------------------------
        assert camera.shape == (3, 3)
        camera_tensor = torch.from_numpy(camera).float()

        keypoints_list = []
        key_fmt = None
        for i in range(N):
            dr = detection_results_list[i]
            assert isinstance(dr, Base_Detect_Engine.Single_Image_Results)
            kp = torch.from_numpy(dr.keypoints).float()  # (J,3)
            keypoints_list.append(kp)
            if key_fmt is None:
                key_fmt = dr.keypoints_format
            else:
                assert key_fmt == dr.keypoints_format

        keypoints_tensor = torch.stack(keypoints_list, dim=0)  # (N,J,3)

        # ---------------------------------------------------------
        # 3. 初始化 SMPL_Param_Seq
        # ---------------------------------------------------------
        if smpl_param_init is not None:
            assert isinstance(smpl_param_init, SMPL_Seq_Forward_Module.SMPL_Param_Seq)
            assert smpl_param_init.num_batch == N
            smpl_param = smpl_param_init
        else:
            smpl_param = SMPL_Seq_Forward_Module.SMPL_Param_Seq(dim_shape=self.dim_shape, gender='neutral', num_batch=N)
        # print(smpl_param.transl[N//2])
        # exit()
        smpl_param.transl += positions_nose
        # 设置可优化项
        smpl_param.orient = smpl_param.orient.clone().detach().requires_grad_(True)
        smpl_param.transl = smpl_param.transl.clone().detach().requires_grad_(True)
        smpl_param.pose = smpl_param.pose.clone().detach().requires_grad_(True)
        smpl_param.beta = smpl_param.beta.clone().detach().requires_grad_(True)
        # ---------------------------------------------------------
        # 4. 优化超参
        # ---------------------------------------------------------
        
        N_FRAME  = N
        LR_LBFGS = 0.5
        MAX_ITER_PER_CLOSURE = 60
        MAX_CLOSURE = 60
        LOSS_PERCENT_STOP_CLOSURE = 0.025
        smpl_param_height = SMPL_Seq_Forward_Module.SMPL_Param()

        optimizer = torch.optim.LBFGS(
            [smpl_param.orient, smpl_param.transl, smpl_param.pose, smpl_param.beta],
            line_search_fn='strong_wolfe', 
            lr=LR_LBFGS, 
            max_iter=MAX_ITER_PER_CLOSURE
        )

        loss_prev = 1e9

        WEIGHT_VR_3D = 2 * N_FRAME
        Weight_of_Losses_Seq = {
            'body_2d_det': 2 * N_FRAME / 20,
            'shape': 2e-4,
            'loss_vr': WEIGHT_VR_3D,
            'loss_vr_2d': WEIGHT_VR_3D / 10,
            "loss_foot_penetration": N_FRAME * 0.5,
            "loss_foot_distance": N_FRAME * 4,
            "loss_static_conf": N_FRAME / 20,
            "loss_height": N_FRAME * 10 * 0.5,
        }

        keypoints_weight: Dict[str, float] = { 'Nose': 1, 'L_Eye': 0, 'R_Eye': 0, 'L_Ear': 0, 'R_Ear': 0, \
                                              'L_Shoulder': 4, 'R_Shoulder': 4, 'L_Elbow': 1.5, 'R_Elbow': 1.5, 'L_Wrist': 1, 'R_Wrist': 1, \
                                              'L_Hip': 0, 'R_Hip': 0, 'L_Knee': 1, 'R_Knee': 1, 'L_Ankle': 4, 'R_Ankle': 4, 'L_BigToe': 1.2, 'L_SmallToe': 1.2, \
                                              'L_Heel': 4, 'R_BigToe': 1.2, 'R_SmallToe': 1.2, 'R_Heel': 4 }
        keypoints_weight: Dict[str, float] = {
            'Nose': 1.0,
            'L_Eye': 0.,
            'R_Eye': 0.,
            'L_Ear': 0.,
            'R_Ear': 0.,
            'L_Shoulder': 3,
            'R_Shoulder': 3,
            'L_Hip': 2,
            'R_Hip': 2,
            'L_Ankle': 2.,
            'R_Ankle': 2.,
            'L_Heel': 1.5,
            'R_Heel': 1.5,
            'L_BigToe': 1.5,
            'R_BigToe': 1.5,
            'L_SmallToe': 1.5,
            'R_SmallToe': 1.5,
            'L_Wrist': 1.5,
            'R_Wrist': 1.5,
            'L_Elbow': 1.2,
            'R_Elbow': 1.2,
        }
        
        self.weight_of_losses = Weight_of_Losses_Seq
        # ---------------------------------------------------------
        # 5. closure
        # ---------------------------------------------------------
        def closure():
            optimizer.zero_grad()
            
            verts_smpl, joints_54 = self.forward_smpl_param_batch(smpl_param)

            if person_height is not None:
                smpl_param_height.beta = smpl_param.beta
                verts_t_pose, _ = self.forward_smpl_param(smpl_param_height)

                y = verts_t_pose[..., 1]        # (B, V)
                y_sorted, _ = y.sort()
                height = y_sorted[-1] - y_sorted[0]  # scalar
                height_loss = F.l1_loss(height, person_height)
                losses_all = {'loss_height': height_loss}

            joints_54 = joints_54.reshape((-1, 54, 3))
            # print(j)

            # Keypoints loss
            losses_all = {}
            losses_kp2d, joints_2d_54_current = self.compute_loss_keypoints_2d_from_single_view_seq_by_keypoints_weight(
                camera_K=camera_tensor,
                keypoints_detected=keypoints_tensor,
                joints_54_current=joints_54,
                keypoints_format=key_fmt,
                keypoints_weight=keypoints_weight
            )
            losses_all.update(losses_kp2d)

            # VR 对齐损失
            point_cam = positions_nose
            point_proj = point_cam @ camera_tensor.T
            point_2d = point_proj[:, :2] / (point_proj[:, 2:] + 1e-9)

            nose_2d = joints_2d_54_current[:, 24, :]
            nose_3d = joints_54[:, 24, :]

            loss_vr_2d = torch.norm(point_2d - nose_2d, dim=1).mean()
            
            # print("point_2d nose_2 nose_detect", point_2d[N // 2], nose_2d[N // 2], keypoints_tensor[N // 2, 0, :2])
            # print("point_3d nose_smpl", positions_nose[N // 2], nose_3d[N // 2])
            loss_vr_3d = torch.norm(nose_3d - positions_nose, dim=1).mean()
            # loss_prior = self.compute_loss_prior(smpl_param)
            # losses_all.update(loss_prior if isinstance(loss_prior, dict) else {'loss_prior': loss_prior})

            foot_idx = [7, 8, 10, 11]
            joints_feet_cam = joints_54[:, foot_idx, :]   # (N,4,3)

            # 平面距离：   distance = n^T x + d   (正值：在地面上方)
            dist_foot = joints_feet_cam @ n_cam_tensor   # (N,4)
            dist_foot = dist_foot + d_cam_tensor

            # 1) 穿地惩罚：如果 dist < 0
            loss_foot_penetration = torch.relu(-dist_foot).mean()

            # 2) 贴地约束：dist 越接近 0 越好，但只对接触帧约束
            contact_mask = (dist_foot < 0.05).float()
            print(dist_foot[N // 2], contact_mask[N // 2])
            loss_foot_distance = (dist_foot.abs() * contact_mask).mean()

            # 加入 losses
            losses_all["loss_foot_penetration"] = loss_foot_penetration
            losses_all["loss_foot_distance"] = loss_foot_distance


            losses_all['loss_vr'] = loss_vr_3d
            losses_all['loss_vr_2d'] = loss_vr_2d

            # print('losses_all:', {k: v.item() for k, v in losses_all.items()})
            loss = self.sum_losses_with_weight(losses_all, self.weight_of_losses)
            loss.backward()
            return loss

        # ---------------------------------------------------------
        # 6. 优化循环
        # ---------------------------------------------------------
        print("opt multi-frame...")
        for _ in range(MAX_CLOSURE):
            optimizer.step(closure)
            loss = closure().item()
            if loss_prev - loss < loss_prev * LOSS_PERCENT_STOP_CLOSURE:
                break
            loss_prev = loss

        smpl_param.orient.requires_grad_(False)
        smpl_param.transl.requires_grad_(False)
        smpl_param.pose.requires_grad_(False)
        smpl_param.beta.requires_grad_(False)
        smpl_param.scale.requires_grad_(False)

        return smpl_param
    
    def fit_from_multi_frame_seq(
        self,
        detection_results_list: List[Base_Detect_Engine.Result_Single_Human_Single_Frame],
        camera: np.ndarray,
        # cam_ext: Tuple[np.ndarray, np.ndarray],
        # imu_data_seq: List[Dict[str, np.ndarray]],
        # VR2camera: Dict[str, np.ndarray],
        # plane_cam: Tuple[np.ndarray, float],
        smpl_param_init: Optional[SMPL_Param_Seq] = None,
    ) -> SMPL_Param_Seq:
        """
        Multi-frame fitting using:
        - 2D keypoint reprojection
        - VR nose alignment (3D + 2D)
        - foot-ground constraints
        """
        # _ = cam_ext  # kept in signature for compatibility with existing call chain
        assert isinstance(detection_results_list, list) and len(detection_results_list) > 0
        n_frame = len(detection_results_list)
        # assert len(imu_data_seq) == n_frame, (len(imu_data_seq), n_frame)

        # ---------------------------------------------------------
        # 1. IMU/VR position in camera coordinates
        # ---------------------------------------------------------
        # vr_depth = np.array([0.0, -0.065, 0.065], dtype=np.float32)
        # r_vr_to_cam = VR2camera["R"]
        # t_vr_to_cam = VR2camera["T"]
        # _ = plane_cam
        # t_bias = torch.from_numpy(VR2camera["t_bias"]).float() + torch.from_numpy(vr_depth).float()

        # from lib_dynamic_hvip.dataset.vr_collect.lib_vr_collect.utils import get_vr_bais_by_t

        # positions_vr_camera = []
        # for imu in imu_data_seq:
        #     position = imu["position"].ravel()
        #     rotation = imu["rotation"]
        #     position = position + get_vr_bais_by_t(*rotation, lengths=t_bias.numpy())
        #     position = position[None] @ r_vr_to_cam.T + t_vr_to_cam.ravel()
        #     positions_vr_camera.append(torch.tensor(position[0]).float())
        # positions_vr_camera = torch.stack(positions_vr_camera, dim=0)  # (N, 3)

        # ---------------------------------------------------------
        # 2. Prepare camera + keypoints tensor
        # ---------------------------------------------------------
        assert camera.shape == (3, 3), camera.shape
        camera_tensor = torch.from_numpy(camera).float()

        keypoints_list = []
        key_fmt = None
        for dr in detection_results_list:
            assert isinstance(dr, Base_Detect_Engine.Result_Single_Human_Single_Frame)
            kp = torch.from_numpy(dr.keypoints).float()

            # Expand hip width slightly for more stable optimization.
            idx_lhip, idx_rhip = 11, 12
            center = (kp[idx_lhip, :2] + kp[idx_rhip, :2]) / 2.0
            scale = 1.15
            kp[idx_lhip, :2] = center + (kp[idx_lhip, :2] - center) * scale
            kp[idx_rhip, :2] = center + (kp[idx_rhip, :2] - center) * scale
            keypoints_list.append(kp)

            if key_fmt is None:
                key_fmt = dr.keypoints_format
            else:
                assert key_fmt == dr.keypoints_format
        keypoints_tensor = torch.stack(keypoints_list, dim=0)  # (N, J, 3)

        # ---------------------------------------------------------
        # 3. Init SMPL sequence
        # ---------------------------------------------------------
        if smpl_param_init is not None:
            assert isinstance(smpl_param_init, SMPL_Seq_Forward_Module.SMPL_Param_Seq)
            assert smpl_param_init.num_batch == n_frame
            smpl_param = smpl_param_init
        else:
            smpl_param = SMPL_Seq_Forward_Module.SMPL_Param_Seq(
                dim_shape=self.dim_shape,
                gender="neutral",
                num_batch=n_frame,
            )

        # Initialize shape with the provided beta prior.
        shape_init = torch.tensor(
            [0.08221327, -0.00588347, 0.03931884, -0.0198693, 0.00749705,
             -0.01965939, -0.00093912, -0.01050037, 0.00110335, 0.00892171],
            dtype=smpl_param.beta.dtype,
            device=smpl_param.beta.device,
        )
        smpl_param.beta = torch.zeros_like(smpl_param.beta)
        smpl_param.beta[: min(10, smpl_param.beta.shape[0])] = shape_init[: min(10, smpl_param.beta.shape[0])]
        shape_anchor = smpl_param.beta.clone().detach()

        # smpl_param.transl += positions_vr_camera
        smpl_param.orient = smpl_param.orient.clone().detach().requires_grad_(True)
        smpl_param.transl = smpl_param.transl.clone().detach().requires_grad_(True)
        smpl_param.pose = smpl_param.pose.clone().detach().requires_grad_(True)
        smpl_param.beta = smpl_param.beta.clone().detach().requires_grad_(True)
        smpl_param.scale = smpl_param.scale.clone().detach()

        # ---------------------------------------------------------
        # 4. Optimizer settings
        # ---------------------------------------------------------
        lr_lbfgs = 0.5
        max_iter_per_closure = 60
        max_closure = 80
        stop_ratio = 0.01
        fps_assumed = 30.0
        max_transl_speed_mps = 3.0
        cube_stage_steps = 120
        cube_stage_lr = 0.03
        cube_side_length = 0.4
        cube_outside_penalty_weight = 4.0
        cube_speed_penalty_weight = 0.5
        cube_depth_penalty_weight = 3.0
        cube_min_depth = 0.5
        shape_soft_bound = 0.35
        shape_hard_clip = 0.55
        shape_norm_soft_bound = 0.9

        base_weight_of_losses = {
            "body_2d_det": n_frame * 1,
            "loss_pose_diff": n_frame * 0.0,
            "loss_transl_diff": n_frame * 0.0,
            "loss_pose_prior": n_frame * 0.05,
            "loss_shape_prior": n_frame * 6.0,
            "loss_shape_bound": n_frame * 20.0,
            "loss_shape_anchor": n_frame * 30.0,
            "loss_shape_norm": n_frame * 15.0,
        }
        self.weight_of_losses = dict(base_weight_of_losses)
        keypoints_weight = {}
        # keypoints_weight = {
        #     "Nose": 1.0,
        #     "L_Eye": 0.0,
        #     "R_Eye": 0.0,
        #     "L_Ear": 0.0,
        #     "R_Ear": 0.0,
        #     "L_Shoulder": 4.0,
        #     "R_Shoulder": 4.0,
        #     "L_Hip": 0,
        #     "R_Hip": 0,
        #     "L_Ankle": 3.0,
        #     "R_Ankle": 3.0,
        #     "L_Heel": 2.0,
        #     "R_Heel": 2.0,
        #     "L_BigToe": 2.0,
        #     "R_BigToe": 2.0,
        #     "L_SmallToe": 2.0,
        #     "R_SmallToe": 2.0,
        #     "L_Wrist": 1.5,
        #     "R_Wrist": 1.5,
        #     "L_Elbow": 1.2,
        #     "R_Elbow": 1.2,
        # }

        # ---------------------------------------------------------
        # 4.1 Stage-1: transl only with cube-bbox overlap objective
        # ---------------------------------------------------------
        print("stage-1 transl optimize with cube-bbox overlap...")
        bbox_xyxy = torch.zeros((n_frame, 4), dtype=keypoints_tensor.dtype, device=keypoints_tensor.device)
        bbox_frame_weight = torch.zeros((n_frame,), dtype=keypoints_tensor.dtype, device=keypoints_tensor.device)
        for idx_frame in range(n_frame):
            kp = keypoints_tensor[idx_frame]
            conf_mask = kp[:, 2] > 0.05
            kp_xy = kp[conf_mask][:, :2] if conf_mask.any() else kp[:, :2]
            x_min, _ = kp_xy[:, 0].min(dim=0)
            x_max, _ = kp_xy[:, 0].max(dim=0)
            y_min, _ = kp_xy[:, 1].min(dim=0)
            y_max, _ = kp_xy[:, 1].max(dim=0)
            bbox_xyxy[idx_frame] = torch.stack([x_min, y_min, x_max, y_max])
            if conf_mask.any():
                bbox_frame_weight[idx_frame] = kp[conf_mask, 2].mean().clamp(min=0.05, max=1.0)
            else:
                bbox_frame_weight[idx_frame] = 0.05

        bbox_frame_weight = bbox_frame_weight / (bbox_frame_weight.sum() + 1e-8)
        bbox_w = (bbox_xyxy[:, 2] - bbox_xyxy[:, 0]).clamp_min(1.0)
        bbox_h = (bbox_xyxy[:, 3] - bbox_xyxy[:, 1]).clamp_min(1.0)
        bbox_area = bbox_w * bbox_h

        half = cube_side_length * 0.5 # 0.4 m length 3D cube
        cube_offsets = torch.tensor(
            [
                [-half, -half, -half],
                [-half, -half, half],
                [-half, half, -half],
                [-half, half, half],
                [half, -half, -half],
                [half, -half, half],
                [half, half, -half],
                [half, half, half],
            ],
            dtype=smpl_param.transl.dtype,
            device=smpl_param.transl.device,
        )

        cube_optimizer = torch.optim.Adam([smpl_param.transl], lr=cube_stage_lr)
        cube_speed_start_step = cube_stage_steps // 2
        for step_idx in range(cube_stage_steps):
            smpl_param.transl.grad = None  # clear the grad
            cube_optimizer.zero_grad()
            
            cube_vertices = smpl_param.transl[:, None, :] + cube_offsets[None, :, :]  # (N, 8, 3)
            cube_proj = cube_vertices @ camera_tensor.T
            cube_z = cube_vertices[:, :, 2]
            cube_z_min, _ = cube_z.min(dim=1)
            z_safe = cube_z.clamp_min(cube_min_depth)
            cube_xy = cube_proj[:, :, :2] / z_safe[:, :, None]

            cube_x_min, _ = cube_xy[:, :, 0].min(dim=1)
            cube_x_max, _ = cube_xy[:, :, 0].max(dim=1)
            cube_y_min, _ = cube_xy[:, :, 1].min(dim=1)
            cube_y_max, _ = cube_xy[:, :, 1].max(dim=1)

            inter_w = torch.relu(torch.minimum(cube_x_max, bbox_xyxy[:, 2]) - torch.maximum(cube_x_min, bbox_xyxy[:, 0]))
            inter_h = torch.relu(torch.minimum(cube_y_max, bbox_xyxy[:, 3]) - torch.maximum(cube_y_min, bbox_xyxy[:, 1]))
            inter_area = inter_w * inter_h

            cube_w = (cube_x_max - cube_x_min).clamp_min(1.0)
            cube_h = (cube_y_max - cube_y_min).clamp_min(1.0)
            cube_area = cube_w * cube_h

            # bounded geometric loss:
            # 1) higher overlap -> lower loss (1 - IoU)
            # 2) outside area ratio (normalized by cube area) penalizes bbox overflow
            iou = inter_area / (bbox_area + cube_area - inter_area + 1e-6)  # intersection / union 
            outside_ratio = torch.relu(cube_area - inter_area) / (cube_area + 1e-6)
            geom_loss_per_frame = (1.0 - iou) + cube_outside_penalty_weight * outside_ratio
            loss_geom = (geom_loss_per_frame * bbox_frame_weight).sum()

            loss_depth = (torch.relu(cube_min_depth - cube_z_min) * bbox_frame_weight).sum()
            loss_cube = loss_geom + cube_depth_penalty_weight * loss_depth

            if step_idx >= cube_speed_start_step and n_frame > 1:
                transl_speed = torch.norm(smpl_param.transl[1:] - smpl_param.transl[:-1], dim=1) * fps_assumed
                loss_transl_speed = torch.relu(transl_speed - max_transl_speed_mps).mean()
                loss_cube = loss_cube + cube_speed_penalty_weight * loss_transl_speed
            else:
                loss_transl_speed = torch.zeros((), dtype=smpl_param.transl.dtype, device=smpl_param.transl.device)

            loss_cube.backward()
            cube_optimizer.step()
            with torch.no_grad():
                smpl_param.transl[:, 0:2].clamp_(-3.0, 3.0)
                smpl_param.transl[:, 2:3].clamp_(cube_min_depth, 12.0)

            print(
                "[cube_stage]"
                f" step={step_idx + 1:03d}/{cube_stage_steps}"
                f" total={loss_cube.detach().cpu().item():.4f}"
                f" iou_loss={((1.0 - iou) * bbox_frame_weight).sum().detach().cpu().item():.4f}"
                f" outside={((outside_ratio * bbox_frame_weight).sum()).detach().cpu().item():.4f}"
                f" depth={loss_depth.detach().cpu().item():.4f}"
                f" speed={loss_transl_speed.detach().cpu().item():.4f}"
            )

        latest_total_loss: float = float("inf")
        latest_losses: Dict[str, float] = {}
        current_shape_soft_bound = shape_soft_bound
        current_shape_hard_clip = shape_hard_clip

        def make_closure(optimizer_current: torch.optim.Optimizer):
            def closure():
                nonlocal latest_total_loss, latest_losses, current_shape_soft_bound
                for p in [smpl_param.orient, smpl_param.transl, smpl_param.pose, smpl_param.beta]:
                    p.grad = None
                optimizer_current.zero_grad()
                
                _, joints_54 = self.forward_smpl_param_batch(smpl_param)
                joints_54 = joints_54.reshape((-1, 54, 3))

                losses_all = {}
                losses_kp2d, joints_2d_54_current = self.compute_loss_keypoints_2d_from_single_view_seq_by_keypoints_weight(
                    camera_K=camera_tensor,
                    keypoints_detected=keypoints_tensor,
                    joints_54_current=joints_54,
                    keypoints_format=key_fmt,
                    keypoints_weight=keypoints_weight,
                )
                losses_all.update(losses_kp2d)

                # point_proj = positions_vr_camera @ camera_tensor.T
                # point_2d = point_proj[:, :2] / (point_proj[:, 2:] + 1e-9)
                # nose_2d = joints_2d_54_current[:, 24, :]
                # nose_3d = joints_54[:, 24, :]

                # losses_all["loss_vr_2d"] = torch.norm(point_2d - nose_2d, dim=1).mean()
                # losses_all["loss_vr"] = torch.norm(nose_3d - positions_vr_camera, dim=1).mean()

                # Temporal smoothness
                if n_frame > 1:
                    transl_diff = smpl_param.transl[1:] - smpl_param.transl[:-1]  # (N-1, 3), m/frame
                    transl_speed = torch.norm(transl_diff, dim=1) * fps_assumed
                    losses_all["loss_transl_diff"] = torch.relu(transl_speed - max_transl_speed_mps).mean()

                    pose_diff = (smpl_param.pose[1:] - smpl_param.pose[:-1]).reshape(-1, 23, 3)  # (N-1, 23, 3)
                    losses_all["loss_pose_diff"] = torch.norm(pose_diff, dim=2).mean()
                else:
                    zero_loss = torch.zeros((), dtype=smpl_param.pose.dtype, device=smpl_param.pose.device)
                    losses_all["loss_pose_diff"] = zero_loss
                    losses_all["loss_transl_diff"] = zero_loss

                losses_all["loss_pose_prior"] = torch.norm(smpl_param.pose, dim=1).mean()
                losses_all["loss_shape_prior"] = torch.mean(smpl_param.beta ** 2)
                losses_all["loss_shape_bound"] = torch.mean(torch.relu(torch.abs(smpl_param.beta) - current_shape_soft_bound) ** 2)
                losses_all["loss_shape_anchor"] = torch.mean((smpl_param.beta - shape_anchor) ** 2)
                losses_all["loss_shape_norm"] = torch.relu(torch.norm(smpl_param.beta) - shape_norm_soft_bound) ** 2

                loss = self.sum_losses_with_weight(losses_all, self.weight_of_losses)
                latest_total_loss = loss.detach().cpu().item()
                latest_losses = {k: v.detach().cpu().item() for k, v in losses_all.items()}
                loss.backward()
                return loss
            return closure

        # ---------------------------------------------------------
        # 5. Optimize (stage-2: joint optimization)
        # ---------------------------------------------------------
        optimization_stages = [
            (
                "stage2_full",
                [smpl_param.orient, smpl_param.transl, smpl_param.pose, smpl_param.beta],
                max_closure,
                stop_ratio,
                {
                    "body_2d_det": 1.0,
                    "loss_pose_diff": 1.0,
                    "loss_transl_diff": 1.0,
                    "loss_pose_prior": 1.0,
                    "loss_shape_prior": 3.0,
                    "loss_shape_bound": 12.0,
                    "loss_shape_anchor": 16.0,
                    "loss_shape_norm": 8.0,
                },
            ),
            (
                "stage3_refine_2d",
                [smpl_param.orient, smpl_param.transl, smpl_param.pose],
                max(30, max_closure // 2),
                stop_ratio * 0.7,
                {
                    "body_2d_det": 2.0,
                    "loss_pose_diff": 0.8,
                    "loss_transl_diff": 1.0,
                    "loss_pose_prior": 0.5,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 0.0,
                    "loss_shape_norm": 0.0,
                },
            ),
        ]
        stage_shape_soft_bound = {"stage2_full": 0.35, "stage3_refine_2d": 0.45}
        stage_shape_hard_clip = {"stage2_full": 0.55, "stage3_refine_2d": 0.65}

        for stage_name, stage_params, stage_max_closure, stage_stop_ratio, stage_weight_mult in optimization_stages:
            self.weight_of_losses = {
                k: base_weight_of_losses[k] * stage_weight_mult.get(k, 1.0)
                for k in base_weight_of_losses.keys()
            }
            current_shape_soft_bound = stage_shape_soft_bound.get(stage_name, shape_soft_bound)
            current_shape_hard_clip = stage_shape_hard_clip.get(stage_name, shape_hard_clip)
            optimizer = torch.optim.LBFGS(
                stage_params,
                line_search_fn="strong_wolfe",
                lr=lr_lbfgs,
                max_iter=max_iter_per_closure,
            )
            closure = make_closure(optimizer)

            print(f"opt multi-frame ({stage_name})...")
            loss_prev = 1e9
            for iter_idx in range(stage_max_closure):
                optimizer.step(closure)
                with torch.no_grad():
                    smpl_param.beta.clamp_(-current_shape_hard_clip, current_shape_hard_clip)
                loss = latest_total_loss
                dict_losses_str = ", ".join([f"{k}={v:.4f}" for k, v in sorted(latest_losses.items())])
                print(f"[fit][{stage_name}] iter={iter_idx + 1:03d}/{stage_max_closure} total={loss:.4f} | {dict_losses_str}")
                if loss_prev - loss < loss_prev * stage_stop_ratio:
                    break
                loss_prev = loss

        # report per-frame 2D reprojection loss for debugging
        with torch.no_grad():
            _, joints_54_final = self.forward_smpl_param_batch(smpl_param)
            joints_54_final = joints_54_final.reshape((-1, 54, 3))

            index_54_to_std = []
            index_kp_to_std = []
            weights_list = []
            for name_joint, index_joint in key_fmt.items():
                if name_joint in SMPL_ALL_54.keys():
                    index_54_to_std.append(SMPL_ALL_54[name_joint])
                    index_kp_to_std.append(index_joint)
                    weights_list.append(keypoints_weight.get(name_joint, 1.0))

            index_54_to_std = torch.tensor(index_54_to_std, dtype=torch.long, device=joints_54_final.device)
            index_kp_to_std = torch.tensor(index_kp_to_std, dtype=torch.long, device=joints_54_final.device)
            weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=joints_54_final.device)

            joints_2d_54 = joints_54_final @ camera_tensor.T
            joints_2d_54 = joints_2d_54[:, :, 0:2] / joints_2d_54[:, :, 2:3].clamp_min(1e-6)
            joints_2d_std = joints_2d_54[:, index_54_to_std, :]
            keypoints_detected_std = keypoints_tensor[:, index_kp_to_std, :]
            per_joint = torch.norm(joints_2d_std - keypoints_detected_std[:, :, :2], dim=2)

            w_norm = weights_tensor / (weights_tensor.sum() + 1e-8)
            per_frame_body2d = (per_joint * keypoints_detected_std[:, :, 2] * w_norm.unsqueeze(0)).sum(dim=1)
            smpl_param.loss_per_frame = per_frame_body2d.detach().cpu()
            print("[fit] per-frame body_2d_det first10:", smpl_param.loss_per_frame[:10].tolist())
            print(
                "[fit] per-frame body_2d_det stats:"
                f" mean={smpl_param.loss_per_frame.mean().item():.4f},"
                f" median={smpl_param.loss_per_frame.median().item():.4f},"
                f" max={smpl_param.loss_per_frame.max().item():.4f},"
                f" shape={smpl_param.beta}"
            )

        smpl_param.orient.requires_grad_(False)
        smpl_param.transl.requires_grad_(False)
        smpl_param.pose.requires_grad_(False)
        smpl_param.beta.requires_grad_(False)
        smpl_param.scale.requires_grad_(False)
        return smpl_param

    def fit_from_multi_frame_seq_erp(
            self,
            detection_results_list: List[Base_Detect_Engine.Result_Single_Human_Single_Frame],
            camera: Optional[np.ndarray],
            smpl_param_init: Optional[SMPL_Param_Seq] = None,
            camera_model: str = "pinhole",
            image_wh: Optional[Tuple[int, int]] = None,
        ) -> SMPL_Param_Seq:
            """
            Multi-frame SMPL fitting.
    
            Supports two camera models:
            "pinhole"  — original behaviour, requires camera (3x3 K matrix).
            "equirect" — equirectangular panorama, requires image_wh=(W, H).
    
            Args:
                detection_results_list : list of Result_Single_Human_Single_Frame
                camera     : (3,3) pinhole K matrix. Pass None for equirect mode.
                smpl_param_init        : optional SMPL_Param_Seq initialisation.
                camera_model           : "pinhole" | "equirect"
                image_wh               : (W, H) of the equirect image. Required when
                                        camera_model == "equirect".
            """
            assert isinstance(detection_results_list, list) and len(detection_results_list) > 0
            n_frame = len(detection_results_list)
    
            # ---------------------------------------------------------
            # 1. Camera setup
            # ---------------------------------------------------------
            if camera_model == "pinhole":
                assert camera is not None and camera.shape == (3, 3), \
                    f"Pinhole mode requires a (3,3) camera matrix, got {camera}"
                camera_tensor = torch.from_numpy(camera).float()
                erp_W, erp_H  = None, None
            elif camera_model == "equirect":
                assert image_wh is not None, \
                    "equirect mode requires image_wh=(W, H)"
                camera_tensor = None
                erp_W, erp_H  = image_wh
            else:
                raise ValueError(f"Unknown camera_model: {camera_model!r}")
    
            # ---------------------------------------------------------
            # 2. Prepare keypoints tensor
            # ---------------------------------------------------------
            keypoints_list = []
            key_fmt        = None
            for dr in detection_results_list:
                assert isinstance(dr, Base_Detect_Engine.Result_Single_Human_Single_Frame)
                kp = torch.from_numpy(dr.keypoints).float()
    
                # Slightly expand hip width for more stable optimization.
                idx_lhip, idx_rhip = 11, 12
                center = (kp[idx_lhip, :2] + kp[idx_rhip, :2]) / 2.0
                scale  = 1.15
                kp[idx_lhip, :2] = center + (kp[idx_lhip, :2] - center) * scale
                kp[idx_rhip, :2] = center + (kp[idx_rhip, :2] - center) * scale
                keypoints_list.append(kp)
    
                if key_fmt is None:
                    key_fmt = dr.keypoints_format
                else:
                    assert key_fmt == dr.keypoints_format
    
            keypoints_tensor = torch.stack(keypoints_list, dim=0)  # (N, J, 3)
    
            # ---------------------------------------------------------
            # 3. Init SMPL sequence
            # ---------------------------------------------------------
            if smpl_param_init is not None:
                assert isinstance(smpl_param_init, SMPL_Seq_Forward_Module.SMPL_Param_Seq)
                assert smpl_param_init.num_batch == n_frame
                smpl_param = smpl_param_init
            else:
                smpl_param = SMPL_Seq_Forward_Module.SMPL_Param_Seq(
                    dim_shape=self.dim_shape,
                    gender="neutral",
                    num_batch=n_frame,
                )
    
            # Shape prior anchor
            shape_init = torch.tensor(
                [0.08221327, -0.00588347,  0.03931884, -0.0198693,   0.00749705,
                -0.01965939, -0.00093912, -0.01050037,  0.00110335,  0.00892171],
                dtype=smpl_param.beta.dtype,
                device=smpl_param.beta.device,
            )
            smpl_param.beta = torch.zeros_like(smpl_param.beta)
            smpl_param.beta[: min(10, smpl_param.beta.shape[0])] = \
                shape_init[: min(10, smpl_param.beta.shape[0])]
            shape_anchor = smpl_param.beta.clone().detach()
    
            smpl_param.orient = smpl_param.orient.clone().detach().requires_grad_(True)
            smpl_param.transl = smpl_param.transl.clone().detach().requires_grad_(True)
            smpl_param.pose   = smpl_param.pose.clone().detach().requires_grad_(True)
            smpl_param.beta   = smpl_param.beta.clone().detach().requires_grad_(True)
            smpl_param.scale  = smpl_param.scale.clone().detach()
    
            # ---------------------------------------------------------
            # 4. Optimizer hyper-parameters
            # ---------------------------------------------------------
            lr_lbfgs               = 0.5
            max_iter_per_closure   = 60
            max_closure            = 80
            stop_ratio             = 0.01
            fps_assumed            = 30.0
            max_transl_speed_mps   = 3.0
            cube_stage_steps       = 120
            cube_stage_lr          = 0.03
            cube_side_length       = 0.4
            cube_outside_penalty_weight = 4.0
            cube_speed_penalty_weight   = 0.5
            cube_depth_penalty_weight   = 3.0
            cube_min_depth         = 0.5   # metres (pinhole) / metres from origin (equirect)
            shape_soft_bound       = 0.35
            shape_hard_clip        = 0.55
            shape_norm_soft_bound  = 0.9
    
            base_weight_of_losses = {
                "body_2d_det":       n_frame * 1,
                "loss_pose_diff":    n_frame * 0.0,
                "loss_transl_diff":  n_frame * 0.0,
                "loss_pose_prior":   n_frame * 0.05,
                "loss_shape_prior":  n_frame * 6.0,
                "loss_shape_bound":  n_frame * 20.0,
                "loss_shape_anchor": n_frame * 30.0,
                "loss_shape_norm":   n_frame * 15.0,
            }
            self.weight_of_losses = dict(base_weight_of_losses)
            keypoints_weight = {}
    
            # ---------------------------------------------------------
            # 4.1  Stage-1: transl init with cube-bbox IoU objective
            # ---------------------------------------------------------
            print("stage-1 transl optimize with cube-bbox overlap...")
    
            # Build per-frame bounding boxes from keypoints
            bbox_xyxy        = torch.zeros((n_frame, 4), dtype=keypoints_tensor.dtype,
                                        device=keypoints_tensor.device)
            bbox_frame_weight = torch.zeros((n_frame,),  dtype=keypoints_tensor.dtype,
                                            device=keypoints_tensor.device)
            for idx_frame in range(n_frame):
                kp        = keypoints_tensor[idx_frame]
                conf_mask = kp[:, 2] > 0.05
                kp_xy     = kp[conf_mask][:, :2] if conf_mask.any() else kp[:, :2]
                x_min, _  = kp_xy[:, 0].min(dim=0)
                x_max, _  = kp_xy[:, 0].max(dim=0)
                y_min, _  = kp_xy[:, 1].min(dim=0)
                y_max, _  = kp_xy[:, 1].max(dim=0)
                bbox_xyxy[idx_frame] = torch.stack([x_min, y_min, x_max, y_max])
                bbox_frame_weight[idx_frame] = (
                    kp[conf_mask, 2].mean().clamp(min=0.05, max=1.0)
                    if conf_mask.any() else torch.tensor(0.05)
                )
    
            bbox_frame_weight = bbox_frame_weight / (bbox_frame_weight.sum() + 1e-8)
            bbox_w    = (bbox_xyxy[:, 2] - bbox_xyxy[:, 0]).clamp_min(1.0)
            bbox_h    = (bbox_xyxy[:, 3] - bbox_xyxy[:, 1]).clamp_min(1.0)
            bbox_area = bbox_w * bbox_h
    
            half = cube_side_length * 0.5
            cube_offsets = torch.tensor(
                [[-half, -half, -half], [-half, -half,  half],
                [-half,  half, -half], [-half,  half,  half],
                [ half, -half, -half], [ half, -half,  half],
                [ half,  half, -half], [ half,  half,  half]],
                dtype=smpl_param.transl.dtype,
                device=smpl_param.transl.device,
            )
            
            with torch.no_grad():
                u_c = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2.0  # (N,)
                v_c = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2.0  # (N,)

                lon = (u_c / erp_W - 0.5) * (2.0 * math.pi)
                lat = (v_c / erp_H - 0.5) * math.pi   # Y-down

                dir_x = torch.sin(lon) * torch.cos(lat)
                dir_y = -torch.sin(lat)
                dir_z = torch.cos(lon) * torch.cos(lat)

                # 用 bbox 高度估算距离（人体约 1.7m）
                bbox_h_px = (bbox_xyxy[:, 3] - bbox_xyxy[:, 1]).clamp_min(1.0)
                fov_h = bbox_h_px / erp_H * math.pi
                dist_init = 1.7 / (2.0 * torch.tan(fov_h / 2.0).clamp_min(1e-3))
                dist_init = dist_init.clamp(0.5, 15.0)

                smpl_param.transl.data[:, 0] = dir_x * dist_init
                smpl_param.transl.data[:, 1] = dir_y * dist_init
                smpl_param.transl.data[:, 2] = dir_z * dist_init

            print(f"[init] transl from bbox: {smpl_param.transl.data[0].tolist()}")
    
            cube_optimizer      = torch.optim.Adam([smpl_param.transl], lr=cube_stage_lr)
            cube_speed_start_step = cube_stage_steps // 2
    
            for step_idx in range(cube_stage_steps):
                smpl_param.transl.grad = None
                cube_optimizer.zero_grad()
    
                cube_vertices = smpl_param.transl[:, None, :] + cube_offsets[None, :, :]  # (N,8,3)
    
                # ---- projection (camera-model-aware) ----
                if camera_model == "pinhole":
                    cube_proj  = cube_vertices @ camera_tensor.T           # (N,8,3)
                    cube_z     = cube_vertices[:, :, 2]
                    z_safe     = cube_z.clamp_min(cube_min_depth)
                    cube_xy    = cube_proj[:, :, :2] / z_safe[:, :, None]  # (N,8,2)
                    cube_z_min = cube_z.min(dim=1).values                  # (N,)
                else:  # equirect
                    cube_xy    = SMPL_Seq_Forward_Module.erp_project(
                        cube_vertices, erp_W, erp_H
                    )  # (N,8,2)
                    # Use distance from origin instead of Z-depth
                    cube_dist  = torch.norm(cube_vertices, dim=-1)         # (N,8)
                    cube_z_min = cube_dist.min(dim=1).values               # (N,)
    
                cube_x_min, _ = cube_xy[:, :, 0].min(dim=1)
                cube_x_max, _ = cube_xy[:, :, 0].max(dim=1)
                cube_y_min, _ = cube_xy[:, :, 1].min(dim=1)
                cube_y_max, _ = cube_xy[:, :, 1].max(dim=1)
    
                inter_w    = torch.relu(torch.minimum(cube_x_max, bbox_xyxy[:, 2])
                                        - torch.maximum(cube_x_min, bbox_xyxy[:, 0]))
                inter_h    = torch.relu(torch.minimum(cube_y_max, bbox_xyxy[:, 3])
                                        - torch.maximum(cube_y_min, bbox_xyxy[:, 1]))
                inter_area = inter_w * inter_h
    
                cube_w    = (cube_x_max - cube_x_min).clamp_min(1.0)
                cube_h    = (cube_y_max - cube_y_min).clamp_min(1.0)
                cube_area = cube_w * cube_h
    
                iou           = inter_area / (bbox_area + cube_area - inter_area + 1e-6)
                outside_ratio = torch.relu(cube_area - inter_area) / (cube_area + 1e-6)
                geom_loss_per_frame = (1.0 - iou) + cube_outside_penalty_weight * outside_ratio
                loss_geom  = (geom_loss_per_frame * bbox_frame_weight).sum()
                loss_depth = (torch.relu(cube_min_depth - cube_z_min) * bbox_frame_weight).sum()
                loss_cube  = loss_geom + cube_depth_penalty_weight * loss_depth
    
                if step_idx >= cube_speed_start_step and n_frame > 1:
                    transl_speed        = torch.norm(
                        smpl_param.transl[1:] - smpl_param.transl[:-1], dim=1
                    ) * fps_assumed
                    loss_transl_speed   = torch.relu(transl_speed - max_transl_speed_mps).mean()
                    loss_cube           = loss_cube + cube_speed_penalty_weight * loss_transl_speed
                else:
                    loss_transl_speed = torch.zeros((), dtype=smpl_param.transl.dtype,
                                                    device=smpl_param.transl.device)
    
                loss_cube.backward()
                cube_optimizer.step()
    
                with torch.no_grad():
                    if camera_model == "pinhole":
                        smpl_param.transl[:, 0:2].clamp_(-3.0, 3.0)
                        smpl_param.transl[:, 2:3].clamp_(cube_min_depth, 12.0)
                    else:  # equirect: constrain by distance from origin
                        dist  = torch.norm(smpl_param.transl, dim=1, keepdim=True).clamp_min(1e-6)
                        # Clamp distance to [cube_min_depth, 15m]
                        scale = dist.clamp(cube_min_depth, 15.0)
                        smpl_param.transl.data = smpl_param.transl / dist * scale
    
                print(
                    f"[cube_stage] step={step_idx + 1:03d}/{cube_stage_steps}"
                    f" total={loss_cube.detach().cpu().item():.4f}"
                    f" iou_loss={((1.0 - iou) * bbox_frame_weight).sum().detach().cpu().item():.4f}"
                    f" outside={((outside_ratio * bbox_frame_weight).sum()).detach().cpu().item():.4f}"
                    f" depth={loss_depth.detach().cpu().item():.4f}"
                    f" speed={loss_transl_speed.detach().cpu().item():.4f}"
                )
    
            latest_total_loss: float    = float("inf")
            latest_losses: Dict[str, float] = {}
            current_shape_soft_bound    = shape_soft_bound
            current_shape_hard_clip     = shape_hard_clip
    
            # ---------------------------------------------------------
            # 4.2  closure factory (stage 2+)
            # ---------------------------------------------------------
            def make_closure(optimizer_current: torch.optim.Optimizer):
                def closure():
                    nonlocal latest_total_loss, latest_losses, current_shape_soft_bound
                    for p in [smpl_param.orient, smpl_param.transl,
                            smpl_param.pose,   smpl_param.beta]:
                        p.grad = None
                    optimizer_current.zero_grad()
    
                    _, joints_54 = self.forward_smpl_param_batch(smpl_param)
                    joints_54    = joints_54.reshape((-1, 54, 3))
    
                    losses_all = {}
    
                    # ---- 2D reprojection (camera-model-aware) ----
                    if camera_model == "pinhole":
                        losses_kp2d, _ = \
                            self.compute_loss_keypoints_2d_from_single_view_seq_by_keypoints_weight(
                                camera_K=camera_tensor,
                                keypoints_detected=keypoints_tensor,
                                joints_54_current=joints_54,
                                keypoints_format=key_fmt,
                                keypoints_weight=keypoints_weight,
                            )
                    else:  # equirect
                        losses_kp2d, _ = self.compute_loss_keypoints_2d_erp(
                            image_W=erp_W,
                            image_H=erp_H,
                            joints_54_current=joints_54,
                            keypoints_detected=keypoints_tensor,
                            keypoints_format=key_fmt,
                            keypoints_weight=keypoints_weight,
                        )
                    losses_all.update(losses_kp2d)
    
                    # ---- Temporal smoothness ----
                    if n_frame > 1:
                        transl_diff   = smpl_param.transl[1:] - smpl_param.transl[:-1]
                        transl_speed  = torch.norm(transl_diff, dim=1) * fps_assumed
                        losses_all["loss_transl_diff"] = \
                            torch.relu(transl_speed - max_transl_speed_mps).mean()
    
                        pose_diff = (smpl_param.pose[1:] - smpl_param.pose[:-1]).reshape(-1, 23, 3)
                        losses_all["loss_pose_diff"] = torch.norm(pose_diff, dim=2).mean()
                    else:
                        zero = torch.zeros((), dtype=smpl_param.pose.dtype,
                                        device=smpl_param.pose.device)
                        losses_all["loss_pose_diff"]   = zero
                        losses_all["loss_transl_diff"]  = zero
    
                    # ---- Shape / pose priors ----
                    losses_all["loss_pose_prior"]   = torch.norm(smpl_param.pose, dim=1).mean()
                    losses_all["loss_shape_prior"]  = torch.mean(smpl_param.beta ** 2)
                    losses_all["loss_shape_bound"]  = torch.mean(
                        torch.relu(torch.abs(smpl_param.beta) - current_shape_soft_bound) ** 2
                    )
                    losses_all["loss_shape_anchor"] = torch.mean(
                        (smpl_param.beta - shape_anchor) ** 2
                    )
                    losses_all["loss_shape_norm"]   = (
                        torch.relu(torch.norm(smpl_param.beta) - shape_norm_soft_bound) ** 2
                    )
    
                    loss = self.sum_losses_with_weight(losses_all, self.weight_of_losses)
                    latest_total_loss = loss.detach().cpu().item()
                    latest_losses     = {k: v.detach().cpu().item()
                                        for k, v in losses_all.items()}
                    loss.backward()
                    return loss
                return closure
    
            # ---------------------------------------------------------
            # 5. Multi-stage optimization
            # ---------------------------------------------------------
            optimization_stages = [
                (
                    "stage2_full",
                    [smpl_param.orient, smpl_param.transl, smpl_param.pose, smpl_param.beta],
                    max_closure,
                    stop_ratio,
                    {
                        "body_2d_det":       1.0,
                        "loss_pose_diff":    1.0,
                        "loss_transl_diff":  1.0,
                        "loss_pose_prior":   1.0,
                        "loss_shape_prior":  3.0,
                        "loss_shape_bound":  12.0,
                        "loss_shape_anchor": 16.0,
                        "loss_shape_norm":   8.0,
                    },
                ),
                (
                    "stage3_refine_2d",
                    [smpl_param.orient, smpl_param.transl, smpl_param.pose],
                    max(30, max_closure // 2),
                    stop_ratio * 0.7,
                    {
                        "body_2d_det":       2.0,
                        "loss_pose_diff":    0.8,
                        "loss_transl_diff":  1.0,
                        "loss_pose_prior":   0.5,
                        "loss_shape_prior":  0.0,
                        "loss_shape_bound":  0.0,
                        "loss_shape_anchor": 0.0,
                        "loss_shape_norm":   0.0,
                    },
                ),
            ]
            stage_shape_soft_bound = {"stage2_full": 0.35, "stage3_refine_2d": 0.45}
            stage_shape_hard_clip  = {"stage2_full": 0.55, "stage3_refine_2d": 0.65}
    
            for (stage_name, stage_params,
                stage_max_closure, stage_stop_ratio,
                stage_weight_mult) in optimization_stages:
    
                self.weight_of_losses = {
                    k: base_weight_of_losses[k] * stage_weight_mult.get(k, 1.0)
                    for k in base_weight_of_losses
                }
                current_shape_soft_bound = stage_shape_soft_bound.get(stage_name, shape_soft_bound)
                current_shape_hard_clip  = stage_shape_hard_clip.get(stage_name, shape_hard_clip)
    
                optimizer = torch.optim.LBFGS(
                    stage_params,
                    line_search_fn="strong_wolfe",
                    lr=lr_lbfgs,
                    max_iter=max_iter_per_closure,
                )
                closure = make_closure(optimizer)
    
                print(f"opt multi-frame ({stage_name})...")
                loss_prev = 1e9
                for iter_idx in range(stage_max_closure):
                    optimizer.step(closure)
                    with torch.no_grad():
                        smpl_param.beta.clamp_(-current_shape_hard_clip, current_shape_hard_clip)
                    loss = latest_total_loss
                    dict_losses_str = ", ".join(
                        f"{k}={v:.4f}" for k, v in sorted(latest_losses.items())
                    )
                    print(f"[fit][{stage_name}] iter={iter_idx + 1:03d}/{stage_max_closure}"
                        f" total={loss:.4f} | {dict_losses_str}")
                    if loss_prev - loss < loss_prev * stage_stop_ratio:
                        break
                    loss_prev = loss
    
            # ---------------------------------------------------------
            # 6. Per-frame debug report
            # ---------------------------------------------------------
            with torch.no_grad():
                _, joints_54_final = self.forward_smpl_param_batch(smpl_param)
                joints_54_final    = joints_54_final.reshape((-1, 54, 3))
    
                index_54_to_std, index_kp_to_std, weights_list = [], [], []
                for name_joint, index_joint in key_fmt.items():
                    if name_joint in SMPL_ALL_54:
                        index_54_to_std.append(SMPL_ALL_54[name_joint])
                        index_kp_to_std.append(index_joint)
                        weights_list.append(keypoints_weight.get(name_joint, 1.0))
    
                dev = joints_54_final.device
                index_54_to_std = torch.tensor(index_54_to_std, dtype=torch.long,    device=dev)
                index_kp_to_std = torch.tensor(index_kp_to_std, dtype=torch.long,    device=dev)
                weights_tensor  = torch.tensor(weights_list,     dtype=torch.float32, device=dev)
    
                # ---- projection (camera-model-aware) ----
                if camera_model == "pinhole":
                    joints_2d_54 = joints_54_final @ camera_tensor.T
                    joints_2d_54 = joints_2d_54[:, :, 0:2] / joints_2d_54[:, :, 2:3].clamp_min(1e-6)
                else:  # equirect
                    joints_2d_54 = SMPL_Seq_Forward_Module.erp_project(
                        joints_54_final, erp_W, erp_H
                    )
    
                joints_2d_std          = joints_2d_54[:, index_54_to_std, :]
                keypoints_detected_std = keypoints_tensor[:, index_kp_to_std, :]
                per_joint              = torch.norm(
                    joints_2d_std - keypoints_detected_std[:, :, :2], dim=2
                )
                w_norm          = weights_tensor / (weights_tensor.sum() + 1e-8)
                per_frame_body2d = (
                    per_joint * keypoints_detected_std[:, :, 2] * w_norm.unsqueeze(0)
                ).sum(dim=1)
    
                smpl_param.loss_per_frame = per_frame_body2d.detach().cpu()
                print("[fit] per-frame body_2d_det first10:",
                    smpl_param.loss_per_frame[:10].tolist())
                print(
                    f"[fit] per-frame body_2d_det stats:"
                    f" mean={smpl_param.loss_per_frame.mean().item():.4f},"
                    f" median={smpl_param.loss_per_frame.median().item():.4f},"
                    f" max={smpl_param.loss_per_frame.max().item():.4f},"
                    f" shape={smpl_param.beta}"
                )
    
            smpl_param.orient.requires_grad_(False)
            smpl_param.transl.requires_grad_(False)
            smpl_param.pose.requires_grad_(False)
            smpl_param.beta.requires_grad_(False)
            smpl_param.scale.requires_grad_(False)
            return smpl_param