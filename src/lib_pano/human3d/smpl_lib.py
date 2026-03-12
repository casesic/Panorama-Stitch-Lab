
from typing import Dict, Tuple, List, Any, Optional, Union

import os.path as osp
import numpy as np

import pickle
import cv2 as cv
import smplx
import smplx.utils
import torch
import copy

from icecream import ic

from human_body_prior.train.vposer_smpl import VPoser
from lib_dynamic_hvip.smpl.skeleton import get_3d_torso_center_from_3d_joint
from lib_dynamic_hvip.smpl.smpl_full_54.get_full_joints import (
    get_full_joints_by_model_full_tensor,
    get_full_joints_by_model_full_np
)
from lib_dynamic_hvip.smpl.smpl_full_54.smpl_model_full import SMPL_Full
import lib_dynamic_hvip.constants_skeleton as Lib_Const_Skelenton
from lib_dynamic_hvip.network.rotate_representation.to_rotmat import (
    rot6d_to_rotmat,
    aa_to_rotmat,
    quat_to_rotmat
)
from lib_dynamic_hvip.network.rotate_representation.from_rotmat import (
    rotmat_to_aa
)
import lib_dynamic_hvip.local_setting as local_setting 


'''
Originally, smpl_lib refers to SMPL model library only
but some transform functions are also placed here
such as:
update_root_pose_0_via_R,
add_RT_to_smpl_param,
add_RT_to_smpl_param_tensor_batch,
add_RT_to_smpl_param_with_known_root
When I realized that these functions are more related to SMPL transform rather than SMPL model loading,
I make a new file smpl_transform.py to place these functions
but I did not move these functions from smpl_lib.py to smpl_transform.py,
I just import them from smpl_lib.py to smpl_transform.py and re-export them there
'''

PATH_SMPL_MODELS = osp.join(local_setting.PATH_MODEL_LIB, 'smpl_models')

def get_vposer(device_str:str='cpu') -> (VPoser):
    vposer = VPoser(512, 32, [3, 21]).eval()
    path_weight = osp.join(PATH_SMPL_MODELS, 'TR00_E096.pt')
    assert osp.isfile(path_weight), path_weight
    vposer.load_state_dict(torch.load(path_weight, map_location=device_str))
    return vposer

def get_smpl_full(version:str='v100', gender:str='neutral') -> (SMPL_Full):
    assert osp.isdir(PATH_SMPL_MODELS), PATH_SMPL_MODELS

    PATH_SMPL_FULL = osp.join(PATH_SMPL_MODELS, 'smpl_full')
    PATH_J_REGRESSOR_EXTRA = osp.join(PATH_SMPL_FULL, 'J_regressor_extra.npy')

    assert osp.isdir(PATH_SMPL_FULL), PATH_SMPL_FULL
    assert osp.isfile(PATH_J_REGRESSOR_EXTRA), PATH_J_REGRESSOR_EXTRA
    assert gender in ['neutral', 'female', 'male']

    smpl_model = SMPL_Full(PATH_SMPL_FULL, J_reg_extra9_path=PATH_J_REGRESSOR_EXTRA, str_version=version, gender=gender)
    #smpl_f = SMPL_Full(PATH_SMPL_FULL, J_reg_extra9_path=PATH_J_REGRESSOR_EXTRA, str_version=version, gender='female')
    #smpl_m = SMPL_Full(PATH_SMPL_FULL, J_reg_extra9_path=PATH_J_REGRESSOR_EXTRA, str_version=version, gender='male')
    return smpl_model

def get_the_matrix_of_smplx_to_smpl() -> (np.ndarray):
    assert osp.isdir(PATH_SMPL_MODELS), PATH_SMPL_MODELS

    #this pkl is copyed from PyMAF-X-smplx
    path_transfer = osp.join(PATH_SMPL_MODELS, 'transfer', 'smplx_to_smpl.pkl')
    assert osp.isfile(path_transfer), path_transfer

    with open(path_transfer, 'rb') as f:
        dict_transfer_smplx2smpl_from_pymaf = pickle.load(f)

    assert set(dict_transfer_smplx2smpl_from_pymaf.keys()) == \
        set(['bc', 'closest_points_on_smpl', 'matrix', 'closest_faces', 'valid_vertices', 'mapping']), \
        dict_transfer_smplx2smpl_from_pymaf.keys()
    
    matrix_smplx_to_smpl:np.ndarray = dict_transfer_smplx2smpl_from_pymaf['matrix']
    assert type(matrix_smplx_to_smpl) == np.ndarray, type(matrix_smplx_to_smpl)
    assert matrix_smplx_to_smpl.shape == (6890, 10475), matrix_smplx_to_smpl.shape

    return matrix_smplx_to_smpl

def get_official_smpl(gender:str='neutral') -> (smplx.SMPL):
    path_folder_smpl = osp.join(PATH_SMPL_MODELS, 'smpl')
    assert osp.isdir(path_folder_smpl), path_folder_smpl

    smpl_model_official = smplx.SMPL(
        gender=gender,
        model_path=path_folder_smpl
    )
    return smpl_model_official

def get_official_smplh(gender:str='neutral') -> (smplx.SMPLH):
    path_folder_smplh = osp.join(PATH_SMPL_MODELS, 'smplh')
    assert osp.isdir(path_folder_smplh), path_folder_smplh
    smplh_model_official = smplx.SMPLH(
        model_path=path_folder_smplh, 
        ext='npz', 
        gender=gender, 
        use_pca=True, 
        num_pca_comps=45, 
        flat_hand_mean=True
    )
    return smplh_model_official

def get_smplx_official(gender:str='neutral') -> (smplx.SMPLX):
    path_folder_smplx = osp.join(PATH_SMPL_MODELS, 'smplx')
    assert osp.isdir(path_folder_smplx), path_folder_smplx
    smplx_model_official = smplx.SMPLXLayer(
        gender=gender, 
        model_path=path_folder_smplx,
        ext='npz',
        num_betas=10,
        use_pca=False, 
        batch_size=1, 
        use_face_contour=True, 
        num_pca_comps=45
    )
    return smplx_model_official

def get_smpl_official(gender:str='neutral') -> (smplx.SMPL):
    path_folder_smpl = osp.join(PATH_SMPL_MODELS, 'smpl')
    smpl_model_official = smplx.SMPL(
        gender=gender,
        model_path=path_folder_smpl,
        num_betas=10
    )
    return smpl_model_official

def forward_smplx_official_overwrite_member(
        smplx_model:smplx.SMPLX,
        dict_smplx_param:Dict[str, torch.Tensor]    
    ):
    N = dict_smplx_param['body_pose'].shape[0]
    for i in range(N):
        dict_smplx_param_one = {key: value[i] for key, value in dict_smplx_param.items()}
        smplx_model.reset_params(**dict_smplx_param_one)

        assert isinstance(dict_smplx_param_one['body_pose'], torch.Tensor), type(dict_smplx_param_one['body_pose'])

        model_output = smplx_model.forward(
            return_verts=True,   
            body_pose=dict_smplx_param_one['body_pose'],
            return_full_pose=True
        ) 
        assert isinstance(model_output, smplx.utils.SMPLXOutput), type(model_output)
        assert isinstance(model_output.vertices, torch.Tensor), type(model_output.vertices)
        smplx_verts = model_output.vertices.detach().cpu().squeeze().numpy()
        raise NotImplementedError('not finished yet')
    return

def forward_smplx_official(
        smplx_model:smplx.SMPLXLayer,
        poses:np.ndarray, 
        shapes:np.ndarray,
        trans:np.ndarray=np.zeros((3)),
        scale:float=1.0,
        matrix_smplx2smpl:Optional[np.ndarray]=None,
        gender:str='neutral',
    ) -> (np.ndarray):
    '''
    @param: scale
    @param: trans
    @param: matrix_smplx2smpl, if matrix_smplx2smpl is not None, it will transfer the smplx result to smpl result,
    at that time, matrix_smplx2smpl should be a numpy matrix with shape (6890, 10475)
    '''
    assert isinstance(smplx_model, smplx.SMPLX), type(smplx_model)
    assert poses.shape == (72,), poses.shape
    assert shapes.shape == (10,), shapes.shape
    assert trans.shape == (3,), trans.shape
    assert isinstance(scale, (float))
    assert gender == smplx_model.gender

    global_orient_aa = poses[:3]
    global_orient_R = cv.Rodrigues(global_orient_aa)[0]

    body_poses_as_Rs = np.zeros((21, 3, 3))
    for i in range(21):
        body_poses_as_Rs[i, :, :] = cv.Rodrigues(poses[3+i*3:3+(i+1)*3])[0]

    assert global_orient_R.shape == (3, 3), global_orient_R.shape

    smplx_output = smplx_model.forward(
        betas=torch.from_numpy(shapes).to(torch.float32)[None],
        body_pose=torch.from_numpy(body_poses_as_Rs.reshape(1, 21, 3, 3)).to(torch.float32),
        global_orient=torch.from_numpy(global_orient_R.reshape(1, 3, 3)).to(torch.float32),
        pose2rot=False,
        get_skin=True
    )
    assert isinstance(smplx_output.vertices, torch.Tensor), type(smplx_output.vertices)
    verts_smplx = smplx_output.vertices[0].detach().cpu().numpy()
    
    assert isinstance(verts_smplx, np.ndarray), type(verts_smplx)

    assert verts_smplx.shape == (10475, 3), verts_smplx.shape

    #apply scale
    verts_smplx = verts_smplx * scale
    #apply trans
    verts_smplx = verts_smplx + trans.reshape((1, 3))

    if matrix_smplx2smpl is not None:
        verts_smpl = matrix_smplx2smpl @ verts_smplx
        assert verts_smpl.shape == (6890, 3), verts_smpl.shape
        return verts_smpl
    else:
        return verts_smplx

def forward_smpl_full(
        smpl_model:SMPL_Full, 
        poses:np.ndarray, 
        shapes:np.ndarray, 
        trans:np.ndarray=np.zeros((3)), 
        trans_torso:np.ndarray=np.zeros((3)),
        scale:Union[float, np.ndarray]=1.0,
        gender:str='neutral',
        ret_joints_54:bool=False,
        pose2rot:bool=True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    '''
    either trans or trans_torso should be provided
    which means do not provide both
    '''
    assert poses.shape == (72,), poses.shape
    assert shapes.shape == (10,), shapes.shape
    assert trans.shape == (3,), trans.shape
    if isinstance(scale, (float)):
        pass
    else:
        assert isinstance(scale, np.ndarray), type(scale)
        assert scale.shape == (1,), scale.shape


    poses_tensor = torch.from_numpy(poses).to(torch.float32)[None]
    betas_tensor = torch.from_numpy(shapes).to(torch.float32)[None]
    
    assert poses_tensor.shape == torch.Size([1, 72])
    assert betas_tensor.shape == torch.Size([1, 10])
    assert smpl_model.gender == gender, (smpl_model.gender, gender)

    dict_smpl_out = smpl_model.forward(poses=poses_tensor, betas=betas_tensor, pose2rot=pose2rot)
    verts_smpl_tensor = dict_smpl_out['verts'] #[1, 6890, 3]
    joints_54_tensor = dict_smpl_out['j3d'] #[1, 54, 3]
    verts_smpl_np:np.ndarray = verts_smpl_tensor[0, :, :].cpu().numpy()
    joints_54:np.ndarray = joints_54_tensor[0, :, :].cpu().numpy()
    assert verts_smpl_np.shape == (6890, 3), verts_smpl_np
    assert joints_54.shape == (54, 3), joints_54

    if np.average(np.abs(trans)) > 1e-5:
        assert np.average(np.abs(trans_torso)) < 1e-5, trans_torso

        verts_smpl_np = verts_smpl_np * scale + trans[None]
        joints_54 = joints_54 * scale + trans[None]
    elif np.average(np.abs(trans_torso)) > 1e-5:
        assert np.average(np.abs(trans)) < 1e-5, trans

        torso_center_3d_src = get_3d_torso_center_from_3d_joint(joints_54)
        # index_torso_tmp = [1, 2, 16, 17]
        # torso_center_3d_src = np.average(joints_54_tensor[index_torso_tmp, :], axis=0)
        torso_center_3d_target = get_3d_torso_center_from_3d_joint(joints_54 + trans_torso)
        # print(torso_center_3d_src)
        # print(torso_center_3d_src2)
        # print(np.max(verts_smpl_np[:, 1]), np.min(verts_smpl_np[:, 1]))
        verts_smpl_np = (verts_smpl_np - torso_center_3d_src) * scale + torso_center_3d_target
        joints_54 = (joints_54 - torso_center_3d_src) * scale + torso_center_3d_target
    else:
        assert np.average(np.abs(trans)) < 1e-5, trans
        assert np.average(np.abs(trans_torso)) < 1e-5, trans
        verts_smpl_np = verts_smpl_np * scale
    
    assert verts_smpl_np.shape == (6890, 3), verts_smpl_np
    assert joints_54.shape == (54, 3), joints_54


    if ret_joints_54:
        return verts_smpl_np, joints_54
    else:
        return verts_smpl_np
    

def forward_smpl_full_batch_tensor(
        smpl_model, 
        poses: torch.Tensor, 
        shapes: torch.Tensor, 
        trans: Optional[torch.Tensor] = None, 
        scale: Optional[torch.Tensor] = None,
        ret_joints_54: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
    Tensor version of SMPL forward pass supporting gradient propagation
    Args:
        smpl_model: SMPL model instance
        poses: Body poses [B, 72] or [B, 24, 3]
        shapes: Shape parameters [B, 10]
        trans: Global translation [B, 3]
        scale: Scaling factor [B]
        ret_joints_54: Whether to return 54 joints
    Returns:
        Vertices and optionally joints
    '''
    
    B = poses.shape[0]
    
    # Convert pose to flat representation if needed
    if poses.dim() == 3 and poses.shape[1:] == (24, 3):
        poses = poses.reshape(B, 72)
    assert poses.shape == (B, 72), f"Pose shape {poses.shape} != (B, 72)"
    assert shapes.shape == (B, 10), f"Shape shape {shapes.shape} != (B, 10)"
    
    # Handle scale
    if scale is None:
        scale = torch.ones(B, dtype=poses.dtype, device=poses.device)
    else:
        assert scale.shape == (B,), f"Scale shape {scale.shape} != (B,)"
    
    # Handle translation
    if trans is None:
        trans = torch.zeros(B, 3, dtype=poses.dtype, device=poses.device)
    else:
        assert trans.shape == (B, 3), f"Trans shape {trans.shape} != (B, 3)"
    
    # Run SMPL model
    dict_smpl_out = smpl_model.forward(poses=poses, betas=shapes)
    verts = dict_smpl_out['verts']  # [B, 6890, 3]
    joints_54 = dict_smpl_out['j3d']  # [B, 54, 3]
    
    # Apply scaling and translation
    scale = scale.view(B, 1, 1)
    trans = trans.view(B, 1, 3)
    
    # Only apply transformation if needed
    if not torch.allclose(trans, torch.zeros_like(trans), atol=1e-5):
        verts = verts * scale + trans
        joints_54 = joints_54 * scale + trans
    else:
        verts = verts * scale
        joints_54 = joints_54 * scale
    
    # Return results
    if ret_joints_54:
        return verts, joints_54
    else:
        return verts

def forward_smpl_full_batch(
        smpl_model:SMPL_Full, 
        poses:np.ndarray, 
        shapes:np.ndarray, 
        trans:Optional[np.ndarray]=None, 
        trans_torso:Optional[np.ndarray]=None,
        scale:Optional[np.ndarray]=None,
        gender:str='neutral',
        ret_joints_54:bool=False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    '''
    either trans or trans_torso should be provided
    which means do not provide both
    '''

    B = poses.shape[0]

    if poses.shape in [(B, 24, 3)]:
        poses = poses.copy().reshape(B, 72)
    assert poses.shape == (B, 72), poses.shape
    assert shapes.shape == (B, 10), shapes.shape
    if scale is not None:
        assert scale.shape == (B,), scale.shape
    else:
        scale = np.ones((B,))
    assert gender == smpl_model.gender, (smpl_model.gender, gender)
    if trans is not None:
        assert trans.shape == (B, 3), trans.shape
    else:
        trans = np.zeros((B, 3))
    if trans_torso is not None:
        assert trans_torso.shape == (B, 3), trans_torso.shape
    else:
        trans_torso = np.zeros((B, 3))

    poses_tensor = torch.from_numpy(poses).to(torch.float32)
    betas_tensor = torch.from_numpy(shapes).to(torch.float32)

    dict_smpl_out = smpl_model.forward(poses=poses_tensor, betas=betas_tensor)
    verts_smpl_tensor = dict_smpl_out['verts'] #[B, 6890, 3]
    joints_54_tensor = dict_smpl_out['j3d'] #[B, 54, 3]
    verts_smpl_np:np.ndarray = verts_smpl_tensor.cpu().numpy()
    joints_54:np.ndarray = joints_54_tensor.cpu().numpy()
    assert verts_smpl_np.shape == (B, 6890, 3), verts_smpl_np
    assert joints_54.shape == (B, 54, 3), joints_54

    if not np.allclose(trans, np.zeros_like(trans), rtol=1e-5, atol=1e-5):
        assert np.allclose(trans_torso, np.zeros_like(trans_torso), rtol=1e-4, atol=1e-4), trans_torso

        verts_smpl_np = verts_smpl_np * scale.reshape(B, 1, 1) + trans.reshape(B, 1, 3)
        joints_54 = joints_54 * scale.reshape(B, 1, 1) + trans.reshape(B, 1, 3)
    elif not np.allclose(trans_torso, np.zeros_like(trans_torso), rtol=1e-5, atol=1e-5):
        assert np.allclose(trans, np.zeros_like(trans), rtol=1e-4, atol=1e-4), trans

        torso_center_3d_src = np.array(
            [get_3d_torso_center_from_3d_joint(joints_54[i]) for i in range(B)]
        )
        assert torso_center_3d_src.shape == (B, 3), torso_center_3d_src
        torso_center_3d_target = np.array(
            [get_3d_torso_center_from_3d_joint(joints_54[i] + trans_torso[i].reshape(1, 3)) for i in range(B)]
        )
        assert torso_center_3d_target.shape == (B, 3), torso_center_3d_target
        verts_smpl_np = (verts_smpl_np - torso_center_3d_src.reshape(B, 1, 3)) * scale.reshape(B, 1, 1) \
            + torso_center_3d_target.reshape(B, 1, 3)
        joints_54 = (joints_54 - torso_center_3d_src.reshape(B, 1, 3)) * scale.reshape(B, 1, 1) \
            + torso_center_3d_target.reshape(B, 1, 3)
    else:
        assert np.allclose(trans, np.zeros_like(trans), rtol=1e-4, atol=1e-4), (trans, trans_torso)
        assert np.allclose(trans_torso, np.zeros_like(trans_torso), rtol=1e-4, atol=1e-4), (trans, trans_torso)
        verts_smpl_np = verts_smpl_np * scale.reshape(B, 1, 1)

    assert verts_smpl_np.shape == (B, 6890, 3), verts_smpl_np.shape
    assert joints_54.shape == (B, 54, 3), joints_54.shape

    if ret_joints_54:
        return verts_smpl_np, joints_54
    else:
        return verts_smpl_np
    

def forward_smpl_full_list(
        smpl_model:SMPL_Full, 
        list_smpl_param:List[Dict[str, np.ndarray]],
        ret_joints_54:bool=False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    '''
    either trans or trans_torso should be provided
    which means do not provide both
    '''
    
    '''
    smpl_model:SMPL_Full, 
    poses:np.ndarray, 
    shapes:np.ndarray, 
    trans:Optional[np.ndarray]=None, 
    trans_torso:Optional[np.ndarray]=None,
    scale:Optional[np.ndarray]=None,
    gender:str='neutral',
    '''
    
    num_smpl = len(list_smpl_param)
    assert num_smpl > 0, num_smpl
    
    poses_batch = np.zeros((num_smpl, 72), dtype=np.float32)
    shapes_batch = np.zeros((num_smpl, 10), dtype=np.float32)
    if 'trans' in list_smpl_param[0]:
        flag_use_trans = True
        trans_batch = np.zeros((num_smpl, 3), dtype=np.float32)
        trans_torso_batch = None
    else:
        assert 'trans_torso' in list_smpl_param[0]
        flag_use_trans = False
        trans_torso_batch = np.zeros((num_smpl, 3), dtype=np.float32)
        trans_batch = None
    scale_batch = np.ones((num_smpl,), dtype=np.float32)
    gender = None
    
    for index_smpl in range(num_smpl):
        poses_batch[index_smpl] = list_smpl_param[index_smpl]['poses']
        shapes_batch[index_smpl] = list_smpl_param[index_smpl]['shapes']
        if flag_use_trans:
            assert isinstance(trans_batch, np.ndarray), type(trans_batch)
            trans_batch[index_smpl] = list_smpl_param[index_smpl]['trans']
        else:
            assert isinstance(trans_torso_batch, np.ndarray), type(trans_batch)
            trans_torso_batch[index_smpl] = list_smpl_param[index_smpl]['trans_torso']
        scale_batch[index_smpl] = list_smpl_param[index_smpl]['scale']
        
        if gender is None:
            gender = list_smpl_param[index_smpl]['gender']
        else:
            assert gender == list_smpl_param[index_smpl]['gender']
            
    assert poses_batch.shape == (num_smpl, 72), poses_batch.shape
    assert shapes_batch.shape == (num_smpl, 10), shapes_batch.shape
    assert scale_batch.shape == (num_smpl,), scale_batch.shape

    if flag_use_trans:
        assert isinstance(trans_batch, np.ndarray), type(trans_batch)
        assert trans_batch.shape == (num_smpl, 3), trans_batch.shape
    else:
        assert isinstance(trans_torso_batch, np.ndarray), type(trans_batch)
        assert trans_torso_batch.shape == (num_smpl, 3), trans_torso_batch.shape
        
    batch_ret = forward_smpl_full_batch(
        smpl_model=smpl_model, 
        poses=poses_batch, 
        shapes=shapes_batch, 
        trans=trans_batch,
        trans_torso=trans_torso_batch, 
        scale=scale_batch, 
        ret_joints_54=ret_joints_54
    )
    return batch_ret
        

def forward_smpl_full_tensor(
        smpl_model:SMPL_Full, 
        poses:torch.Tensor, 
        shapes:torch.Tensor, 
        trans:torch.Tensor=torch.zeros((3)),
        scale:Union[torch.Tensor, float]=1.0,
        ret_joints_54:bool=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
    either trans or trans_torso should be provided
    which means do not provide both
    '''
    assert poses.shape == torch.Size([72]), poses.shape
    assert shapes.shape == torch.Size([10]), shapes.shape
    assert trans.shape == torch.Size([3]), trans.shape

    if isinstance(scale, float):
        scale = torch.tensor(scale, dtype=torch.float32)
    assert isinstance(scale, torch.Tensor), scale
    assert scale.shape == (1,), scale.shape

    dict_smpl_out = smpl_model.forward(
        poses=poses[None], 
        betas=shapes[None]
    )
    verts_smpl_tensor = dict_smpl_out['verts'][0, :, :] #[1, 6890, 3] to [6890, 3]
    joints_54_tensor = dict_smpl_out['j3d'][0, :, :] #[1, 54, 3] to [54, 3]
    
    assert verts_smpl_tensor.shape == torch.Size([6890, 3]), verts_smpl_tensor.shape
    assert joints_54_tensor.shape == torch.Size([54, 3]), joints_54_tensor.shape

    verts_smpl_tensor = verts_smpl_tensor * scale + trans
    joints_54_tensor = joints_54_tensor * scale + trans

    if ret_joints_54:
        return verts_smpl_tensor, joints_54_tensor
    else:
        return verts_smpl_tensor
    

def forward_smpl_full_tensor_batch(
        smpl_model:SMPL_Full, 
        poses:torch.Tensor, 
        shapes:torch.Tensor, 
        trans:torch.Tensor,
        scale:Optional[torch.Tensor]=None,
        ret_joints_54:bool=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
    either trans or trans_torso should be provided
    which means do not provide both
    '''
    B = poses.shape[0]
    if poses.shape == (B, 24*3):
        pose2rot = True
    elif poses.shape == (B, 24, 3, 3):
        pose2rot = False
    elif poses.shape == (B, 24*3*3):
        poses = poses.reshape((B, 24, 3, 3))
        pose2rot = False
    else:
        raise ValueError(poses.shape)

    assert shapes.shape == torch.Size([B, 10]), shapes.shape
    assert trans.shape == torch.Size([B, 3]), trans.shape
    if scale is not None:
        assert scale.shape in [(B, 1), (B,)], scale.shape

    dict_smpl_out = smpl_model.forward(
        poses=poses, 
        betas=shapes,
        pose2rot=pose2rot
    )
    
    verts_smpl_tensor = dict_smpl_out['verts'] #[B, 6890, 3]
    joints_54_tensor = dict_smpl_out['j3d'] #[B, 54, 3]


    assert verts_smpl_tensor.shape == torch.Size([B, 6890, 3]), verts_smpl_tensor.shape
    assert joints_54_tensor.shape == torch.Size([B, 54, 3]), joints_54_tensor.shape
    
    if scale is not None:
        verts_smpl_tensor = verts_smpl_tensor * scale.reshape((B, 1, 1)) + trans.reshape((B, 1, 3))
        joints_54_tensor = joints_54_tensor * scale.reshape((B, 1, 1)) + trans.reshape((B, 1, 3))
    else:
        verts_smpl_tensor = verts_smpl_tensor + trans.reshape((B, 1, 3))
        joints_54_tensor = joints_54_tensor + trans.reshape((B, 1, 3))  
    
    if ret_joints_54:
        return verts_smpl_tensor, joints_54_tensor
    else:
        return verts_smpl_tensor
    
    
    
def forward_smpl_full_batch_as_list(
        list_smpl_param:List[Dict[str, Any]], 
        smpl_model:SMPL_Full,
        ret_joints_54:bool=False
    ) -> Union[Tuple[List[np.ndarray], List[np.ndarray]], List[np.ndarray]]:
    '''
    return List[verts [6890, 3]], List[joints [54, 3]](Optional)
    '''
    if not ret_joints_54:
        list_smpl_verts:List[np.ndarray] = []
        for dict_smpl_param in list_smpl_param:
            smpl_verts = forward_smpl_full(
                **dict_smpl_param, 
                smpl_model=smpl_model,
                ret_joints_54=False,
                gender=smpl_model.gender,
            )
            assert isinstance(smpl_verts, np.ndarray), type(smpl_verts)
            list_smpl_verts.append(smpl_verts)
        return list_smpl_verts
    else:
        list_joints_54 = []
        list_verts_smpl = []
        for smpl_param in list_smpl_param:
            assert ret_joints_54, ret_joints_54
            assert 'gender' not in smpl_param, smpl_param.keys()
            verts_smpl, joints_54 = forward_smpl_full(**smpl_param, smpl_model=smpl_model, ret_joints_54=True, gender=smpl_model.gender)
            list_verts_smpl.append(verts_smpl)
            list_joints_54.append(joints_54)
    
        return list_verts_smpl, list_joints_54


def get_full_joint_batch(list_verts_smpl:List[np.ndarray], smpl_model:SMPL_Full) -> (List[np.ndarray]):
    '''
    @param list_verts_smpl: List[verts [6890, 3]]
    @return List[joints [54, 3]]
    '''
    list_joints_3d = []
    for verts_smpl in list_verts_smpl:
        joints_3d = get_full_joints_by_model_full_tensor(
            torch.from_numpy(verts_smpl).to(torch.float32), 
            smpl_model, 
            num_joint=54
        ).cpu().numpy()
        list_joints_3d.append(joints_3d)
    return list_joints_3d

def update_root_pose_0_via_R(root_0_pose:np.ndarray, R:np.ndarray):
    assert root_0_pose.shape == (3,), root_0_pose.shape
    assert R.shape == (3, 3), R.shape

    root_0_pose = root_0_pose.astype(np.float64)
    R = R.astype(np.float64)

    orient_matrix_ori, ____ = cv.Rodrigues(root_0_pose.reshape((3, 1)))
    assert orient_matrix_ori.shape == (3, 3), orient_matrix_ori.shape

    orient_matrix_new = np.matmul(R, orient_matrix_ori)

    orient_root_0_new, ____ = cv.Rodrigues(orient_matrix_new)
    assert orient_root_0_new.shape == (3, 1), orient_root_0_new.shape
    
    return orient_root_0_new.reshape((3))

def add_RT_to_smpl_param(
    dict_smpl_param:Dict[str, Any], 
    RT:np.ndarray,
    smpl_model:SMPL_Full
) -> (Dict[str, Any]):
    '''
    RT is [R|T] with shape (4, 4)
    '''
    assert isinstance(dict_smpl_param, dict), type(dict_smpl_param)
    assert set(dict_smpl_param.keys()) >= set(['poses', 'shapes', 'trans']), dict_smpl_param.keys()
    
    assert RT.shape == (4, 4), RT.shape
    R = RT[0:3, 0:3]
    T = RT[0:3, 3]
    
    ____, joints_54 = forward_smpl_full(**dict_smpl_param, ret_joints_54=True, smpl_model=smpl_model)
    J_ROOT = joints_54[0]
    smpl_param_new = {
        k: v.copy() if isinstance(v, np.ndarray) else copy.copy(v)
        for k, v in dict_smpl_param.items()
    }
    smpl_param_new['poses'][0:3] = update_root_pose_0_via_R(smpl_param_new['poses'][0:3], R)
    trans_bias = J_ROOT - np.matmul(J_ROOT, R.T)
    smpl_param_new['trans'] = smpl_param_new['trans'] + T - trans_bias

    return smpl_param_new

def add_RT_to_smpl_param_tensor_batch(
    dict_smpl_param:Dict[str, torch.Tensor], 
    RT_batch:Union[np.ndarray, torch.Tensor],
    J_root:Union[np.ndarray, torch.Tensor, None]=None,
    smpl_model:Optional[SMPL_Full]=None
) -> (Dict[str, torch.Tensor]):
    
    assert isinstance(dict_smpl_param, dict), type(dict_smpl_param)
    assert set(dict_smpl_param.keys()) >= set(['poses', 'shapes', 'trans']), dict_smpl_param.keys()
    if isinstance(RT_batch, np.ndarray):
        RT_batch = torch.from_numpy(RT_batch)
    B = RT_batch.shape[0]
    assert RT_batch.shape == (B, 4, 4), RT_batch.shape
    R_batch = RT_batch[:, 0:3, 0:3]
    T_batch = RT_batch[:, 0:3, 3]

    if J_root is not None:
        if isinstance(J_root, np.ndarray):
            J_root = torch.from_numpy(J_root)
        assert J_root.shape == (B, 3), J_root.shape
    else:
        assert smpl_model is not None, 'if J_root is None, smpl_model should be provided'
        ____, joints_54 = forward_smpl_full_tensor_batch(
            smpl_model=smpl_model,
            **dict_smpl_param,
            ret_joints_54=True
        )
        assert joints_54.shape == (B, 54, 3), joints_54.shape
        J_root = joints_54[:, 0, :] #[B, 3]
        assert J_root.shape == (B, 3), J_root.shape

    smpl_param_new = {k: v.clone() if isinstance(v, torch.Tensor) else copy.copy(v) for k, v in dict_smpl_param.items()}

    if dict_smpl_param['poses'].shape == (B, 24, 3, 3):
        smpl_param_new['poses'][:, 0, :, :] = torch.matmul(R_batch, dict_smpl_param['poses'][:, 0, :, :])
    elif dict_smpl_param['poses'].shape == (B, 24*3*3):
        orient_rotmat = dict_smpl_param['poses'][:, 0:9].reshape(B, 3, 3)
        orient_rotmat_new = torch.matmul(R_batch, orient_rotmat)
        smpl_param_new['poses'][:, 0:9] = orient_rotmat_new.reshape(B, 9)
    elif dict_smpl_param['poses'].shape == (B, 72):
        poses_new_rotmat = aa_to_rotmat(dict_smpl_param['poses'].reshape(B*24, 3))
        poses_new_rotmat = poses_new_rotmat.reshape(B, 24, 3, 3)
        poses_new_rotmat[:, 0, :, :] = torch.matmul(R_batch, poses_new_rotmat[:, 0, :, :])
        poses_new_aa = rotmat_to_aa(poses_new_rotmat.reshape(B*24, 3, 3))
        smpl_param_new['poses'] = poses_new_aa.reshape(B, 72)
    else:
        raise ValueError('unknown poses shape: {}'.format(dict_smpl_param['poses'].shape))
    
    trans_bias = J_root - torch.matmul(J_root.reshape(B, 1, 3), R_batch.transpose(1, 2)).reshape(B, 3)
    smpl_param_new['trans'] = dict_smpl_param['trans'] + T_batch - trans_bias
    return smpl_param_new
    
def add_RT_to_smpl_param_with_known_root(
    dict_smpl_param:Dict[str, Any], 
    J_ROOT:np.ndarray,
    RT:np.ndarray
) -> (Dict[str, Any]):
    '''
    RT is [R|T] with shape (4, 4)
    '''
    assert isinstance(dict_smpl_param, dict), type(dict_smpl_param)
    assert set(dict_smpl_param.keys()) >= set(['poses', 'shapes', 'trans']), dict_smpl_param.keys()
    
    assert RT.shape == (4, 4), RT.shape
    R = RT[0:3, 0:3]
    T = RT[0:3, 3]
    
    smpl_param_new = {
        k: v.copy() if isinstance(v, np.ndarray) else copy.copy(v)
        for k, v in dict_smpl_param.items()
    }
    smpl_param_new['poses'][0:3] = update_root_pose_0_via_R(smpl_param_new['poses'][0:3], R)
    trans_bias = J_ROOT - np.matmul(J_ROOT, R.T)
    smpl_param_new['trans'] = smpl_param_new['trans'] + T - trans_bias

    return smpl_param_new

