from typing import List, Tuple, Dict, Union, Optional
from math import cos,sin,pi,acos
import numpy as np



import torch



def get_std_camera_K_by_fov(fov_in_degree:float, img_size:Tuple[int, int]) -> np.ndarray:
    '''
    @param fov: the field of view of the camera
    @param img_size: the image size of the camera
    '''
    W, H = img_size
    cx = W / 2
    cy = H / 2
    #tan(fov/2) = (max(W, H) / 2) / f
    #f = (max(W, H) / 2) / tan(fov/2)
    
    f = (max(W, H) / 2) / np.tan(np.radians(fov_in_degree) / 2)
    
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    return K

def get_fov_by_camera_K(K:np.ndarray, img_size_wh:Tuple[int, int]) -> float:
    '''
    @param K: the camera intrinsic matrix, shape (3, 3)
    @param img_size: the image size of the camera (W, H)
    @return: the field of view in degree
    '''
    assert K.shape == (3, 3), K.shape
    W, H = img_size_wh
    fov_x = 2 * np.arctan(W / (2 * K[0, 0])) * 180 / np.pi
    return fov_x

def get_rotate_space(space_src:np.ndarray, space_dst:np.ndarray) -> np.ndarray:
    '''
    space_src [3, 3] each line is an unit vector
    space_dist [3, 3] each line is an unit vector
    '''

    '''
    space_src is written as S here and
    space_dist is written as D here.
    Then we have:
    S * R^T = D
    R^T = S^(-1) * D
    R = (S^(-1) * D)^T
    '''
    R = np.matmul(np.linalg.inv(space_src), space_dst).T
    return R

def flip_image_xy_by_edit_K(K:np.ndarray, image_size:Tuple[int, int]) -> np.ndarray:
    '''
    image_size is (W, H)
    '''
    assert K.shape == (3, 3), K.shape
    W, H = image_size
    K_new = K.copy()
    K_new[0, 0] = - K[1, 1]
    K_new[1, 1] = - K[0, 0]
    K_new[0, 2] = W - K[0, 2]
    K_new[1, 2] = H - K[1, 2]
    return K_new

def flip_image_xy_by_edit_Ks(Ks:np.ndarray, image_size:Tuple[int, int]) -> np.ndarray:
    '''
    Ks is (N, 3, 3), where N is the number of cameras
    image_size is (W, H)
    '''
    N = Ks.shape[0]
    assert Ks.shape == (N, 3, 3), Ks.shape
    W, H = image_size
    Ks_new = Ks.copy()
    Ks_new[:, 0, 0] = - Ks[:, 1, 1]
    Ks_new[:, 1, 1] = - Ks[:, 0, 0]
    Ks_new[:, 0, 2] = W - Ks[:, 0, 2]
    Ks_new[:, 1, 2] = H - Ks[:, 1, 2]
    return Ks_new


def get_rotate_X(theta:float):
    rotate = np.array(
        [[1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]]
    )
    return rotate

def get_rotate_X_tensor_batch(theta:torch.Tensor):
    B = theta.shape[0]
    assert theta.shape == (B,), theta.shape
    rotate = torch.zeros((B, 3, 3), device=theta.device, dtype=theta.dtype)
    rotate[:, 0, 0] = 1
    rotate[:, 1, 1] = torch.cos(theta)
    rotate[:, 1, 2] = -torch.sin(theta)
    rotate[:, 2, 1] = torch.sin(theta)
    rotate[:, 2, 2] = torch.cos(theta)
    return rotate

def get_rotate_Y(theta:float):
    rotate = np.array(
        [[np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]]
    )
    return rotate

def get_rotate_Z(theta:float):
    rotate = np.array(
        [[np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]]
    )
    return rotate

def get_R_by_euler(x_rotate:float, y_rotate:float, z_rotate:float):
    """
    :param x_rotate: in radians
    :param y_rotate: in radians
    :param z_rotate: in radians
    :return: R [3, 3]
    """
    R = get_rotate_Y(y_rotate) @ get_rotate_X(x_rotate) @ get_rotate_Z(z_rotate)
    return R

def get_R_by_euler_XYZ(x_rotate:float, y_rotate:float, z_rotate:float):
    """
    :param x_rotate: in radians
    :param y_rotate: in radians
    :param z_rotate: in radians
    :return: R [3, 3]
    """
    R = get_rotate_Z(z_rotate) @ get_rotate_Y(y_rotate) @ get_rotate_X(x_rotate)
    return R

def get_euler_by_R(R:np.ndarray):
    """
    :param R: [3, 3]
    :return: x_rotate, y_rotate, z_rotate in radians
    """
    x_rotate = np.arctan2(R[2, 1], R[2, 2])
    y_rotate = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    z_rotate = np.arctan2(R[1, 0], R[0, 0])
    R_EULER = np.array([x_rotate, y_rotate, z_rotate])
    return R_EULER

def get_camera_std_euler_by_R(R:np.ndarray):
    """
    :param R: [3, 3]
    :return: x_rotate, y_rotate, z_rotate in radians
    can be combine by
        R = get_rotate_Y(y_rotate) @ get_rotate_X(x_rotate) @ get_rotate_Z(z_rotate) 
        where initially,
        the camera is looking at the z-axis
        and the y-axis is up
        and the x-axis is right
    """


    
    #initially, the camera is looking at the z-axis
    points_XYZO_camera_init = np.array([
        [1, 0, 0], #x point of local camera coordinate
        [0, 1, 0], #y point of local camera coordinate
        [0, 0, 1], #z point of local camera coordinate
        [0, 0, 0], #o point of local camera coordinate
    ])

    points_XYZO_camera_current = points_XYZO_camera_init @ R.T
    #project Z to XoZ
    point_Z_on_XoZ = points_XYZO_camera_current[2, 0:3]
    point_Z_on_XoZ[1] = 0

    #euler Y is the angle beteen O-Z and O-Z_on_XoZ
    euler_Y = np.arctan2(point_Z_on_XoZ[0], point_Z_on_XoZ[2])

    points_XYZO_camera_current_without_euler_Y = points_XYZO_camera_init @ R.T @ get_rotate_Y(-euler_Y).T
    points_Z_without_euler_Y = points_XYZO_camera_current_without_euler_Y[2, 0:3]
    assert abs(points_Z_without_euler_Y[0] - 0) < 1e-7, points_Z_without_euler_Y[0]

    #euler X is the angle beteen O-Z and O-Z_on_XoZ
    euler_X = - np.arctan(points_Z_without_euler_Y[1]/points_Z_without_euler_Y[2])

    points_XYZO_camera_current_without_euler_YX = points_XYZO_camera_init @ R.T @ get_rotate_Y(-euler_Y).T @ get_rotate_X(-euler_X).T
    assert np.allclose(points_XYZO_camera_current_without_euler_YX[2, 0:1], np.zeros(2)), points_XYZO_camera_current_without_euler_YX[2, 0:1]

    #euler Z is the angle beteen O-X and O-X_on_XoY
    point_X_camera_without_euler_XY = points_XYZO_camera_current_without_euler_YX[0, 0:3]
    euler_Z = np.arctan(point_X_camera_without_euler_XY[1]/point_X_camera_without_euler_XY[0]) + np.pi

    euler_Z_try_1 = euler_Z
    euler_Z_try_2 = euler_Z + np.pi

    def get_points_XYZO_camera_current_without_euler_YXZ(_euler_Z):
        return points_XYZO_camera_init @ R.T @ get_rotate_Y(-euler_Y).T @ get_rotate_X(-euler_X).T @ get_rotate_Z(-_euler_Z).T
    
    if np.allclose(get_points_XYZO_camera_current_without_euler_YXZ(euler_Z_try_1)[0:3, 0:3], np.identity(3), atol=1e-4, rtol=1e-4):
        euler_Z = euler_Z_try_1
    elif np.allclose(get_points_XYZO_camera_current_without_euler_YXZ(euler_Z_try_2)[0:3, 0:3], np.identity(3), atol=1e-4, rtol=1e-4):
        euler_Z = euler_Z_try_2
    else:
        raise Exception("euler_Z not found")
    
    R_reconstruct = get_rotate_Y(euler_Y) @ get_rotate_X(euler_X) @ get_rotate_Z(euler_Z)

    assert np.allclose(R_reconstruct, R, atol=1e-4, rtol=1e-4), (R_reconstruct,  R)

    return euler_X, euler_Y, euler_Z

def get_upright_orient_euler_by_R(R:np.ndarray):
    '''
    originally, the orientation is look at y-negative (0, -1, 0),
    with the default roll-label-vector is (1, 0, 0)

    for any rotation R, we want to get the euler angles (roll, pitch, yaw)
    first, we apply a roll rotate around y-negative axis,
    which means the orientation is still looking at (0, -1, 0), 
    and the roll-label-vector is rotated to (cos(roll), 0, sin(roll))
    then, we apply a pitch rotate the z-axis, 
    and then a yaw rotate around x-axis

    this function compute the (roll, pitch, yaw) from R
    '''
    assert R.shape == (3, 3), R.shape
    

    R_from_default_smpl_orient_to_current = R
    R_from_default_upright_orient_to_current = R @ get_rotate_X(np.pi) #left multiply
    
    del R

    points_XYZO_orient_init = np.array([
        [1, 0, 0], #roll-label-vector
        [0, -1, 0], #orientation vector
        [0, 0, 1], #z point of local orient coordinate
        [0, 0, 0], #o point of local orient coordinate
    ])

    points_XYZO_orient_current = points_XYZO_orient_init @ R_from_default_upright_orient_to_current.T
    assert points_XYZO_orient_current.shape == (4, 3), points_XYZO_orient_current.shape

    #we first solve the last euler angle: yaw (around x-axis)
    #since only yaw and pitch affect the orientation vector
    #we can solve yaw only by the orientation vector
    #project the orientation vector to y-o-z plane
    point_orient_on_YoZ = points_XYZO_orient_current[1, 0:3].copy()
    point_orient_on_YoZ[0] = 0
    point_orient_on_YoZ = point_orient_on_YoZ / np.linalg.norm(point_orient_on_YoZ)
    '''
    after project to y-o-z plane, it is like:
        |
    ----.---->z
        |
        v y
    and the anti-clockwise is positive direction
    
    if z <= 0, then cos(yaw)  = -1.0 * point_orient_on_YoZ.y, yaw = arccos(- point_orient_on_YoZ.y)
    if z >  0, then cos(-yaw) = -1.0 * point_orient_on_YoZ.y, yaw = -1.0 * arccos(- point_orient_on_YoZ.y)
    '''
    if point_orient_on_YoZ[2] <= 0:
        #yaw in (0, pi)
        yaw = np.arccos(-point_orient_on_YoZ[1])
    else:
        #yaw in (-pi, 0)
        yaw = - np.arccos(-point_orient_on_YoZ[1])


    #remove the yaw rotate
    #right multiply by R.T
    points_XYZO_orient_current_without_yaw = points_XYZO_orient_init @ R_from_default_upright_orient_to_current.T @ get_rotate_X(-yaw).T
    #thus we assert the orientation vector is in the x-o-y plane
    # print(point_orient_on_YoZ)
    # print(points_XYZO_orient_current_without_yaw)
    assert abs(points_XYZO_orient_current_without_yaw[1, 2]) < 1e-7, points_XYZO_orient_current_without_yaw[1, 2]
    
    #then we solve the pitch (around z-axis)
    #project the orientation vector to x-o-y plane
    point_orient_on_XoY = points_XYZO_orient_current_without_yaw[1, 0:3].copy()
    point_orient_on_XoY[2] = 0
    point_orient_on_XoY /= np.linalg.norm(point_orient_on_XoY)
    '''
    after project to x-o-y plane, it is like:
        |
    ----.----> x
        |
        v y
    where the clockwise is positive direction
    if x >= 0, then cos(pitch) = -1.0 * point_orient_on_XoY.y, pitch = arccos(- point_orient_on_XoY.y)
    if x <  0, then cos(-pitch)= -1.0 * point_orient_on_XoY.y, pitch = -1.0 * arccos(- point_orient_on_XoY.y)
    '''
    if point_orient_on_XoY[0] >= 0:
        #pitch in [0, pi]
        pitch = np.arccos(- point_orient_on_XoY[1])
    else:
        #pitch in [-pi, 0]
        pitch = - np.arccos(- point_orient_on_XoY[1])
    
    #remove the pitch rotate0
    #right multiply by R.T
    points_XYZO_orient_current_without_yaw_pitch = points_XYZO_orient_init @ R_from_default_upright_orient_to_current.T @ \
        get_rotate_X(-yaw).T @ get_rotate_Z(-pitch).T
    #thus we assert the orientation vector is in the y-negative direction
    assert np.allclose(points_XYZO_orient_current_without_yaw_pitch[1, 0:3], np.array([0, -1, 0]), atol=1e-4), (
        points_XYZO_orient_current_without_yaw_pitch[1, 0:3],
        points_XYZO_orient_current_without_yaw_pitch[1, 0:3] - np.array([0, -1, 0])
    )

    #finally, we solve the roll (around y-negative axis)
    #we use the roll-label-vector to solve the roll angle
    point_roll_label_vector_after_remove_yaw_pitch = points_XYZO_orient_current_without_yaw_pitch[0, 0:3]
    point_roll_label_vector_after_remove_yaw_pitch[1] = 0
    point_roll_label_vector_after_remove_yaw_pitch /= np.linalg.norm(point_roll_label_vector_after_remove_yaw_pitch)
    '''
    after remove yaw and pitch, the roll-label-vector is like:
        ^z
        |
        |
    ----.----> x
        |
    where clockwise is positive direction
    if z <= 0, then cos(roll)  = P.x, roll = arccos(P.x)
    if z >  0, then cos(-roll) = P.x, roll = -1.0 * arccos(P.x)
    '''
    if point_roll_label_vector_after_remove_yaw_pitch[2] <= 0:
        #roll in [0, pi]
        roll = np.arccos(point_roll_label_vector_after_remove_yaw_pitch[0])
    else:
        #roll in [-pi, 0]
        roll = - np.arccos(point_roll_label_vector_after_remove_yaw_pitch[0])

    #check the result
    R_reconstruct = get_rotate_X(yaw) @ get_rotate_Z(pitch) @ get_rotate_Y(roll) @ get_rotate_X(np.pi)
    assert np.allclose(R_reconstruct, R_from_default_smpl_orient_to_current, atol=1e-4, rtol=1e-4), (R_reconstruct,  R_from_default_smpl_orient_to_current)
    # print(np.degrees(roll), np.degrees(pitch), np.degrees(yaw))
    # exit()

    direction_rot_z = pitch
    direction_rot_y = yaw

    return roll, direction_rot_z, direction_rot_y

def convert_direction_and_roll_to_5d(roll:float, direction_rot_z:float, direction_rot_y:float) -> np.ndarray:
    '''
    direction_rot_z: in radians
    direction_rot_y: in radians
    roll: in radians
    return: 5d vector np.ndarray with shape (5,)
    '''
    rot_5d = np.zeros((5,))
    R_reconstruct_without_roll = get_rotate_X(direction_rot_y) @ get_rotate_Z(direction_rot_z)
    orientation_init = np.array([0, -1, 0])
    orientation_current = R_reconstruct_without_roll @ orientation_init
    rot_5d[0:3] = orientation_current
    rot_5d[3] = np.cos(roll)
    rot_5d[4] = np.sin(roll)
    return rot_5d


def get_rotate_R(theta:float,u:np.ndarray):
    assert u.shape == (3,), u
    (a,b,c)=u
    rotate = np.array(
        [[(b*b+c*c)*np.cos(theta)+a*a, a*b*(1-np.cos(theta))-c*np.sin(theta), a*c*(1-np.cos(theta))+b*np.sin(theta),0],
        [a*b*(1-np.cos(theta))+c*np.sin(theta), b*b+(1-b*b)*np.cos(theta), b*c*(1-np.cos(theta)-a*np.sin(theta)),0],
        [a*c*(1-np.cos(theta))-b*np.sin(theta), b*c+(1-np.cos(theta))+a*np.sin(theta), c*c+(1-c*c)*np.cos(theta),0],
        [0, 0, 0, 1]]
    )
    return rotate

def rotate_RT(RT:np.ndarray, rotate:np.ndarray):
    assert RT.shape == (4, 4), RT
    assert rotate.shape == (3, 3), rotate
    
    new_RT = np.zeros((4, 4))
    new_RT[:, 0:4] = RT[:, 0:4]

    ori_R = RT[0:3, 0:3]
    new_R = np.matmul(rotate, ori_R)
    
    new_RT[0:3, 0:3] = new_R
    return new_RT

def trans_RT(RT:np.ndarray, trans:Union[np.ndarray, List[float]]):
    assert RT.shape == (4, 4)
    if type(trans) == np.ndarray:
        assert trans.shape == (3,), trans.shape
    elif type(trans) == list:
        assert len(trans) == 3
    else:
        raise(NotImplementedError(type(trans)))

    new_RT = RT.copy()
    for i in range(3):
        new_RT[i, 3] += trans[i]
    return new_RT

def combine_RT_44(R:Optional[np.ndarray]=None, T:Optional[np.ndarray]=None):
    assert not (R is None and T is None), (R, T)

    if R is None:
        R = np.identity(3)
    if T is None:
        T = np.zeros((3))

    assert R.shape == (3, 3), R.shape
    assert T.shape == (3,), T.shape

    RT_44 = np.identity(4)

    RT_44[0:3, 0:3] = R
    RT_44[0:3, 3] = T

    return RT_44

def decouple_RTS_from_44(RTS_44:np.ndarray) -> (Tuple[np.ndarray, np.ndarray, float]):
    '''
    assert RTS_44 is like
    a00 a01 a02 t0
    a10 a11 a12 t1
    a20 a21 a22 t2
    0   0   0   1
    '''
    assert RTS_44.shape == (4, 4)
    assert (RTS_44[3, 3] - 1) < 1e-7, RTS_44[3, 3]
    assert np.allclose(RTS_44[3, 0:3], np.zeros((3, 3))), RTS_44[3, 0:3]
    

    T = RTS_44[0:3, 3]
    RS = RTS_44[0:3, 0:3]

    scale_xyz = np.linalg.norm(RS, ord=2, axis=0) #get norm of every column
    scale = np.average(scale_xyz)
    #assert scale xyz are the same
    assert np.allclose(scale_xyz, scale), (scale_xyz, scale)

    R = RS / scale

    return R, T, scale

def make_scale_matrix_44(scale:float):
    assert isinstance(scale, float), type(scale)
    RT_44 = np.identity(4)
    RT_44[0, 0] = scale
    RT_44[1, 1] = scale
    RT_44[2, 2] = scale
    return RT_44


def make_pose_from_looking_at(position:np.ndarray, waitou_zuo:float, taitou:float, zhuantou_zuo:float) -> (np.ndarray):
    base_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0],
    ])

    camera_pose = trans_RT(base_pose, trans=position)

    camera_pose = rotate_RT(camera_pose, get_rotate_Z(waitou_zuo))
    camera_pose = rotate_RT(camera_pose, get_rotate_X(taitou))
    camera_pose = rotate_RT(camera_pose, get_rotate_Y(zhuantou_zuo))

    return camera_pose

def convert_camera_pose_from_camRT_to_renderer(camera_pose:np.ndarray) -> (np.ndarray):
    '''
    shoule equal to for all the objects
        Y *= -1 and Z *= -1
        which means rotate around X by pi
    '''
    
    RT_objects = np.linalg.inv(camera_pose)
    RT_objects_new = np.matmul(RT_objects, combine_RT_44(R=get_rotate_Y(np.pi)))  
    camera_pose_new = np.linalg.inv(RT_objects_new @ combine_RT_44(R=get_rotate_Z(np.pi)))
    return camera_pose_new

def get_RT_from_renderer_camera_pose(camera_pose:np.ndarray) -> (np.ndarray):
    '''
    @param camera_pose: [4, 4], used in blender
    use the equation in get_renderer_camera_pose_from_RT()
    #^ RT = Ry(180) @ (P @ Rz(180))^(-1)
    '''
    assert camera_pose.shape == (4, 4), camera_pose.shape
    
    RT = combine_RT_44(get_rotate_Y(np.pi)) @ np.linalg.inv(camera_pose @ combine_RT_44(get_rotate_Z(np.pi)))
    return RT

def get_renderer_camera_pose_from_RT(RT:np.ndarray) -> (np.ndarray):
    '''
    tested for pyrender and blender

    set P = camera_pose
    #^ P = (Ry(180) @ RT)^(-1) @ Rz(180)
    #& which means
    after RT, the objects should be added a rotation around Y by 180
    afterall, the camera should be rotated around Z by 180 (u*=-1, v*=-1)

    #& BTW we have
    P @ Rz(180) = (Ry(180) @ RT)^(-1)
    (P @ Rz(180))^(-1) = Ry(180) @ RT
    #^ RT = Ry(180) @ (P @ Rz(180))^(-1)
    '''
    assert RT.shape == (4, 4), RT.shape
    camera_pose = np.linalg.inv(
        combine_RT_44(get_rotate_Y(np.pi)) @ RT
    ) @ combine_RT_44(get_rotate_Z(np.pi))
    return camera_pose

def calVectorFrom2Planes(Params_1, Params_2):
    '''计算空间两平面交线的方向向量
    输入:两空间平面方程的参数Params_1(a_1,b_1,c_1,d_1),Params_2(a_2,b_2,c_2,d_2);
    输出:空间两平面交线的方向向量s
    DEMO:   import geomeas as gm
            import numpy as np
            Param_1 = np.array([0, 0, -4361.9337, 362040.4971])
            Param_2 = np.array([-2180.41, 6939.63, 2165.499, -283785.5822])
            print(gm.Vector().calLineVictorFrom2Planes(Param_1, Param_2))
    参考资料:邱维声. 解析几何（第三版）[M]. 北京:北京大学出版社，2015.30-36,48-51.
    '''
    a_1 = np.array(Params_1[0])
    b_1 = np.array(Params_1[1])
    c_1 = np.array(Params_1[2])
    a_2 = np.array(Params_2[0])
    b_2 = np.array(Params_2[1])
    c_2 = np.array(Params_2[2])
    m = b_1 * c_2 - b_2 * c_1
    n = c_1 * a_2 - a_1 * c_2
    p = a_1 * b_2 - a_2 * b_1
    return m, n, p

def get_weak_perspective_camera_setting_by_objects_and_ground(
        verts:np.ndarray,
        ground:np.ndarray,
        FoV_in_degree:float=0.5,
        rough_ratio_on_image:float=0.75,
        image_size:Tuple[int, int]=(2880, 2880)
    ):
    '''
    @return: camera_K
    @return: RT, the "object to camera" matrix
    '''

    num_verts = verts.shape[0]
    assert verts.shape == (num_verts, 3), verts.shape
    assert ground.shape == (4,), ground.shape
    assert FoV_in_degree < 45, 'you say you want to use weak perspective camera, but the FoV is too large'

    #& get the center of the verts
    center = np.mean(verts, axis=0)
    #& get the height of the verts, 
    #   should project to the y-axis of the ground vector first
    #   and then compute the max - min
    verts_y = np.dot(verts, ground[0:3])
    height = np.max(verts_y) - np.min(verts_y)

    #& get the distance from the center to the camera
    height_extend = height / rough_ratio_on_image
    distance = height_extend / np.tan(FoV_in_degree / 2 * np.pi / 180)

    vector_ground_normal = ground[0:3] / np.linalg.norm(ground[0:3])

    #& get the basis vectors of the camera space, in the world space
    # where the x-axis is parallel to the ground
    #   the x-axis vector should be like [a, b, 0]
    #   the z-axis is also parallel to the ground
    #   the z-axis is pointing to the center of the verts
    #   and the y-axis is vertical to the ground
    vector_camera_x = np.array([ground[1], -ground[0], 0])
    if vector_camera_x[1] > 0:
        vector_camera_x = vector_camera_x * -1
    vector_camera_x = vector_camera_x / np.linalg.norm(vector_camera_x)
    vector_camera_z = np.cross(vector_camera_x, vector_ground_normal)
    vector_camera_y = vector_ground_normal

    R_camera_pose = get_rotate_space(
        space_src=np.identity(3),
        space_dst=np.concatenate([vector_camera_x, vector_camera_y, vector_camera_z]).reshape(3, 3)
    )
    T_camera_pose = center + vector_camera_z * distance

    #& get the camera_K
    W, H = image_size
    F = W / 2 / np.tan(FoV_in_degree / 2 * np.pi / 180)
    camera_K = np.identity(3)
    camera_K[0, 0] = F
    camera_K[1, 1] = F
    camera_K[0, 2] = W / 2
    camera_K[1, 2] = H / 2

    camera_pose = combine_RT_44(R=R_camera_pose, T=T_camera_pose)

    return camera_K, camera_pose





    


def position(point,angle,pitch,distance):
    """计算相机位置"""
    point=np.array(point)
    if angle<0:
        while angle<0:
            angle=angle+360
    else:
        angle=angle%360
         
    if pitch<0:
        while pitch<0:
            pitch=pitch+360
    else:
        pitch=pitch%360

    distance2=cos(pitch*pi/180)*distance
            
    if angle%90==0:
        if angle==0:
            (x,y)=(0,distance2)
        elif angle==90:
            (x,y)=(distance2,0)
        elif angle==180:
            (x,y)=(0,-distance2)
        else:
            (x,y)=(-distance2,0)
    else:
        if angle<=45:
            x=sin(angle*pi/180)*distance2
            y=cos(angle*pi/180)*distance2
        elif angle<90:
            angle=90-angle
            x=cos(angle*pi/180)*distance2
            y=sin(angle*pi/180)*distance2
        elif angle<=135:
            angle=angle-90
            x=cos(angle*pi/180)*distance2
            y=-sin(angle*pi/180)*distance2
        elif angle<180:
            angle=180-angle
            x=sin(angle*pi/180)*distance2
            y=-cos(angle*pi/180)*distance2
        elif angle<=225:
            angle=angle-180
            x=-sin(angle*pi/180)*distance2
            y=-cos(angle*pi/180)*distance2
        elif angle<270:
            angle=270-angle
            x=-cos(angle*pi/180)*distance2
            y=-sin(angle*pi/180)*distance2
        elif angle<315:
            angle=angle-270
            x=-cos(angle*pi/180)*distance2
            y=sin(angle*pi/180)*distance2
        else:
            angle=360-angle
            x=-sin(angle*pi/180)*distance2
            y=cos(angle*pi/180)*distance2
    
    if pitch%90==0:
        if pitch==0 or pitch==90:
            z=distance
        else:
            z=0
    else:
        if pitch<=45:
            z=sin(pitch*pi/180)*distance
        elif pitch<90:
            pitch=90-pitch
            z=cos(pitch*pi/180)*distance
        elif pitch<=135:
            pitch=pitch-90
            z=cos(pitch*pi/180)*distance
        elif pitch<180:
            pitch=180-pitch
            z=sin(pitch*pi/180)*distance
        elif pitch<=225:
            pitch=pitch-180
            z=-sin(pitch*pi/180)*distance
        elif pitch<270:
            pitch=270-pitch
            z=-cos(pitch*pi/180)*distance
        elif pitch<315:
            pitch=pitch-270
            z=-cos(pitch*pi/180)*distance
        else:
            pitch=360-pitch
            z=-sin(pitch*pi/180)*distance
    
    return np.array([x,y,z])

def adjust_camera_and_image_size(camera_K:np.ndarray, W:int, H:int):

    max_side_pixel = max(W, H)
    image_size = (max_side_pixel, max_side_pixel)
    if max(image_size) > 10000:
        #the image is too large (may memory error)
        #we adjust camera_K and image_size
        SCALE_DOWN = 2
        camera_K_adjusted = camera_K.copy()
        camera_K_adjusted[0, 0] = camera_K[0, 0] / SCALE_DOWN
        camera_K_adjusted[1, 1] = camera_K[1, 1] / SCALE_DOWN
        camera_K_adjusted[0, 2] = max_side_pixel / SCALE_DOWN / 2
        camera_K_adjusted[1, 2] = max_side_pixel / SCALE_DOWN / 2
        image_size = (int(max_side_pixel / SCALE_DOWN), int(max_side_pixel / SCALE_DOWN))
    else:
        camera_K_adjusted = camera_K.copy()
        camera_K_adjusted[0, 2] = max_side_pixel / 2
        camera_K_adjusted[1, 2] = max_side_pixel / 2
        image_size= (max_side_pixel, max_side_pixel)
    assert max(image_size) <= 10000, image_size
        
    assert image_size[0] * image_size[1] <= 1.5 * 10000 * 10000, image_size

    return camera_K_adjusted, image_size
