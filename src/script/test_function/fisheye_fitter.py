import torch
import torch.nn as nn
import logging
import os
import pickle
import config_script
import lib_pano.logging_config
import lib_pano.local_setting as ls
import cv2 as cv
import numpy as np
from lib_pano.human3d.smpl_wrapper import SMPLFittingModel
from lib_pano.human3d.priors import prior_shape, prior_pose_l2
from lib_pano.camera.camera_fisheye import omni_project, omni_radtan_project_to_equirect
from lib_pano.human3d.losses import get_reprojection_loss
from lib_pano.human3d.skeletion_constant import INDEX_CONVERT_COCO_TO_SMPLX, SMPL_24_JOINTS
from lib_pano.human3d.keypoints_detector import CombinedDetector, draw_keypoints, save_mesh_overlay
from lib_pano.io.params_read import load_single_K_and_distortion
from lib_pano.logging_config import configure_logging
from lib_pano.human.vposer import load_vposer
from lib_pano.human3d.visual_render import Equirect_Renderer

logger = logging.getLogger(__name__)

RENDER_UPPER_BODY_ONLY = True
USE_VPOSER_PRIOR = True
W_VPOSER = 0
VPOSER_DIR = os.path.join(ls.PROJECT_ROOT, 'tools', 'V02_05')


def filter_upper_body_faces(faces, lbs_weights):
    upper_joint_ids = np.array([3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    
    vert_mask = np.isin(np.argmax(lbs_weights, axis=1), upper_joint_ids)
    face_mask = vert_mask[faces[:, 0]] & vert_mask[faces[:, 1]] & vert_mask[faces[:, 2]]
    filtered = faces[face_mask]

    if filtered.shape[0] < 100:
        logger.warning("Upper-body filter kept too few faces, falling back.")
        return faces
    return filtered

class FisheyeFitter(nn.Module):
    def __init__(self, model_folder, device='cuda', vposer_enc=None, w_vposer=0.0):
        super(FisheyeFitter, self).__init__()
        self.device = device
        self.body_model = SMPLFittingModel(model_folder, device=device)
        self.vposer_enc = vposer_enc
        self.w_vposer = w_vposer
        
        for param in self.body_model.parameters():
            param.requires_grad = False

        if self.vposer_enc is not None:
            self.vposer_enc.to(self.device)
            self.vposer_enc.eval()
            for p in self.vposer_enc.parameters():
                p.requires_grad = False

    def save_smpl_result_as_pkl(self, final_params, output_path, image_path=None):
        SMPLFittingModel.save_smpl_result_as_pkl(final_params, output_path, image_path=image_path)
    
    def load_smpl_result_from_pkl(self, pkl_path):
        return SMPLFittingModel.load_smpl_result_from_pkl(pkl_path)
    

    def load_vertices_faces_and_joints(self, smpl_result):
        with torch.no_grad():
            smpl_out = self.body_model(
                betas=smpl_result['betas'].to(self.device),
                global_orient=smpl_result['global_orient'].to(self.device),
                body_pose=smpl_result['body_pose'].to(self.device),
                transl=smpl_result['transl'].to(self.device),
            )
            vertices_np = smpl_out.vertices.cpu().numpy()  # (1, N, 3)
            joints_np = smpl_out.joints.cpu().numpy()      # (1, J, 3)
            faces_np = np.array(self.body_model.body_model.faces)
        return vertices_np, faces_np, joints_np
    
    def fit_single_frame(self, target_2d, confidences, camera_params, out_size, num_iters=800, lr=0.01):
        '''
        target_2d: (1, N, 2)
        confidences: (1, N)
        camera_params: dict
        out_size: (int, int) - width and height of the output image
        '''
        
        batch_size = target_2d.shape[0]
        opt_betas = nn.Parameter(torch.zeros([batch_size, 10], device=self.device))
        opt_global_orient = nn.Parameter(torch.zeros([batch_size, 3], device=self.device))
        opt_body_pose = nn.Parameter(torch.zeros([batch_size, 63], device=self.device))
        opt_transl = nn.Parameter(torch.zeros([batch_size, 3], device=self.device))
        
        optimizer = torch.optim.Adam([opt_betas, opt_global_orient, opt_body_pose, opt_transl], lr=lr)
        logger.info("Starting optimization...")
        for i in range(num_iters):
            optimizer.zero_grad()
            
            smpl_out = self.body_model(
                betas = opt_betas,
                global_orient = opt_global_orient,
                body_pose = opt_body_pose,
                transl = opt_transl,   
            )
            
            joints_3d = smpl_out.joints
            # TODO: select the needed joints and add the joints Mapper 
            joints_3d_mapped = joints_3d[:, INDEX_CONVERT_COCO_TO_SMPLX['body'], :]
            
            K = camera_params['K']
            D = camera_params['D']
            xi = camera_params['xi']
            # projected_2d  = omni_project(joints_3d_mapped, K, D, xi, R=None)
            projected_2d = omni_project(joints_3d_mapped, K, D, xi, R=None)
            
            loss_reproj = get_reprojection_loss(projected_2d, target_2d, confidences, 'l1')
            loss_shape = prior_shape(opt_betas)
            loss_pose = prior_pose_l2(opt_global_orient) + prior_pose_l2(opt_body_pose)

            if self.vposer_enc is not None and self.w_vposer > 0:
                body_pose_aa = opt_body_pose.view(batch_size, 21, 3)
                q_z = self.vposer_enc(body_pose_aa)
                mu = q_z.mean
                sigma = q_z.scale.clamp_min(1e-6)
                sigma2 = sigma * sigma
                loss_vposer = 0.5 * (mu * mu + sigma2 - torch.log(sigma2) - 1.0)
                loss_vposer = loss_vposer.mean()
            else:
                loss_vposer = torch.zeros([], device=self.device)

            total_loss = loss_reproj + loss_shape + loss_pose + self.w_vposer * loss_vposer
            
            total_loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                logger.info(f"Iter {i:03d} | Total Loss: {total_loss.item():.4f} | "
                            f"Reproj: {loss_reproj.item():.4f} | "
                            f"Shape: {loss_shape.item():.4f} | "
                            f"Pose: {loss_pose.item():.4f} | "
                            f"VPoser: {loss_vposer.item():.4f}")

        with torch.no_grad():
            smpl_out = self.body_model(
                betas=opt_betas,
                global_orient=opt_global_orient,
                body_pose=opt_body_pose,
                transl=opt_transl,
            )
            joints_3d = smpl_out.joints
            joints_3d_mapped = joints_3d[:, INDEX_CONVERT_COCO_TO_SMPLX['body'], :]
            projected_2d = omni_project(joints_3d_mapped, K, D, xi, R=None)

        logger.info("Optimization finished.")
        return {
            'betas': opt_betas.detach(),
            'global_orient': opt_global_orient.detach(),
            'body_pose': opt_body_pose.detach(),
            'transl': opt_transl.detach(),
            'projected_2d': projected_2d.detach(),
        }

def main():
    configure_logging(level='INFO')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_folder = 'D:\\My_Projects\\Panorama-stitch-lab\\tools\\smpl_fitting_external_model\\smpl_models\\'

    vposer_enc = None
    if USE_VPOSER_PRIOR:
        vposer_enc, _ = load_vposer(VPOSER_DIR)
        logger.info(f"VPoser prior enabled. w_vposer={W_VPOSER}")
        
    image_path = os.path.join(ls.IMAGE_ORIGIN6_DIR, 'frame_00001_rotate.png')
    # image_path = os.path.join(ls.DATA_OUTPUT_DIR,'origin_6', 'frame_00001_equirect.png')
    img = cv.imread(image_path)
    image_detect_path = os.path.join(ls.DATA_OUTPUT_DIR, 'test_human', 'detected_keypoints.png')
    detector = CombinedDetector(device=device, min_detection_confidence=0.5)
    engine_out = detector.detect_single_image(img)
    if engine_out.flag_valid:
        img_drawn = draw_keypoints(img, engine_out, draw_hands=True, draw_face=False)
        cv.imwrite(image_detect_path, img_drawn)
        logger.info(f"Keypoints detected and drawn. Saved to {image_detect_path}")
        kpt = torch.from_numpy(engine_out.keypoints).unsqueeze(0).to(device)  
        # Prioritize body alignment
        target_2d = kpt[:, :17, :2]  # from (1, 527, 3) get (1, 17, 2)
        confidences = kpt[:, :17, 2]  # (1, 17)
        # confidences[:, 11:17] = 0.0
    else:
        raise RuntimeError('No valid person detected. Please check input image or detector settings.')
    K, D, xi, _= load_single_K_and_distortion(ls.CAMERA_CALIB_FILE, 'camera6')
    camera_params = {
        'K': torch.tensor(K, dtype=torch.float32, device=device),
        'D': torch.tensor(D, dtype=torch.float32, device=device),
        'xi': torch.tensor([xi], dtype=torch.float32, device=device)
    }
    
    fitter = FisheyeFitter(model_folder, device=device, vposer_enc=vposer_enc, w_vposer=W_VPOSER)
    final_params = fitter.fit_single_frame(target_2d, confidences, camera_params, out_size=(img.shape[1], img.shape[0]))
    logger.info("Fitting completed. Final parameters:")
    logger.info(f"Betas: {final_params['betas']}")
    projected_2d = final_params['projected_2d'].cpu().numpy()[0]  # (17, 2)
    fitted_conf = confidences[0].detach().cpu().numpy()  # (17,)
    fitted_vis_kpts = np.concatenate([projected_2d, fitted_conf[:, None]], axis=1)
    
    # visual (reuse draw_keypoints with is_fitted=True)
    img_fitted = draw_keypoints(img, fitted_vis_kpts, is_fitted=True)
    image_fitted_path = os.path.join(ls.DATA_OUTPUT_DIR, 'test_human', 'fitted_keypoints.png')
    cv.imwrite(image_fitted_path, img_fitted)
    logger.info(f"Fitted keypoints drawn. Saved to {image_fitted_path}")

    pkl_output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'test_human', 'smpl_fit_result.pkl')
    fitter.save_smpl_result_as_pkl(final_params, pkl_output_path, image_path=image_path)
    
    final_params_loaded = fitter.load_smpl_result_from_pkl(pkl_output_path)
    vertices, faces, joints = fitter.load_vertices_faces_and_joints(final_params_loaded)

    faces_for_render = faces
    if RENDER_UPPER_BODY_ONLY:
        lbs_weights = fitter.body_model.body_model.lbs_weights.detach().cpu().numpy()
        faces_for_render = filter_upper_body_faces(faces, lbs_weights=lbs_weights)

    save_mesh_overlay(
        img,
        vertices[0],
        faces_for_render,
        camera_params,
        overlay_path=os.path.join(ls.DATA_OUTPUT_DIR, 'test_human', 'mesh_overlay.png')
    )
    

main()