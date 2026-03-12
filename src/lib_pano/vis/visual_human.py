"""关节/轮廓可视化 + 对比图"""

import os, time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy

from lib_pano.camera.omni_camera import omni_project_np
from lib_pano.human.smpl_model import export_obj, crop_upper_body

# ── 可视化常量 ──────────────────────────────────────────────────
SKEL_PAIRS_7 = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,5),(4,6)]
JOINT_NAMES  = ['head', 'L_sho', 'R_sho', 'L_elb', 'R_elb', 'L_wri', 'R_wri']

C_GREEN  = (0, 220, 60)
C_ORANGE = (30, 140, 255)
C_CYAN   = (255, 200, 0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (160, 160, 160)

THUMB = 0.25  # 默认缩放比例


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── 基础绘制 ────────────────────────────────────────────────────

def save_thumb(img: np.ndarray, path: str, scale: float = THUMB):
    """缩放保存图像"""
    h, w = img.shape[:2]
    out = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imwrite(path, out)
    _log(f"  已保存: {os.path.basename(path)}  ({out.shape[1]}×{out.shape[0]})")


def draw_joints_on_img(img, uv_gt, valid_gt,
                       uv_pred=None, valid_pred=None,
                       names=None, skel=None, scale=THUMB):
    """在缩略图上绘制 GT(绿) + Pred(橙) 关节点和骨架"""
    if names is None: names = JOINT_NAMES
    if skel  is None: skel  = SKEL_PAIRS_7
    s = scale; h, w = img.shape[:2]
    vis = cv2.resize(img.copy(), (int(w*s), int(h*s)))
    for i, (name, ok) in enumerate(zip(names, valid_gt)):
        if ok:
            pt = (int(uv_gt[i,0]*s), int(uv_gt[i,1]*s))
            cv2.circle(vis, pt, 6, C_GREEN, -1)
            cv2.putText(vis, name, (pt[0]+3, pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GREEN, 1)
    for a, b in skel:
        if valid_gt[a] and valid_gt[b]:
            cv2.line(vis, (int(uv_gt[a,0]*s), int(uv_gt[a,1]*s)),
                          (int(uv_gt[b,0]*s), int(uv_gt[b,1]*s)), C_GREEN, 1)
    if uv_pred is not None:
        vp = valid_pred if valid_pred is not None else [True]*len(uv_pred)
        for i, ok in enumerate(vp):
            if ok and 0 <= uv_pred[i,0] < w and 0 <= uv_pred[i,1] < h:
                cv2.circle(vis, (int(uv_pred[i,0]*s), int(uv_pred[i,1]*s)), 5, C_ORANGE, -1)
        for a, b in skel:
            if vp[a] and vp[b]:
                cv2.line(vis, (int(uv_pred[a,0]*s), int(uv_pred[a,1]*s)),
                              (int(uv_pred[b,0]*s), int(uv_pred[b,1]*s)), C_CYAN, 1)
    return vis


def save_joint_overlay(img, gt_uv, valid, text, fname, output_dir,
                       pred_uv=None, font_scale=0.45):
    """绘制关节点叠加图并保存"""
    vp  = [True]*len(pred_uv) if pred_uv is not None else None
    vis = draw_joints_on_img(img, gt_uv, valid, pred_uv, vp)
    cv2.putText(vis, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, C_WHITE, 1)
    cv2.imwrite(os.path.join(output_dir, fname), vis)
    _log(f"  已保存: {fname}")


# ── Silhouette 渲染 ─────────────────────────────────────────────

def render_silhouette(verts_cam, faces, cam, H, W):
    """逐面片光栅化 silhouette，返回 uint8 mask (H,W)"""
    uv    = omni_project_np(verts_cam, **cam)
    # Mei/UCM 全向相机可见性条件：Z + ξ > 0，而非 Z > 0
    # 针孔相机 ξ=0 时退化为普通 z>0 判断
    xi    = cam.get('xi', 0.0)
    z_vis = verts_cam[:, 2] + xi > 0.01
    mask  = np.zeros((H, W), dtype=np.uint8)
    drawn = 0
    for tri in faces:
        ia, ib, ic = tri
        if not (z_vis[ia] and z_vis[ib] and z_vis[ic]):
            continue
        pts2d = np.array([[uv[ia,0], uv[ia,1]],
                          [uv[ib,0], uv[ib,1]],
                          [uv[ic,0], uv[ic,1]]], dtype=np.float32)
        if pts2d[:,0].min() >= W or pts2d[:,0].max() < 0: continue
        if pts2d[:,1].min() >= H or pts2d[:,1].max() < 0: continue
        side = np.ptp(pts2d[:,0]) + np.ptp(pts2d[:,1])  # max-min of x and y
        # if side > min(H, W) * 0.3: continue   # 几何奇异超大三角，跳过
        poly = pts2d.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
        drawn += 1
    _log(f"  silhouette: 有效面={drawn}/{len(faces)}")
    return mask


def _load_obj_mesh(obj_path: str):
    """Load OBJ vertices/faces and triangulate polygon faces with fan split."""
    verts = []
    faces = []
    with open(obj_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                tokens = line.split()[1:]
                idx = [int(tok.split('/')[0]) - 1 for tok in tokens]
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def rigid_fit_render_silhouette(s1: dict, s2: dict, cam: dict, obj_path: str, output_dir: str,
                            mask_name: str = 'step2_silhouette_mask.png',
                            overlay_name: str = 'step2_silhouette_overlay.png'):
    """Render silhouette after rigid_fit and save mask/overlay images."""
    _log("Step 2.5: render silhouette after rigid fit")

    verts_obj, faces = _load_obj_mesh(obj_path)
    r_gl2cv = np.array([[1., 0., 0.],
                        [0., -1., 0.],
                        [0., 0., -1.]], dtype=np.float32)
    verts_cv  = (r_gl2cv @ verts_obj.T).T
    sc        = float(np.exp(s2['log_scale']))
    R_rot     = R_scipy.from_rotvec(s2['rot_aa']).as_matrix().astype(np.float32)
    centroid  = s2['centroid']                                   # (3,) OBJ joints 在 CV 空间的质心
    verts_cam = sc * ((R_rot @ (verts_cv - centroid).T).T) + s2['trans']

    img = s1['img']
    H, W = img.shape[:2]
    sil_mask = render_silhouette(verts_cam, faces, cam, H, W)
    cv2.imwrite(os.path.join(output_dir, mask_name), sil_mask)

    sil_overlay = img.copy()
    sil_overlay[sil_mask > 0] = (
        sil_overlay[sil_mask > 0].astype(np.float32) * 0.5 + np.array([0, 120, 255], np.float32) * 0.5
    ).astype(np.uint8)
    contours, _ = cv2.findContours(sil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sil_overlay, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, overlay_name), sil_overlay)

    _log(f"  已保存: {mask_name} / {overlay_name}")
    return {
        'sil_mask': sil_mask,
        'overlay': sil_overlay,
    }


# ── 对比图 ──────────────────────────────────────────────────────

def make_comparison(output_dir: str):
    """拼接 step1-step4 为横向对比图"""
    _log("生成各步骤对比图 step_comparison.png")
    pairs = [('step1_keypoints.png','Step1'), ('step2_rigid_fit.png','Step2'),
             ('step3_vposer_fit.png','Step3'), ('step4_overlay.png','Step4')]
    imgs = [(cv2.imread(os.path.join(output_dir, fn)), lb) for fn, lb in pairs
            if os.path.exists(os.path.join(output_dir, fn))]
    if not imgs: return
    h_min = min(im.shape[0] for im, _ in imgs)
    tiles = []
    for im, lb in imgs:
        r = cv2.resize(im, (int(im.shape[1]*h_min/im.shape[0]), h_min))
        cv2.putText(r, lb, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WHITE, 2)
        tiles.append(r)
    comp = np.concatenate(tiles, axis=1)
    cv2.imwrite(os.path.join(output_dir, 'step_comparison.png'), comp)
    _log(f"  已保存: step_comparison.png  ({comp.shape[1]}×{comp.shape[0]})")


# ── Step 4: 导出 OBJ + Silhouette 可视化 ────────────────────────

def step4_export(s1, s3, smpl, cam, output_dir: str):
    """导出上半身/全身 OBJ + Silhouette 渲染 + 关节叠加图"""
    from lib_pano.human.detect import SMPL_ALIGN

    _log("Step 4: 导出优化 OBJ + Silhouette 可视化")
    verts_cam = s3['verts_cam']   # (6890,3) OpenCV 相机空间
    uv_all    = s3['uv_all']      # (24,2)
    img       = s1['img']
    H, W      = img.shape[:2]
    lbs_w     = smpl.lbs_weights.cpu().numpy()   # (6890,24)
    R_cv2gl   = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])

    # ── 上半身裁剪（主推荐）────────────────────────
    verts_ub_cam, faces_ub = crop_upper_body(verts_cam, smpl.faces, lbs_w)
    verts_ub_view = (R_cv2gl @ verts_ub_cam.T).T

    # 【推荐】上半身 OBJ — MeshLab/Blender 可直接查看 (Y-up)
    export_obj(verts_ub_view, faces_ub,
               os.path.join(output_dir, 'step4_upper_view_space.obj'),
               comment='SMPL upper body, Y-up OpenGL view space. ★ Recommended for MeshLab/Blender.')
    # 上半身 OBJ — 相机空间（OpenCV）
    export_obj(verts_ub_cam, faces_ub,
               os.path.join(output_dir, 'step4_upper_camera_space.obj'),
               comment='SMPL upper body in OpenCV camera space')
    # ── 全身 OBJ（备用，下半身不可靠）────────────────
    export_obj(verts_cam, smpl.faces,
               os.path.join(output_dir, 'step4_fullbody_camera_space.obj'),
               comment='SMPL full body in OpenCV camera space (lower body unreliable)')

    # ── Silhouette 渲染（只渲染上半身，结果更干净）────
    _log("  渲染上半身 silhouette（逐面片，约需数秒）...")
    sil_mask = render_silhouette(verts_ub_cam, faces_ub, cam, H, W)

    # (1) 纯 silhouette mask（白底，绿=上半身，橙色轮廓）
    sil_white = np.ones((H, W, 3), dtype=np.uint8) * 255
    sil_white[sil_mask > 0] = [60, 200, 80]
    contours, _ = cv2.findContours(sil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sil_white, contours, -1, (0, 100, 200), 4)
    cv2.imwrite(os.path.join(output_dir, 'step4_silhouette_mask.png'), sil_white)

    # (2) 叠加到原图
    vis_sil = img.copy().astype(np.float32)
    sil_m   = sil_mask > 0
    vis_sil[sil_m] = vis_sil[sil_m] * 0.40 + np.array([30, 100, 255], np.float32) * 0.60
    vis_sil = np.clip(vis_sil, 0, 255).astype(np.uint8)
    cv2.drawContours(vis_sil, contours, -1, (255, 80, 0), 3)

    # ── 叠加关节点 ────────────────────────────────────
    vis_full = vis_sil.copy()
    for i in range(24):
        u, v = uv_all[i]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis_full, (int(u), int(v)), 4, C_GRAY, -1)
    for i, idx in enumerate(SMPL_ALIGN):
        u, v = uv_all[idx]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis_full, (int(u), int(v)), 8, C_ORANGE, 2)
    for i in range(7):
        if s1['valid_mask'][i]:
            u, v = s1['target_uv'][i]
            cv2.circle(vis_full, (int(u), int(v)), 8, C_GREEN, 2)
            cv2.line(vis_full, (int(u), int(v)),
                     (int(uv_all[SMPL_ALIGN[i], 0]), int(uv_all[SMPL_ALIGN[i], 1])),
                     C_CYAN, 1)
    cv2.putText(vis_full, f"Step4: Upper Body  RMSE={s3['rmse']:.1f}px",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_WHITE, 2)
    cv2.putText(vis_full, "Blue silhouette=upper body  Green=RTMPose",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_WHITE, 2)
    save_thumb(vis_full, os.path.join(output_dir, 'step4_overlay.png'))
    _log("  step4_overlay.png / step4_silhouette_mask.png / OBJ × 3 已保存")
    _log("  ★ 推荐: step4_upper_view_space.obj (Y-up, MeshLab/Blender 直接打开)")

    
