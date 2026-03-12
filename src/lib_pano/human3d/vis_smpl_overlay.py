"""
SMPL 投影可视化：将保存的 smpl_joints54.npy / smpl_verts.npy 用
omni-radtan 鱼眼模型投影回原图，分别输出两张图：
  vis_smpl_mesh.jpg     —— 仅 Lambert shading 网格叠加
  vis_smpl_skeleton.jpg —— 仅彩色骨骼 + 关节标注叠加

用法：
  python src/lib_pano/human3d/vis_smpl_overlay.py
"""
import sys, json, os, pickle
import numpy as np
import cv2

# 路径设置
sys.path.insert(0, 'src')

CALIB_PATH  = 'data_process/Input/calib_params/camera_calib.json'
CAM_ID      = 'camera6'
IMG_PATH    = 'data_process/Input/Images/origin_6/frame_00001_rotate.png'
SMPL_DIR    = 'data_process/Output/test_segmentation/smpl_out'
PKL_PATH    = 'tools/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
OUT_MESH    = os.path.join(SMPL_DIR, 'vis_smpl_mesh.jpg')
OUT_SKEL    = os.path.join(SMPL_DIR, 'vis_smpl_skeleton.jpg')
OUT_OBJ     = os.path.join(SMPL_DIR, 'vis_smpl_upper.obj')
SCALE       = 3   # 缩小倍数（原图 2880x3840，缩小 3 倍后 960x1280）

# ── 1. 加载标定参数 ────────────────────────────────────────────────────
with open(CALIB_PATH) as f:
    data = json.load(f)
cam = data['camera_configs']['results'][CAM_ID]
xi, fx, fy, cx, cy = cam['K']
k1, k2, p1, p2     = cam['distortion']
print(f"[标定] xi={xi:.4f} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

# ── 2. 加载 SMPL 结果 ──────────────────────────────────────────────────
j54   = np.load(os.path.join(SMPL_DIR, 'smpl_joints54.npy'))[0]  # (54,3)
verts = np.load(os.path.join(SMPL_DIR, 'smpl_verts.npy'))[0]     # (6890,3)
print(f"[数据] joints54 Z 范围: [{j54[:,2].min():.3f}, {j54[:,2].max():.3f}] m")
print(f"[数据] verts    Z 范围: [{verts[:,2].min():.3f}, {verts[:,2].max():.3f}] m")

# ── 3. omni-radtan 投影 ────────────────────────────────────────────────
from lib_pano.camera.omni_camera import omni_project_np
j2d = omni_project_np(j54,   fx=fx, fy=fy, cx=cx, cy=cy,
                      k1=k1, k2=k2, p1=p1, p2=p2, xi=xi)   # (54,2)
v2d = omni_project_np(verts, fx=fx, fy=fy, cx=cx, cy=cy,
                      k1=k1, k2=k2, p1=p1, p2=p2, xi=xi)   # (6890,2)

# ── 4. 读取原图 ────────────────────────────────────────────────────────
img = cv2.imread(IMG_PATH)
assert img is not None, f"无法读取图片: {IMG_PATH}"
H, W = img.shape[:2]
print(f"[图像] {W}×{H}")
print(f"[投影] 关节 x: [{j2d[:,0].min():.0f}, {j2d[:,0].max():.0f}]  "
      f"y: [{j2d[:,1].min():.0f}, {j2d[:,1].max():.0f}]")
print(f"[投影] 在画面内的关节(0-23): "
      f"{sum(1 for x,y in j2d[:24] if 0<=x<W and 0<=y<H)}/24")

# ── 5. 全身网格（Lambert shading + painter's algorithm）───────────────────
with open(PKL_PATH, 'rb') as _f:
    _smpl_data = pickle.load(_f, encoding='latin1')
    faces      = np.array(_smpl_data['f'], dtype=np.int32)           # (F,3)
    lbs_weights= np.array(_smpl_data['weights'], dtype=np.float32)   # (6890,24)

# 用 LBS 主导关节过滤顶点：只保留上半身关节（Thorax/颈/头/肩/肘/腕/手）控制的顶点
# SMPL 24关节: 0=Pelvis,1=L_Hip,2=R_Hip,3=Spine1,4=L_Knee,5=R_Knee,
#              6=Spine2,7=L_Ankle,8=R_Ankle,9=Thorax,10=L_Foot,11=R_Foot,
#              12=Neck,13=L_Collar,14=R_Collar,15=Head,
#              16=L_Sho,17=R_Sho,18=L_Elbow,19=R_Elbow,
#              20=L_Wrist,21=R_Wrist,22=L_Hand,23=R_Hand
UPPER_JOINTS = {9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
dominant = np.argmax(lbs_weights, axis=1)          # (6890,) 每个顶点的主导关节
upper_mask_v = np.isin(dominant, list(UPPER_JOINTS))  # (6890,) bool

# 只保留：三顶点 Z 均 > 0  且  至少一个顶点主导关节属于上半身
front_mask  = (verts[:, 2][faces]).min(axis=1) > 0
upper_mask  = upper_mask_v[faces].any(axis=1)
valid_faces = faces[front_mask & upper_mask]
print(f"[网格] 上半身(LBS)有效面片 {len(valid_faces)}/{len(faces)}")

# 计算每个面片的法向量
v0 = verts[valid_faces[:, 0]]
v1 = verts[valid_faces[:, 1]]
v2 = verts[valid_faces[:, 2]]
normals = np.cross(v1 - v0, v2 - v0)
normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

# 平行光方向（相机坐标系，斜上左前）
light = np.array([-0.4, -0.7, -0.6], dtype=np.float64)
light /= np.linalg.norm(light)

# 两面着色：取点积绝对值，背面不会全黑
dot       = np.abs(normals @ light)           # (M,)
intensity = (0.25 + 0.75 * dot).clip(0, 1)   # ambient=0.25, diffuse=0.75

# painter's algorithm：按面片平均深度由远到近排序
depth = verts[valid_faces].mean(axis=1)[:, 2]
order = np.argsort(-depth)

# 逐面片着色（皮肤底色 BGR），叠加到独立的 mesh 图上
base_bgr   = np.array([110.0, 140.0, 185.0])
img_mesh   = img.copy()
mesh_layer = img.copy()
tri_pts_f  = v2d[valid_faces].astype(np.float32)
for k in order:
    tri = tri_pts_f[k].astype(np.int32)
    c   = (base_bgr * intensity[k]).clip(0, 255).astype(np.uint8)
    cv2.fillConvexPoly(mesh_layer, tri, c.tolist())
img_mesh = cv2.addWeighted(img, 0.20, mesh_layer, 0.80, 0)

# 在网格图上写 transl 信息
transl = j54[0]   # Pelvis ≈ transl
cv2.putText(img_mesh,
    f"SMPL Pelvis: ({transl[0]:.2f}, {transl[1]:.2f}, {transl[2]:.2f}) m",
    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 4, cv2.LINE_AA)
print("[网格] 渲染完成")

# 保存网格图
out_mesh = cv2.resize(img_mesh, (W//SCALE, H//SCALE), interpolation=cv2.INTER_AREA)
cv2.imwrite(OUT_MESH, out_mesh, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"[完成] 网格图: {os.path.abspath(OUT_MESH)}  ({W//SCALE}x{H//SCALE})")

# ── 5b. 导出上半身 OBJ ────────────────────────────────────────────────
# 只导出 valid_faces 涉及的顶点，重映射索引
used_vids = np.unique(valid_faces)
v_remap   = np.zeros(len(verts), dtype=np.int32)
v_remap[used_vids] = np.arange(len(used_vids)) + 1   # OBJ 1-indexed
with open(OUT_OBJ, 'w') as _obj:
    _obj.write('# SMPL upper body (Y-up, for MeshLab/Blender)\n')
    _obj.write(f'# verts={len(used_vids)}  faces={len(valid_faces)}\n')
    for vid in used_vids:
        x, y, z = verts[vid]
        # OpenCV(X右,Y下,Z前) → Y-up(X右,Y上,Z后)
        _obj.write(f'v {x:.6f} {-y:.6f} {-z:.6f}\n')
    for tri in valid_faces:
        a, b, c = v_remap[tri]
        _obj.write(f'f {a} {b} {c}\n')
print(f"[完成] OBJ:   {os.path.abspath(OUT_OBJ)}  ({len(used_vids)} verts, {len(valid_faces)} faces)")

# ── 6. 骨骼线段（独立图） ────────────────────────────────────────────────
# 有效关节集合（决定骨骼两端都必须在此集合内才画）
# Nose(24)/Head_top(48) 是 extra joint，无 SMPL_PARENT 连线，仅画点
JOINT_LABELS = {
    9:'Thorax', 12:'Neck', 15:'Jaw',
    16:'L_Sho', 17:'R_Sho',
    18:'L_Elbow', 19:'R_Elbow',
    20:'L_Wrist', 21:'R_Wrist',
    24:'Nose', 48:'Head_top',
}
VALID_JOINT_SET = set(idx for idx in JOINT_LABELS if idx < 24)  # {9,12,15,16,17,18,19,20,21}

LEFT_IDX  = {16, 18, 20}
RIGHT_IDX = {17, 19, 21}

def bone_color(child_idx):
    if child_idx in LEFT_IDX:  return (230, 60,  60)   # 蓝（BGR）
    if child_idx in RIGHT_IDX: return (60,  60, 230)   # 红
    return (60, 210, 60)                                # 绿（中轴）

# 显式骨骼表（par, child）：Thorax→Sho 直连，跳过无效的 L/R_Collar 中间节点
DRAW_BONES = [
    (9,  12),  # Thorax → Neck
    (12, 15),  # Neck   → Jaw
    (9,  16),  # Thorax → L_Sho  （跳过 L_Collar）
    (9,  17),  # Thorax → R_Sho  （跳过 R_Collar）
    (16, 18),  # L_Sho  → L_Elbow
    (17, 19),  # R_Sho  → R_Elbow
    (18, 20),  # L_Elbow → L_Wrist
    (19, 21),  # R_Elbow → R_Wrist
]

img_skel = img.copy()
for par, child in DRAW_BONES:
    # 两端必须都是有效关节
    if par not in VALID_JOINT_SET or child not in VALID_JOINT_SET:
        continue
    x1, y1 = j2d[par]
    x2, y2 = j2d[child]
    if (j54[child,2] > 0 and j54[par,2] > 0 and
            0<=x1<W and 0<=y1<H and 0<=x2<W and 0<=y2<H):
        cv2.line(img_skel, (int(x1),int(y1)), (int(x2),int(y2)),
                 bone_color(child), 8)

# ── 7. 关节圆点 + 标注（仅 JOINT_LABELS 内的有效关节）────────────────
for idx, label in JOINT_LABELS.items():
    x, y = j2d[idx]
    if j54[idx,2] > 0 and 0<=x<W and 0<=y<H:
        c = bone_color(idx)
        cv2.circle(img_skel, (int(x), int(y)), 14, c, -1)
        cv2.circle(img_skel, (int(x), int(y)), 14, (255,255,255), 2)
        cv2.putText(img_skel, label, (int(x)+16, int(y)+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2,
                    cv2.LINE_AA)

# 在骨骼图上写 transl 信息
cv2.putText(img_skel,
    f"SMPL Pelvis: ({transl[0]:.2f}, {transl[1]:.2f}, {transl[2]:.2f}) m",
    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 4, cv2.LINE_AA)

# ── 8. 保存骨骼图 ─────────────────────────────────────────────────────
out_skel = cv2.resize(img_skel, (W//SCALE, H//SCALE), interpolation=cv2.INTER_AREA)
cv2.imwrite(OUT_SKEL, out_skel, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"[完成] 骨骼图: {os.path.abspath(OUT_SKEL)}  ({W//SCALE}x{H//SCALE})")
