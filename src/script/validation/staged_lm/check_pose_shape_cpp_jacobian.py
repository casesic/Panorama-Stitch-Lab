"""Compare Pose/Shape combined analytic Jacobian C++ output with Python."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

SHARED=("d_beta_d_latent","d_dynamic_joints_d_beta_latent")
CAMERA=("variable_Jr","d_rotation_pose","d_pose_feature","d_pose_offsets_pose",
"d_dynamic_v_shaped_d_beta_latent","d_v_posed_combined","d_relative_joints_beta",
"d_local_transforms_combined","d_global_transforms_combined","d_rest_transforms_combined",
"d_relative_from_fk","d_relative_from_rest","d_relative_transforms_combined",
"d_selected_transforms_combined","d_skin_from_transform","d_skin_from_vertex",
"d_skinned_vertices_combined","d_vertices_combined","d_vertices_rotpose","d_vertices_beta",
"J_proj","J_uv_combined","J_uv_rotpose","J_beta","J_translation","J_active")
TOL={"d_pose_offsets_pose":2e-6,"d_v_posed_combined":2e-6,"d_local_transforms_combined":2e-6,
"d_global_transforms_combined":2e-6,"d_rest_transforms_combined":2e-6,"d_relative_from_fk":2e-6,
"d_relative_from_rest":2e-6,"d_relative_transforms_combined":2e-6,"d_selected_transforms_combined":2e-6,
"d_skin_from_transform":2e-6,"d_skin_from_vertex":2e-6,"d_skinned_vertices_combined":2e-6,
"d_vertices_combined":2e-6,"d_vertices_rotpose":2e-6,"d_vertices_beta":2e-6,"J_uv_combined":2e-6,
"J_uv_rotpose":2e-6,"J_beta":2e-6,"J_active":2e-6}
def metric(a,b,tol):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));den=np.linalg.norm(a.astype(np.float64));rel=float(np.linalg.norm(d)/(den if den else 1));return {"exact":bool(np.array_equal(a,b)),"max_abs":float(d.max(initial=0)),"mean_abs":float(d.mean()),"rel_frob":rel,"tolerance":tol,"pass":rel<=tol}
def check(ref,out):
 with np.load(ref/"pose_shape_reference.npz") as py:
  results={};first=None;first_beta=None;first_pose=None
  for name in SHARED:
   a=py[name];b=np.fromfile(out/f"{name}.f32",dtype="<f4").reshape(a.shape);results[name]=metric(a,b,TOL.get(name,1e-6))
  for cam in ("cam2","cam4"):
   for name in CAMERA:
    key=f"{cam}_{name}";a=py[key];b=np.fromfile(out/f"{key}.f32",dtype="<f4").reshape(a.shape);results[key]=metric(a,b,TOL.get(name,1e-6))
  for key,value in results.items():
   if not value["pass"]:
    first=first or key
    if "beta" in key or "dynamic" in key:first_beta=first_beta or key
    else:first_pose=first_pose or key
  # Structural invariants are evaluated on both independently produced arrays.
  inv={}
  for source,reader in (("python",lambda k:py[k]),("cpp",lambda k:np.fromfile(out/f"{k}.f32",dtype="<f4").reshape(py[k].shape))):
   inv[source+"_rest_pose_zero"]=all(np.count_nonzero(reader(f"{c}_d_rest_transforms_combined")[...,:66])==0 for c in ("cam2","cam4"))
   inv[source+"_rest_beta_sign"]=all(np.array_equal(reader(f"{c}_d_rest_transforms_combined")[:,:3,3,66:],-reader("d_dynamic_joints_d_beta_latent")) for c in ("cam2","cam4"))
   inv[source+"_relative_joint_shared"]=np.array_equal(reader("cam2_d_relative_joints_beta"),reader("cam4_d_relative_joints_beta"))
   inv[source+"_beta_width_10"]=all(reader(f"{c}_J_beta").shape[-1]==10 for c in ("cam2","cam4"))
  passed=first is None and all(inv.values())
 result={"status":"PASS" if passed else "FAIL","FIRST_DIVERGENCE":first or "NONE","FIRST_BETA_DIVERGENCE":first_beta or "NONE","FIRST_POSE_DIVERGENCE":first_pose or "NONE","metrics":results,"invariants":inv,"POSE_SHAPE_JACOBIAN_CPP_CPU_REFERENCE_READY":"PASS" if passed else "FAIL"}
 out.mkdir(parents=True,exist_ok=True);(out/"pose_shape_jacobian_cpp_parity.json").write_text(json.dumps(result,indent=2)+"\n")
 lines=["# Pose/Shape C++ Jacobian parity","",f"Status: **{result['status']}**",f"FIRST_DIVERGENCE={result['FIRST_DIVERGENCE']}","","| checkpoint | rel_frob | max_abs | pass |","|---|---:|---:|:---:|"]
 lines += [f"| {k} | {v['rel_frob']:.6e} | {v['max_abs']:.6e} | {v['pass']} |" for k,v in results.items()]
 (out/"pose_shape_jacobian_cpp_parity.md").write_text("\n".join(lines)+"\n");print(f"FIRST_DIVERGENCE={result['FIRST_DIVERGENCE']}");print(f"POSE_SHAPE_JACOBIAN_CPP_CPU_REFERENCE_READY={result['POSE_SHAPE_JACOBIAN_CPP_CPU_REFERENCE_READY']}");return 0 if passed else 1
def main():
 p=argparse.ArgumentParser();p.add_argument("--reference_dir",type=Path,required=True);p.add_argument("--cpp_output_dir",type=Path,required=True);a=p.parse_args();return check(a.reference_dir,a.cpp_output_dir)
if __name__=="__main__":raise SystemExit(main())
