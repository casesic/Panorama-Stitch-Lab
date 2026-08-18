"""Export input-only Pose/Shape value/LM fixture."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
FLOATS=("parameters","base_joints","d_joints_d_beta","beta_anchor","body_pose_anchor","cam2_selected_base_v_shaped","cam2_selected_beta_shapedirs","cam2_selected_posedirs","cam2_selected_lbs_weights","cam2_targets","cam2_left_hand_pose","cam2_right_hand_pose","cam2_scale","cam4_selected_base_v_shaped","cam4_selected_beta_shapedirs","cam4_selected_posedirs","cam4_selected_lbs_weights","cam4_targets","cam4_left_hand_pose","cam4_right_hand_pose","cam4_scale")
def export(ref,out=None):
 m=json.loads((ref/"pose_shape_reference_manifest.json").read_text());out=out or ref/"cpp_fixture";out.mkdir(parents=True,exist_ok=True)
 with np.load(ref/"pose_shape_reference.npz") as a:
  for name in FLOATS:(out/f"{name}.f32").write_bytes(np.ascontiguousarray(a[name],dtype="<f4").tobytes())
  if "parents" in a:
   (out/"parents.i64").write_bytes(np.ascontiguousarray(a["parents"],dtype="<i8").tobytes())
  else:
   # Compatibility with reference dumps created before parents became
   # self-contained in pose_shape_reference.npz.
   source=ref.parent.parent/"pose"/f"frame_{int(m['frame']):06d}"/"cpp_fixture"/"parents.i64"
   if not source.is_file(): raise ValueError("POSE_SHAPE_REFERENCE_MISSING_PARENTS")
   (out/"parents.i64").write_bytes(source.read_bytes())
 meta={"schema_version":"1","frame":str(m["frame"]),"parameter_dim":str(m["parameter_dim"]),"frame_parameter_dim":"75","beta_dim":str(m["beta_dim"]),"beta_abs_limit":str(m["beta_abs_limit"]),"pose_anchor_scale":str(m["pose_anchor_scale"]),"shape_anchor_scale":str(m["shape_anchor_scale"]),"charbonnier_eps":str(m["charbonnier_eps"]),"image_width":"5760","image_height":"2880","lm_max_accepted_steps":"1","lm_max_trials":"8","rel_objective_tolerance":"1e-7","scaled_gradient_tolerance":"1e-5","step_tolerance":"1e-5",**{f"{c}_count":str(m["cameras"][c]["point_count"]) for c in ("cam2","cam4")},**{f"{c}_alpha":str(m["aggregation_spec"]["terms"][f"{m['frame']}|{c}"]["alpha_per_point"]) for c in ("cam2","cam4")},"reference_leakage":"NONE"}
 (out/"fixture_meta.txt").write_text("".join(f"{k}={v}\n" for k,v in meta.items()));print("POSE_SHAPE_CPP_FIXTURE",out);return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--reference_dir",type=Path,required=True);p.add_argument("--output_dir",type=Path);a=p.parse_args();export(a.reference_dir,a.output_dir)
if __name__=="__main__":main()
