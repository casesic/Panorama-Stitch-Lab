"""Check Pose/Shape C++ structured normal equations against Python."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
GROUP=("point_weights","row_scales","weighted_residual","weighted_J_active","weighted_J_beta","H_aa","H_ab","H_bb","g_a","g_b")
GLOBAL=("A_data","B_data","C_data","g_frame_data","g_beta_data","A_pose_prior","g_pose_prior","C_shape_prior","g_shape_prior","A_final","B_final","C_final","g_frame_final","g_beta_final","H_dense_structured","g_dense_structured")
def metric(ref,got,name,up=True):
 d=np.abs(ref.astype('f8')-got.astype('f8'));rel=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(ref.astype('f8').ravel()),1e-300));ma=float(d.max(initial=0))
 if name=="point_weights":gate="upstream residual + max_abs<=1e-7";passed=up and ma<=1e-7
 elif name=="row_scales":gate="upstream weights + max_abs<=3e-6";passed=up and ma<=3e-6
 elif name=="weighted_residual":gate="upstream + max_abs<=2e-5";passed=up and ma<=2e-5
 elif name.startswith("weighted_J"):gate="upstream + rel<=2e-5,max_abs<=1e-3";passed=up and rel<=2e-5 and ma<=1e-3
 elif name in ("H_aa","H_ab","H_bb"):gate="Pose/Shape IRLS-amplified intermediate rel<=5e-6";passed=rel<=5e-6
 elif "prior" in name:gate="prior rel<=1e-6 or max_abs<=1e-7";passed=rel<=1e-6 or ma<=1e-7
 elif name.startswith("g_"):gate="cancellation-aware rel<=2e-4,max_abs<=1e-3";passed=rel<=2e-4 and ma<=1e-3
 else:gate="Pose/Shape structured final rel<=5e-6";passed=rel<=5e-6
 return {"exact":bool(np.array_equal(ref,got)),"max_abs":ma,"mean_abs":float(d.mean()),"rel_frob":rel,"gate":gate,"pass":bool(passed and np.isfinite(got).all())}
def check(ref,out):
 comparisons={};order=[]
 with np.load(ref/"pose_shape_reference.npz") as a:
  for cam in ("cam2","cam4"):
   upstream=True
   for name in GROUP:
    key=f"{cam}_{name}";suffix="f32" if name in GROUP[:5] else "f64";got=np.fromfile(out/f"{key}.{suffix}","<f4" if suffix=="f32" else "<f8").reshape(a[key].shape);m=metric(a[key],got,name,upstream);comparisons[key]=m;order.append(key);upstream=upstream and m["pass"]
  for name in GLOBAL:
   got=np.fromfile(out/f"{name}.f64","<f8").reshape(a[name].shape);comparisons[name]=metric(a[name],got,name);order.append(name)
  inv={"cam2_A_opposite_rigid_zero":np.count_nonzero(a["cam2_A_scatter"][6:12])==0 and np.count_nonzero(a["cam2_A_scatter"][:,6:12])==0,"cam4_A_opposite_rigid_zero":np.count_nonzero(a["cam4_A_scatter"][:6])==0 and np.count_nonzero(a["cam4_A_scatter"][:,:6])==0,"cam2_B_opposite_rigid_zero":np.count_nonzero(a["cam2_B_scatter"][6:12])==0,"cam4_B_opposite_rigid_zero":np.count_nonzero(a["cam4_B_scatter"][:6])==0,"global_beta_shapes":a["B_final"].shape==(75,10) and a["C_final"].shape==(10,10) and a["g_beta_final"].shape==(10,),"dense_blocks":np.array_equal(a["H_dense_structured"][:75,75:],a["B_final"]) and np.array_equal(a["H_dense_structured"][75:,:75],a["B_final"].T),"A_symmetry_max_abs":float(np.max(np.abs(a["A_final"]-a["A_final"].T))),"C_symmetry_max_abs":float(np.max(np.abs(a["C_final"]-a["C_final"].T)))}
 first=next((k for k in order if not comparisons[k]["pass"]),"NONE");ok=first=="NONE" and all(v is True or isinstance(v,float) and v<=1e-10 for v in inv.values())
 result={"status":"PASS" if ok else "FAIL","FIRST_DIVERGENCE":first,"metrics":comparisons,"invariants":inv,"dtype_boundary":"weighted r/Ja/Jb float32 -> H/g float64","POSE_SHAPE_NORMAL_EQUATIONS_CPP_CPU_REFERENCE_READY":"PASS" if ok else "FAIL"}
 (out/"pose_shape_ne_cpp_parity.json").write_text(json.dumps(result,indent=2)+"\n");rows=["# Pose/Shape structured normal equations parity","",f"FIRST_DIVERGENCE={first}","","| checkpoint | rel | max abs | pass |","|---|---:|---:|:---:|"]+[f"| {k} | {v['rel_frob']:.6e} | {v['max_abs']:.6e} | {v['pass']} |" for k,v in comparisons.items()];(out/"pose_shape_ne_cpp_parity.md").write_text("\n".join(rows)+"\n");print(f"FIRST_DIVERGENCE={first}");print(f"POSE_SHAPE_NORMAL_EQUATIONS_CPP_CPU_REFERENCE_READY={result['POSE_SHAPE_NORMAL_EQUATIONS_CPP_CPU_REFERENCE_READY']}");return 0 if ok else 1
def main():
 p=argparse.ArgumentParser();p.add_argument("--reference_dir",type=Path,required=True);p.add_argument("--cpp_output",type=Path,required=True);x=p.parse_args();return check(x.reference_dir,x.cpp_output)
if __name__=="__main__":raise SystemExit(main())
