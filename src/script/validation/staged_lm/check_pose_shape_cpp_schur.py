"""Check isolated and integrated Pose/Shape Schur C++ solves."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
FIELDS=("A_damped","C_damped","Y","y","schur_matrix","schur_rhs","delta_beta","frame_rhs_after_beta","delta_frame","full_delta")
def metric(a,b,rel_tol,abs_tol):
 d=np.abs(a-b);rel=float(np.linalg.norm(d)/max(np.linalg.norm(a),1e-300));ma=float(d.max(initial=0));return {"rel_l2":rel,"max_abs":ma,"pass":bool(rel<=rel_tol and ma<=abs_tol)}
def read(out,key,shape):return np.fromfile(out/f"{key}.f64","<f8").reshape(shape)
def main():
 p=argparse.ArgumentParser();p.add_argument("--reference_dir",type=Path,required=True);p.add_argument("--kernel_output",type=Path,required=True);p.add_argument("--integration_output",type=Path,required=True);x=p.parse_args();kernel={};integration={};order=[];diagnostics={}
 with np.load(x.reference_dir/"schur_reference.npz") as z:
  for i in range(3):
   for f in FIELDS:
    key=f"case{i}_{f}";kernel[key]=metric(z[key],read(x.kernel_output,key,z[key].shape),1e-10,1e-10);order.append(key)
   cpp=read(x.kernel_output,f"case{i}_full_delta",z[f"case{i}_full_delta"].shape);dense=read(x.kernel_output,f"case{i}_dense_delta",cpp.shape);kernel[f"case{i}_dense_equivalence"]=metric(cpp,dense,1e-10,1e-10);order.append(f"case{i}_dense_equivalence")
   lr=read(x.kernel_output,f"case{i}_linear_system_residual",(85,));sr=read(x.kernel_output,f"case{i}_schur_residual",(10,));kernel[f"case{i}_solve_residual"]={"rel_l2":float(np.linalg.norm(lr)/max(np.linalg.norm(np.r_[z['g_frame'],z['g_beta']]),1e-300)),"max_abs":float(np.max(np.abs(lr))),"pass":float(np.linalg.norm(lr)/max(np.linalg.norm(np.r_[z['g_frame'],z['g_beta']]),1e-300))<=1e-10};order.append(f"case{i}_solve_residual");kernel[f"case{i}_schur_residual"]={"rel_l2":float(np.linalg.norm(sr)/max(np.linalg.norm(z[f'case{i}_schur_rhs']),1e-300)),"max_abs":float(np.max(np.abs(sr))),"pass":float(np.linalg.norm(sr)/max(np.linalg.norm(z[f'case{i}_schur_rhs']),1e-300))<=1e-10};order.append(f"case{i}_schur_residual")
   for f in ("delta_beta","delta_frame","full_delta"):
    key=f"case{i}_{f}";integration[key]=metric(z[key],read(x.integration_output,key,z[key].shape),1e-4,1e-4)
   S=read(x.kernel_output,f"case{i}_schur_matrix",(10,10));meta=dict(line.split("=",1) for line in (x.kernel_output/f"case{i}_meta.txt").read_text().splitlines());diagnostics[f"case{i}"]={"schur_symmetry_max_abs":float(np.max(np.abs(S-S.T))),"A_factor_method":meta["A_factor_method"],"schur_factor_method":meta["schur_factor_method"],"fallback_used":meta["fallback_used"],"timing_ms":{k:float(v) for k,v in meta.items() if k.endswith("_ms")}}
 first=next((k for k in order if not kernel[k]["pass"]),"NONE");ifirst=next((k for k,v in integration.items() if not v["pass"]),"NONE");ok=first==ifirst=="NONE";result={"SCHUR_KERNEL_FIRST_DIVERGENCE":first,"SCHUR_INTEGRATION_FIRST_DIVERGENCE":ifirst,"kernel_metrics":kernel,"integration_metrics":integration,"diagnostics":diagnostics,"POSE_SHAPE_SCHUR_CPP_CPU_REFERENCE_READY":"PASS" if ok else "FAIL"};(x.kernel_output/"pose_shape_schur_cpp_parity.json").write_text(json.dumps(result,indent=2)+"\n");(x.kernel_output/"pose_shape_schur_cpp_parity.md").write_text(f"# Pose/Shape Schur parity\n\nSCHUR_KERNEL_FIRST_DIVERGENCE={first}\n\nSCHUR_INTEGRATION_FIRST_DIVERGENCE={ifirst}\n");print(f"SCHUR_KERNEL_FIRST_DIVERGENCE={first}");print(f"SCHUR_INTEGRATION_FIRST_DIVERGENCE={ifirst}");print(f"POSE_SHAPE_SCHUR_CPP_CPU_REFERENCE_READY={result['POSE_SHAPE_SCHUR_CPP_CPU_REFERENCE_READY']}");return 0 if ok else 1
if __name__=="__main__":raise SystemExit(main())
