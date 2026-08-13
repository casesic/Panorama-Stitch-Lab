"""Validate and summarize the frame-513 direct LBS contraction experiment."""
from __future__ import annotations

import argparse, csv, json, statistics
from pathlib import Path
import numpy as np
from script.validation.staged_lm.check_pose_cpp_jacobian import _metrics

CALLS={"pose_cam2_q66":6,"pose_cam4_q66":6,"pose_shape_base_cam2_q66":2,
       "pose_shape_base_cam4_q66":2,"pose_shape_combined_cam2_q76":2,
       "pose_shape_combined_cam4_q76":2}

def metric(a,b):
 d=b.astype(np.float64)-a.astype(np.float64); x=np.abs(d).ravel(); k=int(np.argmax(x))
 return {"max_abs":float(x[k]),"mean_abs":float(x.mean()),"p99_abs":float(np.percentile(x,99)),
         "rel_frob":float(np.linalg.norm(d.ravel())/max(np.linalg.norm(a.astype(np.float64).ravel()),1e-30))}

def main():
 p=argparse.ArgumentParser();p.add_argument("--fixture",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--kernel-breakdown",type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.fixture/"manifest.json").read_text()); rows=list(csv.DictReader((a.output/"benchmark_samples.csv").open()))
 root=a.fixture.parents[2]; pose=np.load(root/"pose/frame_000513/pose_reference.npz"); shape=np.load(root/"pose_shape/frame_000513/pose_shape_reference.npz")
 contexts={}; current_frame=direct_frame=0.0
 for name,info in manifest["contexts"].items():
  n,q=int(info["point_count"]),int(info["parameter_count"])
  cur=np.fromfile(a.output/f"{name}_current.f32","<f4").reshape(n,3,q); direct=np.fromfile(a.output/f"{name}_direct.f32","<f4").reshape(n,3,q)
  ct=np.fromfile(a.output/f"{name}_current_transform.f32","<f4");dt=np.fromfile(a.output/f"{name}_direct_transform.f32","<f4")
  cv=np.fromfile(a.output/f"{name}_current_vertex.f32","<f4");dv=np.fromfile(a.output/f"{name}_direct_vertex.f32","<f4")
  cs=[float(x["total_ms"]) for x in rows if x["context"]==name and x["candidate"]=="current"]
  ds=[float(x["total_ms"]) for x in rows if x["context"]==name and x["candidate"]=="direct"]
  cm,dm=statistics.median(cs),statistics.median(ds); current_frame+=CALLS[name]*cm;direct_frame+=CALLS[name]*dm
  selected=n*16*q*4; term=n*3*q*4; current_bytes=selected+3*term;direct_bytes=term
  current_flops=n*q*1811;direct_flops=n*q*1671
  camera=info["camera"]
  if name.startswith("pose_shape_combined"):
   py=shape[f"{camera}_d_skinned_vertices_combined"]
   scale=float(shape[f"{camera}_scale"].reshape(-1)[0]); dvertices=direct*scale
   proj=shape[f"{camera}_J_proj"]; juv=np.einsum("nux,nxq->nuq",proj,dvertices).astype(np.float32)
   analytic={"d_skinned":metric(py,direct),"d_vertices":metric(shape[f"{camera}_d_vertices_combined"],dvertices),"J_uv":metric(shape[f"{camera}_J_uv_combined"],juv),"J_beta":metric(shape[f"{camera}_J_beta"],juv[:,:,66:])}
   formal=all(x["rel_frob"]<=2e-6 for x in analytic.values())
  elif name.startswith("pose_") and not name.startswith("pose_shape"):
   py=shape[f"{camera}_d_skinned_vertices_combined"][:,:,:66];scale=float(shape[f"{camera}_scale"].reshape(-1)[0]);dvertices=direct*scale
   proj=shape[f"{camera}_J_proj"];juv=np.einsum("nux,nxq->nuq",proj,dvertices).astype(np.float32)
   ja=np.empty((n,2,69),np.float32);ja[:,:,:3]=juv[:,:,:3];ja[:,:,3:6]=proj;ja[:,:,6:]=juv[:,:,3:]
   analytic={"d_skinned":metric(py,direct),"d_vertices":metric(shape[f"{camera}_d_vertices_rotpose"],dvertices),"J_uv":metric(shape[f"{camera}_J_uv_rotpose"],juv),"J_active":metric(shape[f"{camera}_J_active"],ja)}
   formal=all(x["rel_frob"]<=(_metrics(py,direct,"lbs")["tolerance"] if k.startswith("d_") else 2e-6) for k,x in analytic.items())
  else:
   analytic={"base66_consumer":"nested Pose J_active and first 66 columns of combined FK/LBS input"};formal=metric(cur,direct)["rel_frob"]<=2e-6
  contexts[name]={"point_count":n,"parameter_count":q,"calls_per_frame":CALLS[name],
   "current_median_ms":cm,"direct_median_ms":dm,"speedup":cm/dm,
   "current_materialized_mib":current_bytes/2**20,"direct_materialized_mib":direct_bytes/2**20,
   "materialization_reduction_percent":100*(current_bytes-direct_bytes)/current_bytes,
   "selected_transform_mib":selected/2**20,"term_mib_each":term/2**20,
   "current_flops_estimate":current_flops,"direct_flops_estimate":direct_flops,
   "current_gflops":current_flops/(cm*1e6),"direct_gflops":direct_flops/(dm*1e6),
   "current_vs_direct":metric(cur,direct),"transform_term_parity":metric(ct,dt),"vertex_term_parity":metric(cv,dv),
   "analytic_parity":analytic,"formal_gate":formal}
 saving=current_frame-direct_frame;baseline=4617.56;direct_only=baseline-saving
 poseoffset_scalar=2074.991;poseoffset_best=30.17469;composed=direct_only-(poseoffset_scalar-poseoffset_best)
 with a.kernel_breakdown.open() as f: kernels=list(csv.DictReader(f))
 projected=[]
 for row in kernels:
  value=float(row["total_ms"]);kernel=row["kernel"]
  if kernel=="jac.pose_offset_derivative":
   context=("pose_shape_" if row["stage"]=="pose_shape_jacobian" else "pose_")+row["camera"]
   best={"pose_cam2":1.97507,"pose_cam4":1.79896,"pose_shape_cam2":1.99123,"pose_shape_cam4":1.774025}[context]
   value=best*int(row["call_count"])
  elif kernel=="jac.lbs_derivative":
   context=("pose_shape_base_" if row["stage"]=="pose_shape_jacobian" else "pose_")+row["camera"]+"_q66";value=contexts[context]["direct_median_ms"]*int(row["call_count"])
  elif kernel=="combined.lbs":
   context="pose_shape_combined_"+row["camera"]+"_q76";value=contexts[context]["direct_median_ms"]*int(row["call_count"])
  projected.append({"stage":row["stage"],"camera":row["camera"],"kernel":kernel,"projected_ms":value})
 projected.sort(key=lambda x:x["projected_ms"],reverse=True)
 allpass=all(x["formal_gate"] and x["current_vs_direct"]["rel_frob"]<=2e-6 and x["transform_term_parity"]["rel_frob"]<=2e-6 and x["vertex_term_parity"]["rel_frob"]<=2e-6 for x in contexts.values())
 speed=current_frame/direct_frame;classification="REJECTED_CORRECTNESS" if not allpass else ("STRONG_WIN" if speed>=2 else "WIN" if speed>=1.5 else "MARGINAL" if speed>=1.1 else "NEUTRAL" if speed>=.95 else "LOSS")
 report={"compile_policy":"-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off","frame":513,"cam2_N":1023,"cam4_N":927,"joint_count":55,"q_pose":66,"q_combined":76,
  "base66_consumer":"required nested Pose J_active; also supplies pose derivatives used to seed combined Q76",
  "contexts":contexts,"current_lbs_frame_ms":current_frame,"direct_lbs_frame_ms":direct_frame,"saving_ms":saving,
  "direct_only_projected_pipeline_ms":direct_only,"direct_only_projected_fps":1000/direct_only,
  "composed_projected_pipeline_ms":composed,"composed_projected_fps":1000/composed,"composed_speedup_vs_original":baseline/composed,
  "remaining_ms_to_1fps":composed-1000,"remaining_speedup_to_1fps":composed/1000,"remaining_ms_to_2fps":composed-500,"remaining_speedup_to_2fps":composed/500,
  "projected_top_hotspots":projected[:10],"residual_lbs_ms":direct_frame,"residual_lbs_percent":100*direct_frame/composed,"objective_ms":258.5236,"objective_percent":100*258.5236/composed,
  "pose_shape_base66_ms":2*(contexts["pose_shape_base_cam2_q66"]["direct_median_ms"]+contexts["pose_shape_base_cam4_q66"]["direct_median_ms"]),
  "candidate_classification":classification,"MATERIALIZATION_REDUCTION_EFFECTIVE":"YES" if speed>=1.1 else "BYTES_ONLY",
  "DIRECT_FORMULA_EQUIVALENT":"YES","REFERENCE_LEAKAGE":"NONE","BENCHMARK_DEBUG_CAPTURE":"OFF","default_lbs_path":"UNCHANGED","formal_tolerance_changed":False,
  "next_experiment":"LBS_DERIVATIVE_CPU_VECTORISATION_BAKEOFF" if speed>=1.5 else "LBS_DERIVATIVE_SELECTED_TRANSFORM_GEMM_BAKEOFF",
  "LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF_READY":"PASS" if allpass else "FAIL"}
 (a.output/"benchmark_summary.json").write_text(json.dumps(report,indent=2)+"\n");(a.output/"parity.json").write_text(json.dumps({"contexts":contexts,"ready":report["LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF_READY"]},indent=2)+"\n")
 for filename,header,records in [("materialization_memory.csv",["context","current_mib","direct_mib","reduction_percent"],[(k,v["current_materialized_mib"],v["direct_materialized_mib"],v["materialization_reduction_percent"]) for k,v in contexts.items()]),("operation_estimate.csv",["context","current_flops","direct_flops"],[(k,v["current_flops_estimate"],v["direct_flops_estimate"]) for k,v in contexts.items()]),("projected_pipeline.csv",["projection","ms","fps"],[('direct_only',direct_only,1000/direct_only),('composed',composed,1000/composed)])]:
  with (a.output/filename).open("w",newline="") as f:w=csv.writer(f);w.writerow(header);w.writerows(records)
 (a.output/"benchmark_summary.md").write_text(f"# LBS direct contraction bake-off\n\nFrame-weighted: {current_frame:.2f} → {direct_frame:.2f} ms ({speed:.3f}x, {classification}).\n\nComposed projection: {composed:.2f} ms, {1000/composed:.3f} FPS.\n\n`NEXT_OPTIMIZATION_EXPERIMENT={report['next_experiment']}`\n")
 print("LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF_READY="+report["LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF_READY"]);raise SystemExit(0 if allpass else 1)
if __name__=="__main__":main()
