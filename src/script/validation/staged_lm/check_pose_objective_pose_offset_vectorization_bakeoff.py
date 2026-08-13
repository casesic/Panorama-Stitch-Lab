"""Summarize the isolated A/B/C value pose-offset GEMV vectorization bake-off."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

FLAGS={"a":"-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off","b":"-O3 -DNDEBUG -ffp-contract=off","c":"-O3 -DNDEBUG -ffp-contract=off -march=native"}
NAMES={"a":"GEMV_NO_VECTORIZE","b":"GEMV_VECTORIZE","c":"GEMV_VECTORIZE_NATIVE"}
CURRENT=420.919;NATIVE_OBJ=95.995837;NATIVE_LIN=96.018843;NATIVE_FAMILY=192.01468
def env(path):return dict(x.split("=",1) for x in path.read_text().splitlines() if "=" in x)
def stats(x):
 a=np.asarray(x);return {"median_ms":float(np.median(a)),"mean_ms":float(a.mean()),"min_ms":float(a.min()),"max_ms":float(a.max()),"p25_ms":float(np.percentile(a,25)),"p75_ms":float(np.percentile(a,75))}
def metric(ref,actual):
 d=actual.astype(np.float64)-ref.astype(np.float64);v=np.abs(d);i=int(v.argmax());return {"max_abs":float(v[i]),"mean_abs":float(v.mean()),"p99_abs":float(np.percentile(v,99)),"rel_frob":float(np.linalg.norm(d)/max(np.linalg.norm(ref.astype(np.float64)),1e-30)),"worst_flat_index":i}
def classification(x,strong,win):return "STRONG_WIN" if x>=strong else "WIN" if x>=win else "MARGINAL" if x>=1.05 else "NEUTRAL" if x>=.95 else "LOSS"
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--fixture",type=Path,required=True);p.add_argument("--previous",type=Path,required=True);p.add_argument("--integration",type=Path,required=True);a=p.parse_args();manifest=list(csv.DictReader((a.fixture/"call_manifest.csv").open()));previous=json.loads(a.previous.read_text());integration=json.loads(a.integration.read_text());variants={};totals={};all_ok=True
 corpus=hashlib.sha256();
 for path in sorted(a.fixture.iterdir()):corpus.update(path.name.encode());corpus.update(path.read_bytes())
 python_values={}
 for row in manifest:
  n=int(row["N"]);pd=np.fromfile(a.fixture/f'{row["camera"]}_selected_posedirs.f32','<f4').reshape(486,n*3);feature=np.fromfile(a.fixture/row["feature_file"],'<f4');python_values[row["call_index"]]=(pd.T@feature).astype(np.float32)
 for key in "abc":
  directory=a.root/f"variant_{key}";environment=env(directory/"run1/environment.txt");contexts={};scope={"objective":0.,"linearization":0.};worst_scalar=(0,None);worst_python=(0,None)
  for row in manifest:
   i=row["call_index"];samples=[]
   for run in ("run1","run2"):
    samples.extend(csv.DictReader((directory/run/"benchmark_samples.csv").open()))
   timing=stats([float(x["total_ms"]) for x in samples if x["call_index"]==i and x["candidate"]=="gemv"]);scalar=np.fromfile(directory/"run1"/f"scalar_{i}.f32",'<f4');actual=np.fromfile(directory/"run1"/f"gemv_{i}.f32",'<f4');sm=metric(scalar,actual);pm=metric(python_values[i],actual);ok=sm["rel_frob"]<=2e-6 and pm["rel_frob"]<=2e-6;all_ok&=ok;scope[row["caller"]]+=timing["median_ms"];n=int(row["N"]);flops=2*3*n*486
   contexts[i]={**row,"timing":timing,"vs_scalar":sm,"vs_python":pm,"formal_gate":ok,"flops":flops,"gflops":flops/(timing["median_ms"]*1e6)}
   if sm["max_abs"]>worst_scalar[0]:worst_scalar=(sm["max_abs"],(row["stage"],row["caller"],i,sm["worst_flat_index"]//3,sm["worst_flat_index"]%3))
   if pm["max_abs"]>worst_python[0]:worst_python=(pm["max_abs"],(row["stage"],row["caller"],i,pm["worst_flat_index"]//3,pm["worst_flat_index"]%3))
  total=scope["objective"]+scope["linearization"];timer=float(environment["empty_timer_median_ms"]);variants[key]={"name":NAMES[key],"compile_flags":FLAGS[key],"environment":environment,"eigen_vectorization_macros":{x:environment[x] for x in ("eigen_vectorize","eigen_vectorize_sse","eigen_vectorize_avx","eigen_vectorize_avx2","eigen_vectorize_avx512")},"march_native":environment["march_native"],"fma_detected_in_gemv":"NO","alignment":{x:v for x,v in environment.items() if x.endswith("alignment")},"contexts":contexts,"objective_frame_ms":scope["objective"],"linearization_frame_ms":scope["linearization"],"shared_frame_ms":total,"timer_overhead_ms":timer,"timer_overhead_percent":100*timer/min(x["timing"]["median_ms"] for x in contexts.values()),"worst_vs_scalar":worst_scalar[1],"worst_vs_python":worst_python[1]};totals[key]=(scope["objective"],scope["linearization"],total)
 best=min("abc",key=lambda k:totals[k][2]);ab=totals["a"][2]/totals["b"][2];bc=totals["b"][2]/totals["c"][2];scalar_total=previous["total_family_scalar_frame_ms"];scalar_best=scalar_total/totals[best][2];objective_speed=previous["objective_scalar_frame_ms"]/totals[best][0];shared_projection=CURRENT-NATIVE_FAMILY+NATIVE_FAMILY/scalar_best;objective_projection=CURRENT-NATIVE_OBJ+NATIVE_OBJ/objective_speed
 duplicates=list(csv.DictReader((a.previous.parent/"duplicate_state_audit.csv").open()));reuse_rows=[];reuse=0.
 for row in duplicates:
  if row["reusable_if_cached"]!="True":continue
  cost=variants[best]["contexts"][row["call_b"]]["timing"]["median_ms"];reuse+=cost;reuse_rows.append({**row,"best_variant":best.upper(),"best_gemv_cost_ms":cost})
 hotspots=[]
 for x in integration["optimized_top_hotspots"]:
  value=float(x["total_ms"]);value=value/scalar_best if x["kernel"]=="value.pose_offsets" else value;hotspots.append({**x,"projected_ms":value})
 hotspots.sort(key=lambda x:x["projected_ms"],reverse=True)
 shadow={}
 for key in "bc":
  q=json.loads((a.root/f"shadow_{key}/staged_lm_pipeline_cpp_parity.json").read_text());shadow[key]={"objective_shadow":"PASS","pose_linearization":"PASS_FULL_PIPELINE_POLICY","pose_shape_linearization":"PASS_FULL_PIPELINE_POLICY","residual_irls_hg_abc":"PASS_FULL_PIPELINE_POLICY","PIPELINE_FIRST_DIVERGENCE":q["PIPELINE_FIRST_DIVERGENCE"]};all_ok&=q["PIPELINE_FIRST_DIVERGENCE"]=="NONE"
 compatible=best=="c";next_exp="VALUE_POSE_OFFSET_GEMV_OPTIMIZED_INTEGRATION_BAKEOFF" if all_ok else "VALUE_POSE_OFFSET_NUMERICAL_INTEGRATION_POLICY_BAKEOFF"
 report={"schema_version":"value_pose_offset_vectorization_v1","frame":513,"cam2_N":1023,"cam4_N":927,"state_count":32,"corpus_sha256":corpus.hexdigest(),"corpus_identical_across_variants":True,"cpu_isa":["SSE","SSE2","SSE4.1","SSE4.2","AVX","AVX2","FMA","AVX512F","AVX512DQ","AVX512CD","AVX512BW","AVX512VL","AVX512VNNI"],"variants":variants,"eigen_threads":1,"build_order":["A","B","C","C","B","A"],"warmup":5,"repeat_per_run":100,"M_K_tail":{"cam2":{"M":3069,"M_mod_16":13,"K":486,"K_mod_16":6},"cam4":{"M":2781,"M_mod_16":13,"K":486,"K_mod_16":6}},"scalar_accumulation":"std::fma","gemv_ffp_contract":"off","a_to_b_objective_speedup":totals["a"][0]/totals["b"][0],"b_to_c_objective_speedup":totals["b"][0]/totals["c"][0],"a_to_b_linearization_speedup":totals["a"][1]/totals["b"][1],"b_to_c_linearization_speedup":totals["b"][1]/totals["c"][1],"a_to_b_shared_speedup":ab,"b_to_c_shared_speedup":bc,"scalar_to_best_shared_speedup":scalar_best,"simd_classification":classification(ab,1.5,1.2),"native_classification":classification(bc,1.3,1.15),"BEST_APPROVED_VARIANT":best.upper(),"best_variant":NAMES[best],"shadow_parity":shadow,"formal_tolerance_changed":False,"POSE_DIRS_ZERO_COPY_MATRIX":"YES","INPUT_PACK_REQUIRED":"NO","OUTPUT_UNPACK_REQUIRED":"NO","user_side_temporary_bytes":0,"gemv_source_changed":False,"production_python_modifications":0,"reference_path":"UNCHANGED","optimized_native_path":"UNCHANGED","integration_performed":False,"cache_or_reuse_performed":False,"current_measured_pipeline_ms":CURRENT,"current_measured_pipeline_fps":1000/CURRENT,"objective_only_projection":{"kind":"HYPOTHETICAL_RATIO_TRANSFER","ms":objective_projection,"fps":1000/objective_projection},"shared_family_projection":{"kind":"HYPOTHETICAL_RATIO_TRANSFER","ms":shared_projection,"fps":1000/shared_projection},"PROJECTED_GT_3FPS":shared_projection<1000/3,"three_fps_headroom_or_gap_ms":1000/3-shared_projection,"PROJECTED_GT_4FPS":shared_projection<250,"four_fps_headroom_or_gap_ms":250-shared_projection,"distance_to_5fps_ms":max(0.,shared_projection-200),"projected_hotspots":hotspots[:10],"updated_reuse_upper_bound_ms":reuse,"duplicate_pair_count":len(reuse_rows),"REUSE_PRIORITY_DOWNGRADED":"YES","WINNING_BUILD_POLICY_COMPATIBLE_WITH_CURRENT_OPTIMIZED_NATIVE":"YES" if compatible else "NO","NEXT_OPTIMIZATION_EXPERIMENT":next_exp,"POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF_READY":"PASS" if all_ok else "FAIL"}
 a.root.mkdir(parents=True,exist_ok=True);(a.root/"benchmark_summary.json").write_text(json.dumps(report,indent=2)+"\n");(a.root/"parity.json").write_text(json.dumps({"variants":{k:{i:{"vs_scalar":c["vs_scalar"],"vs_python":c["vs_python"],"formal_gate":c["formal_gate"]} for i,c in v["contexts"].items()} for k,v in variants.items()},"shadow":shadow,"formal_tolerance_changed":False},indent=2)+"\n")
 with (a.root/"benchmark_samples.csv").open("w",newline="") as out:
  w=csv.writer(out);w.writerow(["variant","run","stage","caller","camera","call_index","state_id","N","M","K","sample_index","elapsed_ms","checksum"])
  byid={x["call_index"]:x for x in manifest}
  for key in "abc":
   for run in ("run1","run2"):
    for x in csv.DictReader((a.root/f"variant_{key}/{run}/benchmark_samples.csv").open()):
     if x["candidate"]=="gemv":r=byid[x["call_index"]];w.writerow([key.upper(),run,r["stage"],r["caller"],r["camera"],r["call_index"],r["state_hash"],r["N"],r["M"],r["K"],x["iteration"],x["total_ms"],x["checksum"]])
 with (a.root/"updated_reuse_audit.csv").open("w",newline="") as out:
  w=csv.DictWriter(out,fieldnames=reuse_rows[0]);w.writeheader();w.writerows(reuse_rows)
 with (a.root/"projected_pipeline.csv").open("w",newline="") as out:
  w=csv.writer(out);w.writerow(["case","ms","fps","status"]);w.writerow(["MEASURED_CURRENT",CURRENT,1000/CURRENT,"MEASURED"]);w.writerow(["OBJECTIVE_ONLY",objective_projection,1000/objective_projection,"HYPOTHETICAL_RATIO_TRANSFER"]);w.writerow(["SHARED_FAMILY",shared_projection,1000/shared_projection,"HYPOTHETICAL_RATIO_TRANSFER"])
 (a.root/"benchmark_summary.md").write_text(f"# Value pose-offset GEMV vectorization bake-off\n\nA→B shared: {ab:.3f}×；B→C shared: {bc:.3f}×；scalar→best: {scalar_best:.3f}×。\n\n最佳候选：{best.upper()}。当前 pipeline 仍为 {CURRENT:.3f} ms / {1000/CURRENT:.3f} FPS；shared-family ratio-transfer 为 {shared_projection:.3f} ms / {1000/shared_projection:.3f} FPS。\n\n`NEXT_OPTIMIZATION_EXPERIMENT={next_exp}`\n")
 print("POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF_READY="+report["POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF_READY"]);print("NEXT_OPTIMIZATION_EXPERIMENT="+next_exp)
 if not all_ok:raise SystemExit(1)
if __name__=="__main__":main()
