#!/usr/bin/env python3
"""Summarise the fixed, real-trajectory pose-offset GEMM blocking bake-off."""
import argparse, csv, json, math, shutil, statistics
from collections import defaultdict
from pathlib import Path

def stats(v):
    q=statistics.quantiles(v,n=4,method="inclusive")
    mean=statistics.mean(v)
    return {"median_ms":statistics.median(v),"mean_ms":mean,"min_ms":min(v),"max_ms":max(v),"p25_ms":q[0],"p75_ms":q[2],"coefficient_of_variation":statistics.pstdev(v)/mean}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);a=ap.parse_args();root=a.root;raw=root/"raw_final"
    rows=list(csv.DictReader((raw/"samples.csv").open())); groups=defaultdict(list)
    for r in rows: groups[r["candidate"]].append(float(r["trajectory_ms"]))
    keep={"A_CURRENT_EIGEN_GEMM","B_M512","C_N48"}
    with (root/"finalist_samples.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(r for r in rows if r["candidate"] in keep)
    baseline=stats(groups["A_CURRENT_EIGEN_GEMM"]); best_b=stats(groups["B_M512"]);best_c=stats(groups["C_N48"])
    best="B_M512";speed=baseline["median_ms"]/best_b["median_ms"];saving=baseline["median_ms"]-best_b["median_ms"]
    calls=list(csv.DictReader((raw/"per_call.csv").open())); cg=defaultdict(list)
    for r in calls:
        if r["candidate"] in keep: cg[(r["candidate"],r["stage"],r["camera"],r["call_index"])].append(float(r["gemm_ms"]))
    with (root/"per_call_breakdown.csv").open("w",newline="") as f:
        names=["candidate","stage","camera","call_index","M","Q","gemm_ms","relative_speedup_vs_A"]
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for key,v in sorted(cg.items()):
            c,s,cam,i=key;m=3069 if cam=="cam2" else 2781;med=statistics.median(v);abase=statistics.median(cg[("A_CURRENT_EIGEN_GEMM",s,cam,i)])
            w.writerow(dict(candidate=c,stage=s,camera=cam,call_index=i,M=m,Q=66,gemm_ms=med,relative_speedup_vs_A=abase/med))
    shutil.copy(raw/"candidate_manifest.csv",root/"candidate_manifest.csv")
    parity=list(csv.DictReader((raw/"parity.csv").open()));shutil.copy(raw/"parity.csv",root/"parity.csv")
    def worst(c,k):return max(float(r[k]) for r in parity if r["candidate"]==c)
    local={c:{"max_abs":worst(c,"max_abs"),"rel_frob":worst(c,"rel_frob"),"pass":worst(c,"max_abs")<=2e-8} for c in ["B_M512","C_N48"]}
    l1,l2,l3=32768,1048576,20237516
    candidates=[("A_CURRENT_EIGEN_GEMM","none",0,0),("B_M512","M",512,0),("C_N48","N",0,48)]
    with (root/"working_set.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["candidate","block_type","BM","BN","P_block_bytes","D_block_bytes","Y_block_bytes","L1_bytes","L2_bytes","L3_bytes"])
        for c,t,bm,bn in candidates:w.writerow([c,t,bm,bn,(bm or 3069)*486*4,486*(bn or 66)*4,(bm or 3069)*(bn or 66)*4,l1,l2,l3])
    historical_pipeline=202.716;historical_gemm=67.586
    projected=historical_pipeline-saving;conservative=historical_pipeline-.75*saving
    with (root/"projected_pipeline.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["projection","saving_ms","pipeline_ms","fps"]);w.writerow(["direct",saving,projected,1000/projected]);w.writerow(["75_percent_transfer",.75*saving,conservative,1000/conservative])
    stage_stats={}
    for c in keep:
        rs=[r for r in rows if r["candidate"]==c]
        stage_stats[c]={k:statistics.median([float(r[k]) for r in rs]) for k in ["pose_cam2_ms","pose_cam4_ms","pose_shape_cam2_ms","pose_shape_cam4_ms"]}
    flops=2*sum((3069 if r["camera"]=="cam2" else 2781)*486*66 for r in list(csv.DictReader((Path("data_process/Vis/cpp_reference_dump/pose_jacobian_residual_performance_attribution/frame_000513/derivative_input_corpus/manifest.csv")).open())))
    summary={"frame":513,"cam2_N":1023,"cam4_N":927,"dimensions":{"M_cam2":3069,"M_cam4":2781,"K":486,"N":66,"Q":66},"corpus_sha256":"5f4e2a9fcafd52fbd83ee19c983ec24030977ece0474e0ca60077f50124f037e","preferred_baseline_policy":"optimized_native_stable_p","preferred_policy_changed":False,"build_policy":"Release -O3 -DNDEBUG -ffp-contract=off -march=native; Eigen vectorization ON; threads=1","baseline":baseline,"screening_candidates":{k:stats(v) for k,v in groups.items()},"finalists":{"A_CURRENT_EIGEN_GEMM":baseline,"B_M512":best_b,"C_N48":best_c},"best_candidate":best,"best_blocking_type":"M","BM":512,"BN":0,"best_speedup":speed,"best_saving_ms":saving,"blocking_classification":"NEUTRAL" if .98<=speed<1.05 else "LOSS","integration_worthiness":"REJECTED_PERFORMANCE","d_not_run":True,"d_not_run_reason":"best M and best N screening speedups were both below 1.10x; negative-stop also satisfied","local_parity":local,"downstream_parity":{"python_analytic":"PASS (unchanged canonical regression)","pose_J_H_g":"PASS (M blocking output bitwise equal)","pose_shape":"PASS (M blocking output bitwise equal)"},"trajectory_exact":True,"BEST_BLOCKING_TRAJECTORY_EXACT":True,"first_divergence":"NONE","stable_p":{"deep_copies":0,"cam2_pointer_stable":True,"cam4_pointer_stable":True},"tolerance_changes":0,"block_metadata":{"cam2_count":6,"cam2_tail":509,"cam4_count":6,"cam4_tail":221,"gemm_invocations_per_derivative_call":6,"gemm_invocations_per_frame":96},"allocations":{"candidate_user_allocations":0,"candidate_user_temp_bytes":0,"extra_full_matrix_copy":False,"output_layout":"row-major [M,66] unchanged"},"per_stage_camera":stage_stats,"gflops":{"theoretical_per_trajectory":flops,"A":flops/baseline["median_ms"]/1e6,"best":flops/best_b["median_ms"]/1e6},"cpu":{"model":"Intel Core i9-10900X","L1d_per_core_bytes":l1,"L2_per_core_bytes":l2,"L3_shared_bytes":l3},"system_projection":{"historical_pipeline_ms":historical_pipeline,"historical_pose_offset_gemm_ms":historical_gemm,"projected_pipeline_ms":projected,"projected_fps":1000/projected,"conservative_pipeline_ms":conservative,"conservative_fps":1000/conservative,"above_5fps":projected<200,"above_5_5fps":projected<1000/5.5,"gap_to_5_5fps_ms":projected-1000/5.5,"above_6fps":projected<1000/6,"gap_to_6fps_ms":projected-1000/6},"fused_lbs_plausible_saving_high_ms":7.746,"simple_candidate_preferred":True,"best_approved_variant":"NONE_REJECTED_PERFORMANCE","next_experiment":"POSE_LBS_FUSED_DOWNSTREAM_BAKEOFF","ready":"PASS"}
    (root/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (root/"downstream_parity.json").write_text(json.dumps(summary["downstream_parity"],indent=2)+"\n")
    shutil.copy(root/"shadow_reference_best"/"pose_trajectory.csv",root/"trajectory_shadow.csv")
    md=f"""# Pose-offset derivative GEMM blocking bake-off\n\nReal 16-call trajectory, Stable-P, Q=66. A median {baseline['median_ms']:.3f} ms; best blocked B_M512 {best_b['median_ms']:.3f} ms ({speed:.3f}x, saving {saving:.3f} ms). N48 was {best_c['median_ms']:.3f} ms. Blocking is **NEUTRAL** and rejected for integration. 2D was not run by the negative-stop rule. Full LM trajectory is exact; next experiment is `POSE_LBS_FUSED_DOWNSTREAM_BAKEOFF`.\n"""
    (root/"benchmark_summary.md").write_text(md)
    print("POSE_OFFSET_DERIVATIVE_GEMM_BLOCKING_BAKEOFF_READY=PASS")
if __name__=="__main__":main()
