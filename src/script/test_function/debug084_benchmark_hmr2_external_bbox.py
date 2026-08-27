"""Benchmark detector-free batched HMR2 initialization on real YOLO boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch

from lib_pano.human3d.hmr2_initializer import HMR2Initializer


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


def _load_boxes(path: Path) -> np.ndarray:
    if path.is_dir():
        boxes=[]
        for item in sorted(path.glob("person_*.npz")):
            with np.load(item) as data: boxes.append(np.asarray(data["bbox_xyxy"],np.float32))
        return np.stack(boxes) if boxes else np.zeros((0,4),np.float32)
    data=json.loads(path.read_text())
    if "boxes_xyxy" in data: return np.asarray(data["boxes_xyxy"],np.float32)
    if data.get("schema_version")=="densekp_pairs_v2":
        return np.asarray([p["bbox_cam4"] for p in data["persons"]],np.float32)
    raise ValueError("boxes source must be an NPZ directory or JSON with boxes_xyxy")


def _timed(initializer,image,boxes,warmup,repeat):
    prepared=initializer.prepare(image,boxes)
    for _ in range(warmup):
        batch=initializer.transfer(prepared); initializer.forward(batch)
    torch.cuda.synchronize()
    before_mb=torch.cuda.memory_allocated()/2**20; torch.cuda.reset_peak_memory_stats()
    rows=[]; last_results=None
    for _ in range(repeat):
        total=time.perf_counter()
        start=time.perf_counter(); prepared=initializer.prepare(image,boxes); preprocess=(time.perf_counter()-start)*1000
        torch.cuda.synchronize(); start=time.perf_counter(); batch=initializer.transfer(prepared);torch.cuda.synchronize();h2d=(time.perf_counter()-start)*1000
        start=time.perf_counter(); output=initializer.forward(batch);torch.cuda.synchronize();forward=(time.perf_counter()-start)*1000
        start=time.perf_counter();last_results=initializer.postprocess(prepared,output);torch.cuda.synchronize();post=(time.perf_counter()-start)*1000
        rows.append({"preprocess_ms":preprocess,"h2d_ms":h2d,"forward_ms":forward,"postprocess_ms":post,"total_ms":(time.perf_counter()-total)*1000})
    peak_mb=torch.cuda.max_memory_allocated()/2**20
    summary={"batch_size":len(boxes),"allocated_before_mb":before_mb,"peak_allocated_mb":peak_mb}
    for field in ("preprocess_ms","h2d_ms","forward_ms","postprocess_ms","total_ms"):
        values=[r[field] for r in rows]; summary[field.replace("_ms","_p50_ms")]=statistics.median(values);summary[field.replace("_ms","_p90_ms")]=_percentile(values,90)
    summary["ms_per_person"]=summary["total_p50_ms"]/len(boxes)
    return summary,last_results


def _repeated_single(initializer,image,boxes,warmup,repeat):
    for _ in range(warmup):
        for box in boxes: initializer.infer(image,box[None])
    torch.cuda.synchronize(); values=[]
    for _ in range(repeat):
        start=time.perf_counter()
        for box in boxes: initializer.infer(image,box[None])
        torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
    return {"p50_ms":statistics.median(values),"p90_ms":_percentile(values,90)}


def _native(result):
    return {"person_index":result.person_index,"bbox_xyxy":result.bbox_xyxy.tolist(),
            "global_orient":result.global_orient.tolist(),"body_pose":result.body_pose.tolist(),
            "betas":result.betas.tolist(),"pred_cam":result.pred_cam.tolist(),
            "pred_cam_t":result.pred_cam_t.tolist(),"pred_cam_t_full":result.pred_cam_t_full.tolist(),
            "pred_vertices_shape":list(result.pred_vertices_shape),"pred_vertices_min":result.pred_vertices_min,"pred_vertices_max":result.pred_vertices_max}


def _validate_results(results, expected_count):
    if len(results) != expected_count:
        raise RuntimeError(
            f"HMR2 output count mismatch: expected={expected_count} actual={len(results)}"
        )
    fields = (
        "bbox_xyxy", "global_orient", "body_pose", "betas", "pred_cam",
        "pred_cam_t", "pred_cam_t_full", "pred_keypoints_2d",
    )
    for result in results:
        for field in fields:
            if not np.isfinite(np.asarray(getattr(result, field))).all():
                raise RuntimeError(
                    f"non-finite HMR2 output: person={result.person_index} field={field}"
                )


def _visualize(image,boxes,results,path):
    from hmr2.datasets.utils import gen_trans_from_patch_cv
    vis=image.copy();colors=[(0,255,0),(255,0,0),(0,0,255),(0,255,255)]
    for index,(box,result) in enumerate(zip(boxes,results)):
        color=colors[index%len(colors)];x1,y1,x2,y2=map(int,box);cv2.rectangle(vis,(x1,y1),(x2,y2),color,4)
        center=(box[2:]+box[:2])/2; scale=(box[2:]-box[:2])/200
        target=np.asarray([192,256],np.float32); expanded=np.array([max(scale[0]*200,scale[1]*200*target[0]/target[1]),max(scale[1]*200,scale[0]*200*target[1]/target[0])]);size=float(expanded.max())
        transform=gen_trans_from_patch_cv(center[0],center[1],size,size,256,256,1.,0.);inverse=cv2.invertAffineTransform(transform)
        crop=(result.pred_keypoints_2d+.5)*256; points=np.c_[crop,np.ones(len(crop))]@inverse.T
        for x,y in points:
            if np.isfinite(x) and np.isfinite(y): cv2.circle(vis,(int(x),int(y)),3,color,-1)
        cv2.putText(vis,f"P{index}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2)
    cv2.imwrite(str(path),vis)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--image",type=Path,required=True);parser.add_argument("--boxes",type=Path,required=True);parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-sizes",nargs="+",type=int,default=[1,2,4]);parser.add_argument("--warmup",type=int,default=5);parser.add_argument("--repeat",type=int,default=30);args=parser.parse_args()
    image=cv2.imread(str(args.image));boxes=_load_boxes(args.boxes)
    if image is None: raise FileNotFoundError(args.image)
    if len(boxes)<max(args.batch_sizes): raise ValueError(f"need {max(args.batch_sizes)} boxes, got {len(boxes)}")
    args.output.mkdir(parents=True,exist_ok=True);initializer=HMR2Initializer(args.checkpoint)
    batches=[];outputs={}
    for size in args.batch_sizes:
        summary,results=_timed(initializer,image,boxes[:size],args.warmup,args.repeat);batches.append(summary);outputs[size]=results
    workloads=[]
    by_size={row["batch_size"]:row for row in batches}
    for size in (2,4):
        if size not in by_size or size>len(boxes): continue
        repeated=_repeated_single(initializer,image,boxes[:size],args.warmup,args.repeat)
        batched=by_size[size]["total_p50_ms"]
        workloads.append({"person_count":size,"repeated_b1_p50_ms":repeated["p50_ms"],"batched_p50_ms":batched,"speedup":repeated["p50_ms"]/batched})
    final=outputs[max(args.batch_sizes)]
    _validate_results(final, max(args.batch_sizes))
    (args.output/"hmr2_initializations.json").write_text(json.dumps({"persons":[_native(x) for x in final]},indent=2)+"\n")
    _visualize(image,boxes[:len(final)],final,args.output/"hmr2_external_bbox_sanity.jpg")
    import onnxruntime,torchvision
    report={"environment":{"torch":torch.__version__,"torchvision":torchvision.__version__,"numpy":np.__version__,"onnxruntime":onnxruntime.__version__,"device":torch.cuda.get_device_name()},"checkpoint":str(args.checkpoint.resolve()),"model":{"parameter_count":initializer.parameter_count,"checkpoint_size_bytes":args.checkpoint.stat().st_size,"resident_gpu_memory_mb":batches[0]["allocated_before_mb"]},"benchmark":{"warmup":args.warmup,"repeat":args.repeat,"precision":"float32"},"bbox_source":str(args.boxes),"bbox_format":"xyxy full-image pixels","batch_input_shape_template":["B",3,256,256],"vit_backbone_input_shape_template":["B",3,256,192],"correctness":{"output_count":len(final),"all_finite":True},"batches":batches,"batched_vs_repeated_b1":workloads,"load_count":initializer.load_count}
    (args.output/"benchmark.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
