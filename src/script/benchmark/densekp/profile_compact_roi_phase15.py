"""A/B benchmark for the calibration-derived baseline and compact ROI layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[4]
DELIVERY_ROOT = ROOT / "Circular_Video/01_recv/20260708"
sys.path.insert(0, str(DELIVERY_ROOT))

from densekp_delivery.config import DenseKPConfig
from densekp_delivery.runtime import DenseKPRuntime
import densekp_delivery.pipeline as pipeline


def percentile(values, fraction):
    ordered=sorted(values); index=min(len(ordered)-1,int(round((len(ordered)-1)*fraction)))
    return ordered[index]


def run_strategy(args, strategy):
    pipeline.DETECTION_ROI_STRATEGY=strategy
    root=args.output/strategy; root.mkdir(parents=True,exist_ok=True)
    config=DenseKPConfig(
        backend="ort",onnx_path=str(args.onnx),fisheye_cam2=str(args.cam2),
        fisheye_cam4=str(args.cam4),out_dir=str(root/"warmup"),camera="new",
        erp_remap_backend="cuda",erp_remap_cuda_library=str(args.cuda_remap),
        calibration_cache_dir=str(args.calibration_cache),output_coord_space="erp",
        vis_option="none",save_option="debug",device="cuda",batch_size=8,
        profile_runtime=True,
    )
    rows=[]
    with DenseKPRuntime.from_config(config) as runtime:
        runtime.process(config)
        for repeat in range(args.repeats):
            frame_config=config.for_frame(args.cam2,args.cam4,root/f"repeat_{repeat:02d}")
            runtime.process(frame_config)
            profile=json.loads((Path(frame_config.out_dir)/"result/densekp_runtime_profile.json").read_text())
            result=json.loads((Path(frame_config.out_dir)/"result/densekp_pairs.json").read_text())
            rows.append({**profile,"matched_person_count":result["matched_person_count"],
                         "valid_count":sum(sum(p["valid"]) for p in result["persons"])})
    keys=("detector_inference_ms","densekp_inference_ms","densekp_runner_total_ms","runtime_total_ms")
    return {"strategy":strategy,"repeats":rows,
            "p50":{k:statistics.median(float(r[k]) for r in rows) for k in keys},
            "p90":{k:percentile([float(r[k]) for r in rows],.9) for k in keys},
            "geometry":{k:rows[0][k] for k in ("roi_count","roi_source_pixels_total","roi_source_pixel_ratio","roi_mask_pixels_total","roi_fill_ratio","detector_call_count")}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cam2",type=Path,required=True);parser.add_argument("--cam4",type=Path,required=True)
    parser.add_argument("--onnx",type=Path,required=True);parser.add_argument("--cuda-remap",type=Path,required=True)
    parser.add_argument("--calibration-cache",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--strategies",nargs="+",choices=("component_bbox","compact_strips"),default=("component_bbox","compact_strips"))
    args=parser.parse_args()
    report={s:run_strategy(args,s) for s in args.strategies}
    (args.output/"compact_roi_ab.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
