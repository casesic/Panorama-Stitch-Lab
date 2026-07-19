#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.benchmark.staged_lm.sam3dbody_init_compatibility import audit_compatibility


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Audit SAM 3D Body initialization compatibility")
    p.add_argument("--frames",required=True);p.add_argument("--hmr_dir")
    p.add_argument("--sam3dbody_recon_dir",required=True);p.add_argument("--sam3d_dense_dir",required=True)
    p.add_argument("--bbox_dir",required=True);p.add_argument("--mhr_to_smplx_mapping",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--reproj_objective",default="charbonnier_irls")
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--audit_only",action="store_true");mode.add_argument("--run_bakeoff",action="store_true")
    p.add_argument("--output_json",required=True);return p


def main() -> int:
    args=parser().parse_args();frames=[int(v) for v in args.frames.split(",")]
    result=audit_compatibility(frames=frames,recon_dir=Path(args.sam3dbody_recon_dir),
        dense_dir=Path(args.sam3d_dense_dir),bbox_dir=Path(args.bbox_dir),mapping_path=Path(args.mhr_to_smplx_mapping))
    result["exact_configuration"]={"frames":frames,"device":args.device,"objective":args.reproj_objective,
        "mode":"run_bakeoff" if args.run_bakeoff else "audit_only"}
    if args.run_bakeoff:
        result["bakeoff"]={"status":"SKIPPED_INSUFFICIENT_COMPATIBILITY"} if not result["bakeoff_allowed"] else {
            "status":"NOT_IMPLEMENTED_UNEXPECTED_GATE_STATE"}
    path=Path(args.output_json);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"[CLASSIFICATION] {result['classification']}");print(f"[RECOMMENDATION] {result['recommendation']}")
    print(f"[BAKEOFF_ALLOWED] {str(result['bakeoff_allowed']).lower()}");print(f"[SAVE] {path}");return 0


if __name__=="__main__":raise SystemExit(main())
