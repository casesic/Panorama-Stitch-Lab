from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two mask fill region summaries.")
    parser.add_argument("--before_summary", type=str, required=True)
    parser.add_argument("--after_summary", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    before = load_summary(Path(args.before_summary))
    after = load_summary(Path(args.after_summary))

    before_frames: dict = before.get("frames", {})
    after_frames: dict = after.get("frames", {})

    comparison: dict = {"frames": {}}
    agg = {
        "underfill_before": [],
        "underfill_after": [],
        "overfill_before": [],
        "overfill_after": [],
    }

    for frame_key in sorted(before_frames.keys()):
        bf = before_frames[frame_key]
        af = after_frames.get(frame_key, {})
        frame_comp = {}

        for cam in sorted(set(bf.keys()) | set(af.keys())):
            bs = bf.get(cam, {})
            as_ = af.get(cam, {})

            ub = bs.get("underfill_area", 0)
            ua = as_.get("underfill_area", 0)
            ob = bs.get("overfill_area", 0)
            oa = as_.get("overfill_area", 0)

            u_delta = ua - ub
            o_delta = oa - ob

            frame_comp[cam] = {
                "underfill_area_before": ub,
                "underfill_area_after": ua,
                "underfill_delta": u_delta,
                "underfill_delta_ratio": u_delta / max(ub, 1),
                "overfill_area_before": ob,
                "overfill_area_after": oa,
                "overfill_delta": o_delta,
                "overfill_delta_ratio": o_delta / max(ob, 1),
            }

            agg["underfill_before"].append(ub)
            agg["underfill_after"].append(ua)
            agg["overfill_before"].append(ob)
            agg["overfill_after"].append(oa)

        comparison["frames"][frame_key] = frame_comp

    def _mean(vals):
        return sum(vals) / max(len(vals), 1)

    comparison["summary"] = {
        "mean_underfill_before": _mean(agg["underfill_before"]),
        "mean_underfill_after": _mean(agg["underfill_after"]),
        "mean_underfill_delta": _mean(agg["underfill_after"]) - _mean(agg["underfill_before"]),
        "mean_overfill_before": _mean(agg["overfill_before"]),
        "mean_overfill_after": _mean(agg["overfill_after"]),
        "mean_overfill_delta": _mean(agg["overfill_after"]) - _mean(agg["overfill_before"]),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    s = comparison["summary"]
    print(f"mean underfill: {s['mean_underfill_before']:.0f} → {s['mean_underfill_after']:.0f}  "
          f"(Δ={s['mean_underfill_delta']:+.0f})")
    print(f"mean overfill:  {s['mean_overfill_before']:.0f} → {s['mean_overfill_after']:.0f}  "
          f"(Δ={s['mean_overfill_delta']:+.0f})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug012_compare_mask_fill_summary.py \
  --before_summary data_process/Output/20260311_riding/vis/mask_fill_regions/kp_densekp_bbox_filter/summary.json \
  --after_summary data_process/Output/20260311_riding/vis/mask_fill_regions/kp_densekp_bbox_filter_maskfill_stage3/summary.json \
  --output_json data_process/Output/20260311_riding/vis/mask_fill_regions/comparison_stage3_vs_no_stage3.json
'''
