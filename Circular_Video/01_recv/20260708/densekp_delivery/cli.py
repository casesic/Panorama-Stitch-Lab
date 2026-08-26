from __future__ import annotations

import argparse

try:
    from .config import DenseKPConfig
    from .pipeline import run_pipeline
except ImportError:
    import sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).resolve().parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from densekp_delivery.config import DenseKPConfig
    from densekp_delivery.pipeline import run_pipeline

VALID_BACKENDS = ("pytorch", "ort")
VALID_CACHE_MODES = ("read_write", "refresh", "read_only", "off")
VALID_VIS_OPTIONS = ("none", "det", "densekp", "pair", "all")
VALID_SAVE_OPTIONS = ("none", "npz", "debug", "all")
VALID_COORD_SPACES = ("erp", "erp_rotate", "fisheye", "both")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DenseKP Delivery — detect, extract, match 138 dense keypoints from fisheye image pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--backend", type=str, required=True, choices=VALID_BACKENDS,
                   help="Inference backend: pytorch, ort, or mock (for testing)")
    p.add_argument("--fisheye_cam2", type=str, required=True,
                   help="Path to cam2 fisheye image")
    p.add_argument("--fisheye_cam4", type=str, required=True,
                   help="Path to cam4 fisheye image")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory root")

    g_model = p.add_argument_group("Model paths")
    g_model.add_argument("--checkpoint", type=str, default=None,
                         help="Path to DenseKP PyTorch checkpoint (backend=pytorch)")
    g_model.add_argument("--onnx_path", type=str, default=None,
                         help="Path to DenseKP ONNX model (backend=ort)")

    g_cam = p.add_argument_group("Camera / fisheye-to-ERP")
    g_cam.add_argument("--camera", type=str, default="old", choices=["old", "new"],
                       help="Camera parameter set (default: old)")
    g_cam.add_argument("--erp_width", type=int, default=5760,
                       help="ERP output width (default: 5760)")
    g_cam.add_argument("--erp_height", type=int, default=2880,
                       help="ERP output height (default: 2880)")
    g_cam.add_argument(
        "--erp-remap-backend", choices=("cpu", "cuda"), default="cuda",
        help="ERP remap backend (default: cuda, with explicit CPU fallback).",
    )
    g_cam.add_argument(
        "--erp-remap-cuda-library", default=None,
        help="Path to libdensekp_erp_remap_cuda.so.",
    )

    g_cache = p.add_argument_group("Cache control")
    g_cache.add_argument("--cache_mode", type=str, default="read_write",
                         choices=VALID_CACHE_MODES,
                         help="Cache mode (default: read_write)")
    g_cache.add_argument(
        "--calibration_cache_dir", type=str, default=None,
        help="Shared ERP LUT/mask cache; keyed by calibration and image size.",
    )

    g_det = p.add_argument_group("Detection")
    g_det.add_argument("--detector_backend", type=str, default="rtmlib_yolox",
                       help="Person detector backend (default: rtmlib_yolox)")
    g_det.add_argument("--detector_threshold", type=float, default=0.25,
                       help="Detection score threshold (default: 0.25)")

    g_match = p.add_argument_group("Person matching")
    g_match.add_argument("--person_idx_cam2", type=int, default=None,
                         help="Manually specify person index for cam2")
    g_match.add_argument("--person_idx_cam4", type=int, default=None,
                         help="Manually specify person index for cam4")

    g_out = p.add_argument_group("Output")
    g_out.add_argument("--output_coord_space", type=str, default="erp",
                       choices=VALID_COORD_SPACES,
                       help="Output coordinate space: erp, erp_rotate, fisheye, both (default: erp)")
    g_out.add_argument("--vis_option", type=str, default="none",
                       choices=VALID_VIS_OPTIONS,
                       help="Visualization output (default: none)")
    g_out.add_argument("--save_option", type=str, default="none",
                       choices=VALID_SAVE_OPTIONS,
                       help="Additional save output (default: none)")

    p.add_argument("--device", type=str, default="cuda",
                   help="Inference device: cuda or cpu (default: cuda)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size for DenseKP inference (default: 1)")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging")
    p.add_argument(
        "--profile_runtime", action="store_true",
        help="Write result/densekp_runtime_profile.json.",
    )
    p.add_argument("--dry_run", action="store_true",
                   help="Parse args only, do not run pipeline")

    args = p.parse_args(argv)

    if args.backend == "pytorch" and not args.checkpoint:
        p.error("--checkpoint is required when backend=pytorch")
    if args.backend == "ort" and not args.onnx_path:
        p.error("--onnx_path is required when backend=ort")

    return args


# Compatibility for older scripts.  Production code must use DenseKPConfig.
_parse_args = parse_args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.dry_run:
        print("Dry run — arguments:")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v!r}")
        return
    run_pipeline(DenseKPConfig.from_namespace(args))


if __name__ == "__main__":
    main()
