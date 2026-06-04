__all__ = [
    "DualCamERPFitter",
    "init_smplx_from_smpl54_seq",
    "slice_smplx_seq",
]


def __getattr__(name):
    if name == "DualCamERPFitter":
        from .erp_dual import DualCamERPFitter

        return DualCamERPFitter
    if name in {"init_smplx_from_smpl54_seq", "slice_smplx_seq"}:
        from .init_utils import init_smplx_from_smpl54_seq, slice_smplx_seq

        return {
            "init_smplx_from_smpl54_seq": init_smplx_from_smpl54_seq,
            "slice_smplx_seq": slice_smplx_seq,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
