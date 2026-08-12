"""Package verified C++ NE blocks as Schur integration inputs (no delta)."""
from pathlib import Path
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument("--ne_output",type=Path,required=True);p.add_argument("--kernel_fixture",type=Path,required=True);p.add_argument("--output",type=Path,required=True);x=p.parse_args();x.output.mkdir(parents=True,exist_ok=True)
 mapping={"A.f64":"A_final.f64","B.f64":"B_final.f64","C.f64":"C_final.f64","g_frame.f64":"g_frame_final.f64","g_beta.f64":"g_beta_final.f64"}
 for dst,src in mapping.items():(x.output/dst).write_bytes((x.ne_output/src).read_bytes())
 (x.output/"fixture_meta.txt").write_text((x.kernel_fixture/"fixture_meta.txt").read_text().replace("no_delta_reference_leakage=TRUE","no_delta_reference_leakage=TRUE\nsource=C++_verified_normal_equations"))
if __name__=="__main__":main()
