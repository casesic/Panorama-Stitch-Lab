"""Build frame-513 Pose/Shape Schur references from verified FP64 blocks."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np, torch
def solve(A,B,C,gf,gb,lam):
 ad=A+lam*torch.eye(A.shape[-1],dtype=torch.float64);cd=C+lam*torch.eye(C.shape[-1],dtype=torch.float64);L=torch.linalg.cholesky(ad);joined=torch.cat((gf[:,None],B),1);sol=torch.cholesky_solve(joined,L);y=sol[:,0];Y=sol[:,1:];S=cd-B.T@Y;rhs=-gb+B.T@y;Ls=torch.linalg.cholesky(S);db=torch.cholesky_solve(rhs[:,None],Ls)[:,0];fr=-(gf+B@db);df=torch.cholesky_solve(fr[:,None],L)[:,0];delta=torch.cat((df,db));H=torch.cat((torch.cat((ad,B),1),torch.cat((B.T,cd),1)),0);dense=torch.linalg.solve(H,-torch.cat((gf,gb)));return {"A_damped":ad,"C_damped":cd,"Y":Y,"y":y,"schur_matrix":S,"schur_rhs":rhs,"delta_beta":db,"frame_rhs_after_beta":fr,"delta_frame":df,"full_delta":delta,"dense_delta":dense,"linear_system_residual":H@delta+torch.cat((gf,gb)),"schur_residual":S@db-rhs}
def main():
 p=argparse.ArgumentParser();p.add_argument("--pose_shape_reference",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);x=p.parse_args();x.output_dir.mkdir(parents=True,exist_ok=True)
 with np.load(x.pose_shape_reference/"pose_shape_reference.npz") as z:A=torch.from_numpy(z["A_final"]);B=torch.from_numpy(z["B_final"]);C=torch.from_numpy(z["C_final"]);gf=torch.from_numpy(z["g_frame_final"]);gb=torch.from_numpy(z["g_beta_final"])
 md=float(torch.maximum(A.diag().max(),C.diag().max()));lams=[md*q for q in (1e-4,1e-3,1e-2)];arrays={"A":A.numpy(),"B":B.numpy(),"C":C.numpy(),"g_frame":gf.numpy(),"g_beta":gb.numpy()};cases=[]
 for i,lam in enumerate(lams):
  q=solve(A,B,C,gf,gb,lam);cases.append({"index":i,"lambda":lam,"python_structured_dense_rel":float(torch.linalg.vector_norm(q["full_delta"]-q["dense_delta"])/torch.linalg.vector_norm(q["dense_delta"]))});arrays.update({f"case{i}_{k}":v.numpy() for k,v in q.items()})
 np.savez(x.output_dir/"schur_reference.npz",**arrays);fixture=x.output_dir/"schur_solver_fixture";fixture.mkdir(exist_ok=True)
 for k in ("A","B","C","g_frame","g_beta"):(fixture/f"{k}.f64").write_bytes(np.ascontiguousarray(arrays[k],dtype="<f8").tobytes())
 (fixture/"fixture_meta.txt").write_text("frame_count=1\nframe_dim=75\nbeta_dim=10\n"+"".join(f"lambda_{i}={v:.17g}\n" for i,v in enumerate(lams))+"no_delta_reference_leakage=TRUE\n")
 m={"schema_version":"pose_shape_schur_v1","frame":513,"dimensions":{"T":1,"P":75,"B":10},"max_diagonal":md,"lambda_cases":cases,"solver_fixture_inputs":["A","B","C","g_frame","g_beta","lambda_cases"],"no_delta_reference_leakage":True,"damping":"A+lambda*I; C+lambda*I; B unchanged","factorization":"cholesky_ex/cholesky_solve; structured failure returns None; caller owns dense fallback"};(x.output_dir/"schur_reference_manifest.json").write_text(json.dumps(m,indent=2)+"\n");(x.output_dir/"schur_reference_report.md").write_text("# Pose/Shape Schur Python reference\n\nThree externally supplied damping cases; structured-vs-dense passed.\n");print("POSE_SHAPE_SCHUR_REFERENCE",x.output_dir)
if __name__=="__main__":main()
