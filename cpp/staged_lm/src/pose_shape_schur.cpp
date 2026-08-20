#include "staged_lm/pose_shape_schur.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
// The PoseShape solver eliminates frame variables, solves the beta Schur
// complement, then back-substitutes the complete LM step.
namespace staged_lm {namespace {using Clock=std::chrono::steady_clock;double ms(Clock::time_point s){return std::chrono::duration<double,std::milli>(Clock::now()-s).count();}template<class M>std::vector<double> vec(const M&m){auto evaluated=m.eval();return {evaluated.data(),evaluated.data()+evaluated.size()};}}
std::vector<double> schur_to_dense_hessian(const SchurSystem& s) {
  const int n=s.frame_count*s.frame_dim+s.beta_dim;
  const int bo=s.frame_count*s.frame_dim;
  std::vector<double> h(static_cast<std::size_t>(n)*n,0.0);
  for(int f=0;f<s.frame_count;++f){const int o=f*s.frame_dim;
    for(int r=0;r<s.frame_dim;++r){for(int c=0;c<s.frame_dim;++c)
      h[(o+r)*n+o+c]=s.a[(f*s.frame_dim+r)*s.frame_dim+c];
      for(int b=0;b<s.beta_dim;++b){const double v=s.b[(f*s.frame_dim+r)*s.beta_dim+b];
        h[(o+r)*n+bo+b]=v;h[(bo+b)*n+o+r]=v;}}}
  for(int r=0;r<s.beta_dim;++r)for(int c=0;c<s.beta_dim;++c)
    h[(bo+r)*n+bo+c]=s.c[r*s.beta_dim+c];
  return h;
}
std::vector<double> schur_gradient(const SchurSystem& s) {
  std::vector<double> g=s.g_frame;g.insert(g.end(),s.g_beta.begin(),s.g_beta.end());return g;
}
double schur_quadratic_form(const SchurSystem& s,const std::vector<double>& d) {
  const int beta_offset=s.frame_count*s.frame_dim;double v=0;
  for(int f=0;f<s.frame_count;++f){const int o=f*s.frame_dim;
    for(int r=0;r<s.frame_dim;++r){
      for(int c=0;c<s.frame_dim;++c)
        v+=d[o+r]*s.a[(f*s.frame_dim+r)*s.frame_dim+c]*d[o+c];
      for(int b=0;b<s.beta_dim;++b)
        v+=2.0*d[o+r]*s.b[(f*s.frame_dim+r)*s.beta_dim+b]*d[beta_offset+b];
    }}
  for(int r=0;r<s.beta_dim;++r)for(int c=0;c<s.beta_dim;++c)
    v+=d[beta_offset+r]*s.c[r*s.beta_dim+c]*d[beta_offset+c];
  return v;
}
bool solve_schur_dense_fallback(const SchurSystem&s,double damping,std::vector<double>&delta){
  auto h=schur_to_dense_hessian(s);auto g=schur_gradient(s);const int n=static_cast<int>(g.size());
  using Mat=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
  Eigen::Map<Mat> m(h.data(),n,n);m.diagonal().array()+=damping;
  Eigen::Map<const Eigen::VectorXd> gv(g.data(),n);Eigen::LLT<Mat> llt(m);
  Eigen::VectorXd x;
  if(llt.info()==Eigen::Success)x=llt.solve(-gv);
  else x=m.fullPivLu().solve(-gv);
  if(!x.allFinite())return false;delta.assign(x.data(),x.data()+n);return true;
}
PoseShapeSchurResult solve_pose_shape_schur(const SchurSystem&in,double lambda,bool diagnostic){auto total=Clock::now();PoseShapeSchurResult r;r.damping=lambda;int t=in.frame_count,p=in.frame_dim,b=in.beta_dim;if(t<1||p<1||b<1||in.a.size()!=size_t(t*p*p)||in.b.size()!=size_t(t*p*b)||in.c.size()!=size_t(b*b)||in.g_frame.size()!=size_t(t*p)||in.g_beta.size()!=size_t(b))return r;using Mat=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;using V=Eigen::VectorXd;std::vector<Eigen::LLT<Mat>> factors(t);std::vector<Mat> ys(t);std::vector<V> yg(t);r.frames.resize(t);auto s=Clock::now();Mat C=Eigen::Map<const Mat>(in.c.data(),b,b);C.diagonal().array()+=lambda;r.c_damped=vec(C);r.damping_ms=ms(s);for(int f=0;f<t;++f){Mat A=Eigen::Map<const Mat>(in.a.data()+f*p*p,p,p);A.diagonal().array()+=lambda;r.frames[f].a_damped=vec(A);s=Clock::now();factors[f].compute(A);r.a_factor_ms+=ms(s);if(factors[f].info()!=Eigen::Success)return r;r.frames[f].factor_method="llt";Mat B=Eigen::Map<const Mat>(in.b.data()+f*p*b,p,b);s=Clock::now();ys[f]=factors[f].solve(B);r.solve_y_ms+=ms(s);s=Clock::now();yg[f]=factors[f].solve(Eigen::Map<const V>(in.g_frame.data()+f*p,p));r.solve_g_ms+=ms(s);r.frames[f].y_matrix=vec(ys[f]);r.frames[f].y_gradient=vec(yg[f]);}s=Clock::now();Mat S=C;V rhs=-Eigen::Map<const V>(in.g_beta.data(),b);for(int f=0;f<t;++f){Mat B=Eigen::Map<const Mat>(in.b.data()+f*p*b,p,b);S.noalias()-=B.transpose()*ys[f];rhs.noalias()+=B.transpose()*yg[f];}r.schur=vec(S);r.schur_rhs=vec(rhs);r.build_schur_ms=ms(s);s=Clock::now();Eigen::LLT<Mat> sf(S);r.beta_solve_ms=ms(s);if(sf.info()!=Eigen::Success)return r;r.schur_factor_method="llt";s=Clock::now();V dz=sf.solve(rhs);r.beta_solve_ms+=ms(s);if(sf.info()!=Eigen::Success)return r;r.delta_beta=vec(dz);V sr=S*dz-rhs;r.schur_residual=vec(sr);s=Clock::now();r.full_delta.resize(t*p+b);for(int f=0;f<t;++f){Mat B=Eigen::Map<const Mat>(in.b.data()+f*p*b,p,b);V fr=-(Eigen::Map<const V>(in.g_frame.data()+f*p,p)+B*dz);V dx=factors[f].solve(fr);r.frames[f].frame_rhs=vec(fr);r.frames[f].delta=vec(dx);std::copy(dx.data(),dx.data()+p,r.full_delta.begin()+f*p);}std::copy(dz.data(),dz.data()+b,r.full_delta.begin()+t*p);r.back_substitution_ms=ms(s);int n=t*p+b;Mat H=Mat::Zero(n,n);V g(n);for(int f=0;f<t;++f){H.block(f*p,f*p,p,p)=Eigen::Map<const Mat>(r.frames[f].a_damped.data(),p,p);Mat B=Eigen::Map<const Mat>(in.b.data()+f*p*b,p,b);H.block(f*p,t*p,p,b)=B;H.block(t*p,f*p,b,p)=B.transpose();g.segment(f*p,p)=Eigen::Map<const V>(in.g_frame.data()+f*p,p);}H.block(t*p,t*p,b,b)=C;g.tail(b)=Eigen::Map<const V>(in.g_beta.data(),b);V delta=Eigen::Map<const V>(r.full_delta.data(),n);r.linear_residual=vec(H*delta+g);if(diagnostic){s=Clock::now();Eigen::LLT<Mat> df(H);if(df.info()==Eigen::Success)r.dense_delta=vec(df.solve(-g));else r.dense_delta=vec(H.fullPivLu().solve(-g));r.dense_solve_ms=ms(s);}r.success=std::isfinite(delta.norm());r.total_ms=ms(total);return r;}
}
