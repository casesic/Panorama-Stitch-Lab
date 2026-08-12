#include "staged_lm/pose_shape_jacobian.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace staged_lm {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point s) {
  return std::chrono::duration<double, std::milli>(Clock::now() - s).count();
}
std::size_t ix4(int joint, int element, int q, int width) {
  return (static_cast<std::size_t>(joint) * 16 + element) * width + q;
}
float product_derivative(const float* dl, const float* r, const float* l,
                         const float* dr, int row, int col, int q, int width) {
  float a = 0.0F, b = 0.0F;
  for (int k = 0; k < 4; ++k) {
    a = std::fma(dl[(row * 4 + k) * width + q], r[k * 4 + col], a);
    b = std::fma(l[row * 4 + k], dr[(k * 4 + col) * width + q], b);
  }
  return a + b;
}
}  // namespace

PoseShapeCameraJacobianResult evaluate_pose_shape_camera_with_jacobian(
    const PoseModelInput& model, const PoseShapeCameraInput& input,
    const std::vector<float>& dynamic_v_shaped,
    const std::vector<float>& d_joints_beta,
    const std::vector<float>& d_beta) {
  const auto total = Clock::now();
  if (d_beta.size() != kPoseShapeBetaDimension ||
      d_joints_beta.size() != kPoseJointCount * 3 * kPoseShapeBetaDimension) {
    throw std::runtime_error("pose/shape derivative input mismatch");
  }
  PoseShapeCameraJacobianResult out;
  PoseCameraInput pi = input.pose;
  pi.selected_v_shaped = dynamic_v_shaped;
  out.pose = evaluate_pose_camera_with_jacobian(model, pi);
  out.d_beta_d_latent = d_beta;
  const int n = pi.point_count, p = kPoseRotationParameterDimension;
  const int b = kPoseShapeBetaDimension, qn = kPoseShapeCombinedDimension;

  auto start = Clock::now();
  out.d_dynamic_joints_beta = d_joints_beta;
  out.d_dynamic_v_shaped_beta.resize(static_cast<std::size_t>(n) * 3 * b);
  for (int i = 0; i < n * 3; ++i)
    for (int z = 0; z < b; ++z)
      out.d_dynamic_v_shaped_beta[i * b + z] =
          input.selected_beta_shapedirs[i * b + z] * d_beta[z];
  out.d_relative_joints_beta = d_joints_beta;
  for (int j = 1; j < kPoseJointCount; ++j) {
    const int parent = static_cast<int>(model.parents[j]);
    for (int xyz = 0; xyz < 3; ++xyz)
      for (int z = 0; z < b; ++z)
        out.d_relative_joints_beta[(j * 3 + xyz) * b + z] -=
            d_joints_beta[(parent * 3 + xyz) * b + z];
  }
  out.d_v_posed_combined.assign(static_cast<std::size_t>(n) * 3 * qn, 0.0F);
  for (int i = 0; i < n * 3; ++i) {
    for (int q = 0; q < p; ++q)
      out.d_v_posed_combined[i * qn + q] = out.pose.d_v_posed[i * p + q];
    for (int z = 0; z < b; ++z)
      out.d_v_posed_combined[i * qn + p + z] =
          out.d_dynamic_v_shaped_beta[i * b + z];
  }
  out.shape_derivative_ms = ms(start);

  start = Clock::now();
  const std::size_t tc = kPoseJointCount * 16 * qn;
  out.d_local_combined.assign(tc, 0.0F);
  for (int j = 0; j < kPoseJointCount; ++j) {
    for (int e = 0; e < 16; ++e)
      for (int q = 0; q < p; ++q)
        out.d_local_combined[ix4(j,e,q,qn)] = out.pose.d_local_transforms[ix4(j,e,q,p)];
    for (int xyz = 0; xyz < 3; ++xyz)
      for (int z = 0; z < b; ++z)
        out.d_local_combined[ix4(j,xyz*4+3,p+z,qn)] =
            out.d_relative_joints_beta[(j*3+xyz)*b+z];
  }
  out.d_global_combined.assign(tc, 0.0F);
  for (int e = 0; e < 16; ++e)
    for (int q = 0; q < qn; ++q)
      out.d_global_combined[e*qn+q] = out.d_local_combined[e*qn+q];
  for (int j = 1; j < kPoseJointCount; ++j) {
    int parent = static_cast<int>(model.parents[j]);
    const float* pg = out.pose.value.global_transforms.data()+parent*16;
    const float* local = out.pose.value.local_transforms.data()+j*16;
    const float* dpg = out.d_global_combined.data()+parent*16*qn;
    const float* dl = out.d_local_combined.data()+j*16*qn;
    float* dg = out.d_global_combined.data()+j*16*qn;
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)for(int q=0;q<qn;++q)
      dg[(r*4+c)*qn+q]=product_derivative(dpg,local,pg,dl,r,c,q,qn);
  }
  out.d_rest_combined.assign(tc,0.0F);
  for(int j=0;j<kPoseJointCount;++j)for(int xyz=0;xyz<3;++xyz)for(int z=0;z<b;++z)
    out.d_rest_combined[ix4(j,xyz*4+3,p+z,qn)]=-d_joints_beta[(j*3+xyz)*b+z];
  out.d_relative_from_fk.assign(tc,0.0F);
  out.d_relative_from_rest.assign(tc,0.0F);
  out.d_relative_combined.assign(tc,0.0F);
  for(int j=0;j<kPoseJointCount;++j){
    const float* dg=out.d_global_combined.data()+j*16*qn;
    const float* g=out.pose.value.global_transforms.data()+j*16;
    const float* rest=out.pose.value.rest_transforms.data()+j*16;
    const float* dr=out.d_rest_combined.data()+j*16*qn;
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)for(int q=0;q<qn;++q){
      float a=0,bv=0;for(int k=0;k<4;++k){a=std::fma(dg[(r*4+k)*qn+q],rest[k*4+c],a);bv=std::fma(g[r*4+k],dr[(k*4+c)*qn+q],bv);}
      auto ix=ix4(j,r*4+c,q,qn);out.d_relative_from_fk[ix]=a;out.d_relative_from_rest[ix]=bv;out.d_relative_combined[ix]=a+bv;
    }
  }
  out.combined_fk_ms=ms(start);

  start=Clock::now();
  out.d_selected_transforms_combined.resize(static_cast<std::size_t>(n)*16*qn);
  for(int i=0;i<n;++i)for(int e=0;e<16;++e)for(int q=0;q<qn;++q){float v=0;for(int j=0;j<kPoseJointCount;++j)v=std::fma(pi.selected_lbs_weights[i*kPoseJointCount+j],out.d_relative_combined[ix4(j,e,q,qn)],v);out.d_selected_transforms_combined[(static_cast<std::size_t>(i)*16+e)*qn+q]=v;}
  const std::size_t vc=static_cast<std::size_t>(n)*3*qn;
  out.d_skin_from_transform.resize(vc);out.d_skin_from_vertex.resize(vc);out.d_skinned_combined.resize(vc);out.d_vertices_combined.resize(vc);
  for(int i=0;i<n;++i){float h[4]={out.pose.value.v_posed[i*3],out.pose.value.v_posed[i*3+1],out.pose.value.v_posed[i*3+2],1};for(int xyz=0;xyz<3;++xyz)for(int q=0;q<qn;++q){float a=0,bv=0;for(int k=0;k<4;++k){a=std::fma(out.d_selected_transforms_combined[((static_cast<std::size_t>(i)*16+xyz*4+k)*qn)+q],h[k],a);float dh=k<3?out.d_v_posed_combined[((static_cast<std::size_t>(i)*3+k)*qn)+q]:0;bv=std::fma(out.pose.value.selected_transforms[i*16+xyz*4+k],dh,bv);}auto ix=((static_cast<std::size_t>(i)*3+xyz)*qn)+q;out.d_skin_from_transform[ix]=a;out.d_skin_from_vertex[ix]=bv;out.d_skinned_combined[ix]=a+bv;out.d_vertices_combined[ix]=(a+bv)*pi.scale;}}
  out.d_vertices_rotpose.resize(static_cast<std::size_t>(n)*3*p);out.d_vertices_beta.resize(static_cast<std::size_t>(n)*3*b);
  for(int i=0;i<n*3;++i){for(int q=0;q<p;++q)out.d_vertices_rotpose[i*p+q]=out.d_vertices_combined[i*qn+q];for(int z=0;z<b;++z)out.d_vertices_beta[i*b+z]=out.d_vertices_combined[i*qn+p+z];}
  out.combined_lbs_ms=ms(start);

  start=Clock::now();out.uv_combined.resize(static_cast<std::size_t>(n)*2*qn);out.beta_jacobian.resize(static_cast<std::size_t>(n)*2*b);
  for(int i=0;i<n;++i)for(int uv=0;uv<2;++uv)for(int q=0;q<qn;++q){float v=0;for(int xyz=0;xyz<3;++xyz)v=std::fma(out.pose.projection_jacobian[i*6+uv*3+xyz],out.d_vertices_combined[((static_cast<std::size_t>(i)*3+xyz)*qn)+q],v);out.uv_combined[((static_cast<std::size_t>(i)*2+uv)*qn)+q]=v;if(q>=p)out.beta_jacobian[((static_cast<std::size_t>(i)*2+uv)*b)+q-p]=v;}
  out.erp_chain_ms=ms(start);
  start=Clock::now();out.active_jacobian=out.pose.active_jacobian;out.assemble_ms=ms(start);out.total_ms=ms(total);return out;
}
}  // namespace staged_lm
