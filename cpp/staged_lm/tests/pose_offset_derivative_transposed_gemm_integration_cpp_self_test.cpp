#include "staged_lm/experimental/pose_offset_gemm.h"
#include "staged_lm/pose_forward.h"
#include "staged_lm/pose_jacobian.h"
#include "staged_lm/performance.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  for (int n : {1023, 927}) {
    staged_lm::PoseCameraInput camera;
    camera.selected_posedirs.resize(static_cast<std::size_t>(n) * 3 * 486);
    for (std::size_t i=0;i<camera.selected_posedirs.size();++i)
      camera.selected_posedirs[i]=std::sin(float(i)*0.0001F);
    staged_lm::promote_selected_posedirs_to_stable_owner(camera);
    const float* stable=staged_lm::selected_posedirs_view(camera).data();
    auto copy=camera;
    if(staged_lm::selected_posedirs_view(copy).data()!=stable)return 1;
    std::vector<float>d(486*66);for(std::size_t i=0;i<d.size();++i)d[i]=std::cos(float(i)*0.0003F);
    std::vector<float>a,b;staged_lm::experimental::PoseOffsetGemmTiming ta,tb;
    staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(
        staged_lm::selected_posedirs_view(copy),d,n,a,&ta);
    staged_lm::experimental::compute_pose_offset_derivative_transposed_eigen_gemm(
        staged_lm::selected_posedirs_view(copy),d,n,b,&tb);
    if(a!=b||a.size()!=static_cast<std::size_t>(n)*3*66)return 2;
    for(float x:b)if(!std::isfinite(x))return 3;
    if(tb.input_pack_ms!=0||tb.output_unpack_ms!=0)return 4;
  }
  staged_lm::reset_kernel_dispatch_counters();
  staged_lm::set_experimental_pose_offset_gemm(true);
  staged_lm::set_experimental_pose_offset_transposed_gemm(true);
  std::vector<float>p(3*486),d(486*66),y;
  staged_lm::compute_pose_offset_derivative_dispatched(p,d,1,y);
  const auto&c=staged_lm::kernel_dispatch_counters();
  if(c.pose_offset_transposed_calls!=1||c.pose_offset_optimized_calls!=0||
      c.pose_offset_scalar_calls!=0)return 5;
  std::cout<<"POSE_OFFSET_DERIVATIVE_TRANSPOSED_GEMM_INTEGRATION_CPP_SELF_TEST=PASS\n";
  return 0;
}
