#include "staged_lm/performance.h"
#include "staged_lm/pose_jacobian.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <vector>

#ifdef EIGEN_VECTORIZE
constexpr bool kEigenVectorize = true;
#else
constexpr bool kEigenVectorize = false;
#endif

int main() {
  using namespace staged_lm;
  if (kEigenVectorize != static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE) ||
      Eigen::nbThreads() != 1) return 1;
  const bool optimized = STAGED_LM_EXPECT_MARCH_NATIVE;
  set_experimental_pose_offset_gemm(optimized);
  set_experimental_selected_transform_gemm(optimized);
  reset_kernel_dispatch_counters();

  std::vector<float> posedirs(486 * 3, 0.001F);
  std::vector<float> feature(486 * 66, 0.002F), pose_output;
  compute_pose_offset_derivative_dispatched(posedirs, feature, 1, pose_output);
  std::vector<float> weights(55, 1.0F / 55.0F);
  std::vector<float> derivative(55 * 16 * 66, 0.003F);
  std::vector<float> transforms(16, 0.0F), vertices(3, 0.0F);
  std::vector<float> vertex_derivative(3 * 66, 0.0F);
  transforms[0] = transforms[5] = transforms[10] = transforms[15] = 1.0F;
  LbsDerivativeMaterializedResult lbs;
  compute_lbs_derivative_materialized(weights, derivative, transforms, vertices,
      vertex_derivative, 1, 66, lbs);
  const auto& counters = kernel_dispatch_counters();
  if (optimized) {
    if (counters.pose_offset_optimized_calls != 1 ||
        counters.lbs_selected_transform_optimized_calls != 1 ||
        counters.pose_offset_scalar_calls ||
        counters.lbs_selected_transform_scalar_calls) return 1;
  } else if (counters.pose_offset_scalar_calls != 1 ||
             counters.lbs_selected_transform_scalar_calls != 1 ||
             counters.pose_offset_optimized_calls ||
             counters.lbs_selected_transform_optimized_calls) return 1;
  for (float value : pose_output) if (!std::isfinite(value)) return 1;
  for (float value : lbs.skinned_derivatives) if (!std::isfinite(value)) return 1;
  std::cout << "STAGED_LM_OPTIMIZED_KERNEL_INTEGRATION_CPP_SELF_TEST=PASS\n";
  return 0;
}
