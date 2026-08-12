#include "staged_lm/pose_shape_normal_equations.h"
#include <cmath>
#include <iostream>
int main() {
  using namespace staged_lm;
  PoseShapeCameraJacobianResult cam2, cam4;
  for (auto* camera : {&cam2, &cam4}) {
    camera->pose.value.raw_residual = {1, 2};
    camera->pose.active_jacobian.assign(2 * 69, 0);
    camera->beta_jacobian.assign(20, 0);
    camera->pose.active_jacobian[0] = 1;
    camera->pose.active_jacobian[69 + 6] = 2;
    camera->beta_jacobian[0] = 3;
    camera->beta_jacobian[10 + 1] = 4;
  }
  std::vector<float> parameters(85, 0), pose_anchor(63, 0), beta_anchor(10, 0);
  PoseShapeNormalEquationConfig config;
  config.cam2_alpha_per_point = config.cam4_alpha_per_point = 1;
  config.pose_anchor_scale = 2;
  config.shape_anchor_scale = 3;
  const auto result = build_pose_shape_normal_equations(
      cam2, cam4, parameters, pose_anchor, beta_anchor, config);
  if (result.a_final.size() != 75 * 75 || result.b_final.size() != 75 * 10
      || result.c_final.size() != 100 || result.h_dense_structured.size() != 85 * 85) return 1;
  if (result.b_data[0] == 0 || result.b_data[6 * 10] == 0
      || result.b_data[12 * 10 + 1] == 0) return 2;
  if (result.c_data[0] == 0 || result.g_beta_data.size() != 10) return 3;
  for (double value : result.h_dense_structured) if (!std::isfinite(value)) return 4;
  for (int i = 0; i < 85; ++i) for (int j = 0; j < 85; ++j)
    if (std::abs(result.h_dense_structured[i * 85 + j]
                 - result.h_dense_structured[j * 85 + i]) > 1e-12) return 5;
  std::cout << "POSE_SHAPE_NE_CPP_SELF_TEST=PASS\n";
  return 0;
}
