#pragma once

#include "staged_lm/pose_jacobian.h"

#include <string>
#include <vector>

namespace staged_lm {

struct PoseNormalEquationConfig {
  float charbonnier_eps = 1.0F;
  float cam2_alpha_per_point = 0.0F;
  float cam4_alpha_per_point = 0.0F;
  float pose_anchor_scale = 1.0F;
};

struct PoseCameraNormalEquations {
  std::vector<float> point_weights;
  std::vector<float> row_scales;
  std::vector<float> weighted_residual;
  std::vector<float> weighted_jacobian_active;
  std::vector<double> hessian_active;
  std::vector<double> gradient_active;
  double irls_ms = 0.0;
  double row_scaling_ms = 0.0;
  double fp64_cast_ms = 0.0;
  double active_normal_equations_ms = 0.0;
};

struct PoseNormalEquationsResult {
  PoseCameraNormalEquations cam2;
  PoseCameraNormalEquations cam4;
  std::vector<double> hessian_data_frame;
  std::vector<double> gradient_data_frame;
  std::vector<float> pose_anchor_delta;
  std::vector<float> pose_anchor_residual;
  std::vector<double> hessian_pose_anchor;
  std::vector<double> gradient_pose_anchor;
  std::vector<double> hessian_frame;
  std::vector<double> gradient_frame;
  double scatter_ms = 0.0;
  double anchor_ms = 0.0;
  double total_normal_equations_ms = 0.0;
};

void scatter_pose_active_to_frame(
    const PoseCameraNormalEquations& active, const std::string& camera,
    std::vector<double>& hessian_frame, std::vector<double>& gradient_frame);

PoseNormalEquationsResult build_pose_normal_equations(
    const PoseCameraJacobianResult& cam2,
    const PoseCameraJacobianResult& cam4,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const PoseNormalEquationConfig& config);

}  // namespace staged_lm
