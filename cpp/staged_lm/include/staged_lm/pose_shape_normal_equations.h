#pragma once
#include "staged_lm/pose_normal_equations.h"
#include "staged_lm/pose_shape_jacobian.h"
namespace staged_lm
{
  struct PoseShapeNormalEquationConfig : PoseNormalEquationConfig
  {
    float shape_anchor_scale = 1.0F;
  };
  struct PoseShapeCameraNormalEquations
  {
    PoseCameraNormalEquations active;
    std::vector<float> weighted_jacobian_beta;
    std::vector<double> h_active_beta, h_beta_beta, g_beta;
    double beta_scaling_ms = 0, fp64_beta_cast_ms = 0, h_aa_ms = 0, h_ab_ms = 0, h_bb_ms = 0, gradient_ms = 0;
  };
  struct PoseShapeNormalEquationsResult
  {
    PoseShapeCameraNormalEquations cam2, cam4;
    std::vector<double> a_data, b_data, c_data, g_frame_data, g_beta_data;
    std::vector<double> a_pose_prior, g_pose_prior, c_shape_prior, g_shape_prior;
    std::vector<double> a_final, b_final, c_final, g_frame_final, g_beta_final;
    std::vector<double> h_dense_structured, g_dense_structured;
    double scatter_accumulate_ms = 0, pose_prior_ms = 0, shape_prior_ms = 0, total_ms = 0;
  };
  PoseShapeNormalEquationsResult build_pose_shape_normal_equations(
      const PoseShapeCameraJacobianResult &, const PoseShapeCameraJacobianResult &,
      const std::vector<float> &parameters, const std::vector<float> &body_pose_anchor,
      const std::vector<float> &beta_anchor, const PoseShapeNormalEquationConfig &);
}
