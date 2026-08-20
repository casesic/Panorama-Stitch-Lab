#pragma once

#include "staged_lm/pose_jacobian.h"
#include "staged_lm/pose_shape_forward.h"

namespace staged_lm {

constexpr int kPoseShapeBetaDimension = 10;
constexpr int kPoseShapeCombinedDimension =
    kPoseRotationParameterDimension + kPoseShapeBetaDimension;

struct PoseShapeCameraJacobianResult {
  PoseCameraJacobianResult pose;
  std::vector<float> d_beta_d_latent;
  std::vector<float> d_dynamic_v_shaped_beta;
  std::vector<float> d_dynamic_joints_beta;
  std::vector<float> d_relative_joints_beta;
  std::vector<float> d_v_posed_combined;
  std::vector<float> d_local_combined;
  std::vector<float> d_global_combined;
  std::vector<float> d_rest_combined;
  std::vector<float> d_relative_from_fk;
  std::vector<float> d_relative_from_rest;
  std::vector<float> d_relative_combined;
  std::vector<float> d_selected_transforms_combined;
  std::vector<float> d_skin_from_transform;
  std::vector<float> d_skin_from_vertex;
  std::vector<float> d_skinned_combined;
  std::vector<float> d_vertices_combined;
  std::vector<float> observation_d_vertices_combined;
  std::vector<float> d_vertices_rotpose;
  std::vector<float> d_vertices_beta;
  std::vector<float> observation_d_vertices_beta;
  std::vector<float> uv_combined;
  std::vector<float> beta_jacobian;
  std::vector<float> active_jacobian;
  double shape_derivative_ms = 0.0;
  double combined_fk_ms = 0.0;
  double combined_lbs_ms = 0.0;
  double erp_chain_ms = 0.0;
  double assemble_ms = 0.0;
  double total_ms = 0.0;
};

PoseShapeCameraJacobianResult evaluate_pose_shape_camera_with_jacobian(
    const PoseModelInput& dynamic_model,
    const PoseShapeCameraInput& input,
    const std::vector<float>& dynamic_v_shaped,
    const std::vector<float>& d_dynamic_joints_beta,
    const std::vector<float>& d_beta_d_latent);
PoseShapeCameraJacobianResult evaluate_pose_shape_camera_with_jacobian_from_value(
    const PoseModelInput& dynamic_model,const PoseShapeCameraInput& input,
    const std::vector<float>& dynamic_v_shaped,
    const std::vector<float>& d_dynamic_joints_beta,
    const std::vector<float>& d_beta_d_latent,PoseCameraResult value);

}  // namespace staged_lm
