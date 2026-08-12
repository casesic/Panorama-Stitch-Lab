#pragma once

#include "staged_lm/pose_forward.h"

#include <vector>

namespace staged_lm {

constexpr int kPoseRotationParameterDimension = 66;
constexpr int kPoseActiveDimension = 69;

struct PoseCameraJacobianResult {
  PoseCameraResult value;
  std::vector<float> variable_right_jacobians;
  std::vector<float> d_rotation;
  std::vector<float> d_pose_feature;
  std::vector<float> d_pose_offsets;
  std::vector<float> d_v_posed;
  std::vector<float> d_local_transforms;
  std::vector<float> d_global_transforms;
  std::vector<float> d_relative_transforms;
  std::vector<float> d_selected_transforms;
  std::vector<float> d_skin_from_transform;
  std::vector<float> d_skin_from_vertex;
  std::vector<float> d_skinned_vertices;
  std::vector<float> d_vertices_rotpose;
  std::vector<float> projection_jacobian;
  std::vector<float> uv_rotpose_jacobian;
  std::vector<float> translation_jacobian;
  std::vector<float> active_jacobian;
  std::vector<float> frame_jacobian;
  double rotation_derivative_ms = 0.0;
  double pose_feature_derivative_ms = 0.0;
  double pose_offset_derivative_ms = 0.0;
  double fk_derivative_ms = 0.0;
  double lbs_derivative_ms = 0.0;
  double erp_jacobian_chain_ms = 0.0;
  double assemble_active_ms = 0.0;
  double total_jacobian_ms = 0.0;
};

PoseCameraJacobianResult evaluate_pose_camera_with_jacobian(
    const PoseModelInput& model, const PoseCameraInput& input);

}  // namespace staged_lm
