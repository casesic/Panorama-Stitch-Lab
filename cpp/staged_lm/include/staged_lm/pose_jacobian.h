#pragma once

#include "staged_lm/pose_forward.h"
#include <vector>

namespace staged_lm {

constexpr int kPoseRotationParameterDimension = 66;
constexpr int kPoseActiveDimension = 69;

struct LbsDerivativeMaterializedResult {
  std::vector<float> selected_transform_derivatives;
  std::vector<float> transform_term;
  std::vector<float> vertex_term;
  std::vector<float> skinned_derivatives;
  double selected_transform_ms = 0.0;
  double downstream_ms = 0.0;
};

void compute_lbs_derivative_from_selected_transforms(
    const std::vector<float>& selected_transform_derivatives,
    const std::vector<float>& selected_transforms,
    const std::vector<float>& posed_vertices,
    const std::vector<float>& posed_vertex_derivatives,
    int point_count, int parameter_count,
    std::vector<float>& transform_term,
    std::vector<float>& vertex_term,
    std::vector<float>& skinned_derivatives);

void compute_lbs_derivative_materialized(
    const std::vector<float>& weights,
    const std::vector<float>& relative_transform_derivatives,
    const std::vector<float>& selected_transforms,
    const std::vector<float>& posed_vertices,
    const std::vector<float>& posed_vertex_derivatives,
    int point_count, int parameter_count,
    LbsDerivativeMaterializedResult& output);

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
  std::vector<float> observation_d_vertices_rotpose;
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
  double cuda_allocation_ms = 0.0;
  double cuda_memcpy_ms = 0.0;
  double cuda_h2d_ms = 0.0;
  double cuda_d2h_ms = 0.0;
  double cuda_kernel_ms = 0.0;
};

PoseCameraJacobianResult evaluate_pose_camera_with_jacobian(
    const PoseModelInput& model, const PoseCameraInput& input);
PoseCameraJacobianResult evaluate_pose_camera_with_jacobian_from_value(
    const PoseModelInput& model, const PoseCameraInput& input,
    PoseCameraResult value);

}  // namespace staged_lm
