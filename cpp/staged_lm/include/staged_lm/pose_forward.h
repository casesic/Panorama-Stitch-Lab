#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace staged_lm {

constexpr int kPoseParameterDimension = 75;
constexpr int kPoseJointCount = 55;
constexpr int kPoseFeatureDimension = 486;

struct PoseModelInput {
  std::vector<float> joints;
  std::vector<std::int64_t> parents;
};

struct PoseCameraInput {
  std::string camera;
  std::vector<float> parameters;
  std::vector<float> left_hand_pose;
  std::vector<float> right_hand_pose;
  float scale = 1.0F;
  std::vector<float> selected_v_shaped;
  std::vector<float> selected_posedirs;
  std::shared_ptr<const std::vector<float>> stable_selected_posedirs;
  std::vector<float> selected_lbs_weights;
  std::vector<float> targets;
  int point_count = 0;
  int image_width = 0;
  int image_height = 0;
};

const std::vector<float>& selected_posedirs_view(const PoseCameraInput& input);
void promote_selected_posedirs_to_stable_owner(PoseCameraInput& input);

struct PoseCameraResult {
  std::vector<float> axis_angles;
  std::vector<float> rotations;
  std::vector<float> pose_feature;
  std::vector<float> pose_offsets;
  std::vector<float> v_posed;
  std::vector<float> relative_joints;
  std::vector<float> local_transforms;
  std::vector<float> global_transforms;
  std::vector<float> rest_transforms;
  std::vector<float> relative_transforms;
  std::vector<float> selected_transforms;
  std::vector<float> skinned_vertices;
  std::vector<float> vertices;
  std::vector<float> uv;
  std::vector<float> raw_residual;
  double rotations_ms = 0.0;
  double pose_offsets_ms = 0.0;
  double fk_ms = 0.0;
  double lbs_ms = 0.0;
  double erp_ms = 0.0;
  double total_value_forward_ms = 0.0;
};

PoseCameraResult evaluate_pose_camera_value(
    const PoseModelInput& model, const PoseCameraInput& input);

void compute_value_pose_offset_scalar(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& pose_feature,
    int point_count,
    std::vector<float>& output);
void set_experimental_value_pose_offset_gemv(bool enabled);

}  // namespace staged_lm
