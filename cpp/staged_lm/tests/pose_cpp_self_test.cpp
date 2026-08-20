#include "staged_lm/pose_forward.h"
#include "staged_lm/pose_jacobian.h"
#include "staged_lm/pose_normal_equations.h"

#include <cmath>
#include <iostream>

int main() {
  staged_lm::PoseModelInput model;
  model.joints.assign(staged_lm::kPoseJointCount * 3, 0.0F);
  model.parents.resize(staged_lm::kPoseJointCount);
  model.parents[0] = -1;
  for (int joint = 1; joint < staged_lm::kPoseJointCount; ++joint) {
    model.parents[joint] = joint - 1;
    model.joints[joint * 3 + 1] = static_cast<float>(joint) * 0.01F;
  }
  staged_lm::PoseCameraInput input;
  input.camera = "cam2";
  input.parameters.assign(staged_lm::kPoseParameterDimension, 0.0F);
  input.left_hand_pose.assign(45, 0.0F);
  input.right_hand_pose.assign(45, 0.0F);
  input.scale = 1.0F;
  input.point_count = 1;
  input.selected_v_shaped = {0.1F, 0.2F, 2.0F};
  input.selected_posedirs.assign(staged_lm::kPoseFeatureDimension * 3, 0.0F);
  input.selected_lbs_weights.assign(staged_lm::kPoseJointCount, 0.0F);
  input.selected_lbs_weights[0] = 1.0F;
  input.targets = {2880.0F, 1440.0F};
  input.image_width = 5760;
  input.image_height = 2880;
  const auto result = staged_lm::evaluate_pose_camera_value(model, input);
  for (float value : result.vertices) {
    if (!std::isfinite(value)) {
      std::cerr << "POSE_VALUE_NONFINITE\n";
      return 1;
    }
  }
  for (int joint = 0; joint < staged_lm::kPoseJointCount; ++joint) {
    const float* local = result.local_transforms.data() + joint * 16;
    if (local[12] != 0.0F || local[13] != 0.0F || local[14] != 0.0F
        || local[15] != 1.0F) {
      std::cerr << "POSE_HOMOGENEOUS_ROW_FAILED\n";
      return 1;
    }
  }
  float weight_sum = 0.0F;
  for (float value : input.selected_lbs_weights) {
    weight_sum += value;
  }
  if (std::abs(weight_sum - 1.0F) > 1.0e-6F) {
    std::cerr << "POSE_LBS_WEIGHT_SUM_FAILED\n";
    return 1;
  }
  const auto jacobian = staged_lm::evaluate_pose_camera_with_jacobian(model, input);
  for (float value : jacobian.d_skinned_vertices) {
    if (!std::isfinite(value)) {
      std::cerr << "POSE_DERIVATIVE_NONFINITE\n";
      return 1;
    }
  }
  for (int feature = 0; feature < staged_lm::kPoseFeatureDimension; ++feature) {
    for (int root = 0; root < 3; ++root) {
      if (jacobian.d_pose_feature[
              feature * staged_lm::kPoseRotationParameterDimension + root] != 0.0F) {
        std::cerr << "POSE_ROOT_FEATURE_DERIVATIVE_NONZERO\n";
        return 1;
      }
    }
  }
  if (jacobian.active_jacobian.size()
          != static_cast<std::size_t>(input.point_count * 2 * 69)
      || jacobian.frame_jacobian.size()
          != static_cast<std::size_t>(input.point_count * 2 * 75)) {
    std::cerr << "POSE_JACOBIAN_SHAPE_FAILED\n";
    return 1;
  }
  for (int row = 0; row < input.point_count * 2; ++row) {
    for (int column = 6; column < 12; ++column) {
      if (jacobian.frame_jacobian[row * 75 + column] != 0.0F) {
        std::cerr << "POSE_CAM2_SCATTER_FAILED\n";
        return 1;
      }
    }
  }
  input.camera = "cam4";
  const auto cam4_jacobian = staged_lm::evaluate_pose_camera_with_jacobian(model, input);
  for (int row = 0; row < input.point_count * 2; ++row) {
    for (int column = 0; column < 6; ++column) {
      if (cam4_jacobian.frame_jacobian[row * 75 + column] != 0.0F) {
        std::cerr << "POSE_CAM4_SCATTER_FAILED\n";
        return 1;
      }
    }
    for (int body = 0; body < 63; ++body) {
      if (jacobian.frame_jacobian[row * 75 + 12 + body]
              != jacobian.active_jacobian[row * 69 + 6 + body]
          || cam4_jacobian.frame_jacobian[row * 75 + 12 + body]
              != cam4_jacobian.active_jacobian[row * 69 + 6 + body]) {
        std::cerr << "POSE_SHARED_BODY_SCATTER_FAILED\n";
        return 1;
      }
    }
  }
  staged_lm::PoseNormalEquationConfig normal_config;
  normal_config.charbonnier_eps = 1.0F;
  normal_config.cam2_alpha_per_point = 1.0F;
  normal_config.cam4_alpha_per_point = 1.0F;
  normal_config.pose_anchor_scale = 0.25F;
  std::vector<float> anchor(63, 0.0F);
  const auto normal = staged_lm::build_pose_normal_equations(
      jacobian, cam4_jacobian, input.parameters, anchor, normal_config);
  if (normal.hessian_frame.size() != 75 * 75 || normal.gradient_frame.size() != 75) {
    std::cerr << "POSE_NORMAL_EQUATION_SHAPE_FAILED\n";
    return 1;
  }
  for (float scale : normal.cam2.row_scales) {
    if (!std::isfinite(scale) || scale < 0.0F) {
      std::cerr << "POSE_IRLS_SCALE_FAILED\n";
      return 1;
    }
  }
  for (int row = 0; row < 75; ++row) {
    for (int column = 0; column < 75; ++column) {
      const double value = normal.hessian_frame[row * 75 + column];
      if (!std::isfinite(value)
          || std::abs(value - normal.hessian_frame[column * 75 + row]) > 1.0e-9) {
        std::cerr << "POSE_NORMAL_EQUATION_SYMMETRY_FAILED\n";
        return 1;
      }
      if ((row < 12 || column < 12)
          && normal.hessian_pose_anchor[row * 75 + column] != 0.0) {
        std::cerr << "POSE_ANCHOR_RIGID_BLOCK_FAILED\n";
        return 1;
      }
    }
    if (row < 12 && normal.gradient_pose_anchor[row] != 0.0) {
      std::cerr << "POSE_ANCHOR_RIGID_GRADIENT_FAILED\n";
      return 1;
    }
  }
  for (int rigid = 6; rigid < 12; ++rigid) {
    if (normal.cam2.gradient_active.size() != 69
        || normal.hessian_data_frame[rigid * 75 + rigid] !=
               normal.cam4.hessian_active[(rigid - 6) * 69 + rigid - 6]) {
      std::cerr << "POSE_CAM2_CAM4_NORMAL_SCATTER_FAILED\n";
      return 1;
    }
  }
  std::cout << "POSE_CPP_SELF_TEST_PASS\n";
  return 0;
}
