#include "staged_lm/densekp.h"
#include "staged_lm/pose_forward.h"
#include "staged_lm/pose_jacobian.h"
#include "staged_lm/erp.h"

#include <cmath>
#include <iostream>
#include <memory>

int main() {
  auto regressor = std::make_shared<staged_lm::DenseKPRegressor>();
  regressor->densekp_count = 3;
  regressor->support_count = 2;
  regressor->support_vertex_ids = {4, 9};
  const std::vector<float> dense_weights = {
      1.0F, 0.0F, 0.25F, 0.75F, 0.0F, 1.0F};
  staged_lm::build_densekp_sparse_representation(*regressor, dense_weights);
  staged_lm::DenseKPObservation observation;
  observation.regressor = regressor;
  observation.densekp_ids = {2, 1};
  observation.targets.assign(4, 0.0F);
  staged_lm::validate_densekp_observation(observation, 2);

  const std::vector<float> points = {1, 2, 3, 5, 6, 7};
  std::vector<float> output;
  staged_lm::apply_densekp_operator(observation, points, 3, 1, output);
  const std::vector<float> expected = {5, 6, 7, 4, 5, 6};
  if (output != expected) {
    std::cerr << "DENSEKP_POINT_OPERATOR_FAILED\n";
    return 1;
  }
  std::vector<float> sparse_output;

  // Two parameters per xyz component, checking R(dV/dq) layout and values.
  std::vector<float> jacobian(2 * 3 * 2);
  for (std::size_t index = 0; index < jacobian.size(); ++index) {
    jacobian[index] = static_cast<float>(index + 1);
  }
  staged_lm::apply_densekp_operator(observation, jacobian, 3, 2, output);
  for (int component = 0; component < 3; ++component) {
    for (int parameter = 0; parameter < 2; ++parameter) {
      const float expected_last = jacobian[(3 + component) * 2 + parameter];
      const float expected_mix = 0.25F * jacobian[component * 2 + parameter]
          + 0.75F * jacobian[(3 + component) * 2 + parameter];
      if (output[(component * 2) + parameter] != expected_last ||
          output[((3 + component) * 2) + parameter] != expected_mix) {
        std::cerr << "DENSEKP_JACOBIAN_OPERATOR_FAILED\n";
        return 1;
      }
    }
  }
  for (int parameter_count : {66, 76}) {
    std::vector<float> support_derivatives(2 * 3 * parameter_count);
    for (std::size_t index = 0; index < support_derivatives.size(); ++index)
      support_derivatives[index] =
          static_cast<float>((static_cast<int>(index) % 29) - 14) * 0.03125F;
    std::vector<float> dense_derivatives;
    staged_lm::apply_densekp_operator(
        observation, support_derivatives, 3, parameter_count, dense_derivatives);
    if (dense_derivatives.empty()) return 1;
  }

  staged_lm::PoseModelInput model;
  model.joints.assign(staged_lm::kPoseJointCount * 3, 0.0F);
  model.parents.resize(staged_lm::kPoseJointCount);
  model.parents[0] = -1;
  for (int joint = 1; joint < staged_lm::kPoseJointCount; ++joint)
    model.parents[joint] = joint - 1;
  staged_lm::PoseCameraInput input;
  input.camera = "cam2";
  input.parameters.assign(staged_lm::kPoseParameterDimension, 0.0F);
  input.left_hand_pose.assign(45, 0.0F);
  input.right_hand_pose.assign(45, 0.0F);
  input.scale = 1.0F;
  input.point_count = 2;
  input.selected_v_shaped = {1, 2, 10, 5, 6, 10};
  input.selected_posedirs.assign(
      staged_lm::kPoseFeatureDimension * input.point_count * 3, 0.0F);
  input.selected_lbs_weights.assign(
      input.point_count * staged_lm::kPoseJointCount, 0.0F);
  input.selected_lbs_weights[0] = 1.0F;
  input.selected_lbs_weights[staged_lm::kPoseJointCount] = 1.0F;
  input.image_width = 5760;
  input.image_height = 2880;
  input.densekp_observation = std::make_shared<staged_lm::DenseKPObservation>(
      observation);
  const auto value = staged_lm::evaluate_pose_camera_value(model, input);
  const auto pose_jacobian = staged_lm::evaluate_pose_camera_with_jacobian(model, input);
  if (value.vertices.size() != 2 * 3 || value.observation_points.size() != 2 * 3 ||
      value.raw_residual.size() != 2 * 2 ||
      pose_jacobian.d_vertices_rotpose.size() !=
          2 * 3 * staged_lm::kPoseRotationParameterDimension ||
      pose_jacobian.observation_d_vertices_rotpose.size() !=
          2 * 3 * staged_lm::kPoseRotationParameterDimension ||
      pose_jacobian.active_jacobian.size() != 2 * 2 * staged_lm::kPoseActiveDimension) {
    std::cerr << "DENSEKP_SUPPORT_OBSERVATION_SPLIT_FAILED\n";
    return 1;
  }
  std::cout << "DENSEKP_CPP_SELF_TEST_PASS\n";
  return 0;
}
