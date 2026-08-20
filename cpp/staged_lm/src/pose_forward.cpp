#include "staged_lm/pose_forward.h"

#include "staged_lm/erp.h"
#include "staged_lm/so3.h"
#include "pose_kernels.h"

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace staged_lm {

// Pose value forward maps a 75-D frame state through rotations, pose offsets,
// FK, LBS, and ERP projection to two-dimensional residuals.
const std::vector<float>& selected_posedirs_view(const PoseCameraInput& input) {
  return input.stable_selected_posedirs ? *input.stable_selected_posedirs
                                       : input.selected_posedirs;
}

void promote_selected_posedirs_to_stable_owner(PoseCameraInput& input) {
  if (input.stable_selected_posedirs) return;
  auto owner = std::make_shared<std::vector<float>>(
      std::move(input.selected_posedirs));
  input.stable_selected_posedirs = std::move(owner);
}
namespace {

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

float matmul_entry(
    const float* left, const float* right, int inner_count,
    int left_stride, int right_stride) {
  float value = 0.0F;
  for (int inner = 0; inner < inner_count; ++inner) {
    value = std::fma(
        left[inner * left_stride], right[inner * right_stride], value);
  }
  return value;
}

void multiply4(const float* left, const float* right, float* output) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      output[row * 4 + column] = matmul_entry(
          left + row * 4, right + column, 4, 1, 4);
    }
  }
}

void validate_input(const PoseModelInput& model, const PoseCameraInput& input) {
  const int observation_count = pose_observation_count(input);
  if (input.parameters.size() != kPoseParameterDimension
      || model.joints.size() != kPoseJointCount * 3
      || model.parents.size() != kPoseJointCount
      || input.left_hand_pose.size() != 45
      || input.right_hand_pose.size() != 45
      || input.selected_v_shaped.size() != static_cast<std::size_t>(input.point_count * 3)
      || selected_posedirs_view(input).size()
          != static_cast<std::size_t>(kPoseFeatureDimension * input.point_count * 3)
      || input.selected_lbs_weights.size()
          != static_cast<std::size_t>(input.point_count * kPoseJointCount)
      || (!input.densekp_observation &&
          input.targets.size() != static_cast<std::size_t>(observation_count * 2))) {
    throw std::runtime_error("invalid Pose value input shape");
  }
  if (input.densekp_observation) {
    validate_densekp_observation(*input.densekp_observation, input.point_count);
  }
  if (model.parents[0] != -1) {
    throw std::runtime_error("Pose root parent must be -1");
  }
  for (int joint = 1; joint < kPoseJointCount; ++joint) {
    if (model.parents[joint] < 0 || model.parents[joint] >= joint) {
      throw std::runtime_error("invalid Pose parent traversal order");
    }
  }
}

}  // namespace

int pose_observation_count(const PoseCameraInput& input) {
  return input.densekp_observation
      ? static_cast<int>(input.densekp_observation->densekp_ids.size())
      : input.point_count;
}

PoseCameraResult evaluate_pose_camera_value(
    const PoseModelInput& model, const PoseCameraInput& input) {
  validate_input(model, input);
  const auto total_start = Clock::now();
  PoseCameraResult result;
  const int n = input.point_count;
  result.axis_angles.assign(kPoseJointCount * 3, 0.0F);
  const int camera_offset = input.camera == "cam2" ? 0
      : input.camera == "cam4" ? 6 : -1;
  if (camera_offset < 0) {
    throw std::runtime_error("unsupported Pose camera");
  }
  for (int xyz = 0; xyz < 3; ++xyz) {
    result.axis_angles[xyz] = input.parameters[camera_offset + xyz];
  }
  for (int component = 0; component < 63; ++component) {
    result.axis_angles[3 + component] = input.parameters[12 + component];
  }
  for (int component = 0; component < 45; ++component) {
    result.axis_angles[(25 * 3) + component] = input.left_hand_pose[component];
    result.axis_angles[(40 * 3) + component] = input.right_hand_pose[component];
  }

  auto start = Clock::now();
  result.rotations.resize(kPoseJointCount * 9);
  for (int joint = 0; joint < kPoseJointCount; ++joint) {
    const Eigen::Vector3f axis_angle(
        result.axis_angles[joint * 3], result.axis_angles[joint * 3 + 1],
        result.axis_angles[joint * 3 + 2]);
    const Eigen::Matrix3f rotation = axis_angle_to_matrix(axis_angle);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        result.rotations[joint * 9 + row * 3 + column] = rotation(row, column);
      }
    }
  }
  result.rotations_ms = elapsed_ms(start);

  start = Clock::now();
  result.pose_feature.resize(kPoseFeatureDimension);
  for (int joint = 1; joint < kPoseJointCount; ++joint) {
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        const int feature = (joint - 1) * 9 + row * 3 + column;
        result.pose_feature[feature] = result.rotations[joint * 9 + row * 3 + column]
            - (row == column ? 1.0F : 0.0F);
      }
    }
  }
  compute_value_pose_offset(
      selected_posedirs_view(input), result.pose_feature, n, result.pose_offsets);
  result.v_posed.resize(n * 3);
  for (int vertex = 0; vertex < n; ++vertex) {
    for (int xyz = 0; xyz < 3; ++xyz) {
      result.v_posed[vertex * 3 + xyz] =
          input.selected_v_shaped[vertex * 3 + xyz] + result.pose_offsets[vertex * 3 + xyz];
    }
  }
  result.pose_offsets_ms = elapsed_ms(start);

  start = Clock::now();
  result.relative_joints = model.joints;
  for (int joint = 1; joint < kPoseJointCount; ++joint) {
    const int parent = static_cast<int>(model.parents[joint]);
    for (int xyz = 0; xyz < 3; ++xyz) {
      result.relative_joints[joint * 3 + xyz] =
          model.joints[joint * 3 + xyz] - model.joints[parent * 3 + xyz];
    }
  }
  result.local_transforms.assign(kPoseJointCount * 16, 0.0F);
  for (int joint = 0; joint < kPoseJointCount; ++joint) {
    float* local = result.local_transforms.data() + joint * 16;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        local[row * 4 + column] = result.rotations[joint * 9 + row * 3 + column];
      }
      local[row * 4 + 3] = result.relative_joints[joint * 3 + row];
    }
    local[15] = 1.0F;
  }
  result.global_transforms.resize(kPoseJointCount * 16);
  for (int index = 0; index < 16; ++index) {
    result.global_transforms[index] = result.local_transforms[index];
  }
  for (int joint = 1; joint < kPoseJointCount; ++joint) {
    multiply4(
        result.global_transforms.data() + model.parents[joint] * 16,
        result.local_transforms.data() + joint * 16,
        result.global_transforms.data() + joint * 16);
  }
  result.relative_transforms = result.global_transforms;
  result.rest_transforms.assign(kPoseJointCount * 16, 0.0F);
  for (int joint = 0; joint < kPoseJointCount; ++joint) {
    float* rest = result.rest_transforms.data() + joint * 16;
    rest[0] = rest[5] = rest[10] = rest[15] = 1.0F;
    rest[3] = -model.joints[joint * 3];
    rest[7] = -model.joints[joint * 3 + 1];
    rest[11] = -model.joints[joint * 3 + 2];
    const float* global = result.global_transforms.data() + joint * 16;
    float transformed_joint[4] = {};
    for (int row = 0; row < 4; ++row) {
      float value = 0.0F;
      for (int xyz = 0; xyz < 3; ++xyz) {
        value = std::fma(global[row * 4 + xyz], model.joints[joint * 3 + xyz], value);
      }
      transformed_joint[row] = value;
    }
    for (int row = 0; row < 4; ++row) {
      result.relative_transforms[joint * 16 + row * 4 + 3] =
          global[row * 4 + 3] - transformed_joint[row];
    }
  }
  result.fk_ms = elapsed_ms(start);

  start = Clock::now();
  result.selected_transforms.resize(n * 16);
  result.skinned_vertices.resize(n * 3);
  for (int vertex = 0; vertex < n; ++vertex) {
    for (int element = 0; element < 16; ++element) {
      float value = 0.0F;
      for (int joint = 0; joint < kPoseJointCount; ++joint) {
        value = std::fma(
            input.selected_lbs_weights[vertex * kPoseJointCount + joint],
            result.relative_transforms[joint * 16 + element], value);
      }
      result.selected_transforms[vertex * 16 + element] = value;
    }
    const float homogeneous[4] = {
        result.v_posed[vertex * 3], result.v_posed[vertex * 3 + 1],
        result.v_posed[vertex * 3 + 2], 1.0F,
    };
    for (int xyz = 0; xyz < 3; ++xyz) {
      float value = 0.0F;
      for (int component = 0; component < 4; ++component) {
        value = std::fma(
            result.selected_transforms[vertex * 16 + xyz * 4 + component],
            homogeneous[component], value);
      }
      result.skinned_vertices[vertex * 3 + xyz] = value;
    }
  }
  result.vertices.resize(n * 3);
  for (int vertex = 0; vertex < n; ++vertex) {
    for (int xyz = 0; xyz < 3; ++xyz) {
      result.vertices[vertex * 3 + xyz] =
          result.skinned_vertices[vertex * 3 + xyz] * input.scale
          + input.parameters[camera_offset + 3 + xyz];
    }
  }
  result.lbs_ms = elapsed_ms(start);

  start = Clock::now();
  const std::vector<float>* targets = &input.targets;
  if (input.densekp_observation) {
    apply_densekp_operator(
        *input.densekp_observation, result.vertices, 3, 1,
        result.observation_points);
    targets = &input.densekp_observation->targets;
  } else {
    result.observation_points = result.vertices;
  }
  const int observation_count = pose_observation_count(input);
  result.uv.resize(observation_count * 2);
  result.raw_residual.resize(observation_count * 2);
  for (int vertex = 0; vertex < observation_count; ++vertex) {
    const Eigen::Vector3f point(
        result.observation_points[vertex * 3],
        result.observation_points[vertex * 3 + 1],
        result.observation_points[vertex * 3 + 2]);
    const Eigen::Vector2f uv = project_unified_erp(
        point, input.image_width, input.image_height);
    for (int component = 0; component < 2; ++component) {
      result.uv[vertex * 2 + component] = uv(component);
      result.raw_residual[vertex * 2 + component] =
          uv(component) - (*targets)[vertex * 2 + component];
    }
  }
  result.erp_ms = elapsed_ms(start);
  result.total_value_forward_ms = elapsed_ms(total_start);
  return result;
}

}  // namespace staged_lm
