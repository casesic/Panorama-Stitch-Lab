#include "staged_lm/pose_jacobian.h"

#include "staged_lm/erp.h"
#include "staged_lm/so3.h"

#include <Eigen/Core>

#include <chrono>
#include <cmath>

namespace staged_lm {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

float derivative_product_entry(
    const float* d_left, const float* right,
    const float* left, const float* d_right,
    int row, int column, int parameter) {
  float first = 0.0F;
  float second = 0.0F;
  for (int inner = 0; inner < 4; ++inner) {
    first = std::fma(
        d_left[(row * 4 + inner) * kPoseRotationParameterDimension + parameter],
        right[inner * 4 + column], first);
    second = std::fma(
        left[row * 4 + inner],
        d_right[(inner * 4 + column) * kPoseRotationParameterDimension + parameter],
        second);
  }
  return first + second;
}

}  // namespace

PoseCameraJacobianResult evaluate_pose_camera_with_jacobian(
    const PoseModelInput& model, const PoseCameraInput& input) {
  const auto total_start = Clock::now();
  PoseCameraJacobianResult result;
  result.value = evaluate_pose_camera_value(model, input);
  const int n = input.point_count;

  auto start = Clock::now();
  result.variable_right_jacobians.resize(22 * 9);
  result.d_rotation.assign(
      kPoseJointCount * 9 * kPoseRotationParameterDimension, 0.0F);
  for (int joint = 0; joint < 22; ++joint) {
    const Eigen::Vector3f phi(
        result.value.axis_angles[joint * 3],
        result.value.axis_angles[joint * 3 + 1],
        result.value.axis_angles[joint * 3 + 2]);
    const Eigen::Matrix3f right = so3_right_jacobian(phi);
    Eigen::Matrix3f rotation;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        result.variable_right_jacobians[joint * 9 + row * 3 + column] =
            right(row, column);
        rotation(row, column) = result.value.rotations[joint * 9 + row * 3 + column];
      }
    }
    const int parameter_start = joint * 3;
    for (int component = 0; component < 3; ++component) {
      const Eigen::Matrix3f basis = skew(right.col(component));
      Eigen::Matrix3f derivative;
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          float value = 0.0F;
          for (int inner = 0; inner < 3; ++inner) {
            value = std::fma(rotation(row, inner), basis(inner, column), value);
          }
          derivative(row, column) = value;
          result.d_rotation[
              ((joint * 3 + row) * 3 + column) * kPoseRotationParameterDimension
              + parameter_start + component] = value;
        }
      }
    }
  }
  result.rotation_derivative_ms = elapsed_ms(start);

  start = Clock::now();
  result.d_pose_feature.resize(kPoseFeatureDimension * kPoseRotationParameterDimension);
  for (int feature = 0; feature < kPoseFeatureDimension; ++feature) {
    const int joint = 1 + feature / 9;
    const int matrix_element = feature % 9;
    for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
      result.d_pose_feature[feature * kPoseRotationParameterDimension + parameter] =
          result.d_rotation[(joint * 9 + matrix_element)
                            * kPoseRotationParameterDimension + parameter];
    }
  }
  result.pose_feature_derivative_ms = elapsed_ms(start);

  start = Clock::now();
  result.d_pose_offsets.resize(n * 3 * kPoseRotationParameterDimension);
  for (int vertex = 0; vertex < n; ++vertex) {
    for (int xyz = 0; xyz < 3; ++xyz) {
      for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
        float derivative = 0.0F;
        for (int feature = 0; feature < kPoseFeatureDimension; ++feature) {
          const std::size_t direction =
              (static_cast<std::size_t>(feature) * n + vertex) * 3 + xyz;
          derivative = std::fma(
              result.d_pose_feature[feature * kPoseRotationParameterDimension + parameter],
              input.selected_posedirs[direction], derivative);
        }
        result.d_pose_offsets[(vertex * 3 + xyz) * kPoseRotationParameterDimension
                              + parameter] = derivative;
      }
    }
  }
  result.d_v_posed = result.d_pose_offsets;
  result.pose_offset_derivative_ms = elapsed_ms(start);

  start = Clock::now();
  const std::size_t transform_derivative_count =
      kPoseJointCount * 16 * kPoseRotationParameterDimension;
  result.d_local_transforms.assign(transform_derivative_count, 0.0F);
  for (int joint = 0; joint < kPoseJointCount; ++joint) {
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
          result.d_local_transforms[
              ((joint * 16 + row * 4 + column) * kPoseRotationParameterDimension)
              + parameter] = result.d_rotation[
                  ((joint * 9 + row * 3 + column) * kPoseRotationParameterDimension)
                  + parameter];
        }
      }
    }
  }
  result.d_global_transforms.assign(transform_derivative_count, 0.0F);
  for (int element = 0; element < 16; ++element) {
    for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
      result.d_global_transforms[element * kPoseRotationParameterDimension + parameter] =
          result.d_local_transforms[element * kPoseRotationParameterDimension + parameter];
    }
  }
  for (int joint = 1; joint < kPoseJointCount; ++joint) {
    const int parent = static_cast<int>(model.parents[joint]);
    const float* parent_global = result.value.global_transforms.data() + parent * 16;
    const float* local = result.value.local_transforms.data() + joint * 16;
    const float* d_parent = result.d_global_transforms.data()
        + parent * 16 * kPoseRotationParameterDimension;
    const float* d_local = result.d_local_transforms.data()
        + joint * 16 * kPoseRotationParameterDimension;
    float* d_global = result.d_global_transforms.data()
        + joint * 16 * kPoseRotationParameterDimension;
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
          d_global[(row * 4 + column) * kPoseRotationParameterDimension + parameter] =
              derivative_product_entry(
                  d_parent, local, parent_global, d_local, row, column, parameter);
        }
      }
    }
  }
  float rest[kPoseJointCount * 16] = {};
  for (int joint = 0; joint < kPoseJointCount; ++joint) {
    float* current = rest + joint * 16;
    current[0] = current[5] = current[10] = current[15] = 1.0F;
    current[3] = -model.joints[joint * 3];
    current[7] = -model.joints[joint * 3 + 1];
    current[11] = -model.joints[joint * 3 + 2];
  }
  result.d_relative_transforms.resize(transform_derivative_count);
  for (int joint = 0; joint < kPoseJointCount; ++joint) {
    const float* d_global = result.d_global_transforms.data()
        + joint * 16 * kPoseRotationParameterDimension;
    const float* current_rest = rest + joint * 16;
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
          float value = 0.0F;
          for (int inner = 0; inner < 4; ++inner) {
            value = std::fma(
                d_global[(row * 4 + inner) * kPoseRotationParameterDimension + parameter],
                current_rest[inner * 4 + column], value);
          }
          result.d_relative_transforms[
              ((joint * 16 + row * 4 + column) * kPoseRotationParameterDimension)
              + parameter] = value;
        }
      }
    }
  }
  result.fk_derivative_ms = elapsed_ms(start);

  start = Clock::now();
  result.d_selected_transforms.resize(
      static_cast<std::size_t>(n) * 16 * kPoseRotationParameterDimension);
  for (int vertex = 0; vertex < n; ++vertex) {
    for (int element = 0; element < 16; ++element) {
      for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
        float value = 0.0F;
        for (int joint = 0; joint < kPoseJointCount; ++joint) {
          value = std::fma(
              input.selected_lbs_weights[vertex * kPoseJointCount + joint],
              result.d_relative_transforms[
                  ((joint * 16 + element) * kPoseRotationParameterDimension) + parameter],
              value);
        }
        result.d_selected_transforms[
            ((vertex * 16 + element) * kPoseRotationParameterDimension) + parameter] = value;
      }
    }
  }
  result.d_skin_from_transform.resize(
      static_cast<std::size_t>(n) * 3 * kPoseRotationParameterDimension);
  result.d_skin_from_vertex.resize(result.d_skin_from_transform.size());
  result.d_skinned_vertices.resize(result.d_skin_from_transform.size());
  for (int vertex = 0; vertex < n; ++vertex) {
    const float homogeneous[4] = {
        result.value.v_posed[vertex * 3], result.value.v_posed[vertex * 3 + 1],
        result.value.v_posed[vertex * 3 + 2], 1.0F,
    };
    for (int xyz = 0; xyz < 3; ++xyz) {
      for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
        float transform_term = 0.0F;
        float vertex_term = 0.0F;
        for (int component = 0; component < 4; ++component) {
          transform_term = std::fma(
              result.d_selected_transforms[
                  ((vertex * 16 + xyz * 4 + component)
                   * kPoseRotationParameterDimension) + parameter],
              homogeneous[component], transform_term);
          const float d_vertex = component < 3
              ? result.d_v_posed[
                  ((vertex * 3 + component) * kPoseRotationParameterDimension) + parameter]
              : 0.0F;
          vertex_term = std::fma(
              result.value.selected_transforms[vertex * 16 + xyz * 4 + component],
              d_vertex, vertex_term);
        }
        const std::size_t index =
            ((static_cast<std::size_t>(vertex) * 3 + xyz)
             * kPoseRotationParameterDimension) + parameter;
        result.d_skin_from_transform[index] = transform_term;
        result.d_skin_from_vertex[index] = vertex_term;
        result.d_skinned_vertices[index] = transform_term + vertex_term;
      }
    }
  }
  result.d_vertices_rotpose = result.d_skinned_vertices;
  for (float& value : result.d_vertices_rotpose) {
    value *= input.scale;
  }
  result.lbs_derivative_ms = elapsed_ms(start);

  start = Clock::now();
  result.projection_jacobian.resize(n * 6);
  result.uv_rotpose_jacobian.resize(
      static_cast<std::size_t>(n) * 2 * kPoseRotationParameterDimension);
  result.translation_jacobian.resize(n * 6);
  for (int vertex = 0; vertex < n; ++vertex) {
    const Eigen::Vector3f point(
        result.value.vertices[vertex * 3], result.value.vertices[vertex * 3 + 1],
        result.value.vertices[vertex * 3 + 2]);
    const auto projection = project_unified_erp_with_jacobian(
        point, input.image_width, input.image_height);
    for (int uv = 0; uv < 2; ++uv) {
      for (int xyz = 0; xyz < 3; ++xyz) {
        result.projection_jacobian[vertex * 6 + uv * 3 + xyz] =
            projection.jacobian(uv, xyz);
        result.translation_jacobian[vertex * 6 + uv * 3 + xyz] =
            projection.jacobian(uv, xyz);
      }
      for (int parameter = 0; parameter < kPoseRotationParameterDimension; ++parameter) {
        float value = 0.0F;
        for (int xyz = 0; xyz < 3; ++xyz) {
          value = std::fma(
              projection.jacobian(uv, xyz),
              result.d_vertices_rotpose[
                  ((vertex * 3 + xyz) * kPoseRotationParameterDimension) + parameter],
              value);
        }
        result.uv_rotpose_jacobian[
            ((vertex * 2 + uv) * kPoseRotationParameterDimension) + parameter] = value;
      }
    }
  }
  result.erp_jacobian_chain_ms = elapsed_ms(start);

  start = Clock::now();
  result.active_jacobian.resize(
      static_cast<std::size_t>(n) * 2 * kPoseActiveDimension);
  result.frame_jacobian.assign(
      static_cast<std::size_t>(n) * 2 * kPoseParameterDimension, 0.0F);
  const int camera_column = input.camera == "cam2" ? 0 : 6;
  for (int row = 0; row < n * 2; ++row) {
    const int vertex = row / 2;
    const int uv = row % 2;
    float* active = result.active_jacobian.data() + row * kPoseActiveDimension;
    const float* uv_rotpose = result.uv_rotpose_jacobian.data()
        + row * kPoseRotationParameterDimension;
    for (int component = 0; component < 3; ++component) {
      active[component] = uv_rotpose[component];
      active[3 + component] =
          result.projection_jacobian[vertex * 6 + uv * 3 + component];
    }
    for (int parameter = 3; parameter < kPoseRotationParameterDimension; ++parameter) {
      active[3 + parameter] = uv_rotpose[parameter];
    }
    float* frame = result.frame_jacobian.data() + row * kPoseParameterDimension;
    for (int component = 0; component < 6; ++component) {
      frame[camera_column + component] = active[component];
    }
    for (int body = 0; body < 63; ++body) {
      frame[12 + body] = active[6 + body];
    }
  }
  result.assemble_active_ms = elapsed_ms(start);
  result.total_jacobian_ms = elapsed_ms(total_start);
  return result;
}

}  // namespace staged_lm
