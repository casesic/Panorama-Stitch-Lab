#include "staged_lm/stage1.h"

#include "staged_lm/erp.h"
#include "staged_lm/so3.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace staged_lm {
namespace {

// Stage1 projects rigidly transformed camera points and builds analytic
// residual/Jacobian blocks for the per-frame camera parameters.
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Eigen::Vector3f vector3_at(const std::vector<float>& values, int index) {
  return Eigen::Vector3f(
      values[3 * index], values[3 * index + 1], values[3 * index + 2]);
}

Eigen::Vector3f matrix_vector_product(
    const Eigen::Matrix3f& matrix, const Eigen::Vector3f& vector) {
  Eigen::Vector3f result;
  for (int row = 0; row < 3; ++row) {
    float accumulated = 0.0F;
    for (int inner = 0; inner < 3; ++inner) {
      accumulated = std::fma(matrix(row, inner), vector(inner), accumulated);
    }
    result(row) = accumulated;
  }
  return result;
}

Eigen::Matrix<float, 2, 3> multiply2x3_3x3(
    const Eigen::Matrix<float, 2, 3>& left, const Eigen::Matrix3f& right) {
  Eigen::Matrix<float, 2, 3> result;
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 3; ++column) {
      float accumulated = 0.0F;
      for (int inner = 0; inner < 3; ++inner) {
        accumulated = std::fma(left(row, inner), right(inner, column), accumulated);
      }
      result(row, column) = accumulated;
    }
  }
  return result;
}

}  // namespace

Stage1CameraResult evaluate_stage1_camera(const Stage1CameraInput& input) {
  if (input.point_count < 0
      || input.local_vertices.size() != static_cast<std::size_t>(input.point_count * 3)
      || input.targets.size() != static_cast<std::size_t>(input.point_count * 2)) {
    throw std::runtime_error("invalid Stage1 camera input shape");
  }
  const int camera_offset = input.camera == "cam2" ? kCam2Offset
      : input.camera == "cam4" ? kCam4Offset : -1;
  if (camera_offset < 0) {
    throw std::runtime_error("unsupported camera: " + input.camera);
  }

  Stage1CameraResult result;
  const int n = input.point_count;
  result.points.resize(n * 3);
  result.uv.resize(n * 2);
  result.projection_jacobian.resize(n * 6);
  result.point_orientation_jacobian.resize(n * 9);
  result.orientation_jacobian.resize(n * 6);
  result.translation_jacobian.resize(n * 6);
  result.frame_jacobian.assign(n * 2 * kStage1ParameterDimension, 0.0F);
  result.raw_residual.resize(n * 2);
  result.point_weights.resize(n);
  result.row_scales.resize(n * 2);
  result.weighted_residual.resize(n * 2);
  result.weighted_jacobian.resize(n * 2 * kStage1ParameterDimension);

  auto start = Clock::now();
  result.rotation = axis_angle_to_matrix(input.orientation);
  result.right_jacobian = so3_right_jacobian(input.orientation);
  result.so3_ms = elapsed_ms(start);

  start = Clock::now();
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3f rotated = matrix_vector_product(
        result.rotation, vector3_at(input.local_vertices, i));
    Eigen::Vector3f point;
    for (int xyz = 0; xyz < 3; ++xyz) {
      point(xyz) = (rotated(xyz) + input.rotation_center(xyz))
          + input.translation(xyz);
    }
    for (int xyz = 0; xyz < 3; ++xyz) {
      result.points[3 * i + xyz] = point(xyz);
    }
  }
  result.point_transform_ms = elapsed_ms(start);

  start = Clock::now();
  for (int i = 0; i < n; ++i) {
    const ErpProjectionResult projection = project_unified_erp_with_jacobian(
        vector3_at(result.points, i), input.image_width, input.image_height);
    for (int uv = 0; uv < 2; ++uv) {
      result.uv[2 * i + uv] = projection.uv(uv);
      for (int xyz = 0; xyz < 3; ++xyz) {
        result.projection_jacobian[i * 6 + uv * 3 + xyz] =
            projection.jacobian(uv, xyz);
      }
    }
  }
  result.erp_ms = elapsed_ms(start);

  start = Clock::now();
  for (int i = 0; i < n; ++i) {
    const Eigen::Matrix3f point_jacobian = rotated_point_jacobian(
        input.orientation, vector3_at(input.local_vertices, i));
    Eigen::Matrix<float, 2, 3> projection_jacobian;
    for (int uv = 0; uv < 2; ++uv) {
      for (int xyz = 0; xyz < 3; ++xyz) {
        projection_jacobian(uv, xyz) =
            result.projection_jacobian[i * 6 + uv * 3 + xyz];
      }
    }
    const Eigen::Matrix<float, 2, 3> orientation_jacobian =
        multiply2x3_3x3(projection_jacobian, point_jacobian);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        result.point_orientation_jacobian[i * 9 + row * 3 + column] =
            point_jacobian(row, column);
      }
    }
    for (int uv = 0; uv < 2; ++uv) {
      const int output_row = 2 * i + uv;
      for (int xyz = 0; xyz < 3; ++xyz) {
        result.orientation_jacobian[i * 6 + uv * 3 + xyz] =
            orientation_jacobian(uv, xyz);
        result.translation_jacobian[i * 6 + uv * 3 + xyz] =
            projection_jacobian(uv, xyz);
        result.frame_jacobian[output_row * kStage1ParameterDimension
                              + camera_offset + xyz] = orientation_jacobian(uv, xyz);
        result.frame_jacobian[output_row * kStage1ParameterDimension
                              + camera_offset + 3 + xyz] = projection_jacobian(uv, xyz);
      }
    }
  }
  result.rigid_jacobian_ms = elapsed_ms(start);

  start = Clock::now();
  for (int i = 0; i < n; ++i) {
    const float residual_u = result.uv[2 * i] - input.targets[2 * i];
    const float residual_v = result.uv[2 * i + 1] - input.targets[2 * i + 1];
    result.raw_residual[2 * i] = residual_u;
    result.raw_residual[2 * i + 1] = residual_v;
    const float norm = std::sqrt(residual_u * residual_u + residual_v * residual_v);
    const float weight = input.alpha_per_point
        / std::sqrt(norm * norm + input.irls_epsilon * input.irls_epsilon);
    const float scale = std::sqrt(weight);
    result.point_weights[i] = weight;
    result.row_scales[2 * i] = scale;
    result.row_scales[2 * i + 1] = scale;
    for (int uv = 0; uv < 2; ++uv) {
      const int row = 2 * i + uv;
      result.weighted_residual[row] = result.raw_residual[row] * scale;
      for (int column = 0; column < kStage1ParameterDimension; ++column) {
        result.weighted_jacobian[row * kStage1ParameterDimension + column] =
            result.frame_jacobian[row * kStage1ParameterDimension + column] * scale;
      }
    }
  }
  result.irls_ms = elapsed_ms(start);
  return result;
}

Stage1FrameResult evaluate_stage1_frame(
    const Stage1CameraInput& cam2_input, const Stage1CameraInput& cam4_input) {
  Stage1FrameResult result;
  result.cam2 = evaluate_stage1_camera(cam2_input);
  result.cam4 = evaluate_stage1_camera(cam4_input);
  const int cam2_rows = cam2_input.point_count * 2;
  const int cam4_rows = cam4_input.point_count * 2;
  const int row_count = cam2_rows + cam4_rows;
  Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> jacobian(row_count, 12);
  Eigen::VectorXd residual(row_count);
  for (int row = 0; row < row_count; ++row) {
    const Stage1CameraResult& camera = row < cam2_rows ? result.cam2 : result.cam4;
    const int camera_row = row < cam2_rows ? row : row - cam2_rows;
    residual(row) = static_cast<double>(camera.weighted_residual[camera_row]);
    for (int column = 0; column < 12; ++column) {
      jacobian(row, column) = static_cast<double>(
          camera.weighted_jacobian[camera_row * 12 + column]);
    }
  }
  const auto start = Clock::now();
  result.hessian = jacobian.transpose() * jacobian;
  result.gradient = jacobian.transpose() * residual;
  result.normal_equations_ms = elapsed_ms(start);
  return result;
}

}  // namespace staged_lm
