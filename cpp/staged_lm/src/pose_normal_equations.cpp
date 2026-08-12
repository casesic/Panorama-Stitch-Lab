#include "staged_lm/pose_normal_equations.h"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace staged_lm {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

PoseCameraNormalEquations build_camera(
    const PoseCameraJacobianResult& input, float alpha, float eps) {
  PoseCameraNormalEquations result;
  const int point_count = static_cast<int>(input.value.raw_residual.size() / 2);
  const int row_count = point_count * 2;
  if (input.value.raw_residual.size() != static_cast<std::size_t>(row_count)
      || input.active_jacobian.size()
             != static_cast<std::size_t>(row_count * kPoseActiveDimension)) {
    throw std::runtime_error("Pose normal-equation input shape mismatch");
  }

  auto started = Clock::now();
  result.point_weights.resize(point_count);
  result.row_scales.resize(row_count);
  for (int point = 0; point < point_count; ++point) {
    const float u = input.value.raw_residual[point * 2];
    const float v = input.value.raw_residual[point * 2 + 1];
    const float weight = alpha / std::sqrt(u * u + v * v + eps * eps);
    result.point_weights[point] = weight;
    const float scale = std::sqrt(weight);
    result.row_scales[point * 2] = scale;
    result.row_scales[point * 2 + 1] = scale;
  }
  result.irls_ms = milliseconds(started);

  started = Clock::now();
  result.weighted_residual.resize(row_count);
  result.weighted_jacobian_active.resize(row_count * kPoseActiveDimension);
  for (int row = 0; row < row_count; ++row) {
    const float scale = result.row_scales[row];
    result.weighted_residual[row] = input.value.raw_residual[row] * scale;
    for (int column = 0; column < kPoseActiveDimension; ++column) {
      result.weighted_jacobian_active[row * kPoseActiveDimension + column] =
          input.active_jacobian[row * kPoseActiveDimension + column] * scale;
    }
  }
  result.row_scaling_ms = milliseconds(started);

  started = Clock::now();
  Eigen::Matrix<double, Eigen::Dynamic, kPoseActiveDimension, Eigen::RowMajor> j64(
      row_count, kPoseActiveDimension);
  Eigen::VectorXd r64(row_count);
  for (int row = 0; row < row_count; ++row) {
    r64[row] = static_cast<double>(result.weighted_residual[row]);
    for (int column = 0; column < kPoseActiveDimension; ++column) {
      j64(row, column) = static_cast<double>(
          result.weighted_jacobian_active[row * kPoseActiveDimension + column]);
    }
  }
  result.fp64_cast_ms = milliseconds(started);

  started = Clock::now();
  const Eigen::Matrix<double, kPoseActiveDimension, kPoseActiveDimension, Eigen::RowMajor>
      hessian = j64.transpose() * j64;
  const Eigen::Matrix<double, kPoseActiveDimension, 1> gradient = j64.transpose() * r64;
  result.hessian_active.assign(hessian.data(), hessian.data() + hessian.size());
  result.gradient_active.assign(gradient.data(), gradient.data() + gradient.size());
  result.active_normal_equations_ms = milliseconds(started);
  return result;
}

std::vector<int> active_indices(const std::string& camera) {
  const int rigid_start = camera == "cam2" ? 0 : camera == "cam4" ? 6 : -1;
  if (rigid_start < 0) {
    throw std::runtime_error("unknown Pose camera: " + camera);
  }
  std::vector<int> indices;
  indices.reserve(kPoseActiveDimension);
  for (int index = 0; index < 6; ++index) {
    indices.push_back(rigid_start + index);
  }
  for (int index = 12; index < kPoseParameterDimension; ++index) {
    indices.push_back(index);
  }
  return indices;
}

}  // namespace

void scatter_pose_active_to_frame(
    const PoseCameraNormalEquations& active, const std::string& camera,
    std::vector<double>& hessian_frame, std::vector<double>& gradient_frame) {
  if (hessian_frame.size() != kPoseParameterDimension * kPoseParameterDimension
      || gradient_frame.size() != kPoseParameterDimension) {
    throw std::runtime_error("Pose frame normal-equation shape mismatch");
  }
  const std::vector<int> indices = active_indices(camera);
  for (int row = 0; row < kPoseActiveDimension; ++row) {
    gradient_frame[indices[row]] += active.gradient_active[row];
    for (int column = 0; column < kPoseActiveDimension; ++column) {
      hessian_frame[indices[row] * kPoseParameterDimension + indices[column]] +=
          active.hessian_active[row * kPoseActiveDimension + column];
    }
  }
}

PoseNormalEquationsResult build_pose_normal_equations(
    const PoseCameraJacobianResult& cam2,
    const PoseCameraJacobianResult& cam4,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const PoseNormalEquationConfig& config) {
  if (parameters.size() != kPoseParameterDimension || body_pose_anchor.size() != 63) {
    throw std::runtime_error("Pose anchor input shape mismatch");
  }
  const auto total_started = Clock::now();
  PoseNormalEquationsResult result;
  result.cam2 = build_camera(cam2, config.cam2_alpha_per_point, config.charbonnier_eps);
  result.cam4 = build_camera(cam4, config.cam4_alpha_per_point, config.charbonnier_eps);

  auto started = Clock::now();
  result.hessian_data_frame.assign(kPoseParameterDimension * kPoseParameterDimension, 0.0);
  result.gradient_data_frame.assign(kPoseParameterDimension, 0.0);
  scatter_pose_active_to_frame(
      result.cam2, "cam2", result.hessian_data_frame, result.gradient_data_frame);
  scatter_pose_active_to_frame(
      result.cam4, "cam4", result.hessian_data_frame, result.gradient_data_frame);
  result.scatter_ms = milliseconds(started);

  started = Clock::now();
  result.pose_anchor_delta.resize(63);
  result.pose_anchor_residual.resize(63);
  result.hessian_pose_anchor.assign(kPoseParameterDimension * kPoseParameterDimension, 0.0);
  result.gradient_pose_anchor.assign(kPoseParameterDimension, 0.0);
  const double scale_squared = static_cast<double>(config.pose_anchor_scale)
                               * static_cast<double>(config.pose_anchor_scale);
  for (int body = 0; body < 63; ++body) {
    const float delta = parameters[12 + body] - body_pose_anchor[body];
    result.pose_anchor_delta[body] = delta;
    result.pose_anchor_residual[body] = delta * config.pose_anchor_scale;
    result.hessian_pose_anchor[(12 + body) * kPoseParameterDimension + 12 + body] =
        scale_squared;
    result.gradient_pose_anchor[12 + body] = static_cast<double>(delta) * scale_squared;
  }
  result.hessian_frame = result.hessian_data_frame;
  result.gradient_frame = result.gradient_data_frame;
  for (std::size_t index = 0; index < result.hessian_frame.size(); ++index) {
    result.hessian_frame[index] += result.hessian_pose_anchor[index];
  }
  for (std::size_t index = 0; index < result.gradient_frame.size(); ++index) {
    result.gradient_frame[index] += result.gradient_pose_anchor[index];
  }
  result.anchor_ms = milliseconds(started);
  result.total_normal_equations_ms = milliseconds(total_started);
  return result;
}

}  // namespace staged_lm
