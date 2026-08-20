#include "pose_kernels.h"

#include "staged_lm/pose_forward.h"
#include "staged_lm/pose_jacobian.h"

#include <Eigen/Core>

#include <stdexcept>

namespace staged_lm {

void compute_value_pose_offset(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& pose_feature,
    int point_count,
    std::vector<float>& output) {
  const int rows = point_count * 3;
  if (rows == 0) {
    output.clear();
    return;
  }
  if (selected_posedirs.size() !=
          static_cast<std::size_t>(rows * kPoseFeatureDimension) ||
      pose_feature.size() != kPoseFeatureDimension ||
      pose_feature.data() == output.data() ||
      selected_posedirs.data() == output.data()) {
    throw std::runtime_error("INVALID_VALUE_POSE_OFFSET_INPUT");
  }
  output.resize(rows);
  using Matrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  Eigen::Map<const Matrix> posedirs(
      selected_posedirs.data(), rows, kPoseFeatureDimension);
  Eigen::Map<const Eigen::VectorXf> feature(
      pose_feature.data(), kPoseFeatureDimension);
  Eigen::Map<Eigen::VectorXf> offsets(output.data(), rows);
  offsets.noalias() = posedirs * feature;
}

void compute_pose_offset_derivative(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    std::vector<float>& output) {
  const int rows = point_count * 3;
  if (rows == 0) {
    output.clear();
    return;
  }
  if (selected_posedirs.size() !=
          static_cast<std::size_t>(rows * kPoseFeatureDimension) ||
      d_pose_feature.size() != static_cast<std::size_t>(
          kPoseFeatureDimension * kPoseRotationParameterDimension)) {
    throw std::runtime_error("POSE_OFFSET_DERIVATIVE_INPUT_SHAPE_MISMATCH");
  }
  if (selected_posedirs.data() == output.data() ||
      d_pose_feature.data() == output.data()) {
    throw std::runtime_error("POSE_OFFSET_DERIVATIVE_OUTPUT_ALIASES_INPUT");
  }
  output.resize(
      static_cast<std::size_t>(rows) * kPoseRotationParameterDimension);
  using Posedirs =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using FeatureDerivative =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Output =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const Posedirs> posedirs(
      selected_posedirs.data(), rows, kPoseFeatureDimension);
  const Eigen::Map<const FeatureDerivative> derivative(
      d_pose_feature.data(), kPoseFeatureDimension,
      kPoseRotationParameterDimension);
  Eigen::Map<Output> result(
      output.data(), rows, kPoseRotationParameterDimension);
  result.noalias() = posedirs * derivative;
}

void compute_selected_transform_derivative(
    const std::vector<float>& weights,
    const std::vector<float>& relative_transform_derivatives,
    int point_count,
    int parameter_count,
    std::vector<float>& output) {
  const int columns = 16 * parameter_count;
  if (point_count == 0) {
    output.clear();
    return;
  }
  if (weights.size() !=
          static_cast<std::size_t>(point_count) * kPoseJointCount ||
      relative_transform_derivatives.size() !=
          static_cast<std::size_t>(kPoseJointCount) * columns ||
      output.data() == weights.data() ||
      output.data() == relative_transform_derivatives.data()) {
    throw std::runtime_error(
        "SELECTED_TRANSFORM_DERIVATIVE_INPUT_OR_ALIAS_VIOLATION");
  }
  output.resize(static_cast<std::size_t>(point_count) * columns);
  using Matrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const Matrix> weights_matrix(
      weights.data(), point_count, kPoseJointCount);
  const Eigen::Map<const Matrix> transform_derivatives(
      relative_transform_derivatives.data(), kPoseJointCount, columns);
  Eigen::Map<Matrix> selected_derivatives(
      output.data(), point_count, columns);
  selected_derivatives.noalias() = weights_matrix * transform_derivatives;
}

}  // namespace staged_lm
