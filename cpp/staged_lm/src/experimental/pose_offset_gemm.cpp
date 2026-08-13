#include "staged_lm/experimental/pose_offset_gemm.h"

#include "staged_lm/pose_forward.h"
#include "staged_lm/pose_jacobian.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace staged_lm::experimental {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace

void compute_pose_offset_derivative_eigen_gemm(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    std::vector<float>& output,
    PoseOffsetGemmTiming* timing) {
  const int rows = point_count * 3;
  if (selected_posedirs.size() !=
          static_cast<std::size_t>(rows * kPoseFeatureDimension) ||
      d_pose_feature.size() != static_cast<std::size_t>(
          kPoseFeatureDimension * kPoseRotationParameterDimension)) {
    throw std::runtime_error("POSE_OFFSET_GEMM_INPUT_SHAPE_MISMATCH");
  }
  if (selected_posedirs.data() == output.data() ||
      d_pose_feature.data() == output.data()) {
    throw std::runtime_error("POSE_OFFSET_GEMM_OUTPUT_ALIASES_INPUT");
  }
  const auto total_started = Clock::now();
  if (timing) {
    timing->capacity_before = output.capacity();
    timing->pointer_before = output.data();
  }
  const auto prepare_started = Clock::now();
  output.resize(static_cast<std::size_t>(rows) * kPoseRotationParameterDimension);
  if (timing) {
    timing->output_preparation_ms = elapsed_ms(prepare_started);
    timing->capacity_after = output.capacity();
    timing->pointer_after = output.data();
  }
  const auto setup_started = Clock::now();
  using Posedirs = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using FeatureDerivative =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Output = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const Posedirs> posedirs(
      selected_posedirs.data(), rows, kPoseFeatureDimension);
  const Eigen::Map<const FeatureDerivative> derivative(
      d_pose_feature.data(), kPoseFeatureDimension,
      kPoseRotationParameterDimension);
  Eigen::Map<Output> result(
      output.data(), rows, kPoseRotationParameterDimension);
  if (timing) timing->view_setup_ms = elapsed_ms(setup_started);
  const auto gemm_started = Clock::now();
  result.noalias() = posedirs * derivative;
  if (timing != nullptr) {
    timing->input_pack_ms = 0.0;
    timing->gemm_core_ms = elapsed_ms(gemm_started);
    timing->output_unpack_ms = 0.0;
    timing->total_ms = elapsed_ms(total_started);
  }
}

void compute_pose_offset_derivative_transposed_eigen_gemm(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    std::vector<float>& output,
    PoseOffsetGemmTiming* timing) {
  const int rows = point_count * 3;
  if (selected_posedirs.size() !=
          static_cast<std::size_t>(rows * kPoseFeatureDimension) ||
      d_pose_feature.size() != static_cast<std::size_t>(
          kPoseFeatureDimension * kPoseRotationParameterDimension)) {
    throw std::runtime_error("POSE_OFFSET_GEMM_INPUT_SHAPE_MISMATCH");
  }
  const auto total_started = Clock::now();
  if (timing) {
    timing->capacity_before = output.capacity();
    timing->pointer_before = output.data();
  }
  const auto prepare_started = Clock::now();
  output.resize(static_cast<std::size_t>(rows) * kPoseRotationParameterDimension);
  if (timing) {
    timing->output_preparation_ms = elapsed_ms(prepare_started);
    timing->capacity_after = output.capacity();
    timing->pointer_after = output.data();
  }
  const auto setup_started = Clock::now();
  using Posedirs = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using FeatureDerivative = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // A row-major rows x 66 buffer has the same bytes as a column-major
  // 66 x rows matrix, so this output view is zero-copy.
  using TransposedOutput = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  const Eigen::Map<const Posedirs> posedirs(
      selected_posedirs.data(), rows, kPoseFeatureDimension);
  const Eigen::Map<const FeatureDerivative> derivative(
      d_pose_feature.data(), kPoseFeatureDimension,
      kPoseRotationParameterDimension);
  Eigen::Map<TransposedOutput> result_t(
      output.data(), kPoseRotationParameterDimension, rows);
  if (timing) timing->view_setup_ms = elapsed_ms(setup_started);
  const auto gemm_started = Clock::now();
  result_t.noalias() = derivative.transpose() * posedirs.transpose();
  if (timing) {
    timing->input_pack_ms = 0.0;
    timing->gemm_core_ms = elapsed_ms(gemm_started);
    timing->output_unpack_ms = 0.0;
    timing->total_ms = elapsed_ms(total_started);
  }
}

void compute_pose_offset_derivative_blocked_eigen_gemm(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    int row_block_size,
    int column_block_size,
    std::vector<float>& output,
    PoseOffsetGemmTiming* timing) {
  const int rows = point_count * 3;
  constexpr int columns = kPoseRotationParameterDimension;
  if (selected_posedirs.size() !=
          static_cast<std::size_t>(rows * kPoseFeatureDimension) ||
      d_pose_feature.size() != static_cast<std::size_t>(
          kPoseFeatureDimension * columns) ||
      row_block_size < 0 || column_block_size < 0) {
    throw std::runtime_error("POSE_OFFSET_BLOCKED_GEMM_INPUT_MISMATCH");
  }
  if (selected_posedirs.data() == output.data() ||
      d_pose_feature.data() == output.data()) {
    throw std::runtime_error("POSE_OFFSET_GEMM_OUTPUT_ALIASES_INPUT");
  }
  const auto total_started = Clock::now();
  if (timing) {
    timing->capacity_before = output.capacity();
    timing->pointer_before = output.data();
  }
  const auto prepare_started = Clock::now();
  output.resize(static_cast<std::size_t>(rows) * columns);
  if (timing) {
    timing->output_preparation_ms = elapsed_ms(prepare_started);
    timing->capacity_after = output.capacity();
    timing->pointer_after = output.data();
  }
  const auto setup_started = Clock::now();
  using Posedirs = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using FeatureDerivative = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Output = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const Posedirs> posedirs(selected_posedirs.data(), rows,
                                            kPoseFeatureDimension);
  const Eigen::Map<const FeatureDerivative> derivative(
      d_pose_feature.data(), kPoseFeatureDimension, columns);
  Eigen::Map<Output> result(output.data(), rows, columns);
  if (timing) timing->view_setup_ms = elapsed_ms(setup_started);
  const auto gemm_started = Clock::now();
  const int bm = row_block_size == 0 ? rows : row_block_size;
  const int bn = column_block_size == 0 ? columns : column_block_size;
  for (int row = 0; row < rows; row += bm) {
    const int row_count = std::min(bm, rows - row);
    for (int column = 0; column < columns; column += bn) {
      const int column_count = std::min(bn, columns - column);
      result.block(row, column, row_count, column_count).noalias() =
          posedirs.middleRows(row, row_count) *
          derivative.middleCols(column, column_count);
    }
  }
  if (timing) {
    timing->input_pack_ms = 0.0;
    timing->gemm_core_ms = elapsed_ms(gemm_started);
    timing->output_unpack_ms = 0.0;
    timing->total_ms = elapsed_ms(total_started);
  }
}

}  // namespace staged_lm::experimental
