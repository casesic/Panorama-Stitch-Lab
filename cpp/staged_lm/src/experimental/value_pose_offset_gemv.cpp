#include "staged_lm/experimental/value_pose_offset_gemv.h"

#include "staged_lm/pose_forward.h"

#include <Eigen/Core>
#include <chrono>
#include <stdexcept>

namespace staged_lm::experimental {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
}

void compute_value_pose_offset_eigen_gemv(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& pose_feature,
    int point_count, std::vector<float>& output,
    ValuePoseOffsetGemvTiming* timing) {
  const auto total = Clock::now();
  const int rows = point_count * 3;
  if (selected_posedirs.size() != static_cast<std::size_t>(rows*kPoseFeatureDimension) ||
      pose_feature.size() != kPoseFeatureDimension ||
      pose_feature.data() == output.data() || selected_posedirs.data() == output.data())
    throw std::runtime_error("INVALID_VALUE_POSE_OFFSET_GEMV_INPUT");
  output.resize(rows);
  const auto core = Clock::now();
  using Matrix = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
  Eigen::Map<const Matrix> p(selected_posedirs.data(),rows,kPoseFeatureDimension);
  Eigen::Map<const Eigen::VectorXf> x(pose_feature.data(),kPoseFeatureDimension);
  Eigen::Map<Eigen::VectorXf> y(output.data(),rows);
  y.noalias() = p*x;
  if (timing) {
    timing->gemv_core_ms = ms(core);
    timing->total_ms = ms(total);
  }
}

}  // namespace staged_lm::experimental
