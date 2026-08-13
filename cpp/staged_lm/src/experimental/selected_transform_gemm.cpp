#include "staged_lm/experimental/selected_transform_gemm.h"

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
void compute_selected_transform_derivative_eigen_gemm(
    const std::vector<float>& weights,
    const std::vector<float>& relative_transform_derivatives,
    int point_count, int parameter_count,
    std::vector<float>& output, SelectedTransformGemmTiming* timing) {
  const int columns = 16 * parameter_count;
  if (weights.size()!=static_cast<std::size_t>(point_count)*kPoseJointCount ||
      relative_transform_derivatives.size()!=
          static_cast<std::size_t>(kPoseJointCount)*columns ||
      output.data()==weights.data() || output.data()==relative_transform_derivatives.data())
    throw std::runtime_error("SELECTED_TRANSFORM_GEMM_INPUT_OR_ALIAS_VIOLATION");
  const auto total=Clock::now();
  output.resize(static_cast<std::size_t>(point_count)*columns);
  using Matrix=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
  const Eigen::Map<const Matrix> w(weights.data(),point_count,kPoseJointCount);
  const Eigen::Map<const Matrix> da(relative_transform_derivatives.data(),
                                    kPoseJointCount,columns);
  Eigen::Map<Matrix> dt(output.data(),point_count,columns);
  const auto core=Clock::now();dt.noalias()=w*da;
  if(timing){timing->input_pack_ms=0;timing->gemm_core_ms=ms(core);
    timing->output_unpack_ms=0;timing->total_ms=ms(total);}
}
}  // namespace staged_lm::experimental
