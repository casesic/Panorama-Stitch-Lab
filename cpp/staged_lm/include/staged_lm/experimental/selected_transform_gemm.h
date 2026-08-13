#pragma once

#include <vector>

namespace staged_lm::experimental {

struct SelectedTransformGemmTiming {
  double input_pack_ms = 0.0;
  double gemm_core_ms = 0.0;
  double output_unpack_ms = 0.0;
  double total_ms = 0.0;
};

void compute_selected_transform_derivative_eigen_gemm(
    const std::vector<float>& weights,
    const std::vector<float>& relative_transform_derivatives,
    int point_count, int parameter_count,
    std::vector<float>& selected_transform_derivatives,
    SelectedTransformGemmTiming* timing = nullptr);

}  // namespace staged_lm::experimental
