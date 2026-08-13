#pragma once

#include <vector>

namespace staged_lm::experimental {

struct LbsDirectDebugTerms {
  std::vector<float> transform_term;
  std::vector<float> vertex_term;
};

void compute_lbs_derivative_direct(
    const std::vector<float>& weights,
    const std::vector<float>& relative_transform_derivatives,
    const std::vector<float>& selected_transforms,
    const std::vector<float>& posed_vertices,
    const std::vector<float>& posed_vertex_derivatives,
    int point_count, int parameter_count,
    std::vector<float>& skinned_derivatives,
    LbsDirectDebugTerms* debug_terms = nullptr);

}  // namespace staged_lm::experimental
