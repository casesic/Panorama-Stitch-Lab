#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace staged_lm {

// Immutable linear observation operator shared by every frame and camera.
// Rows address DenseKP IDs; columns follow support_vertex_ids exactly.
struct DenseKPRegressor {
  int densekp_count = 0;
  int support_count = 0;
  std::vector<std::int32_t> support_vertex_ids;
  std::vector<std::int32_t> row_offsets;      // densekp_count + 1
  std::vector<std::int32_t> support_indices;  // nnz
  std::vector<float> sparse_weights;          // nnz
};

struct DenseKPObservation {
  std::shared_ptr<const DenseKPRegressor> regressor;
  std::vector<std::int32_t> densekp_ids;
  std::vector<float> targets;  // densekp_ids.size() * 2, full-ERP pixels
};

void validate_densekp_observation(
    const DenseKPObservation& observation, int support_count);

// source layout: [support, component, parameter], parameter contiguous.
// output layout: [observation, component, parameter].
void apply_densekp_operator(
    const DenseKPObservation& observation,
    const std::vector<float>& source,
    int component_count,
    int parameter_count,
    std::vector<float>& output);

void build_densekp_sparse_representation(
    DenseKPRegressor& regressor, const std::vector<float>& dense_weights);
}  // namespace staged_lm
