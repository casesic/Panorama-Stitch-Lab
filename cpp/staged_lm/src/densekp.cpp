#include "staged_lm/densekp.h"

#include <cmath>
#include <stdexcept>

namespace staged_lm {
void build_densekp_sparse_representation(
    DenseKPRegressor& regressor, const std::vector<float>& dense_weights) {
  if (regressor.densekp_count <= 0 || regressor.support_count <= 0 ||
      dense_weights.size() != static_cast<std::size_t>(
          regressor.densekp_count * regressor.support_count))
    throw std::runtime_error("invalid DenseKP dense asset shape");
  regressor.row_offsets.assign(regressor.densekp_count + 1, 0);
  regressor.support_indices.clear();
  regressor.sparse_weights.clear();
  for (int row = 0; row < regressor.densekp_count; ++row) {
    for (int support = 0; support < regressor.support_count; ++support) {
      const float weight = dense_weights[
          static_cast<std::size_t>(row) * regressor.support_count + support];
      if (weight != 0.0F) {
        regressor.support_indices.push_back(support);
        regressor.sparse_weights.push_back(weight);
      }
    }
    regressor.row_offsets[row + 1] =
        static_cast<std::int32_t>(regressor.support_indices.size());
  }
}

void validate_densekp_observation(
    const DenseKPObservation& observation, int support_count) {
  if (!observation.regressor) {
    throw std::runtime_error("DenseKP observation has no regressor");
  }
  const DenseKPRegressor& regressor = *observation.regressor;
  if (regressor.support_count != support_count || regressor.densekp_count <= 0 ||
      regressor.support_vertex_ids.size() !=
          static_cast<std::size_t>(regressor.support_count) ||
      observation.targets.size() != observation.densekp_ids.size() * 2) {
    throw std::runtime_error("invalid DenseKP operator shape");
  }
  if (regressor.row_offsets.size() !=
           static_cast<std::size_t>(regressor.densekp_count + 1) ||
       regressor.support_indices.size() != regressor.sparse_weights.size() ||
       regressor.row_offsets.back() !=
           static_cast<std::int32_t>(regressor.support_indices.size())) {
    throw std::runtime_error("invalid DenseKP sparse representation");
  }
  for (std::int32_t id : observation.densekp_ids) {
    if (id < 0 || id >= regressor.densekp_count) {
      throw std::runtime_error("DenseKP observation ID out of range");
    }
  }
}

void apply_densekp_operator(
    const DenseKPObservation& observation,
    const std::vector<float>& source,
    int component_count,
    int parameter_count,
    std::vector<float>& output) {
  const DenseKPRegressor& regressor = *observation.regressor;
  if (regressor.row_offsets.empty()) {
    throw std::runtime_error("DenseKP sparse representation is not initialized");
  }
  const std::size_t expected = static_cast<std::size_t>(
      regressor.support_count) * component_count * parameter_count;
  if (component_count <= 0 || parameter_count <= 0 || source.size() != expected) {
    throw std::runtime_error("invalid DenseKP sparse source shape");
  }
  const std::size_t output_count =
      observation.densekp_ids.size() * component_count * parameter_count;
  output.resize(output_count);
  std::fill(output.begin(), output.end(), 0.0F);
  for (std::size_t observation_index = 0;
       observation_index < observation.densekp_ids.size(); ++observation_index) {
    const int row = observation.densekp_ids[observation_index];
    for (int nz = regressor.row_offsets[row]; nz < regressor.row_offsets[row + 1];
         ++nz) {
      const int support = regressor.support_indices[nz];
      const float weight = regressor.sparse_weights[nz];
      for (int component = 0; component < component_count; ++component) {
        const float* source_row = source.data() +
            (static_cast<std::size_t>(support) * component_count + component) *
                parameter_count;
        float* output_row = output.data() +
            (observation_index * component_count + component) * parameter_count;
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
          output_row[parameter] = std::fma(
              weight, source_row[parameter], output_row[parameter]);
        }
      }
    }
  }
}

}  // namespace staged_lm
