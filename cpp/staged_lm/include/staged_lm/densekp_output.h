#pragma once

#include "staged_lm/pose_shape_lm.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace staged_lm {

struct OptimizedDenseKPObservation {
  int point_count = 0;
  std::vector<float> pairs;  // [x_cam2, y_cam2, x_cam4, y_cam4]
  std::vector<std::uint8_t> valid;
};

OptimizedDenseKPObservation evaluate_optimized_densekp(
    const PoseShapeLMProblem& problem,
    const std::vector<float>& final_parameters);

void write_optimized_densekp_json(
    const std::filesystem::path& path,
    const OptimizedDenseKPObservation& observation);

}  // namespace staged_lm
