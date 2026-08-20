#pragma once

#include <vector>

namespace staged_lm {

void compute_value_pose_offset(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& pose_feature,
    int point_count,
    std::vector<float>& output);

void compute_pose_offset_derivative(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    std::vector<float>& output);

void compute_selected_transform_derivative(
    const std::vector<float>& weights,
    const std::vector<float>& relative_transform_derivatives,
    int point_count,
    int parameter_count,
    std::vector<float>& output);

}  // namespace staged_lm
