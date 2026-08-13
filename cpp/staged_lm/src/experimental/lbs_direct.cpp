#include "staged_lm/experimental/lbs_direct.h"

#include "staged_lm/pose_forward.h"

#include <cmath>
#include <stdexcept>

namespace staged_lm::experimental {

void compute_lbs_derivative_direct(
    const std::vector<float>& weights,
    const std::vector<float>& relative_transform_derivatives,
    const std::vector<float>& selected_transforms,
    const std::vector<float>& posed_vertices,
    const std::vector<float>& posed_vertex_derivatives,
    int point_count, int parameter_count,
    std::vector<float>& skinned_derivatives,
    LbsDirectDebugTerms* debug_terms) {
  const std::size_t output_count =
      static_cast<std::size_t>(point_count) * 3 * parameter_count;
  if (weights.size() != static_cast<std::size_t>(point_count) * kPoseJointCount ||
      relative_transform_derivatives.size() !=
          static_cast<std::size_t>(kPoseJointCount) * 16 * parameter_count ||
      selected_transforms.size() != static_cast<std::size_t>(point_count) * 16 ||
      posed_vertices.size() != static_cast<std::size_t>(point_count) * 3 ||
      posed_vertex_derivatives.size() != output_count ||
      skinned_derivatives.data() == weights.data() ||
      skinned_derivatives.data() == relative_transform_derivatives.data() ||
      skinned_derivatives.data() == selected_transforms.data() ||
      skinned_derivatives.data() == posed_vertices.data() ||
      skinned_derivatives.data() == posed_vertex_derivatives.data())
    throw std::runtime_error("LBS_DIRECT_INPUT_OR_ALIAS_CONTRACT_VIOLATION");
  skinned_derivatives.resize(output_count);
  if (debug_terms != nullptr) {
    debug_terms->transform_term.resize(output_count);
    debug_terms->vertex_term.resize(output_count);
  }
  for (int vertex = 0; vertex < point_count; ++vertex) {
    const float homogeneous[4] = {
        posed_vertices[vertex * 3], posed_vertices[vertex * 3 + 1],
        posed_vertices[vertex * 3 + 2], 1.0F};
    for (int parameter = 0; parameter < parameter_count; ++parameter)
      for (int row = 0; row < 3; ++row) {
        float transform_term = 0.0F;
        for (int joint = 0; joint < kPoseJointCount; ++joint) {
          float joint_term = 0.0F;
          for (int column = 0; column < 4; ++column)
            joint_term = std::fma(relative_transform_derivatives[
                ((joint * 16 + row * 4 + column) * parameter_count) + parameter],
                homogeneous[column], joint_term);
          transform_term = std::fma(
              weights[vertex * kPoseJointCount + joint], joint_term,
              transform_term);
        }
        float vertex_term = 0.0F;
        for (int column = 0; column < 3; ++column)
          vertex_term = std::fma(
              selected_transforms[vertex * 16 + row * 4 + column],
              posed_vertex_derivatives[
                  ((static_cast<std::size_t>(vertex) * 3 + column) *
                   parameter_count) + parameter], vertex_term);
        const std::size_t index =
            ((static_cast<std::size_t>(vertex) * 3 + row) * parameter_count) +
            parameter;
        skinned_derivatives[index] = transform_term + vertex_term;
        if (debug_terms != nullptr) {
          debug_terms->transform_term[index] = transform_term;
          debug_terms->vertex_term[index] = vertex_term;
        }
      }
  }
}

}  // namespace staged_lm::experimental
