#include "staged_lm/shape.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace staged_lm {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace

ShapeValueResult evaluate_shape_value(
    const std::vector<float>& latent, float limit,
    const std::vector<float>& base_joints,
    const std::vector<float>& joint_basis, int joint_count) {
  if (base_joints.size() != static_cast<std::size_t>(joint_count * 3)
      || joint_basis.size() != base_joints.size() * latent.size()) {
    throw std::runtime_error("shape input mismatch");
  }
  ShapeValueResult result;
  auto started = Clock::now();
  result.tanh_latent.resize(latent.size());
  result.beta.resize(latent.size());
  for (std::size_t beta = 0; beta < latent.size(); ++beta) {
    result.tanh_latent[beta] = std::tanh(latent[beta]);
    result.beta[beta] = limit * result.tanh_latent[beta];
  }
  result.beta_ms = elapsed_ms(started);
  started = Clock::now();
  result.joints = base_joints;
  for (int joint = 0; joint < joint_count; ++joint) {
    for (int xyz = 0; xyz < 3; ++xyz) {
      float value = base_joints[joint * 3 + xyz];
      for (std::size_t beta = 0; beta < latent.size(); ++beta) {
        value = std::fma(
            joint_basis[(joint * 3 + xyz) * latent.size() + beta],
            result.beta[beta], value);
      }
      result.joints[joint * 3 + xyz] = value;
    }
  }
  result.dynamic_joints_ms = elapsed_ms(started);
  return result;
}

std::vector<float> evaluate_dynamic_vertices(
    const std::vector<float>& base, const std::vector<float>& basis,
    const std::vector<float>& beta, int point_count, double* timing_ms) {
  const auto started = Clock::now();
  if (base.size() != static_cast<std::size_t>(point_count * 3)
      || basis.size() != base.size() * beta.size()) {
    throw std::runtime_error("vertex shape mismatch");
  }
  std::vector<float> result = base;
  for (int point = 0; point < point_count; ++point) {
    for (int xyz = 0; xyz < 3; ++xyz) {
      float value = base[point * 3 + xyz];
      for (std::size_t component = 0; component < beta.size(); ++component) {
        value = std::fma(
            basis[(point * 3 + xyz) * beta.size() + component],
            beta[component], value);
      }
      result[point * 3 + xyz] = value;
    }
  }
  if (timing_ms != nullptr) *timing_ms = elapsed_ms(started);
  return result;
}

}  // namespace staged_lm
