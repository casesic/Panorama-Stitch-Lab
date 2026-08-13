#include "staged_lm/stage1_lm.h"
#include "staged_lm/performance.h"

#include <cmath>

namespace staged_lm {
namespace {

// Stage1 LM adapts the camera problem to generic LM callbacks.
Stage1CameraInput camera_at(
    const Stage1CameraInput& source, const std::vector<float>& parameters,
    int offset) {
  Stage1CameraInput camera = source;
  for (int index = 0; index < 3; ++index) {
    camera.orientation(index) = parameters[offset + index];
    camera.translation(index) = parameters[offset + 3 + index];
  }
  return camera;
}

double camera_objective(const Stage1CameraResult& result, float alpha, float eps) {
  float value = 0.0F;
  for (std::size_t point = 0; point < result.raw_residual.size() / 2; ++point) {
    const float u = result.raw_residual[2 * point];
    const float v = result.raw_residual[2 * point + 1];
    const float norm = std::sqrt(u * u + v * v);
    value += alpha * (std::sqrt(norm * norm + eps * eps) - eps);
  }
  return static_cast<double>(value);
}

}  // namespace

LMRunResult run_stage1_lm(const Stage1LMProblem& problem) {
  set_lm_attribution_stage("stage1");
  std::vector<float> initial(12);
  for (int index = 0; index < 3; ++index) {
    initial[index] = problem.cam2.orientation(index);
    initial[index + 3] = problem.cam2.translation(index);
    initial[index + 6] = problem.cam4.orientation(index);
    initial[index + 9] = problem.cam4.translation(index);
  }
  NormalEquationFunction normal_equations = [&](const std::vector<float>& parameters) {
    const Stage1FrameResult frame = evaluate_stage1_frame(
        camera_at(problem.cam2, parameters, 0),
        camera_at(problem.cam4, parameters, 6));
    DenseNormalEquations equations;
    equations.hessian.resize(144);
    for (int row = 0; row < 12; ++row) {
      for (int column = 0; column < 12; ++column) {
        equations.hessian[row * 12 + column] = frame.hessian(row, column);
      }
    }
    equations.gradient.assign(frame.gradient.data(), frame.gradient.data() + 12);
    equations.jacobian_ms =
        frame.cam2.so3_ms + frame.cam2.point_transform_ms + frame.cam2.erp_ms +
        frame.cam2.rigid_jacobian_ms + frame.cam2.irls_ms +
        frame.cam4.so3_ms + frame.cam4.point_transform_ms + frame.cam4.erp_ms +
        frame.cam4.rigid_jacobian_ms + frame.cam4.irls_ms;
    equations.normal_equations_ms = frame.normal_equations_ms;
    return equations;
  };
  ObjectiveFunction objective = [&](const std::vector<float>& parameters) {
    const Stage1CameraResult cam2 = evaluate_stage1_camera(
        camera_at(problem.cam2, parameters, 0));
    const Stage1CameraResult cam4 = evaluate_stage1_camera(
        camera_at(problem.cam4, parameters, 6));
    const float total = static_cast<float>(
        camera_objective(cam2, problem.cam2.alpha_per_point,
                         problem.cam2.irls_epsilon) +
        camera_objective(cam4, problem.cam4.alpha_per_point,
                         problem.cam4.irls_epsilon));
    return static_cast<double>(total);
  };
  return run_lm(initial, problem.lm_config, normal_equations, objective);
}

}  // namespace staged_lm
