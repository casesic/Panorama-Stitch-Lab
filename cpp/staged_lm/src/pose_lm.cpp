#include "staged_lm/pose_lm.h"

#include "staged_lm/pose_jacobian.h"

#include <chrono>
#include <cmath>

namespace staged_lm {
namespace {

// Pose LM owns the two callbacks consumed by generic LM: a 75-D
// linearization and the true robust objective at the same frame state.
double pose_objective_from_values(
    const PoseCameraResult& cam2, const PoseCameraResult& cam4,
    const std::vector<float>& parameters, const PoseLMProblem& problem) {
  double value = 0.0;
  const PoseCameraResult* cameras[] = {&cam2, &cam4};
  const float alphas[] = {
      problem.normal_config.cam2_alpha_per_point,
      problem.normal_config.cam4_alpha_per_point,
  };
  for (int camera = 0; camera < 2; ++camera) {
    const auto& residual = cameras[camera]->raw_residual;
    for (std::size_t point = 0; point < residual.size() / 2; ++point) {
      const float u = residual[point * 2];
      const float v = residual[point * 2 + 1];
      const float norm = std::sqrt(u * u + v * v);
      const float rho = std::sqrt(
                            norm * norm +
                            problem.normal_config.charbonnier_eps *
                                problem.normal_config.charbonnier_eps) -
                        problem.normal_config.charbonnier_eps;
      value += static_cast<double>(alphas[camera] * rho);
    }
  }
  float anchor_sum = 0.0F;
  for (int body = 0; body < 63; ++body) {
    const float residual =
        (parameters[12 + body] - problem.body_pose_anchor[body]) *
        problem.normal_config.pose_anchor_scale;
    anchor_sum += residual * residual;
  }
  return static_cast<double>(static_cast<float>(value) + 0.5F * anchor_sum);
}

}  // namespace

LMRunResult run_pose_lm(const PoseLMProblem& problem) {
  auto make_camera = [](const PoseCameraInput& source,
                        const std::vector<float>& parameters) {
    PoseCameraInput input = source;
    input.parameters = parameters;
    return input;
  };

  // Evaluate both-camera value/Jacobian and assemble the 75-D frame system.
  NormalEquationFunction normal = [&](const std::vector<float>& parameters) {
    PoseCameraJacobianResult cam2 = evaluate_pose_camera_with_jacobian(
        problem.model, make_camera(problem.cam2, parameters));
    PoseCameraJacobianResult cam4 = evaluate_pose_camera_with_jacobian(
        problem.model, make_camera(problem.cam4, parameters));
    const double jacobian_ms = cam2.total_jacobian_ms + cam4.total_jacobian_ms;
    const PoseNormalEquationsResult equations = build_pose_normal_equations(
        cam2, cam4, parameters, problem.body_pose_anchor, problem.normal_config);
    DenseNormalEquations result;
    result.hessian = equations.hessian_frame;
    result.gradient = equations.gradient_frame;
    result.jacobian_ms = jacobian_ms;
    result.normal_equations_ms = equations.total_normal_equations_ms;
    return result;
  };

  // Evaluate the true robust objective used for LM acceptance.
  ObjectiveFunction objective = [&](const std::vector<float>& parameters) {
    const PoseCameraResult cam2 = evaluate_pose_camera_value(
        problem.model, make_camera(problem.cam2, parameters));
    const PoseCameraResult cam4 = evaluate_pose_camera_value(
        problem.model, make_camera(problem.cam4, parameters));
    return pose_objective_from_values(cam2, cam4, parameters, problem);
  };

  return run_lm(problem.cam2.parameters, problem.lm_config, normal, objective);
}

}  // namespace staged_lm
