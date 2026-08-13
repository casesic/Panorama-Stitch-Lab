#include "staged_lm/pose_lm.h"

#include "staged_lm/forward_reuse_cache.h"
#include "staged_lm/performance.h"
#include "staged_lm/pose_jacobian.h"

#include <chrono>
#include <cmath>

namespace staged_lm {
namespace {

// Pose LM owns the two callbacks consumed by generic LM: a 75-D
// linearization and the true robust objective at the same frame state.
struct PoseForwardPair {
  PoseCameraResult cam2;
  PoseCameraResult cam4;
};

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

std::size_t pose_pair_bytes(const PoseForwardPair& pair) {
  auto camera_bytes = [](const PoseCameraResult& value) {
    return (value.axis_angles.size() + value.rotations.size() +
            value.pose_feature.size() + value.pose_offsets.size() +
            value.v_posed.size() + value.relative_joints.size() +
            value.local_transforms.size() + value.global_transforms.size() +
            value.rest_transforms.size() + value.relative_transforms.size() +
            value.selected_transforms.size() + value.skinned_vertices.size() +
            value.vertices.size() + value.uv.size() +
            value.raw_residual.size()) *
           sizeof(float);
  };
  return camera_bytes(pair.cam2) + camera_bytes(pair.cam4);
}

}  // namespace

LMRunResult run_pose_lm(const PoseLMProblem& problem) {
  set_lm_attribution_stage("pose");
  set_performance_stage("pose");
  const auto reuse_mode = forward_reuse_mode();
  const auto context = next_forward_reuse_context();
  ExactRoleSingleSlotCache<PoseForwardPair> cache("pose", context);

  auto make_camera = [](const PoseCameraInput& source,
                        const std::vector<float>& parameters,
                        bool derivative) {
    const auto started = std::chrono::steady_clock::now();
    PoseCameraInput input = source;
    if (derivative && !source.stable_selected_posedirs) {
      record_selected_posedirs_derivative_copy(
          source.selected_posedirs.size() * sizeof(float),
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started).count());
    }
    input.parameters = parameters;
    return input;
  };

  // Evaluate both-camera value/Jacobian and assemble the 75-D frame system.
  NormalEquationFunction normal = [&](const std::vector<float>& parameters) {
    set_performance_stage("pose_jacobian");
    const auto started = std::chrono::steady_clock::now();

    // Experimental reuse diagnostics are inactive for preferred O4.
    if (reuse_mode ==
        ForwardReuseMode::kRetainPayloadDropBeforeLinearization) {
      (void)cache.discard(
          parameters, ForwardRole::kLinearization, context);
    }
    PoseForwardPair cached;
    const bool hit =
        forward_reuse_consumes(reuse_mode, ForwardRole::kLinearization) &&
        cache.consume(
            parameters, ForwardRole::kLinearization, context, cached);
    if (hit && reuse_mode ==
                   ForwardReuseMode::kObjectiveToLinearizationShadowWarmup) {
      set_performance_stage("pose_shadow_warmup");
      auto scratch2 = evaluate_pose_camera_value(
          problem.model, make_camera(problem.cam2, parameters, false));
      auto scratch4 = evaluate_pose_camera_value(
          problem.model, make_camera(problem.cam4, parameters, false));
      (void)scratch2;
      (void)scratch4;
      set_performance_stage("pose_jacobian");
    }

    PoseCameraJacobianResult cam2 = hit
        ? evaluate_pose_camera_with_jacobian_from_value(
              problem.model, make_camera(problem.cam2, parameters, true),
              std::move(cached.cam2))
        : evaluate_pose_camera_with_jacobian(
              problem.model, make_camera(problem.cam2, parameters, true));
    PoseCameraJacobianResult cam4 = hit
        ? evaluate_pose_camera_with_jacobian_from_value(
              problem.model, make_camera(problem.cam4, parameters, true),
              std::move(cached.cam4))
        : evaluate_pose_camera_with_jacobian(
              problem.model, make_camera(problem.cam4, parameters, true));
    const double jacobian_ms = cam2.total_jacobian_ms + cam4.total_jacobian_ms;
    const PoseNormalEquationsResult equations = build_pose_normal_equations(
        cam2, cam4, parameters, problem.body_pose_anchor, problem.normal_config);
    DenseNormalEquations result;
    result.hessian = equations.hessian_frame;
    result.gradient = equations.gradient_frame;
    result.jacobian_ms = jacobian_ms;
    result.normal_equations_ms = equations.total_normal_equations_ms;
    if (forward_reuse_stores(reuse_mode, ForwardRole::kLinearization) &&
        !hit) {
      PoseForwardPair stored{
          std::move(cam2.value), std::move(cam4.value)};
      const auto bytes = pose_pair_bytes(stored);
      cache.store(
          parameters, ForwardRole::kLinearization, context,
          std::move(stored), bytes);
    }
    (void)started;
    return result;
  };

  // Evaluate the true robust objective used for LM acceptance.
  ObjectiveFunction objective = [&](const std::vector<float>& parameters) {
    set_performance_stage("pose_objective");
    PoseForwardPair values;
    const bool hit =
        forward_reuse_consumes(reuse_mode, ForwardRole::kObjective) &&
        cache.consume(parameters, ForwardRole::kObjective, context, values);
    if (!hit) {
      values.cam2 = evaluate_pose_camera_value(
          problem.model, make_camera(problem.cam2, parameters, false));
      values.cam4 = evaluate_pose_camera_value(
          problem.model, make_camera(problem.cam4, parameters, false));
    }
    const double objective_value = pose_objective_from_values(
        values.cam2, values.cam4, parameters, problem);
    if (forward_reuse_stores(reuse_mode, ForwardRole::kObjective) && !hit) {
      const auto bytes = pose_pair_bytes(values);
      cache.store(
          parameters, ForwardRole::kObjective, context, std::move(values),
          bytes);
    }
    return objective_value;
  };

  return run_lm(problem.cam2.parameters, problem.lm_config, normal, objective);
}

}  // namespace staged_lm
