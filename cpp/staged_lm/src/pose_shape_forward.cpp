#include "staged_lm/pose_shape_forward.h"

#include "staged_lm/performance.h"

#include <chrono>
#include <cmath>

namespace staged_lm {
namespace {

// PoseShape value evaluation expands latent shape, updates selected vertices,
// runs both Pose camera forwards, and evaluates the true robust objective.
using Clock = std::chrono::steady_clock;

double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

double evaluate_pose_shape_objective_from_common(
    const std::vector<float>& parameters, const PoseShapeValueResult& result,
    const std::vector<float>& beta_anchor,
    const std::vector<float>& pose_anchor, float pose_scale,
    float shape_scale, float eps, float cam2_alpha, float cam4_alpha,
    double* primary_out, double* pose_out, double* shape_out) {
  const PoseCameraResult* cameras[] = {&result.cam2, &result.cam4};
  const float alphas[] = {cam2_alpha, cam4_alpha};
  float primary = 0.0F;
  for (int camera = 0; camera < 2; ++camera) {
    for (std::size_t point = 0;
         point < cameras[camera]->raw_residual.size() / 2; ++point) {
      const float u = cameras[camera]->raw_residual[2 * point];
      const float v = cameras[camera]->raw_residual[2 * point + 1];
      const float norm = std::sqrt(u * u + v * v);
      primary += alphas[camera] *
                 (std::sqrt(norm * norm + eps * eps) - eps);
    }
  }

  float pose_sum = 0.0F;
  float shape_sum = 0.0F;
  for (std::size_t body = 0; body < pose_anchor.size(); ++body) {
    const float residual =
        (parameters[12 + body] - pose_anchor[body]) * pose_scale;
    pose_sum += residual * residual;
  }
  for (std::size_t beta = 0; beta < beta_anchor.size(); ++beta) {
    const float residual =
        (result.shape.beta[beta] - beta_anchor[beta]) * shape_scale;
    shape_sum += residual * residual;
  }

  if (primary_out) *primary_out = primary;
  if (pose_out) *pose_out = 0.5F * pose_sum;
  if (shape_out) *shape_out = 0.5F * shape_sum;
  return static_cast<double>(
      static_cast<float>(primary + 0.5F * pose_sum + 0.5F * shape_sum));
}

PoseShapeValueResult evaluate_pose_shape_value(
    const std::vector<float>& parameters,
    const std::vector<float>& base_joints,
    const std::vector<float>& joint_basis,
    const std::vector<float>& beta_anchor,
    const std::vector<float>& pose_anchor, float beta_limit,
    float pose_scale, float shape_scale, float eps, float cam2_alpha,
    float cam4_alpha, const std::vector<std::int64_t>& parents,
    const PoseShapeCameraInput& cam2_input,
    const PoseShapeCameraInput& cam4_input) {
  const auto total_started = Clock::now();
  PoseShapeValueResult result;

  std::vector<float> latent(parameters.begin() + 75, parameters.end());
  result.shape = evaluate_shape_value(
      latent, beta_limit, base_joints, joint_basis, 55);
  record_performance_sample(
      "state", "shape.value", 0,
      result.shape.beta_ms + result.shape.dynamic_joints_ms);

  result.cam2_dynamic_v_shaped = evaluate_dynamic_vertices(
      cam2_input.selected_base_v_shaped,
      cam2_input.selected_beta_shapedirs, result.shape.beta,
      cam2_input.pose.point_count, &result.cam2_shape_ms);
  record_performance_sample(
      "cam2", "shape.dynamic_vertices", cam2_input.pose.point_count,
      result.cam2_shape_ms);
  result.cam4_dynamic_v_shaped = evaluate_dynamic_vertices(
      cam4_input.selected_base_v_shaped,
      cam4_input.selected_beta_shapedirs, result.shape.beta,
      cam4_input.pose.point_count, &result.cam4_shape_ms);
  record_performance_sample(
      "cam4", "shape.dynamic_vertices", cam4_input.pose.point_count,
      result.cam4_shape_ms);
  record_shape_forward_sample(
      parameters, result.shape, result.cam2_dynamic_v_shaped,
      result.cam4_dynamic_v_shaped);

  PoseModelInput model{result.shape.joints, parents};
  auto cam2 = cam2_input.pose;
  cam2.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam2.selected_v_shaped = result.cam2_dynamic_v_shaped;
  auto cam4 = cam4_input.pose;
  cam4.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam4.selected_v_shaped = result.cam4_dynamic_v_shaped;
  result.cam2 = evaluate_pose_camera_value(model, cam2);
  result.cam4 = evaluate_pose_camera_value(model, cam4);

  const auto objective_started = Clock::now();
  result.objective_total = evaluate_pose_shape_objective_from_common(
      parameters, result, beta_anchor, pose_anchor, pose_scale, shape_scale,
      eps, cam2_alpha, cam4_alpha, &result.objective_primary,
      &result.objective_pose_anchor, &result.objective_shape_anchor);
  result.objective_ms = ms(objective_started);
  result.total_ms = ms(total_started);
  return result;
}

}  // namespace staged_lm
