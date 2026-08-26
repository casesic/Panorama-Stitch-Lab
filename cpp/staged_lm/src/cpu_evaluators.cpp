#include "staged_lm/cpu_evaluators.h"

#include "staged_lm/pose_jacobian.h"
#include "staged_lm/pose_shape_schur.h"
#include "staged_lm/shape.h"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace staged_lm {
namespace {

constexpr int kFrameDimension = 75;
constexpr int kBetaDimension = 10;

PoseCameraInput pose_camera_at(
    const PoseCameraInput& source, const std::vector<float>& parameters) {
  PoseCameraInput camera = source;
  camera.parameters = parameters;
  return camera;
}

PoseShapeCameraInput pose_shape_camera_at(
    const PoseShapeCameraInput& source,
    const std::vector<float>& parameters) {
  PoseShapeCameraInput camera = source;
  camera.pose.parameters.assign(
      parameters.begin(), parameters.begin() + kFrameDimension);
  return camera;
}

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

SchurSystem structured_blocks(
    const PoseShapeNormalEquationsResult& equations) {
  return {
      1, kFrameDimension, kBetaDimension,
      equations.a_final, equations.b_final, equations.c_final,
      equations.g_frame_final, equations.g_beta_final,
  };
}

}  // namespace

struct CpuPoseEvaluator::Impl {
  explicit Impl(const PoseLMProblem& source) : problem(source) {}
  const PoseLMProblem& problem;
  std::vector<float> value_parameters;
  PoseCameraResult value_cam2;
  PoseCameraResult value_cam4;
  std::vector<float> linearization_parameters;
  PoseCameraJacobianResult jacobian_cam2;
  PoseCameraJacobianResult jacobian_cam4;
  bool value_ready = false;
  bool linearization_ready = false;
};

CpuPoseEvaluator::CpuPoseEvaluator(const PoseLMProblem& problem)
    : impl_(std::make_unique<Impl>(problem)) {}
CpuPoseEvaluator::~CpuPoseEvaluator() = default;

void CpuPoseEvaluator::forward(const std::vector<float>& parameters) {
  impl_->value_parameters = parameters;
  impl_->value_cam2 = evaluate_pose_camera_value(
      impl_->problem.model, pose_camera_at(impl_->problem.cam2, parameters));
  impl_->value_cam4 = evaluate_pose_camera_value(
      impl_->problem.model, pose_camera_at(impl_->problem.cam4, parameters));
  impl_->value_ready = true;
}

double CpuPoseEvaluator::evaluate_residual() {
  if (!impl_->value_ready) {
    throw std::runtime_error("CPU_POSE_FORWARD_REQUIRED");
  }
  return pose_objective_from_values(
      impl_->value_cam2, impl_->value_cam4, impl_->value_parameters,
      impl_->problem);
}

void CpuPoseEvaluator::linearize(const std::vector<float>& parameters) {
  impl_->linearization_parameters = parameters;
  impl_->jacobian_cam2 = evaluate_pose_camera_with_jacobian(
      impl_->problem.model, pose_camera_at(impl_->problem.cam2, parameters));
  impl_->jacobian_cam4 = evaluate_pose_camera_with_jacobian(
      impl_->problem.model, pose_camera_at(impl_->problem.cam4, parameters));
  impl_->linearization_ready = true;
}

DenseNormalEquations CpuPoseEvaluator::build_normal_equations() {
  if (!impl_->linearization_ready) {
    throw std::runtime_error("CPU_POSE_LINEARIZATION_REQUIRED");
  }
  const double jacobian_ms =
      impl_->jacobian_cam2.total_jacobian_ms +
      impl_->jacobian_cam4.total_jacobian_ms;
  const PoseNormalEquationsResult equations = build_pose_normal_equations(
      impl_->jacobian_cam2, impl_->jacobian_cam4,
      impl_->linearization_parameters, impl_->problem.body_pose_anchor,
      impl_->problem.normal_config);
  DenseNormalEquations result;
  result.hessian = equations.hessian_frame;
  result.gradient = equations.gradient_frame;
  result.jacobian_ms = jacobian_ms;
  result.normal_equations_ms = equations.total_normal_equations_ms;
  return result;
}

struct CpuPoseShapeEvaluator::Impl {
  explicit Impl(const PoseShapeLMProblem& source) : problem(source) {}
  const PoseShapeLMProblem& problem;
  std::vector<float> value_parameters;
  PoseShapeValueResult value;
  std::vector<float> linearization_parameters;
  PoseShapeCameraJacobianResult jacobian_cam2;
  PoseShapeCameraJacobianResult jacobian_cam4;
  bool value_ready = false;
  bool linearization_ready = false;
};

CpuPoseShapeEvaluator::CpuPoseShapeEvaluator(
    const PoseShapeLMProblem& problem)
    : impl_(std::make_unique<Impl>(problem)) {}
CpuPoseShapeEvaluator::~CpuPoseShapeEvaluator() = default;

void CpuPoseShapeEvaluator::forward(const std::vector<float>& parameters) {
  impl_->value_parameters = parameters;
  const PoseShapeCameraInput cam2 =
      pose_shape_camera_at(impl_->problem.cam2, parameters);
  const PoseShapeCameraInput cam4 =
      pose_shape_camera_at(impl_->problem.cam4, parameters);
  impl_->value = evaluate_pose_shape_value(
      parameters, impl_->problem.base_joints, impl_->problem.joint_basis,
      impl_->problem.beta_anchor, impl_->problem.body_pose_anchor,
      impl_->problem.beta_limit,
      impl_->problem.normal_config.pose_anchor_scale,
      impl_->problem.normal_config.shape_anchor_scale,
      impl_->problem.normal_config.charbonnier_eps,
      impl_->problem.normal_config.cam2_alpha_per_point,
      impl_->problem.normal_config.cam4_alpha_per_point,
      impl_->problem.parents, cam2, cam4);
  impl_->value_ready = true;
}

double CpuPoseShapeEvaluator::evaluate_residual() {
  if (!impl_->value_ready) {
    throw std::runtime_error("CPU_POSE_SHAPE_FORWARD_REQUIRED");
  }
  return evaluate_pose_shape_objective_from_common(
      impl_->value_parameters, impl_->value, impl_->problem.beta_anchor,
      impl_->problem.body_pose_anchor,
      impl_->problem.normal_config.pose_anchor_scale,
      impl_->problem.normal_config.shape_anchor_scale,
      impl_->problem.normal_config.charbonnier_eps,
      impl_->problem.normal_config.cam2_alpha_per_point,
      impl_->problem.normal_config.cam4_alpha_per_point);
}

void CpuPoseShapeEvaluator::linearize(
    const std::vector<float>& parameters) {
  impl_->linearization_parameters = parameters;
  const std::vector<float> latent_beta(
      parameters.begin() + kFrameDimension, parameters.end());
  ShapeValueResult shape = evaluate_shape_value(
      latent_beta, impl_->problem.beta_limit, impl_->problem.base_joints,
      impl_->problem.joint_basis, 55);
  std::vector<float> dbeta_dlatent(kBetaDimension);
  std::vector<float> djoints_dlatent(165 * kBetaDimension);
  for (int beta = 0; beta < kBetaDimension; ++beta) {
    dbeta_dlatent[beta] = impl_->problem.beta_limit *
        (1.0F - shape.tanh_latent[beta] * shape.tanh_latent[beta]);
  }
  for (int joint = 0; joint < 165; ++joint) {
    for (int beta = 0; beta < kBetaDimension; ++beta) {
      djoints_dlatent[joint * kBetaDimension + beta] =
          impl_->problem.joint_basis[joint * kBetaDimension + beta] *
          dbeta_dlatent[beta];
    }
  }

  const PoseModelInput model{shape.joints, impl_->problem.parents};
  const PoseShapeCameraInput cam2 =
      pose_shape_camera_at(impl_->problem.cam2, parameters);
  const PoseShapeCameraInput cam4 =
      pose_shape_camera_at(impl_->problem.cam4, parameters);
  const std::vector<float> vertices2 = evaluate_dynamic_vertices(
      cam2.selected_base_v_shaped, cam2.selected_beta_shapedirs,
      shape.beta, cam2.pose.point_count);
  const std::vector<float> vertices4 = evaluate_dynamic_vertices(
      cam4.selected_base_v_shaped, cam4.selected_beta_shapedirs,
      shape.beta, cam4.pose.point_count);
  impl_->jacobian_cam2 = evaluate_pose_shape_camera_with_jacobian(
      model, cam2, vertices2, djoints_dlatent, dbeta_dlatent);
  impl_->jacobian_cam4 = evaluate_pose_shape_camera_with_jacobian(
      model, cam4, vertices4, djoints_dlatent, dbeta_dlatent);
  impl_->linearization_ready = true;
}

DenseNormalEquations CpuPoseShapeEvaluator::build_normal_equations() {
  if (!impl_->linearization_ready) {
    throw std::runtime_error("CPU_POSE_SHAPE_LINEARIZATION_REQUIRED");
  }
  const PoseShapeNormalEquationsResult assembled =
      build_pose_shape_normal_equations(
          impl_->jacobian_cam2, impl_->jacobian_cam4,
          impl_->linearization_parameters, impl_->problem.body_pose_anchor,
          impl_->problem.beta_anchor, impl_->problem.normal_config);

  DenseNormalEquations output;
  output.hessian = assembled.h_dense_structured;
  output.gradient = assembled.g_dense_structured;
  output.jacobian_ms =
      impl_->jacobian_cam2.total_ms + impl_->jacobian_cam4.total_ms;
  output.normal_equations_ms = assembled.total_ms;
  auto system = std::make_shared<SchurSystem>(structured_blocks(assembled));
  output.solve_override = [system](
      double damping, std::vector<double>& delta, std::string& backend) {
    const PoseShapeSchurResult solved =
        solve_pose_shape_schur(*system, damping, false);
    if (solved.success) {
      delta = solved.full_delta;
      backend = "schur";
      return true;
    }
    backend = "dense_fallback";
    return solve_schur_dense_fallback(*system, damping, delta);
  };
  output.quadratic_form_override = [system](
      const std::vector<double>& delta) {
    return schur_quadratic_form(*system, delta);
  };
  return output;
}

}  // namespace staged_lm
