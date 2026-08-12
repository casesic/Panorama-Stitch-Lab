#include "staged_lm/pose_shape_lm.h"

#include "staged_lm/pose_shape_schur.h"
#include "staged_lm/shape.h"

#include <Eigen/Dense>

#include <memory>

namespace staged_lm {
namespace {

constexpr int kFrameDimension = 75;
constexpr int kBetaDimension = 10;
constexpr int kParameterDimension = kFrameDimension + kBetaDimension;

PoseShapeCameraInput camera_at(
    const PoseShapeCameraInput& source, const std::vector<float>& parameters) {
  PoseShapeCameraInput camera = source;
  camera.pose.parameters.assign(
      parameters.begin(), parameters.begin() + kFrameDimension);
  return camera;
}

SchurSystem structured_blocks(const PoseShapeNormalEquationsResult& equations) {
  return {
      1, kFrameDimension, kBetaDimension,
      equations.a_final, equations.b_final, equations.c_final,
      equations.g_frame_final, equations.g_beta_final,
  };
}

double structured_quadratic_form(
    const SchurSystem& system, const std::vector<double>& delta) {
  double result = 0.0;
  for (int row = 0; row < kFrameDimension; ++row) {
    for (int column = 0; column < kFrameDimension; ++column) {
      result += delta[row] * system.a[
          row * kFrameDimension + column] * delta[column];
    }
    for (int beta = 0; beta < kBetaDimension; ++beta) {
      result += 2.0 * delta[row] * system.b[
          row * kBetaDimension + beta] * delta[kFrameDimension + beta];
    }
  }
  for (int row = 0; row < kBetaDimension; ++row) {
    for (int column = 0; column < kBetaDimension; ++column) {
      result += delta[kFrameDimension + row] * system.c[
          row * kBetaDimension + column] * delta[kFrameDimension + column];
    }
  }
  return result;
}

bool solve_dense_fallback(
    const std::vector<double>& hessian, const std::vector<double>& gradient,
    double damping, std::vector<double>& delta) {
  using Matrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const Matrix> mapped_hessian(
      hessian.data(), kParameterDimension, kParameterDimension);
  Matrix matrix = mapped_hessian;
  matrix.diagonal().array() += damping;
  Eigen::Map<const Eigen::VectorXd> mapped_gradient(
      gradient.data(), kParameterDimension);
  Eigen::LLT<Matrix> cholesky(matrix);
  if (cholesky.info() == Eigen::Success) {
    const Eigen::VectorXd solution = cholesky.solve(-mapped_gradient);
    if (cholesky.info() == Eigen::Success && solution.allFinite()) {
      delta.assign(solution.data(), solution.data() + kParameterDimension);
      return true;
    }
  }
  const Eigen::PartialPivLU<Matrix> lu(matrix);
  const Eigen::VectorXd solution = lu.solve(-mapped_gradient);
  if (!solution.allFinite()) return false;
  delta.assign(solution.data(), solution.data() + kParameterDimension);
  return true;
}

}  // namespace

LMRunResult run_pose_shape_lm(const PoseShapeLMProblem& problem) {
  NormalEquationFunction normal_equations =
      [&](const std::vector<float>& parameters) {
        std::vector<float> latent_beta(
            parameters.begin() + kFrameDimension, parameters.end());
        const ShapeValueResult shape = evaluate_shape_value(
            latent_beta, problem.beta_limit, problem.base_joints,
            problem.joint_basis, 55);
        std::vector<float> dbeta_dlatent(kBetaDimension);
        std::vector<float> djoints_dlatent(165 * kBetaDimension);
        for (int beta = 0; beta < kBetaDimension; ++beta) {
          dbeta_dlatent[beta] = problem.beta_limit *
              (1.0F - shape.tanh_latent[beta] * shape.tanh_latent[beta]);
        }
        for (int joint = 0; joint < 165; ++joint) {
          for (int beta = 0; beta < kBetaDimension; ++beta) {
            djoints_dlatent[joint * kBetaDimension + beta] =
                problem.joint_basis[joint * kBetaDimension + beta] *
                dbeta_dlatent[beta];
          }
        }
        const PoseModelInput model{shape.joints, problem.parents};
        const PoseShapeCameraInput cam2 = camera_at(problem.cam2, parameters);
        const PoseShapeCameraInput cam4 = camera_at(problem.cam4, parameters);
        const std::vector<float> vertices2 = evaluate_dynamic_vertices(
            cam2.selected_base_v_shaped, cam2.selected_beta_shapedirs,
            shape.beta, cam2.pose.point_count);
        const std::vector<float> vertices4 = evaluate_dynamic_vertices(
            cam4.selected_base_v_shaped, cam4.selected_beta_shapedirs,
            shape.beta, cam4.pose.point_count);
        const PoseShapeCameraJacobianResult jacobian2 =
            evaluate_pose_shape_camera_with_jacobian(
                model, cam2, vertices2, djoints_dlatent, dbeta_dlatent);
        const PoseShapeCameraJacobianResult jacobian4 =
            evaluate_pose_shape_camera_with_jacobian(
                model, cam4, vertices4, djoints_dlatent, dbeta_dlatent);
        const PoseShapeNormalEquationsResult assembled =
            build_pose_shape_normal_equations(
                jacobian2, jacobian4, parameters, problem.body_pose_anchor,
                problem.beta_anchor, problem.normal_config);

        DenseNormalEquations output;
        output.hessian = assembled.h_dense_structured;
        output.gradient = assembled.g_dense_structured;
        output.jacobian_ms = jacobian2.total_ms + jacobian4.total_ms;
        output.normal_equations_ms = assembled.total_ms;
        auto system = std::make_shared<SchurSystem>(structured_blocks(assembled));
        output.solve_override = [
            system, hessian = output.hessian, gradient = output.gradient](
                double damping, std::vector<double>& delta,
                std::string& backend) {
          const PoseShapeSchurResult schur =
              solve_pose_shape_schur(*system, damping, false);
          if (schur.success) {
            delta = schur.full_delta;
            backend = "schur";
            return true;
          }
          backend = "dense_fallback";
          return solve_dense_fallback(hessian, gradient, damping, delta);
        };
        output.quadratic_form_override = [system](
            const std::vector<double>& delta) {
          return structured_quadratic_form(*system, delta);
        };
        return output;
      };

  ObjectiveFunction objective = [&](const std::vector<float>& parameters) {
    const PoseShapeCameraInput cam2 = camera_at(problem.cam2, parameters);
    const PoseShapeCameraInput cam4 = camera_at(problem.cam4, parameters);
    return evaluate_pose_shape_value(
        parameters, problem.base_joints, problem.joint_basis,
        problem.beta_anchor, problem.body_pose_anchor, problem.beta_limit,
        problem.normal_config.pose_anchor_scale,
        problem.normal_config.shape_anchor_scale,
        problem.normal_config.charbonnier_eps,
        problem.normal_config.cam2_alpha_per_point,
        problem.normal_config.cam4_alpha_per_point,
        problem.parents, cam2, cam4).objective_total;
  };
  return run_lm(
      problem.initial_parameters, problem.lm_config, normal_equations,
      objective);
}

}  // namespace staged_lm
