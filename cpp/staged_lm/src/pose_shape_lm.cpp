#include "staged_lm/pose_shape_lm.h"

#include "staged_lm/forward_reuse_cache.h"
#include "staged_lm/performance.h"
#include "staged_lm/pose_shape_schur.h"
#include "staged_lm/shape.h"

#include <Eigen/Dense>

#include <chrono>
#include <memory>

namespace staged_lm {
namespace {

// PoseShape LM extends the 75-D frame state with ten latent shape parameters
// and exposes structured normal equations to the Schur solver.
constexpr int kFrameDimension = 75;
constexpr int kBetaDimension = 10;
constexpr int kParameterDimension = kFrameDimension + kBetaDimension;

PoseShapeCameraInput camera_at(
    const PoseShapeCameraInput& source, const std::vector<float>& parameters,
    bool derivative) {
  const auto started = std::chrono::steady_clock::now();
  PoseShapeCameraInput camera = source;
  if (derivative && !source.pose.stable_selected_posedirs) {
    record_selected_posedirs_derivative_copy(
        source.pose.selected_posedirs.size() * sizeof(float),
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
  }
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
std::size_t shape_common_bytes(const PoseShapeValueResult& result) {
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
  return (result.shape.tanh_latent.size() + result.shape.beta.size() +
          result.shape.joints.size() +
          result.cam2_dynamic_v_shaped.size() +
          result.cam4_dynamic_v_shaped.size()) *
             sizeof(float) +
         camera_bytes(result.cam2) + camera_bytes(result.cam4);
}

}  // namespace

LMRunResult run_pose_shape_lm(const PoseShapeLMProblem& problem) {
  set_lm_attribution_stage("pose_shape");
  set_performance_stage("pose_shape");
  const auto reuse_mode = forward_reuse_mode();
  const auto context = next_forward_reuse_context();
  ExactRoleSingleSlotCache<PoseShapeValueResult> cache("pose_shape", context);

  // Evaluate shape, both-camera Jacobians, and the structured 85-D system.
  NormalEquationFunction normal_equations =
      [&](const std::vector<float>& parameters) {
        set_performance_stage("pose_shape_jacobian");
        std::vector<float> latent_beta(
            parameters.begin() + kFrameDimension, parameters.end());
        // Experimental reuse diagnostics are inactive for preferred O4.
        if (reuse_mode ==
            ForwardReuseMode::kRetainPayloadDropBeforeLinearization) {
          (void)cache.discard(
              parameters, ForwardRole::kLinearization, context);
        }
        PoseShapeValueResult cached;
        const bool hit =
            forward_reuse_consumes(reuse_mode, ForwardRole::kLinearization) &&
            cache.consume(
                parameters, ForwardRole::kLinearization, context, cached);
        if (hit && reuse_mode ==
                       ForwardReuseMode::kObjectiveToLinearizationShadowWarmup) {
          set_performance_stage("pose_shape_shadow_warmup");
          const PoseShapeCameraInput shadow2 =
              camera_at(problem.cam2, parameters, false);
          const PoseShapeCameraInput shadow4 =
              camera_at(problem.cam4, parameters, false);
          auto scratch = evaluate_pose_shape_value(
              parameters, problem.base_joints, problem.joint_basis,
              problem.beta_anchor, problem.body_pose_anchor, problem.beta_limit,
              problem.normal_config.pose_anchor_scale,
              problem.normal_config.shape_anchor_scale,
              problem.normal_config.charbonnier_eps,
              problem.normal_config.cam2_alpha_per_point,
              problem.normal_config.cam4_alpha_per_point, problem.parents,
              shadow2, shadow4);
          (void)scratch;
          set_performance_stage("pose_shape_jacobian");
        }
        ShapeValueResult shape = hit
            ? std::move(cached.shape)
            : evaluate_shape_value(
                  latent_beta, problem.beta_limit, problem.base_joints,
                  problem.joint_basis, 55);
        if (!hit) {
          record_performance_sample(
              "state", "shape.value", 0,
              shape.beta_ms + shape.dynamic_joints_ms);
        }
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
        const PoseShapeCameraInput cam2 = camera_at(problem.cam2, parameters, true);
        const PoseShapeCameraInput cam4 = camera_at(problem.cam4, parameters, true);
        double vertices2_ms = 0.0;
        std::vector<float> vertices2 = hit
            ? std::move(cached.cam2_dynamic_v_shaped)
            : evaluate_dynamic_vertices(
                  cam2.selected_base_v_shaped,
                  cam2.selected_beta_shapedirs, shape.beta,
                  cam2.pose.point_count, &vertices2_ms);
        if (!hit) {
          record_performance_sample(
              "cam2", "shape.dynamic_vertices", cam2.pose.point_count,
              vertices2_ms);
        }
        double vertices4_ms = 0.0;
        std::vector<float> vertices4 = hit
            ? std::move(cached.cam4_dynamic_v_shaped)
            : evaluate_dynamic_vertices(
                  cam4.selected_base_v_shaped,
                  cam4.selected_beta_shapedirs, shape.beta,
                  cam4.pose.point_count, &vertices4_ms);
        if (!hit) {
          record_performance_sample(
              "cam4", "shape.dynamic_vertices", cam4.pose.point_count,
              vertices4_ms);
        }
        if (!hit) {
          record_shape_forward_sample(
              parameters, shape, vertices2, vertices4);
        }

        PoseShapeCameraJacobianResult jacobian2 = hit
            ? evaluate_pose_shape_camera_with_jacobian_from_value(
                  model, cam2, vertices2, djoints_dlatent, dbeta_dlatent,
                  std::move(cached.cam2))
            : evaluate_pose_shape_camera_with_jacobian(
                  model, cam2, vertices2, djoints_dlatent, dbeta_dlatent);
        PoseShapeCameraJacobianResult jacobian4 = hit
            ? evaluate_pose_shape_camera_with_jacobian_from_value(
                  model, cam4, vertices4, djoints_dlatent, dbeta_dlatent,
                  std::move(cached.cam4))
            : evaluate_pose_shape_camera_with_jacobian(
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
        if (forward_reuse_stores(
                reuse_mode, ForwardRole::kLinearization) &&
            !hit) {
          PoseShapeValueResult stored;
          stored.shape = std::move(shape);
          stored.cam2_dynamic_v_shaped = std::move(vertices2);
          stored.cam4_dynamic_v_shaped = std::move(vertices4);
          stored.cam2 = std::move(jacobian2.pose.value);
          stored.cam4 = std::move(jacobian4.pose.value);
          const auto bytes = shape_common_bytes(stored);
          cache.store(
              parameters, ForwardRole::kLinearization, context,
              std::move(stored), bytes);
        }
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

  // Evaluate the true robust objective used for LM acceptance.
  ObjectiveFunction objective = [&](const std::vector<float>& parameters) {
    set_performance_stage("pose_shape_objective");
    PoseShapeValueResult common;
    const bool hit =
        forward_reuse_consumes(reuse_mode, ForwardRole::kObjective) &&
        cache.consume(parameters, ForwardRole::kObjective, context, common);
    if (!hit) {
      const PoseShapeCameraInput cam2 =
          camera_at(problem.cam2, parameters, false);
      const PoseShapeCameraInput cam4 =
          camera_at(problem.cam4, parameters, false);
      common = evaluate_pose_shape_value(
          parameters, problem.base_joints, problem.joint_basis,
          problem.beta_anchor, problem.body_pose_anchor, problem.beta_limit,
          problem.normal_config.pose_anchor_scale,
          problem.normal_config.shape_anchor_scale,
          problem.normal_config.charbonnier_eps,
          problem.normal_config.cam2_alpha_per_point,
          problem.normal_config.cam4_alpha_per_point, problem.parents, cam2,
          cam4);
    }
    const double value = evaluate_pose_shape_objective_from_common(
        parameters, common, problem.beta_anchor, problem.body_pose_anchor,
        problem.normal_config.pose_anchor_scale,
        problem.normal_config.shape_anchor_scale,
        problem.normal_config.charbonnier_eps,
        problem.normal_config.cam2_alpha_per_point,
        problem.normal_config.cam4_alpha_per_point);
    if (forward_reuse_stores(reuse_mode, ForwardRole::kObjective) && !hit) {
      const auto bytes = shape_common_bytes(common);
      cache.store(
          parameters, ForwardRole::kObjective, context, std::move(common),
          bytes);
    }
    return value;
  };
  return run_lm(
      problem.initial_parameters, problem.lm_config, normal_equations,
      objective);
}

}  // namespace staged_lm
