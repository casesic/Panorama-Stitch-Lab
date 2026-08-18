#include "staged_lm/multiframe.h"

#include "staged_lm/pose_shape_schur.h"
#include "staged_lm/shape.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace staged_lm {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<float> latent_from_anchor(
    const std::vector<float>& anchor, float limit) {
  if (anchor.size() != kStagedLMBetaDimension || limit <= 0.0F) {
    throw std::runtime_error("MULTIFRAME_BETA_INITIALIZATION_MISMATCH");
  }
  std::vector<float> latent(kStagedLMBetaDimension);
  for (int beta = 0; beta < kStagedLMBetaDimension; ++beta) {
    const float normalized = anchor[beta] / limit;
    if (!(std::abs(normalized) < 1.0F)) {
      throw std::runtime_error("MULTIFRAME_BETA_ANCHOR_OUT_OF_RANGE");
    }
    latent[beta] = std::atanh(normalized);
  }
  return latent;
}

std::vector<float> local_parameters(
    const std::vector<float>& parameters, std::size_t frame_index,
    std::size_t frame_count) {
  const std::size_t frame_offset = multiframe_frame_offset(frame_index);
  const std::size_t beta_offset = multiframe_beta_offset(frame_count);
  std::vector<float> local;
  local.reserve(kStagedLMFrameDimension + kStagedLMBetaDimension);
  local.insert(
      local.end(), parameters.begin() + frame_offset,
      parameters.begin() + frame_offset + kStagedLMFrameDimension);
  local.insert(
      local.end(), parameters.begin() + beta_offset,
      parameters.begin() + beta_offset + kStagedLMBetaDimension);
  return local;
}

PoseShapeCameraInput camera_at(
    const PoseShapeCameraInput& source,
    const std::vector<float>& frame_parameters) {
  PoseShapeCameraInput camera = source;
  camera.pose.parameters = frame_parameters;
  return camera;
}

bool solve_dense(
    const std::vector<double>& hessian, const std::vector<double>& gradient,
    double damping, std::vector<double>& delta) {
  using Matrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const int dimension = static_cast<int>(gradient.size());
  Eigen::Map<const Matrix> mapped(hessian.data(), dimension, dimension);
  Matrix matrix = mapped;
  matrix.diagonal().array() += damping;
  Eigen::Map<const Eigen::VectorXd> mapped_gradient(
      gradient.data(), dimension);
  Eigen::LLT<Matrix> cholesky(matrix);
  if (cholesky.info() == Eigen::Success) {
    const Eigen::VectorXd solution = cholesky.solve(-mapped_gradient);
    if (cholesky.info() == Eigen::Success && solution.allFinite()) {
      delta.assign(solution.data(), solution.data() + dimension);
      return true;
    }
  }
  const Eigen::VectorXd solution = matrix.fullPivLu().solve(-mapped_gradient);
  if (!solution.allFinite()) return false;
  delta.assign(solution.data(), solution.data() + dimension);
  return true;
}

double arrowhead_quadratic_form(
    const SchurSystem& system, const std::vector<double>& delta) {
  const int frame_count = system.frame_count;
  const int frame_dim = system.frame_dim;
  const int beta_dim = system.beta_dim;
  const int beta_offset = frame_count * frame_dim;
  double value = 0.0;
  for (int frame = 0; frame < frame_count; ++frame) {
    const int offset = frame * frame_dim;
    for (int row = 0; row < frame_dim; ++row) {
      for (int column = 0; column < frame_dim; ++column) {
        value += delta[offset + row] *
            system.a[(frame * frame_dim + row) * frame_dim + column] *
            delta[offset + column];
      }
      for (int beta = 0; beta < beta_dim; ++beta) {
        value += 2.0 * delta[offset + row] *
            system.b[(frame * frame_dim + row) * beta_dim + beta] *
            delta[beta_offset + beta];
      }
    }
  }
  for (int row = 0; row < beta_dim; ++row) {
    for (int column = 0; column < beta_dim; ++column) {
      value += delta[beta_offset + row] * system.c[row * beta_dim + column] *
          delta[beta_offset + column];
    }
  }
  return value;
}

std::vector<double> dense_arrowhead(const SchurSystem& system) {
  const int dimension =
      system.frame_count * system.frame_dim + system.beta_dim;
  const int beta_offset = system.frame_count * system.frame_dim;
  std::vector<double> dense(dimension * dimension, 0.0);
  for (int frame = 0; frame < system.frame_count; ++frame) {
    const int offset = frame * system.frame_dim;
    for (int row = 0; row < system.frame_dim; ++row) {
      for (int column = 0; column < system.frame_dim; ++column) {
        dense[(offset + row) * dimension + offset + column] =
            system.a[(frame * system.frame_dim + row) * system.frame_dim +
                     column];
      }
      for (int beta = 0; beta < system.beta_dim; ++beta) {
        const double value =
            system.b[(frame * system.frame_dim + row) * system.beta_dim + beta];
        dense[(offset + row) * dimension + beta_offset + beta] = value;
        dense[(beta_offset + beta) * dimension + offset + row] = value;
      }
    }
  }
  for (int row = 0; row < system.beta_dim; ++row) {
    for (int column = 0; column < system.beta_dim; ++column) {
      dense[(beta_offset + row) * dimension + beta_offset + column] =
          system.c[row * system.beta_dim + column];
    }
  }
  return dense;
}

std::vector<double> structured_gradient(const SchurSystem& system) {
  std::vector<double> gradient = system.g_frame;
  gradient.insert(
      gradient.end(), system.g_beta.begin(), system.g_beta.end());
  return gradient;
}

void validate_problem(const MultiFramePoseShapeLMProblem& problem) {
  if (problem.frames.empty()) {
    throw std::runtime_error("MULTIFRAME_REQUIRES_AT_LEAST_ONE_FRAME");
  }
  if (problem.base_joints.size() != 165 ||
      problem.joint_basis.size() != 1650 ||
      problem.beta_anchor.size() != kStagedLMBetaDimension ||
      problem.parents.size() != 55) {
    throw std::runtime_error("MULTIFRAME_SHARED_MODEL_DIMENSION_MISMATCH");
  }
  for (const auto& frame : problem.frames) {
    if (frame.initial_parameters.size() != kStagedLMFrameDimension ||
        frame.body_pose_anchor.size() != 63) {
      throw std::runtime_error("MULTIFRAME_FRAME_DIMENSION_MISMATCH");
    }
  }
}

}  // namespace

std::size_t multiframe_parameter_dimension(std::size_t frame_count) {
  return frame_count * kStagedLMFrameDimension + kStagedLMBetaDimension;
}

std::size_t multiframe_frame_offset(std::size_t frame_index) {
  return frame_index * kStagedLMFrameDimension;
}

std::size_t multiframe_beta_offset(std::size_t frame_count) {
  return frame_count * kStagedLMFrameDimension;
}

LMRunResult run_multiframe_pose_shape_lm(
    const MultiFramePoseShapeLMProblem& problem) {
  validate_problem(problem);
  const std::size_t frame_count = problem.frames.size();
  const std::size_t dimension = multiframe_parameter_dimension(frame_count);
  std::vector<float> initial;
  initial.reserve(dimension);
  for (const auto& frame : problem.frames) {
    initial.insert(
        initial.end(), frame.initial_parameters.begin(),
        frame.initial_parameters.end());
  }
  const std::vector<float> initial_beta =
      latent_from_anchor(problem.beta_anchor, problem.beta_limit);
  initial.insert(initial.end(), initial_beta.begin(), initial_beta.end());

  // This exact delegation is the T=1 compatibility contract. It keeps the
  // original O4 arithmetic order and LM trajectory untouched.
  if (frame_count == 1) {
    PoseShapeLMProblem single;
    single.initial_parameters = initial;
    single.base_joints = problem.base_joints;
    single.joint_basis = problem.joint_basis;
    single.beta_anchor = problem.beta_anchor;
    single.body_pose_anchor = problem.frames[0].body_pose_anchor;
    single.parents = problem.parents;
    single.cam2 = problem.frames[0].cam2;
    single.cam4 = problem.frames[0].cam4;
    single.beta_limit = problem.beta_limit;
    single.normal_config = problem.normal_config;
    single.lm_config = problem.lm_config;
    single.lm_config.parameter_dimension =
        kStagedLMFrameDimension + kStagedLMBetaDimension;
    return run_pose_shape_lm(single);
  }

  NormalEquationFunction normal_equations =
      [&](const std::vector<float>& parameters) {
        const auto total_started = Clock::now();
        const std::size_t beta_offset = multiframe_beta_offset(frame_count);
        std::vector<float> latent_beta(
            parameters.begin() + beta_offset, parameters.end());
        const ShapeValueResult shape = evaluate_shape_value(
            latent_beta, problem.beta_limit, problem.base_joints,
            problem.joint_basis, 55);
        std::vector<float> dbeta_dlatent(kStagedLMBetaDimension);
        std::vector<float> djoints_dlatent(165 * kStagedLMBetaDimension);
        for (int beta = 0; beta < kStagedLMBetaDimension; ++beta) {
          dbeta_dlatent[beta] = problem.beta_limit *
              (1.0F - shape.tanh_latent[beta] * shape.tanh_latent[beta]);
        }
        for (int joint = 0; joint < 165; ++joint) {
          for (int beta = 0; beta < kStagedLMBetaDimension; ++beta) {
            djoints_dlatent[joint * kStagedLMBetaDimension + beta] =
                problem.joint_basis[joint * kStagedLMBetaDimension + beta] *
                dbeta_dlatent[beta];
          }
        }

        SchurSystem system;
        system.frame_count = static_cast<int>(frame_count);
        system.frame_dim = kStagedLMFrameDimension;
        system.beta_dim = kStagedLMBetaDimension;
        system.a.reserve(
            frame_count * kStagedLMFrameDimension * kStagedLMFrameDimension);
        system.b.reserve(
            frame_count * kStagedLMFrameDimension * kStagedLMBetaDimension);
        system.g_frame.reserve(frame_count * kStagedLMFrameDimension);
        system.c.assign(kStagedLMBetaDimension * kStagedLMBetaDimension, 0.0);
        system.g_beta.assign(kStagedLMBetaDimension, 0.0);
        double jacobian_ms = 0.0;
        double assembly_ms = 0.0;
        bool shape_prior_added = false;

        const PoseModelInput model{shape.joints, problem.parents};
        for (std::size_t frame_index = 0; frame_index < frame_count;
             ++frame_index) {
          const auto& frame = problem.frames[frame_index];
          const std::vector<float> local =
              local_parameters(parameters, frame_index, frame_count);
          const std::vector<float> frame75(
              local.begin(), local.begin() + kStagedLMFrameDimension);
          const PoseShapeCameraInput cam2 = camera_at(frame.cam2, frame75);
          const PoseShapeCameraInput cam4 = camera_at(frame.cam4, frame75);
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
                  jacobian2, jacobian4, local, frame.body_pose_anchor,
                  problem.beta_anchor, problem.normal_config);
          system.a.insert(
              system.a.end(), assembled.a_final.begin(), assembled.a_final.end());
          system.b.insert(
              system.b.end(), assembled.b_final.begin(), assembled.b_final.end());
          system.g_frame.insert(
              system.g_frame.end(), assembled.g_frame_final.begin(),
              assembled.g_frame_final.end());
          for (int index = 0; index < kStagedLMBetaDimension *
                                      kStagedLMBetaDimension; ++index) {
            system.c[index] += assembled.c_data[index];
          }
          for (int beta = 0; beta < kStagedLMBetaDimension; ++beta) {
            system.g_beta[beta] += assembled.g_beta_data[beta];
          }
          if (!shape_prior_added) {
            for (int index = 0; index < kStagedLMBetaDimension *
                                        kStagedLMBetaDimension; ++index) {
              system.c[index] += assembled.c_shape_prior[index];
            }
            for (int beta = 0; beta < kStagedLMBetaDimension; ++beta) {
              system.g_beta[beta] += assembled.g_shape_prior[beta];
            }
            shape_prior_added = true;
          }
          jacobian_ms += jacobian2.total_ms + jacobian4.total_ms;
          assembly_ms += assembled.total_ms;
        }

        DenseNormalEquations output;
        output.hessian = dense_arrowhead(system);
        output.gradient = structured_gradient(system);
        output.jacobian_ms = jacobian_ms;
        output.normal_equations_ms = assembly_ms;
        auto shared_system = std::make_shared<SchurSystem>(std::move(system));
        output.solve_override = [
            shared_system, hessian = output.hessian,
            gradient = output.gradient](
                double damping, std::vector<double>& delta,
                std::string& backend) {
          const PoseShapeSchurResult solved =
              solve_pose_shape_schur(*shared_system, damping, false);
          if (solved.success) {
            delta = solved.full_delta;
            backend = "schur";
            return true;
          }
          backend = "dense_fallback";
          return solve_dense(hessian, gradient, damping, delta);
        };
        output.quadratic_form_override = [shared_system](
            const std::vector<double>& delta) {
          return arrowhead_quadratic_form(*shared_system, delta);
        };
        output.normal_equations_ms = std::max(
            0.0, elapsed_ms(total_started) - output.jacobian_ms);
        return output;
      };

  ObjectiveFunction objective = [&](const std::vector<float>& parameters) {
    double objective_sum = 0.0;
    double shared_shape_anchor = 0.0;
    for (std::size_t frame_index = 0; frame_index < frame_count;
         ++frame_index) {
      const auto& frame = problem.frames[frame_index];
      const std::vector<float> local =
          local_parameters(parameters, frame_index, frame_count);
      const std::vector<float> frame75(
          local.begin(), local.begin() + kStagedLMFrameDimension);
      const PoseShapeValueResult value = evaluate_pose_shape_value(
          local, problem.base_joints, problem.joint_basis,
          problem.beta_anchor, frame.body_pose_anchor, problem.beta_limit,
          problem.normal_config.pose_anchor_scale,
          problem.normal_config.shape_anchor_scale,
          problem.normal_config.charbonnier_eps,
          problem.normal_config.cam2_alpha_per_point,
          problem.normal_config.cam4_alpha_per_point, problem.parents,
          camera_at(frame.cam2, frame75), camera_at(frame.cam4, frame75));
      double primary = 0.0;
      double pose_anchor = 0.0;
      double shape_anchor = 0.0;
      (void)evaluate_pose_shape_objective_from_common(
          local, value, problem.beta_anchor, frame.body_pose_anchor,
          problem.normal_config.pose_anchor_scale,
          problem.normal_config.shape_anchor_scale,
          problem.normal_config.charbonnier_eps,
          problem.normal_config.cam2_alpha_per_point,
          problem.normal_config.cam4_alpha_per_point, &primary, &pose_anchor,
          &shape_anchor);
      objective_sum += primary + pose_anchor;
      if (frame_index == 0) shared_shape_anchor = shape_anchor;
    }
    return objective_sum + shared_shape_anchor;
  };

  LMConfig config = problem.lm_config;
  config.parameter_dimension = static_cast<int>(dimension);
  return run_lm(initial, config, normal_equations, objective);
}

MultiFrameStagedLMResult run_multiframe_staged_pipeline(
    const MultiFrameStagedLMProblem& problem) {
  if (problem.frames.empty()) {
    throw std::runtime_error("MULTIFRAME_PIPELINE_REQUIRES_FRAMES");
  }
  const auto total_started = Clock::now();
  MultiFrameStagedLMResult result;
  result.frames.resize(problem.frames.size());
  MultiFramePoseShapeLMProblem pose_shape;
  pose_shape.base_joints = problem.base_joints;
  pose_shape.joint_basis = problem.joint_basis;
  pose_shape.beta_anchor = problem.beta_anchor;
  pose_shape.parents = problem.parents;
  pose_shape.beta_limit = problem.beta_limit;
  pose_shape.normal_config = problem.pose_shape_normal_config;
  pose_shape.lm_config = problem.pose_shape_lm_config;

  auto started = Clock::now();
  for (std::size_t frame = 0; frame < problem.frames.size(); ++frame) {
    result.frames[frame].stage1 = run_stage1_lm(problem.frames[frame].stage1);
    if (result.frames[frame].stage1.final_parameters.size() != 12) {
      throw std::runtime_error("MULTIFRAME_STAGE1_OUTPUT_DIMENSION_MISMATCH");
    }
  }
  result.stage1_total_ms = elapsed_ms(started);

  started = Clock::now();
  for (std::size_t frame = 0; frame < problem.frames.size(); ++frame) {
    auto& output = result.frames[frame];
    const auto& input = problem.frames[frame];
    output.pose_initial_parameters = output.stage1.final_parameters;
    output.pose_initial_parameters.insert(
        output.pose_initial_parameters.end(), input.body_pose_anchor.begin(),
        input.body_pose_anchor.end());
    if (output.pose_initial_parameters.size() != kStagedLMFrameDimension) {
      throw std::runtime_error("MULTIFRAME_STAGE1_POSE_HANDOFF_MISMATCH");
    }
    PoseLMProblem pose_problem = input.pose;
    pose_problem.cam2.parameters = output.pose_initial_parameters;
    pose_problem.cam4.parameters = output.pose_initial_parameters;
    output.pose = run_pose_lm(pose_problem);
    pose_shape.frames.push_back({
        input.pose_shape_cam2, input.pose_shape_cam4, input.body_pose_anchor,
        output.pose.final_parameters});
  }
  result.pose_total_ms = elapsed_ms(started);

  started = Clock::now();
  result.pose_shape = run_multiframe_pose_shape_lm(pose_shape);
  result.pose_shape_total_ms = elapsed_ms(started);
  const std::size_t beta_offset = multiframe_beta_offset(problem.frames.size());
  result.beta_latent.assign(
      result.pose_shape.final_parameters.begin() + beta_offset,
      result.pose_shape.final_parameters.end());
  result.physical_beta.resize(kStagedLMBetaDimension);
  for (int beta = 0; beta < kStagedLMBetaDimension; ++beta) {
    result.physical_beta[beta] = problem.beta_limit *
        std::tanh(result.beta_latent[beta]);
  }
  for (std::size_t frame = 0; frame < problem.frames.size(); ++frame) {
    const std::size_t offset = multiframe_frame_offset(frame);
    result.frames[frame].final_parameters.assign(
        result.pose_shape.final_parameters.begin() + offset,
        result.pose_shape.final_parameters.begin() + offset +
            kStagedLMFrameDimension);
  }
  result.total_ms = elapsed_ms(total_started);
  result.ms_per_frame = result.total_ms / problem.frames.size();
  return result;
}

}  // namespace staged_lm
