#include "staged_lm/multiframe.h"

#include "staged_lm/cuda_evaluators.h"
#include "staged_lm/pose_shape_schur.h"
#include "staged_lm/shape.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

#if STAGED_LM_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace staged_lm {
namespace {

class ProfileRange {
 public:
  explicit ProfileRange(const char* name) {
#if STAGED_LM_HAS_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~ProfileRange() {
#if STAGED_LM_HAS_NVTX
    nvtxRangePop();
#endif
  }
};

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

struct CudaMultiFramePoseShapeWorkspace {
  explicit CudaMultiFramePoseShapeWorkspace(
      const MultiFramePoseShapeLMProblem& source) : problem(source) {
    const std::size_t frame_count = problem.frames.size();
    const std::vector<float> latent = latent_from_anchor(
        problem.beta_anchor, problem.beta_limit);
    local_problems.reserve(frame_count);
    for (const auto& frame : problem.frames) {
      PoseShapeLMProblem local;
      local.initial_parameters = frame.initial_parameters;
      local.initial_parameters.insert(
          local.initial_parameters.end(), latent.begin(), latent.end());
      local.base_joints = problem.base_joints;
      local.joint_basis = problem.joint_basis;
      local.beta_anchor = problem.beta_anchor;
      local.body_pose_anchor = frame.body_pose_anchor;
      local.parents = problem.parents;
      local.cam2 = frame.cam2;
      local.cam4 = frame.cam4;
      local.beta_limit = problem.beta_limit;
      local.normal_config = problem.normal_config;
      local.lm_config = problem.lm_config;
      local.lm_config.parameter_dimension = 85;
      local.evaluator_backend = EvaluatorBackendKind::kCuda;
      local_problems.push_back(std::move(local));
    }
    evaluators.reserve(frame_count);
    for (const auto& local : local_problems) {
      evaluators.push_back(std::make_unique<CudaPoseShapeEvaluator>(local));
    }
    normal_equation_workspace =
        create_cuda_multiframe_pose_shape_ne_workspace(
            static_cast<int>(frame_count));
  }
  ~CudaMultiFramePoseShapeWorkspace() {
    destroy_cuda_multiframe_pose_shape_ne_workspace(
        normal_equation_workspace);
  }

  const MultiFramePoseShapeLMProblem& problem;
  std::vector<PoseShapeLMProblem> local_problems;
  std::vector<std::unique_ptr<CudaPoseShapeEvaluator>> evaluators;
  CudaMultiFramePoseShapeNormalEquationWorkspace*
      normal_equation_workspace = nullptr;
  std::vector<float> parameters;
  bool forward_ready = false;
  bool linearization_ready = false;
};

CudaMultiFramePoseShapeEvaluator::CudaMultiFramePoseShapeEvaluator(
    const MultiFramePoseShapeLMProblem& problem)
    : workspace_(std::make_unique<CudaMultiFramePoseShapeWorkspace>(problem)) {}
CudaMultiFramePoseShapeEvaluator::~CudaMultiFramePoseShapeEvaluator() = default;

void CudaMultiFramePoseShapeEvaluator::forward(
    const std::vector<float>& parameters) {
  ProfileRange profile("multiframe_cuda_pose_shape_forward");
  workspace_->parameters = parameters;
  const std::size_t count = workspace_->evaluators.size();
  for (std::size_t frame = 0; frame < count; ++frame) {
    workspace_->evaluators[frame]->forward(
        local_parameters(parameters, frame, count));
  }
  workspace_->forward_ready = true;
}

double CudaMultiFramePoseShapeEvaluator::evaluate_residual() {
  if (!workspace_->forward_ready) {
    throw std::runtime_error("CUDA_MULTIFRAME_FORWARD_REQUIRED");
  }
  double value = 0.0;
  for (auto& evaluator : workspace_->evaluators)
    value += evaluator->evaluate_residual();
  const std::size_t count = workspace_->evaluators.size();
  if (count > 1) {
    const std::size_t beta_offset = multiframe_beta_offset(count);
    double shape_anchor = 0.0;
    for (int beta = 0; beta < 10; ++beta) {
      const float physical = workspace_->problem.beta_limit *
          std::tanh(workspace_->parameters[beta_offset + beta]);
      const float residual =
          (physical - workspace_->problem.beta_anchor[beta]) *
          workspace_->problem.normal_config.shape_anchor_scale;
      shape_anchor += 0.5 * static_cast<double>(residual) * residual;
    }
    value -= static_cast<double>(count - 1) * shape_anchor;
  }
  return value;
}

void CudaMultiFramePoseShapeEvaluator::linearize(
    const std::vector<float>& parameters) {
  ProfileRange profile("multiframe_cuda_pose_shape_linearize");
  workspace_->parameters = parameters;
  const std::size_t count = workspace_->evaluators.size();
  for (std::size_t frame = 0; frame < count; ++frame) {
    workspace_->evaluators[frame]->linearize(
        local_parameters(parameters, frame, count));
  }
  workspace_->linearization_ready = true;
}

DenseNormalEquations CudaMultiFramePoseShapeEvaluator::build_normal_equations() {
  ProfileRange profile("multiframe_cuda_pose_shape_normal_equations");
  if (!workspace_->linearization_ready) {
    throw std::runtime_error("CUDA_MULTIFRAME_LINEARIZATION_REQUIRED");
  }
  const int frame_count = static_cast<int>(workspace_->evaluators.size());
  SchurSystem system;
  system.frame_count = frame_count;
  system.frame_dim = 75;
  system.beta_dim = 10;
  std::vector<CudaPoseWorkspace*> cam2_workspaces;
  std::vector<CudaPoseWorkspace*> cam4_workspaces;
  std::vector<float> body_pose_anchors;
  cam2_workspaces.reserve(frame_count);
  cam4_workspaces.reserve(frame_count);
  body_pose_anchors.reserve(static_cast<std::size_t>(frame_count) * 63);
  double jacobian_ms = 0.0;
  for (auto& evaluator : workspace_->evaluators) {
    cam2_workspaces.push_back(evaluator->cam2_workspace());
    cam4_workspaces.push_back(evaluator->cam4_workspace());
    jacobian_ms += evaluator->last_jacobian_ms();
  }
  for (const auto& frame : workspace_->problem.frames) {
    body_pose_anchors.insert(body_pose_anchors.end(),
        frame.body_pose_anchor.begin(), frame.body_pose_anchor.end());
  }
  CudaMultiFramePoseShapeNormalEquationResult blocks =
      build_cuda_multiframe_pose_shape_normal_equations(
          workspace_->normal_equation_workspace, cam2_workspaces,
          cam4_workspaces, workspace_->parameters, body_pose_anchors,
          workspace_->problem.beta_anchor, workspace_->problem.beta_limit,
          workspace_->problem.normal_config);
  system.a = std::move(blocks.a);
  system.b = std::move(blocks.b);
  system.c = std::move(blocks.c);
  system.g_frame = std::move(blocks.g_pose);
  system.g_beta = std::move(blocks.g_beta);
  DenseNormalEquations output;
  output.hessian = schur_to_dense_hessian(system);
  output.gradient = schur_gradient(system);
  output.jacobian_ms = jacobian_ms;
  output.normal_equations_ms = blocks.total_ms;
  auto shared = std::make_shared<SchurSystem>(std::move(system));
  output.solve_override = [shared](double damping,
      std::vector<double>& delta, std::string& backend) {
    const PoseShapeSchurResult solved =
        solve_pose_shape_schur(*shared, damping, false);
    if (solved.success) {
      delta = solved.full_delta;
      backend = "schur";
      return true;
    }
    backend = "dense_fallback";
    return solve_schur_dense_fallback(*shared, damping, delta);
  };
  output.quadratic_form_override = [shared](const std::vector<double>& delta) {
    return schur_quadratic_form(*shared, delta);
  };
  return output;
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
    single.evaluator_backend = problem.evaluator_backend;
    return run_pose_shape_lm(single);
  }

  if (problem.evaluator_backend == EvaluatorBackendKind::kCuda) {
    LMConfig config = problem.lm_config;
    config.parameter_dimension = static_cast<int>(dimension);
    std::unique_ptr<CudaMultiFramePoseShapeEvaluator> evaluator;
    {
      ProfileRange profile("multiframe_cuda_pose_shape_workspace");
      evaluator = std::make_unique<CudaMultiFramePoseShapeEvaluator>(problem);
    }
    return run_lm(initial, config, *evaluator);
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
        output.hessian = schur_to_dense_hessian(system);
        output.gradient = schur_gradient(system);
        output.jacobian_ms = jacobian_ms;
        output.normal_equations_ms = assembly_ms;
        auto shared_system = std::make_shared<SchurSystem>(std::move(system));
        output.solve_override = [shared_system](
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
          return solve_schur_dense_fallback(*shared_system, damping, delta);
        };
        output.quadratic_form_override = [shared_system](
            const std::vector<double>& delta) {
          return schur_quadratic_form(*shared_system, delta);
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
  pose_shape.evaluator_backend = problem.evaluator_backend;

  auto started = Clock::now();
  for (std::size_t frame = 0; frame < problem.frames.size(); ++frame) {
    result.frames[frame].stage1 = run_stage1_lm(problem.frames[frame].stage1);
    if (result.frames[frame].stage1.final_parameters.size() != 12) {
      throw std::runtime_error("MULTIFRAME_STAGE1_OUTPUT_DIMENSION_MISMATCH");
    }
  }
  result.stage1_total_ms = elapsed_ms(started);

  started = Clock::now();
  std::vector<PoseLMProblem> pose_problems;
  pose_problems.reserve(problem.frames.size());
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
    pose_problem.evaluator_backend = problem.evaluator_backend;
    pose_problem.cam2.parameters = output.pose_initial_parameters;
    pose_problem.cam4.parameters = output.pose_initial_parameters;
    pose_problems.push_back(std::move(pose_problem));
  }
  if (problem.evaluator_backend == EvaluatorBackendKind::kCuda &&
      problem.batch_cuda_pose) {
    auto pose_results = run_multiframe_pose_lm_cuda(pose_problems);
    for (std::size_t frame = 0; frame < problem.frames.size(); ++frame) {
      result.frames[frame].pose = std::move(pose_results[frame]);
    }
  } else {
    for (std::size_t frame = 0; frame < problem.frames.size(); ++frame) {
      result.frames[frame].pose = run_pose_lm(pose_problems[frame]);
    }
  }
  for (std::size_t frame = 0; frame < problem.frames.size(); ++frame) {
    auto& output = result.frames[frame];
    const auto& input = problem.frames[frame];
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
