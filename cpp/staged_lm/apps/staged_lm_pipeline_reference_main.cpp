#include "staged_lm/binary_io.h"
#include "staged_lm/densekp_output.h"
#include "staged_lm/cuda_evaluators.h"
#include "staged_lm/pipeline.h"
#include "staged_lm/shape.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>

namespace {
using namespace staged_lm;

std::filesystem::path argument(
    int argc, char** argv, const std::string& requested) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == requested) return argv[index + 1];
  }
  throw std::runtime_error("missing " + requested);
}
int integer(const std::map<std::string, std::string>& values,
            const std::string& key) { return std::stoi(values.at(key)); }
float real(const std::map<std::string, std::string>& values,
           const std::string& key) { return std::stof(values.at(key)); }

bool has_flag(int argc, char** argv, const std::string& requested) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == requested) return true;
  }
  return false;
}

std::string optional_argument(
    int argc, char** argv, const std::string& requested,
    const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == requested) return argv[index + 1];
  }
  return fallback;
}

double max_abs_difference(
    const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("CUDA_FORWARD_PARITY_SIZE_MISMATCH");
  }
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result = std::max(
        result, std::abs(static_cast<double>(left[index]) - right[index]));
  }
  return result;
}

double relative_l2_difference(
    const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("CUDA_JACOBIAN_PARITY_SIZE_MISMATCH");
  }
  double difference2 = 0.0, reference2 = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double difference = static_cast<double>(right[index]) - left[index];
    difference2 += difference * difference;
    reference2 += static_cast<double>(left[index]) * left[index];
  }
  return std::sqrt(difference2) / std::max(std::sqrt(reference2), 1.0e-30);
}

double max_abs_difference_double(
    const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size()) throw std::runtime_error("CUDA_NE_PARITY_SIZE");
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result = std::max(result, std::abs(left[index] - right[index]));
  }
  return result;
}

double relative_l2_difference_double(
    const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size()) throw std::runtime_error("CUDA_NE_PARITY_SIZE");
  double difference2 = 0.0, reference2 = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double difference = right[index] - left[index];
    difference2 += difference * difference;
    reference2 += left[index] * left[index];
  }
  return std::sqrt(difference2) / std::max(std::sqrt(reference2), 1.0e-30);
}

std::string precise(double value) {
  std::ostringstream output;
  output << std::scientific << std::setprecision(17) << value;
  return output.str();
}

Stage1CameraInput stage1_camera(
    const std::filesystem::path& fixture,
    const std::map<std::string, std::string>& metadata,
    const std::vector<float>& parameters, const std::string& camera) {
  const int count = integer(metadata, camera + "_count");
  const int offset = camera == "cam2" ? 0 : 6;
  Stage1CameraInput result;
  result.camera = camera;
  for (int index = 0; index < 3; ++index) {
    result.orientation(index) = parameters[offset + index];
    result.translation(index) = parameters[offset + 3 + index];
  }
  result.local_vertices = read_float32_file(
      fixture / ("stage1_" + camera + "_local_vertices.f32"), count * 3);
  const auto center = read_float32_file(
      fixture / ("stage1_" + camera + "_rotation_center.f32"), 3);
  result.rotation_center = Eigen::Vector3f(center[0], center[1], center[2]);
  result.targets = read_float32_file(
      fixture / ("stage1_" + camera + "_targets.f32"), count * 2);
  result.point_count = count;
  result.image_width = integer(metadata, "image_width");
  result.image_height = integer(metadata, "image_height");
  result.irls_epsilon = real(metadata, "irls_epsilon");
  result.alpha_per_point = real(metadata, camera + "_alpha_per_point");
  return result;
}
PoseShapeCameraInput model_camera(
    const std::filesystem::path& fixture,
    const std::map<std::string, std::string>& metadata,
    const std::string& camera,
    const std::shared_ptr<const DenseKPRegressor>& densekp_regressor) {
  const bool densekp = static_cast<bool>(densekp_regressor);
  const int count = densekp ? integer(metadata, camera + "_support_count")
                            : integer(metadata, camera + "_count");
  const int observation_count = densekp
      ? integer(metadata, camera + "_observation_count") : count;
  const std::string prefix = "model_" + camera + "_";
  PoseShapeCameraInput result;
  result.pose.camera = camera;
  result.pose.point_count = count;
  result.pose.left_hand_pose = read_float32_file(
      fixture / (prefix + "left_hand_pose.f32"), 45);
  result.pose.right_hand_pose = read_float32_file(
      fixture / (prefix + "right_hand_pose.f32"), 45);
  result.pose.scale = read_float32_file(fixture / (prefix + "scale.f32"), 1)[0];
  result.pose.selected_posedirs = read_float32_file(
      fixture / (prefix + "selected_posedirs.f32"), 486 * count * 3);
  result.pose.selected_lbs_weights = read_float32_file(
      fixture / (prefix + "selected_lbs_weights.f32"), count * 55);
  const auto targets = read_float32_file(
      fixture / (prefix + "targets.f32"), observation_count * 2);
  if (densekp) {
    auto observation = std::make_shared<DenseKPObservation>();
    observation->regressor = densekp_regressor;
    observation->densekp_ids = read_int32_file(
        fixture / (prefix + "densekp_ids.i32"), observation_count);
    observation->targets = targets;
    result.pose.densekp_observation = std::move(observation);
  } else {
    result.pose.targets = targets;
  }
  result.pose.image_width = integer(metadata, "image_width");
  result.pose.image_height = integer(metadata, "image_height");
  result.selected_base_v_shaped = read_float32_file(
      fixture / (prefix + "selected_base_v_shaped.f32"), count * 3);
  result.selected_beta_shapedirs = read_float32_file(
      fixture / (prefix + "selected_beta_shapedirs.f32"), count * 3 * 10);
  return result;
}

std::map<std::string, std::string> stage_summary(const LMRunResult& result) {
  return {
      {"status", result.status},
      {"initial_objective", std::to_string(result.initial_objective)},
      {"final_objective", std::to_string(result.final_objective)},
      {"initial_lambda", std::to_string(result.initial_lambda)},
      {"final_lambda", std::to_string(result.final_lambda)},
      {"trials", std::to_string(result.trials.size())},
      {"accepted", std::to_string(result.accepted_steps)},
      {"rejected", std::to_string(result.rejected_trials)},
      {"objective_calls", std::to_string(result.objective_call_count)},
      {"linearization_calls", std::to_string(result.normal_equation_call_count)},
      {"compute_ms", std::to_string(result.total_lm_ms)},
      {"jacobian_ms", std::to_string(result.total_jacobian_ms)},
      {"objective_ms", std::to_string(result.total_objective_ms)},
      {"normal_equations_ms", std::to_string(result.total_normal_equations_ms)},
      {"solve_ms", std::to_string(result.total_solve_ms)},
      {"control_ms", std::to_string(result.total_control_ms)},
  };
}

void write_trajectory(const std::filesystem::path& path,
                      const LMRunResult& result) {
  std::ofstream output(path);
  output << std::setprecision(17);
  output << "index,outer,trial,lambda_before,lambda_after,solve_backend,"
            "solve_success,candidate_objective,predicted_reduction,"
            "actual_reduction,rho,accepted\n";
  for (const auto& trial : result.trials) {
    output << trial.index << ',' << trial.outer << ',' << trial.trial << ','
           << trial.lambda_before << ',' << trial.lambda_after << ','
           << trial.solve_backend << ',' << trial.solve_success << ','
           << trial.candidate_objective << ',' << trial.predicted_reduction << ','
           << trial.actual_reduction << ',' << trial.rho << ',' << trial.accepted
           << '\n';
  }
}

#ifdef EIGEN_VECTORIZE
constexpr bool kEigenVectorize = true;
#else
constexpr bool kEigenVectorize = false;
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto fixture = argument(argc, argv, "--fixture");
    const auto output = argument(argc, argv, "--output");
    const bool optimized_densekp_only =
        has_flag(argc, argv, "--optimized-densekp-only");
    const bool cuda_forward_parity =
        has_flag(argc, argv, "--cuda-forward-parity");
    const bool cuda_pose_jacobian_parity =
        has_flag(argc, argv, "--cuda-pose-jacobian-parity");
    const bool cuda_pose_ne_parity =
        has_flag(argc, argv, "--cuda-pose-ne-parity");
    const bool cuda_pose_shape_parity =
        has_flag(argc, argv, "--cuda-pose-shape-parity");
    const EvaluatorBackendKind evaluator_backend = parse_evaluator_backend(
        optional_argument(
            argc, argv, "--evaluator-backend", "cpu").c_str());
    if (kEigenVectorize != static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE)) {
      throw std::runtime_error("EIGEN_VECTORIZE_BUILD_POLICY_MISMATCH");
    }
    if (!kEigenVectorize || !STAGED_LM_EXPECT_MARCH_NATIVE) {
      throw std::runtime_error("OPTIMIZED_NATIVE_BUILD_POLICY_MISMATCH");
    }
    if (Eigen::nbThreads() != 1) {
      throw std::runtime_error("EIGEN_THREAD_POLICY_MISMATCH");
    }
    const std::string densekp_variant_name = "sparse_materialized";
    const auto pipeline_meta = read_key_value_file(fixture / "pipeline_meta.txt");
    const auto stage1_meta = read_key_value_file(
        fixture / "stage1_fixture_meta.txt");
    const auto model_meta = read_key_value_file(fixture / "model_fixture_meta.txt");
    std::shared_ptr<const DenseKPRegressor> densekp_regressor;
    if (model_meta.count("observation_mode") &&
        model_meta.at("observation_mode") == "densekp") {
      auto mutable_regressor = std::make_shared<DenseKPRegressor>();
      mutable_regressor->support_count = integer(model_meta, "densekp_support_count");
      mutable_regressor->densekp_count = integer(model_meta, "densekp_row_count");
      mutable_regressor->support_vertex_ids = read_int32_file(
          fixture / "densekp_support_vertex_ids.i32",
          mutable_regressor->support_count);
      const auto dense_weights = read_float32_file(
          fixture / "densekp_regressor_138x204.f32",
          mutable_regressor->support_count * mutable_regressor->densekp_count);
      build_densekp_sparse_representation(*mutable_regressor, dense_weights);
      densekp_regressor = std::move(mutable_regressor);
    }
    const auto initial = read_float32_file(
        fixture / "stage1_initial_parameters.f32", 12);

    StagedPipelineProblem problem;
    problem.stage1.cam2 = stage1_camera(fixture, stage1_meta, initial, "cam2");
    problem.stage1.cam4 = stage1_camera(fixture, stage1_meta, initial, "cam4");
    const LMConfig common{
        12, integer(pipeline_meta, "stage1_max_accepted_steps"),
        integer(pipeline_meta, "lm_max_trials"),
        std::stod(pipeline_meta.at("rel_objective_tolerance")),
        std::stod(pipeline_meta.at("scaled_gradient_tolerance")),
        std::stod(pipeline_meta.at("step_tolerance"))};
    problem.stage1.lm_config = common;

    problem.pose_shape.base_joints = read_float32_file(
        fixture / "model_base_joints.f32", 165);
    problem.pose_shape.joint_basis = read_float32_file(
        fixture / "model_d_joints_d_beta.f32", 1650);
    problem.pose_shape.beta_anchor = read_float32_file(
        fixture / "model_beta_anchor.f32", 10);
    problem.pose_shape.body_pose_anchor = read_float32_file(
        fixture / "model_body_pose_anchor.f32", 63);
    problem.pose_shape.parents = read_int64_file(
        fixture / "model_parents.i64", 55);
    problem.pose_shape.beta_limit = real(model_meta, "beta_abs_limit");
    problem.pose_shape.cam2 = model_camera(fixture, model_meta, "cam2", densekp_regressor);
    problem.pose_shape.cam4 = model_camera(fixture, model_meta, "cam4", densekp_regressor);
    problem.pose_shape.normal_config = {
        real(model_meta, "charbonnier_eps"),
        real(model_meta, "cam2_alpha"), real(model_meta, "cam4_alpha"),
        real(model_meta, "pose_anchor_scale"),
        real(model_meta, "shape_anchor_scale")};
    problem.pose_shape.lm_config = {
        85, integer(pipeline_meta, "pose_shape_max_accepted_steps"),
        common.max_trials, common.rel_objective_tolerance,
        common.scaled_gradient_tolerance, common.step_tolerance};
    problem.pose_shape.evaluator_backend = evaluator_backend;

    std::vector<float> anchor_joints = problem.pose_shape.base_joints;
    for (int coordinate = 0; coordinate < 165; ++coordinate) {
      for (int beta = 0; beta < 10; ++beta) {
        anchor_joints[coordinate] +=
            problem.pose_shape.joint_basis[coordinate * 10 + beta] *
            problem.pose_shape.beta_anchor[beta];
      }
    }
    problem.pose.model = {anchor_joints, problem.pose_shape.parents};
    problem.pose.body_pose_anchor = problem.pose_shape.body_pose_anchor;
    problem.pose.normal_config = {
        problem.pose_shape.normal_config.charbonnier_eps,
        problem.pose_shape.normal_config.cam2_alpha_per_point,
        problem.pose_shape.normal_config.cam4_alpha_per_point,
        problem.pose_shape.normal_config.pose_anchor_scale};
    problem.pose.lm_config = {
        75, integer(pipeline_meta, "pose_max_accepted_steps"), common.max_trials,
        common.rel_objective_tolerance, common.scaled_gradient_tolerance,
        common.step_tolerance};
    problem.pose.evaluator_backend = evaluator_backend;
    auto prepare_pose_camera = [&](const PoseShapeCameraInput& source) {
      PoseCameraInput camera = source.pose;
      camera.selected_v_shaped = evaluate_dynamic_vertices(
          source.selected_base_v_shaped, source.selected_beta_shapedirs,
          problem.pose_shape.beta_anchor, source.pose.point_count);
      return camera;
    };
    problem.pose.cam2 = prepare_pose_camera(problem.pose_shape.cam2);
    problem.pose.cam4 = prepare_pose_camera(problem.pose_shape.cam4);
    promote_selected_posedirs_to_stable_owner(problem.pose_shape.cam2.pose);
    promote_selected_posedirs_to_stable_owner(problem.pose_shape.cam4.pose);
    problem.pose.cam2.stable_selected_posedirs =
        problem.pose_shape.cam2.pose.stable_selected_posedirs;
    problem.pose.cam4.stable_selected_posedirs =
        problem.pose_shape.cam4.pose.stable_selected_posedirs;
    problem.pose.cam2.selected_posedirs.clear();
    problem.pose.cam4.selected_posedirs.clear();

    if ((cuda_forward_parity || cuda_pose_jacobian_parity || cuda_pose_ne_parity ||
         cuda_pose_shape_parity) &&
        evaluator_backend != EvaluatorBackendKind::kCpu) {
      throw std::runtime_error(
          "CUDA_FORWARD_PARITY_REQUIRES_CPU_TRAJECTORY");
    }
    const StagedPipelineResult result = run_staged_pipeline(problem);
    std::filesystem::create_directories(output);
    if (cuda_forward_parity || cuda_pose_jacobian_parity || cuda_pose_ne_parity ||
        cuda_pose_shape_parity) {
      std::string unavailable_reason;
      if (!cuda_evaluator_available(&unavailable_reason)) {
        throw std::runtime_error(
            "CUDA_EVALUATOR_UNAVAILABLE: " + unavailable_reason);
      }
      auto pose_camera = [](const PoseCameraInput& source,
                            const std::vector<float>& parameters) {
        PoseCameraInput camera = source;
        camera.parameters = parameters;
        return camera;
      };
      const PoseCameraInput pose_cam2 = pose_camera(
          problem.pose.cam2, result.pose.final_parameters);
      const PoseCameraInput pose_cam4 = pose_camera(
          problem.pose.cam4, result.pose.final_parameters);
      const PoseCameraResult pose_cpu2 = evaluate_pose_camera_value(
          problem.pose.model, pose_cam2);
      const PoseCameraResult pose_cpu4 = evaluate_pose_camera_value(
          problem.pose.model, pose_cam4);
      const PoseCameraResult pose_cuda2 = evaluate_pose_camera_value_cuda(
          problem.pose.model, pose_cam2);
      const PoseCameraResult pose_cuda4 = evaluate_pose_camera_value_cuda(
          problem.pose.model, pose_cam4);
      const PoseShapeValueResult shape_cpu = evaluate_pose_shape_value(
          result.pose_shape.final_parameters, problem.pose_shape.base_joints,
          problem.pose_shape.joint_basis, problem.pose_shape.beta_anchor,
          problem.pose_shape.body_pose_anchor, problem.pose_shape.beta_limit,
          problem.pose_shape.normal_config.pose_anchor_scale,
          problem.pose_shape.normal_config.shape_anchor_scale,
          problem.pose_shape.normal_config.charbonnier_eps,
          problem.pose_shape.normal_config.cam2_alpha_per_point,
          problem.pose_shape.normal_config.cam4_alpha_per_point,
          problem.pose_shape.parents, problem.pose_shape.cam2,
          problem.pose_shape.cam4);
      const PoseShapeValueResult shape_cuda = evaluate_pose_shape_value_cuda(
          result.pose_shape.final_parameters, problem.pose_shape);
      const double vertices_error = std::max({
          max_abs_difference(pose_cpu2.vertices, pose_cuda2.vertices),
          max_abs_difference(pose_cpu4.vertices, pose_cuda4.vertices),
          max_abs_difference(shape_cpu.cam2.vertices, shape_cuda.cam2.vertices),
          max_abs_difference(shape_cpu.cam4.vertices, shape_cuda.cam4.vertices)});
      const double densekp_error = std::max({
          max_abs_difference(pose_cpu2.observation_points,
                             pose_cuda2.observation_points),
          max_abs_difference(pose_cpu4.observation_points,
                             pose_cuda4.observation_points),
          max_abs_difference(shape_cpu.cam2.observation_points,
                             shape_cuda.cam2.observation_points),
          max_abs_difference(shape_cpu.cam4.observation_points,
                             shape_cuda.cam4.observation_points)});
      const double residual_error = std::max({
          max_abs_difference(pose_cpu2.raw_residual, pose_cuda2.raw_residual),
          max_abs_difference(pose_cpu4.raw_residual, pose_cuda4.raw_residual),
          max_abs_difference(shape_cpu.cam2.raw_residual,
                             shape_cuda.cam2.raw_residual),
          max_abs_difference(shape_cpu.cam4.raw_residual,
                             shape_cuda.cam4.raw_residual)});
      write_key_value_file(output / "cuda_forward_parity.txt", {
          {"vertices_max_abs", precise(vertices_error)},
          {"densekp_max_abs", precise(densekp_error)},
          {"residual_max_abs", precise(residual_error)},
          {"trajectory_backend", "cpu"},
          {"cuda_pose_linearization", "IMPLEMENTED"},
      });
      if (cuda_pose_jacobian_parity || cuda_pose_ne_parity) {
        const PoseCameraJacobianResult jacobian_cpu2 =
            evaluate_pose_camera_with_jacobian(problem.pose.model, pose_cam2);
        const PoseCameraJacobianResult jacobian_cpu4 =
            evaluate_pose_camera_with_jacobian(problem.pose.model, pose_cam4);
        CudaPoseWorkspace* workspace2 = create_cuda_pose_workspace(
            problem.pose.model, pose_cam2);
        CudaPoseWorkspace* workspace4 = create_cuda_pose_workspace(
            problem.pose.model, pose_cam4);
        // Warm once so the recorded call measures persistent reuse rather than
        // one-time device allocation.
        evaluate_pose_camera_with_jacobian_cuda_with_workspace(
            workspace2, problem.pose.model, pose_cam2);
        evaluate_pose_camera_with_jacobian_cuda_with_workspace(
            workspace4, problem.pose.model, pose_cam4);
        const PoseCameraJacobianResult timed_cuda2 =
            evaluate_pose_camera_with_jacobian_cuda_with_workspace(
                workspace2, problem.pose.model, pose_cam2);
        const PoseCameraJacobianResult timed_cuda4 =
            evaluate_pose_camera_with_jacobian_cuda_with_workspace(
                workspace4, problem.pose.model, pose_cam4);
        const PoseCameraJacobianResult jacobian_cuda2 =
            evaluate_pose_camera_with_jacobian_cuda(problem.pose.model, pose_cam2);
        const PoseCameraJacobianResult jacobian_cuda4 =
            evaluate_pose_camera_with_jacobian_cuda(problem.pose.model, pose_cam4);
        if (cuda_pose_ne_parity) {
          const PoseNormalEquationsResult cpu_ne = build_pose_normal_equations(
              jacobian_cpu2, jacobian_cpu4, result.pose.final_parameters,
              problem.pose.body_pose_anchor, problem.pose.normal_config);
          const PoseNormalEquationsResult cpu_ne_cuda_input =
              build_pose_normal_equations(
                  jacobian_cuda2, jacobian_cuda4,
                  result.pose.final_parameters, problem.pose.body_pose_anchor,
                  problem.pose.normal_config);
          const CudaPoseNormalEquationResult cuda_ne =
              build_cuda_pose_normal_equations(
                  workspace2, workspace4, result.pose.final_parameters,
                  problem.pose.body_pose_anchor, problem.pose.normal_config);
          write_key_value_file(output / "cuda_pose_ne_parity.txt", {
              {"h_max_abs", precise(max_abs_difference_double(
                  cpu_ne.hessian_frame, cuda_ne.equations.hessian))},
              {"h_relative_l2", precise(relative_l2_difference_double(
                  cpu_ne.hessian_frame, cuda_ne.equations.hessian))},
              {"g_max_abs", precise(max_abs_difference_double(
                  cpu_ne.gradient_frame, cuda_ne.equations.gradient))},
              {"g_relative_l2", precise(relative_l2_difference_double(
                  cpu_ne.gradient_frame, cuda_ne.equations.gradient))},
              {"same_input_h_max_abs", precise(max_abs_difference_double(
                  cpu_ne_cuda_input.hessian_frame, cuda_ne.equations.hessian))},
              {"same_input_h_relative_l2", precise(relative_l2_difference_double(
                  cpu_ne_cuda_input.hessian_frame, cuda_ne.equations.hessian))},
              {"same_input_g_max_abs", precise(max_abs_difference_double(
                  cpu_ne_cuda_input.gradient_frame, cuda_ne.equations.gradient))},
              {"same_input_g_relative_l2", precise(relative_l2_difference_double(
                  cpu_ne_cuda_input.gradient_frame, cuda_ne.equations.gradient))},
              {"cpu_normal_equations_ms", precise(cpu_ne.total_normal_equations_ms)},
              {"cuda_gpu_accumulation_ms", precise(cuda_ne.gpu_accumulation_ms)},
              {"cuda_hg_d2h_ms", precise(cuda_ne.hg_d2h_ms)},
              {"cuda_total_normal_equations_ms", precise(
                  cuda_ne.equations.normal_equations_ms)},
          });
        }
        destroy_cuda_pose_workspace(workspace2);
        destroy_cuda_pose_workspace(workspace4);
        write_key_value_file(output / "cuda_pose_jacobian_parity.txt", {
            {"residual_max_abs", precise(std::max(
                max_abs_difference(jacobian_cpu2.value.raw_residual,
                                   jacobian_cuda2.value.raw_residual),
                max_abs_difference(jacobian_cpu4.value.raw_residual,
                                   jacobian_cuda4.value.raw_residual)))},
            {"jacobian_max_abs", precise(std::max(
                max_abs_difference(jacobian_cpu2.active_jacobian,
                                   jacobian_cuda2.active_jacobian),
                max_abs_difference(jacobian_cpu4.active_jacobian,
                                   jacobian_cuda4.active_jacobian)))},
            {"jacobian_relative_l2", precise(std::max(
                relative_l2_difference(jacobian_cpu2.active_jacobian,
                                       jacobian_cuda2.active_jacobian),
                relative_l2_difference(jacobian_cpu4.active_jacobian,
                                       jacobian_cuda4.active_jacobian)))},
            {"cuda_pose_jacobian_ms", precise(
                timed_cuda2.total_jacobian_ms + timed_cuda4.total_jacobian_ms)},
            {"cpu_pose_jacobian_ms", precise(
                jacobian_cpu2.total_jacobian_ms + jacobian_cpu4.total_jacobian_ms)},
            {"cuda_allocation_ms", precise(
                timed_cuda2.cuda_allocation_ms + timed_cuda4.cuda_allocation_ms)},
            {"cuda_memcpy_ms", precise(
                timed_cuda2.cuda_memcpy_ms + timed_cuda4.cuda_memcpy_ms)},
            {"cuda_h2d_ms", precise(
                timed_cuda2.cuda_h2d_ms + timed_cuda4.cuda_h2d_ms)},
            {"cuda_d2h_ms", precise(
                timed_cuda2.cuda_d2h_ms + timed_cuda4.cuda_d2h_ms)},
            {"cuda_kernel_ms", precise(
                timed_cuda2.cuda_kernel_ms + timed_cuda4.cuda_kernel_ms)},
        });
      }
      if (cuda_pose_shape_parity) {
        std::vector<float> d_beta(10), d_joints(1650);
        for (int beta = 0; beta < 10; ++beta) {
          d_beta[beta] = problem.pose_shape.beta_limit *
              (1.0F - shape_cpu.shape.tanh_latent[beta] *
                          shape_cpu.shape.tanh_latent[beta]);
        }
        for (int coordinate = 0; coordinate < 165; ++coordinate)
          for (int beta = 0; beta < 10; ++beta)
            d_joints[coordinate * 10 + beta] =
                problem.pose_shape.joint_basis[coordinate * 10 + beta] *
                d_beta[beta];
        PoseModelInput dynamic_model{shape_cpu.shape.joints,
                                     problem.pose_shape.parents};
        PoseShapeCameraInput shape_cam2 = problem.pose_shape.cam2;
        PoseShapeCameraInput shape_cam4 = problem.pose_shape.cam4;
        shape_cam2.pose.parameters.assign(
            result.pose_shape.final_parameters.begin(),
            result.pose_shape.final_parameters.begin() + 75);
        shape_cam4.pose.parameters = shape_cam2.pose.parameters;
        shape_cam2.pose.selected_v_shaped = shape_cpu.cam2_dynamic_v_shaped;
        shape_cam4.pose.selected_v_shaped = shape_cpu.cam4_dynamic_v_shaped;
        const PoseShapeCameraJacobianResult cpu2 =
            evaluate_pose_shape_camera_with_jacobian(
                dynamic_model, shape_cam2, shape_cpu.cam2_dynamic_v_shaped,
                d_joints, d_beta);
        const PoseShapeCameraJacobianResult cpu4 =
            evaluate_pose_shape_camera_with_jacobian(
                dynamic_model, shape_cam4, shape_cpu.cam4_dynamic_v_shaped,
                d_joints, d_beta);
        CudaPoseWorkspace* ws2 = create_cuda_pose_workspace(
            dynamic_model, shape_cam2.pose);
        CudaPoseWorkspace* ws4 = create_cuda_pose_workspace(
            dynamic_model, shape_cam4.pose);
        evaluate_pose_camera_with_jacobian_cuda_with_workspace(
            ws2, dynamic_model, shape_cam2.pose);
        const std::vector<float> beta2 =
            evaluate_pose_shape_beta_jacobian_cuda_with_workspace(
                ws2, shape_cam2, d_joints, d_beta, true);
        evaluate_pose_camera_with_jacobian_cuda_with_workspace(
            ws4, dynamic_model, shape_cam4.pose);
        const std::vector<float> beta4 =
            evaluate_pose_shape_beta_jacobian_cuda_with_workspace(
                ws4, shape_cam4, d_joints, d_beta, true);
        const CudaPoseShapeNormalEquationResult gpu_blocks =
            build_cuda_pose_shape_normal_equations(
                ws2, ws4, result.pose_shape.final_parameters,
                problem.pose_shape.body_pose_anchor,
                problem.pose_shape.beta_anchor,
                problem.pose_shape.normal_config);
        const PoseShapeNormalEquationsResult cpu_blocks =
            build_pose_shape_normal_equations(
                cpu2, cpu4, result.pose_shape.final_parameters,
                problem.pose_shape.body_pose_anchor,
                problem.pose_shape.beta_anchor,
                problem.pose_shape.normal_config);
        write_key_value_file(output / "cuda_pose_shape_parity.txt", {
            {"residual_max_abs", precise(std::max(
                max_abs_difference(cpu2.pose.value.raw_residual,
                                   shape_cuda.cam2.raw_residual),
                max_abs_difference(cpu4.pose.value.raw_residual,
                                   shape_cuda.cam4.raw_residual)))},
            {"beta_jacobian_max_abs", precise(std::max(
                max_abs_difference(cpu2.beta_jacobian, beta2),
                max_abs_difference(cpu4.beta_jacobian, beta4)))},
            {"beta_jacobian_relative_l2", precise(std::max(
                relative_l2_difference(cpu2.beta_jacobian, beta2),
                relative_l2_difference(cpu4.beta_jacobian, beta4)))},
            {"A_max_abs", precise(max_abs_difference_double(
                cpu_blocks.a_final, gpu_blocks.a))},
            {"A_relative_l2", precise(relative_l2_difference_double(
                cpu_blocks.a_final, gpu_blocks.a))},
            {"B_max_abs", precise(max_abs_difference_double(
                cpu_blocks.b_final, gpu_blocks.b))},
            {"B_relative_l2", precise(relative_l2_difference_double(
                cpu_blocks.b_final, gpu_blocks.b))},
            {"C_max_abs", precise(max_abs_difference_double(
                cpu_blocks.c_final, gpu_blocks.c))},
            {"C_relative_l2", precise(relative_l2_difference_double(
                cpu_blocks.c_final, gpu_blocks.c))},
            {"g_pose_max_abs", precise(max_abs_difference_double(
                cpu_blocks.g_frame_final, gpu_blocks.g_pose))},
            {"g_pose_relative_l2", precise(relative_l2_difference_double(
                cpu_blocks.g_frame_final, gpu_blocks.g_pose))},
            {"g_beta_max_abs", precise(max_abs_difference_double(
                cpu_blocks.g_beta_final, gpu_blocks.g_beta))},
            {"g_beta_relative_l2", precise(relative_l2_difference_double(
                cpu_blocks.g_beta_final, gpu_blocks.g_beta))},
            {"gpu_blocks_ms", precise(gpu_blocks.total_ms)},
        });
        destroy_cuda_pose_workspace(ws2);
        destroy_cuda_pose_workspace(ws4);
      }
    }
    if (densekp_regressor) {
      const OptimizedDenseKPObservation optimized = evaluate_optimized_densekp(
          problem.pose_shape, result.pose_shape.final_parameters);
      write_optimized_densekp_json(output / "optimized_densekp.json", optimized);
    }
    if (optimized_densekp_only) {
      if (!densekp_regressor) {
        throw std::runtime_error(
            "OPTIMIZED_DENSEKP_OUTPUT_REQUIRES_DENSEKP_FIXTURE");
      }
      std::cout << "OPTIMIZED_DENSEKP_CPP_WRITTEN\n";
      return 0;
    }
    write_float32_file(output / "stage1_initial_parameters.f32",
                       result.stage1.initial_parameters);
    write_float32_file(output / "stage1_final_parameters.f32",
                       result.stage1.final_parameters);
    write_float32_file(output / "pose_initial_parameters.f32",
                       result.pose_initial_parameters);
    write_float32_file(output / "pose_final_parameters.f32",
                       result.pose.final_parameters);
    write_float32_file(output / "pose_shape_initial_parameters.f32",
                       result.pose_shape_initial_parameters);
    write_float32_file(output / "pose_shape_final_parameters.f32",
                       result.pose_shape.final_parameters);
    write_float32_file(output / "final_physical_beta.f32",
                       result.final_physical_beta);
    write_key_value_file(output / "stage1_summary.txt", stage_summary(result.stage1));
    write_key_value_file(output / "pose_summary.txt", stage_summary(result.pose));
    write_key_value_file(output / "pose_shape_summary.txt", stage_summary(result.pose_shape));
    write_key_value_file(output / "pipeline_timing.txt", {
        {"stage1_to_pose_ms", std::to_string(result.stage1_to_pose_ms)},
        {"pose_to_pose_shape_ms", std::to_string(result.pose_to_pose_shape_ms)},
        {"final_output_ms", std::to_string(result.final_output_ms)},
        {"compute_ms", std::to_string(result.compute_ms)},
    });
    write_trajectory(output / "stage1_trajectory.csv", result.stage1);
    write_trajectory(output / "pose_trajectory.csv", result.pose);
    write_trajectory(output / "pose_shape_trajectory.csv", result.pose_shape);
    write_key_value_file(output / "build_policy.txt", {
        {"kernel_policy", "optimized_native_stable_p"},
        {"evaluator_backend", evaluator_backend_name(evaluator_backend)},
        {"pose_shape_evaluator_backend", evaluator_backend_name(evaluator_backend)},
        {"densekp_regression_variant", densekp_variant_name},
        {"eigen_vectorize", kEigenVectorize ? "ON" : "OFF"},
        {"march_native", STAGED_LM_EXPECT_MARCH_NATIVE ? "ON" : "OFF"},
        {"eigen_threads", std::to_string(Eigen::nbThreads())},
        {"pose_offset_kernel", "EIGEN_GEMM_NATIVE"},
        {"lbs_selected_transform_kernel", "EIGEN_GEMM_NATIVE"},
        {"value_pose_offset_kernel", "EIGEN_GEMV_NATIVE"},
        {"fast_math", "OFF"}, {"ffp_contract", "OFF"}, {"lto", "OFF"},
    });
    std::cout << "STAGED_LM_PIPELINE_CPP_WRITTEN\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "STAGED_LM_PIPELINE_CPP_ERROR: " << error.what() << '\n';
    return 1;
  }
}
