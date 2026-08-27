#include "staged_lm/binary_io.h"
#include "staged_lm/densekp_output.h"
#include "staged_lm/multiframe.h"
#include "staged_lm/runtime_preparation.h"
#include "staged_lm/shape.h"

#include <Eigen/Core>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
using namespace staged_lm;

std::string argument(int argc, char** argv, const std::string& requested) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == requested) return argv[index + 1];
  }
  throw std::runtime_error("missing " + requested);
}

bool has_flag(int argc, char** argv, const std::string& requested) {
  for (int index = 1; index < argc; ++index)
    if (argv[index] == requested) return true;
  return false;
}

std::string optional_argument(
    int argc, char** argv, const std::string& requested,
    const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index)
    if (argv[index] == requested) return argv[index + 1];
  return fallback;
}

std::vector<std::filesystem::path> fixture_paths(const std::string& value) {
  std::vector<std::filesystem::path> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) result.emplace_back(item);
  }
  if (result.empty()) throw std::runtime_error("empty --fixtures");
  return result;
}

int integer(
    const std::map<std::string, std::string>& values,
    const std::string& key) {
  return std::stoi(values.at(key));
}

float real(
    const std::map<std::string, std::string>& values,
    const std::string& key) {
  return std::stof(values.at(key));
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
    const std::shared_ptr<const DenseKPRegressor>& regressor) {
  const bool densekp = static_cast<bool>(regressor);
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
    observation->regressor = regressor;
    observation->densekp_ids = read_int32_file(
        fixture / (prefix + "densekp_ids.i32"), observation_count);
    observation->targets = targets;
    result.pose.densekp_observation = std::move(observation);
  } else result.pose.targets = targets;
  result.pose.image_width = integer(metadata, "image_width");
  result.pose.image_height = integer(metadata, "image_height");
  result.selected_base_v_shaped = read_float32_file(
      fixture / (prefix + "selected_base_v_shaped.f32"), count * 3);
  result.selected_beta_shapedirs = read_float32_file(
      fixture / (prefix + "selected_beta_shapedirs.f32"), count * 3 * 10);
  promote_selected_posedirs_to_stable_owner(result.pose);
  return result;
}

void write_trajectory(
    const std::filesystem::path& path, const LMRunResult& result) {
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

StagedLMFrameProblem load_frame(
    const std::filesystem::path& fixture,
    const std::vector<float>& base_joints,
    const std::vector<float>& joint_basis,
    const std::vector<float>& beta_anchor,
    const std::vector<std::int64_t>& parents,
    const PoseShapeNormalEquationConfig& normal_config,
    const std::shared_ptr<const DenseKPRegressor>& regressor,
    bool runtime_prepare_stage1,
    double* preparation_ms,
    const std::filesystem::path& preparation_debug) {
  const auto pipeline_meta = read_key_value_file(fixture / "pipeline_meta.txt");
  const auto preparation_required = pipeline_meta.find(
      "runtime_prepare_stage1_required");
  if (preparation_required != pipeline_meta.end()
      && preparation_required->second == "true"
      && !runtime_prepare_stage1) {
    throw std::runtime_error(
        "PERSON_RUNTIME_INPUT_REQUIRES_STAGE1_PREPARATION");
  }
  const auto stage1_meta = read_key_value_file(
      fixture / "stage1_fixture_meta.txt");
  const auto model_meta = read_key_value_file(fixture / "model_fixture_meta.txt");
  const auto initial = read_float32_file(
      fixture / "stage1_initial_parameters.f32", 12);
  const LMConfig stage1_config{
      12, integer(pipeline_meta, "stage1_max_accepted_steps"),
      integer(pipeline_meta, "lm_max_trials"),
      std::stod(pipeline_meta.at("rel_objective_tolerance")),
      std::stod(pipeline_meta.at("scaled_gradient_tolerance")),
      std::stod(pipeline_meta.at("step_tolerance"))};
  StagedLMFrameProblem frame;
  frame.stage1.cam2 = stage1_camera(fixture, stage1_meta, initial, "cam2");
  frame.stage1.cam4 = stage1_camera(fixture, stage1_meta, initial, "cam4");
  frame.stage1.lm_config = stage1_config;
  frame.body_pose_anchor = read_float32_file(
      fixture / "model_body_pose_anchor.f32", 63);
  frame.pose_shape_cam2 = model_camera(fixture, model_meta, "cam2", regressor);
  frame.pose_shape_cam4 = model_camera(fixture, model_meta, "cam4", regressor);

  std::vector<float> anchor_joints = base_joints;
  for (int coordinate = 0; coordinate < 165; ++coordinate) {
    for (int beta = 0; beta < 10; ++beta) {
      anchor_joints[coordinate] +=
          joint_basis[coordinate * 10 + beta] * beta_anchor[beta];
    }
  }
  frame.pose.model = {anchor_joints, parents};
  frame.pose.body_pose_anchor = frame.body_pose_anchor;
  frame.pose.normal_config = {
      normal_config.charbonnier_eps,
      normal_config.cam2_alpha_per_point,
      normal_config.cam4_alpha_per_point,
      normal_config.pose_anchor_scale};
  frame.pose.lm_config = {
      75, integer(pipeline_meta, "pose_max_accepted_steps"),
      stage1_config.max_trials, stage1_config.rel_objective_tolerance,
      stage1_config.scaled_gradient_tolerance, stage1_config.step_tolerance};
  auto pose_camera = [&](const PoseShapeCameraInput& source) {
    PoseCameraInput camera = source.pose;
    camera.selected_v_shaped = evaluate_dynamic_vertices(
        source.selected_base_v_shaped, source.selected_beta_shapedirs,
        beta_anchor, source.pose.point_count);
    return camera;
  };
  frame.pose.cam2 = pose_camera(frame.pose_shape_cam2);
  frame.pose.cam4 = pose_camera(frame.pose_shape_cam4);
  if (runtime_prepare_stage1) {
    std::vector<float> frame_parameters = initial;
    frame_parameters.insert(
        frame_parameters.end(), frame.body_pose_anchor.begin(),
        frame.body_pose_anchor.end());
    const PreparedStage1Geometry prepared = prepare_stage1_geometry(
        frame_parameters, beta_anchor, base_joints, joint_basis, parents,
        frame.pose_shape_cam2, frame.pose_shape_cam4);
    auto install = [](Stage1CameraInput& destination,
                      const PoseShapeCameraInput& source,
                      const PreparedStage1CameraGeometry& geometry) {
      if (source.pose.densekp_observation) {
        apply_densekp_operator(
            *source.pose.densekp_observation, geometry.local_vertices, 3, 1,
            destination.local_vertices);
      } else {
        destination.local_vertices = geometry.local_vertices;
      }
      destination.rotation_center = geometry.rotation_center;
      if (destination.local_vertices.size()
          != static_cast<std::size_t>(destination.point_count * 3)) {
        throw std::runtime_error("runtime-prepared Stage1 count mismatch");
      }
    };
    install(frame.stage1.cam2, frame.pose_shape_cam2, prepared.cam2);
    install(frame.stage1.cam4, frame.pose_shape_cam4, prepared.cam4);
    if (!preparation_debug.empty()) {
      std::filesystem::create_directories(preparation_debug);
      auto dump = [&](const std::string& camera,
                      const PreparedStage1CameraGeometry& geometry,
                      const Stage1CameraInput& installed) {
        write_float32_file(
            preparation_debug / (camera + "_selected_vertices.f32"),
            geometry.selected_vertices);
        write_float32_file(
            preparation_debug / (camera + "_support_local_vertices.f32"),
            geometry.local_vertices);
        write_float32_file(
            preparation_debug / (camera + "_stage1_local_vertices.f32"),
            installed.local_vertices);
        write_float32_file(
            preparation_debug / (camera + "_rotation_center.f32"),
            {geometry.rotation_center(0), geometry.rotation_center(1),
             geometry.rotation_center(2)});
      };
      dump("cam2", prepared.cam2, frame.stage1.cam2);
      dump("cam4", prepared.cam4, frame.stage1.cam4);
    }
    if (preparation_ms) *preparation_ms += prepared.preparation_ms;
  }
  return frame;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto app_started = std::chrono::steady_clock::now();
    const auto fixtures = fixture_paths(argument(argc, argv, "--fixtures"));
    const std::filesystem::path output = argument(argc, argv, "--output");
    const bool optimized_densekp_only =
        has_flag(argc, argv, "--optimized-densekp-only");
    const bool runtime_prepare_stage1 =
        has_flag(argc, argv, "--runtime-prepare-stage1");
    const EvaluatorBackendKind evaluator_backend = parse_evaluator_backend(
        optional_argument(argc, argv, "--evaluator-backend", "cpu").c_str());
    const auto first_model_meta = read_key_value_file(
        fixtures.front() / "model_fixture_meta.txt");
    const auto first_pipeline_meta = read_key_value_file(
        fixtures.front() / "pipeline_meta.txt");
    MultiFrameStagedLMProblem problem;
    problem.evaluator_backend = evaluator_backend;
    problem.batch_cuda_pose = !has_flag(argc, argv, "--sequential-cuda-pose");
    std::shared_ptr<const DenseKPRegressor> densekp_regressor;
    if (first_model_meta.count("observation_mode") &&
        first_model_meta.at("observation_mode") == "densekp") {
      auto value = std::make_shared<DenseKPRegressor>();
      value->support_count = integer(first_model_meta, "densekp_support_count");
      value->densekp_count = integer(first_model_meta, "densekp_row_count");
      value->support_vertex_ids = read_int32_file(
          fixtures.front() / "densekp_support_vertex_ids.i32", value->support_count);
      const auto dense_weights = read_float32_file(
          fixtures.front() / "densekp_regressor_138x204.f32",
          value->support_count * value->densekp_count);
      build_densekp_sparse_representation(*value, dense_weights);
      densekp_regressor = std::move(value);
    }
    problem.base_joints = read_float32_file(
        fixtures.front() / "model_base_joints.f32", 165);
    problem.joint_basis = read_float32_file(
        fixtures.front() / "model_d_joints_d_beta.f32", 1650);
    problem.beta_anchor = read_float32_file(
        fixtures.front() / "model_beta_anchor.f32", 10);
    problem.parents = read_int64_file(
        fixtures.front() / "model_parents.i64", 55);
    problem.beta_limit = real(first_model_meta, "beta_abs_limit");
    problem.pose_shape_normal_config = {
        real(first_model_meta, "charbonnier_eps"),
        real(first_model_meta, "cam2_alpha"),
        real(first_model_meta, "cam4_alpha"),
        real(first_model_meta, "pose_anchor_scale"),
        real(first_model_meta, "shape_anchor_scale")};
    problem.pose_shape_lm_config = {
        static_cast<int>(multiframe_parameter_dimension(fixtures.size())),
        integer(first_pipeline_meta, "pose_shape_max_accepted_steps"),
        integer(first_pipeline_meta, "lm_max_trials"),
        std::stod(first_pipeline_meta.at("rel_objective_tolerance")),
        std::stod(first_pipeline_meta.at("scaled_gradient_tolerance")),
        std::stod(first_pipeline_meta.at("step_tolerance"))};
    double runtime_preparation_ms = 0.0;
    for (std::size_t fixture_index = 0; fixture_index < fixtures.size(); ++fixture_index) {
      const auto& fixture = fixtures[fixture_index];
      problem.frames.push_back(load_frame(
          fixture, problem.base_joints, problem.joint_basis,
          problem.beta_anchor, problem.parents,
          problem.pose_shape_normal_config, densekp_regressor,
          runtime_prepare_stage1, &runtime_preparation_ms,
          runtime_prepare_stage1
              ? output / "runtime_prepared_geometry" /
                    ("frame_" + std::to_string(fixture_index))
              : std::filesystem::path{}));
    }
    const double fixture_model_load_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - app_started).count();

    Eigen::setNbThreads(1);
    const MultiFrameStagedLMResult result =
        run_multiframe_staged_pipeline(problem);
    const auto output_started = std::chrono::steady_clock::now();
    std::filesystem::create_directories(output);
    for (std::size_t frame = 0; frame < result.frames.size(); ++frame) {
      const auto directory = output / ("frame_" + std::to_string(frame));
      std::filesystem::create_directories(directory);
      if (densekp_regressor) {
        PoseShapeLMProblem output_problem;
        output_problem.base_joints = problem.base_joints;
        output_problem.joint_basis = problem.joint_basis;
        output_problem.beta_anchor = problem.beta_anchor;
        output_problem.body_pose_anchor = problem.frames[frame].body_pose_anchor;
        output_problem.parents = problem.parents;
        output_problem.cam2 = problem.frames[frame].pose_shape_cam2;
        output_problem.cam4 = problem.frames[frame].pose_shape_cam4;
        output_problem.beta_limit = problem.beta_limit;
        std::vector<float> parameters = result.frames[frame].final_parameters;
        parameters.insert(
            parameters.end(), result.beta_latent.begin(), result.beta_latent.end());
        write_optimized_densekp_json(
            directory / "optimized_densekp.json",
            evaluate_optimized_densekp(output_problem, parameters));
      }
      if (optimized_densekp_only) continue;
      write_float32_file(
          directory / "stage1_final_parameters.f32",
          result.frames[frame].stage1.final_parameters);
      write_trajectory(
          directory / "stage1_trajectory.csv", result.frames[frame].stage1);
      write_float32_file(
          directory / "pose_final_parameters.f32",
          result.frames[frame].pose.final_parameters);
      write_trajectory(
          directory / "pose_trajectory.csv", result.frames[frame].pose);
      write_key_value_file(directory / "fitting_summary.txt", {
          {"stage1_initial_objective", std::to_string(
              result.frames[frame].stage1.initial_objective)},
          {"stage1_final_objective", std::to_string(
              result.frames[frame].stage1.final_objective)},
          {"stage1_status", result.frames[frame].stage1.status},
          {"stage1_trials", std::to_string(
              result.frames[frame].stage1.trials.size())},
          {"pose_initial_objective", std::to_string(
              result.frames[frame].pose.initial_objective)},
          {"pose_final_objective", std::to_string(
              result.frames[frame].pose.final_objective)},
          {"pose_status", result.frames[frame].pose.status},
          {"pose_trials", std::to_string(
              result.frames[frame].pose.trials.size())},
      });
      write_float32_file(
          directory / "pose_shape_frame_parameters.f32",
          result.frames[frame].final_parameters);
    }
    if (optimized_densekp_only) {
      if (!densekp_regressor)
        throw std::runtime_error(
            "OPTIMIZED_DENSEKP_OUTPUT_REQUIRES_DENSEKP_FIXTURE");
      std::cout << "OPTIMIZED_DENSEKP_CPP_WRITTEN frames="
                << result.frames.size() << '\n';
      return 0;
    }
    write_float32_file(output / "shared_beta_latent.f32", result.beta_latent);
    write_float32_file(output / "shared_physical_beta.f32", result.physical_beta);
    write_float32_file(
        output / "pose_shape_final_parameters.f32",
        result.pose_shape.final_parameters);
    write_trajectory(output / "pose_shape_trajectory.csv", result.pose_shape);
    std::size_t pose_linearizations = 0, pose_objectives = 0;
    for (const auto& frame : result.frames) {
      pose_linearizations += frame.pose.normal_equation_call_count;
      pose_objectives += frame.pose.objective_call_count;
    }
    const std::size_t before_h2d_count =
        pose_linearizations * 6 + pose_objectives * 4;
    const std::size_t before_h2d_bytes =
        pose_linearizations * 2664 + pose_objectives * 1344;
    const std::size_t before_d2h_count = pose_linearizations * 4;
    const std::size_t before_d2h_bytes = pose_linearizations * 77280;
    const auto& transfer = result.frames.front().pose;
    const double output_export_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - output_started).count();
    const double app_profiled_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - app_started).count();
    write_key_value_file(output / "multiframe_summary.txt", {
        {"frame_count", std::to_string(result.frames.size())},
        {"parameter_dimension", std::to_string(
            result.pose_shape.final_parameters.size())},
        {"shared_beta_block_count", "1"},
        {"pose_shape_status", result.pose_shape.status},
        {"pose_shape_initial_objective", std::to_string(
            result.pose_shape.initial_objective)},
        {"pose_shape_final_objective", std::to_string(
            result.pose_shape.final_objective)},
        {"pose_shape_accepted", std::to_string(result.pose_shape.accepted_steps)},
        {"pose_shape_rejected", std::to_string(result.pose_shape.rejected_trials)},
        {"pose_shape_trials", std::to_string(result.pose_shape.trials.size())},
        {"stage1_total_ms", std::to_string(result.stage1_total_ms)},
        {"pose_total_ms", std::to_string(result.pose_total_ms)},
        {"pose_shape_total_ms", std::to_string(result.pose_shape_total_ms)},
        {"total_ms", std::to_string(result.total_ms)},
        {"ms_per_frame", std::to_string(result.ms_per_frame)},
        {"fixture_model_load_ms", std::to_string(fixture_model_load_ms)},
        {"runtime_prepare_stage1", runtime_prepare_stage1 ? "true" : "false"},
        {"runtime_preparation_ms", std::to_string(runtime_preparation_ms)},
        {"output_export_ms", std::to_string(output_export_ms)},
        {"app_profiled_ms", std::to_string(app_profiled_ms)},
        {"kernel_policy", "optimized_native_stable_p"},
        {"densekp_variant", "sparse_materialized"},
        {"evaluator_backend", evaluator_backend_name(evaluator_backend)},
        {"pose_cuda_scheduling", problem.batch_cuda_pose ? "batch" : "sequential"},
        {"pose_transfer_before_h2d_count", std::to_string(before_h2d_count)},
        {"pose_transfer_before_h2d_bytes", std::to_string(before_h2d_bytes)},
        {"pose_transfer_before_d2h_count", std::to_string(before_d2h_count)},
        {"pose_transfer_before_d2h_bytes", std::to_string(before_d2h_bytes)},
        {"pose_transfer_after_h2d_count", std::to_string(transfer.batch_h2d_count)},
        {"pose_transfer_after_h2d_bytes", std::to_string(transfer.batch_h2d_bytes)},
        {"pose_transfer_after_d2h_count", std::to_string(transfer.batch_d2h_count)},
        {"pose_transfer_after_d2h_bytes", std::to_string(transfer.batch_d2h_bytes)},
    });
    std::cout << "MULTI_FRAME_STAGED_LM_READY=PASS frames="
              << result.frames.size() << " parameter_dimension="
              << result.pose_shape.final_parameters.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MULTI_FRAME_STAGED_LM_ERROR: " << error.what() << '\n';
    return 1;
  }
}
