#include "staged_lm/binary_io.h"
#include "staged_lm/pipeline.h"
#include "staged_lm/shape.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

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
    const std::string& camera) {
  const int count = integer(metadata, camera + "_count");
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
  result.pose.targets = read_float32_file(
      fixture / (prefix + "targets.f32"), count * 2);
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto fixture = argument(argc, argv, "--fixture");
    const auto output = argument(argc, argv, "--output");
    const auto pipeline_meta = read_key_value_file(fixture / "pipeline_meta.txt");
    const auto stage1_meta = read_key_value_file(
        fixture / "stage1_fixture_meta.txt");
    const auto model_meta = read_key_value_file(fixture / "model_fixture_meta.txt");
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
    problem.pose_shape.cam2 = model_camera(fixture, model_meta, "cam2");
    problem.pose_shape.cam4 = model_camera(fixture, model_meta, "cam4");
    problem.pose_shape.normal_config = {
        real(model_meta, "charbonnier_eps"),
        real(model_meta, "cam2_alpha"), real(model_meta, "cam4_alpha"),
        real(model_meta, "pose_anchor_scale"),
        real(model_meta, "shape_anchor_scale")};
    problem.pose_shape.lm_config = {
        85, integer(pipeline_meta, "pose_shape_max_accepted_steps"),
        common.max_trials, common.rel_objective_tolerance,
        common.scaled_gradient_tolerance, common.step_tolerance};

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
    auto prepare_pose_camera = [&](const PoseShapeCameraInput& source) {
      PoseCameraInput camera = source.pose;
      camera.selected_v_shaped = evaluate_dynamic_vertices(
          source.selected_base_v_shaped, source.selected_beta_shapedirs,
          problem.pose_shape.beta_anchor, source.pose.point_count);
      return camera;
    };
    problem.pose.cam2 = prepare_pose_camera(problem.pose_shape.cam2);
    problem.pose.cam4 = prepare_pose_camera(problem.pose_shape.cam4);

    const StagedPipelineResult result = run_staged_pipeline(problem);
    std::filesystem::create_directories(output);
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
    std::cout << "STAGED_LM_PIPELINE_CPP_WRITTEN\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "STAGED_LM_PIPELINE_CPP_ERROR: " << error.what() << '\n';
    return 1;
  }
}
