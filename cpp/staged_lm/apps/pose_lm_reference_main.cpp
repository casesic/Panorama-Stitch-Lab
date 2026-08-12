#include "staged_lm/binary_io.h"
#include "staged_lm/pose_lm.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
std::filesystem::path argument(int argc, char** argv, const std::string& name) {
  for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return argv[i + 1];
  throw std::runtime_error("missing argument: " + name);
}
int integer(const std::map<std::string, std::string>& meta, const std::string& key) {
  return std::stoi(meta.at(key));
}
float real(const std::map<std::string, std::string>& meta, const std::string& key) {
  return std::stof(meta.at(key));
}
staged_lm::PoseCameraInput load_camera(
    const std::filesystem::path& fixture, const std::map<std::string, std::string>& meta,
    const std::vector<float>& parameters, const std::string& camera) {
  const int count = integer(meta, camera + "_count");
  staged_lm::PoseCameraInput input;
  input.camera = camera; input.parameters = parameters; input.point_count = count;
  input.left_hand_pose = staged_lm::read_float32_file(fixture / (camera + "_left_hand_pose.f32"), 45);
  input.right_hand_pose = staged_lm::read_float32_file(fixture / (camera + "_right_hand_pose.f32"), 45);
  input.scale = staged_lm::read_float32_file(fixture / (camera + "_scale.f32"), 1)[0];
  input.selected_v_shaped = staged_lm::read_float32_file(fixture / (camera + "_selected_v_shaped.f32"), count * 3);
  input.selected_posedirs = staged_lm::read_float32_file(fixture / (camera + "_selected_posedirs.f32"), staged_lm::kPoseFeatureDimension * count * 3);
  input.selected_lbs_weights = staged_lm::read_float32_file(fixture / (camera + "_selected_lbs_weights.f32"), count * staged_lm::kPoseJointCount);
  input.targets = staged_lm::read_float32_file(fixture / (camera + "_targets.f32"), count * 2);
  input.image_width = integer(meta, "image_width"); input.image_height = integer(meta, "image_height");
  return input;
}
}

int main(int argc, char** argv) {
  try {
    const auto fixture = argument(argc, argv, "--fixture");
    const auto output = argument(argc, argv, "--output");
    const auto meta = staged_lm::read_key_value_file(fixture / "fixture_meta.txt");
    const auto parameters = staged_lm::read_float32_file(fixture / "parameters.f32", 75);
    staged_lm::PoseLMProblem problem;
    problem.model.joints = staged_lm::read_float32_file(fixture / "joints.f32", 55 * 3);
    problem.model.parents = staged_lm::read_int64_file(fixture / "parents.i64", 55);
    problem.cam2 = load_camera(fixture, meta, parameters, "cam2");
    problem.cam4 = load_camera(fixture, meta, parameters, "cam4");
    problem.body_pose_anchor = staged_lm::read_float32_file(fixture / "body_pose_anchor.f32", 63);
    problem.normal_config.charbonnier_eps = real(meta, "charbonnier_eps");
    problem.normal_config.cam2_alpha_per_point = real(meta, "cam2_alpha_per_point");
    problem.normal_config.cam4_alpha_per_point = real(meta, "cam4_alpha_per_point");
    problem.normal_config.pose_anchor_scale = real(meta, "pose_anchor_scale");
    problem.lm_config.parameter_dimension = 75;
    problem.lm_config.max_accepted_steps = integer(meta, "lm_max_accepted_steps");
    problem.lm_config.max_trials = integer(meta, "lm_max_trials");
    problem.lm_config.rel_objective_tolerance = real(meta, "lm_rel_objective_tolerance");
    problem.lm_config.scaled_gradient_tolerance = real(meta, "lm_scaled_gradient_tolerance");
    problem.lm_config.step_tolerance = real(meta, "lm_step_tolerance");
    const staged_lm::LMRunResult result = staged_lm::run_pose_lm(problem);
    std::filesystem::create_directories(output);
    staged_lm::write_float32_file(output / "initial_parameters.f32", result.initial_parameters);
    staged_lm::write_float32_file(output / "final_parameters.f32", result.final_parameters);
    staged_lm::write_float64_file(output / "initial_objective.f64", {result.initial_objective});
    staged_lm::write_float64_file(output / "final_objective.f64", {result.final_objective});
    for (const auto& linearization : result.linearizations) {
      const std::string prefix = "linearization_" + std::to_string(linearization.index);
      staged_lm::write_float32_file(output / (prefix + "_parameters.f32"), linearization.parameters);
      staged_lm::write_float64_file(output / (prefix + "_H.f64"), linearization.hessian);
      staged_lm::write_float64_file(output / (prefix + "_g.f64"), linearization.gradient);
    }
    for (const auto& trial : result.trials) {
      const std::string prefix = "trial_" + std::to_string(trial.index);
      staged_lm::write_float64_file(output / (prefix + "_delta_double.f64"), trial.delta_double);
      staged_lm::write_float32_file(output / (prefix + "_delta_float.f32"), trial.delta_float);
      staged_lm::write_float32_file(output / (prefix + "_candidate.f32"), trial.candidate);
    }
    std::ofstream trajectory(output / "trajectory.tsv");
    trajectory << std::setprecision(17);
    trajectory << "type\tindex\touter\ttrial\tlambda_before\tlambda_after\tobjective\tpredicted\tactual\trho\taccepted\tconvergence\n";
    for (const auto& lin : result.linearizations) trajectory << "linearization\t" << lin.index << '\t' << lin.outer << "\t-1\t" << lin.lambda_before << "\t0\t" << lin.objective << "\t0\t0\t0\t0\t" << lin.convergence << '\n';
    for (const auto& trial : result.trials) trajectory << "trial\t" << trial.index << '\t' << trial.outer << '\t' << trial.trial << '\t' << trial.lambda_before << '\t' << trial.lambda_after << '\t' << trial.candidate_objective << '\t' << trial.predicted_reduction << '\t' << trial.actual_reduction << '\t' << trial.rho << '\t' << int(trial.accepted) << "\tNONE\n";
    staged_lm::write_key_value_file(output / "cpp_output_meta.txt", {
      {"status", result.status}, {"convergence_branch", result.convergence_branch},
      {"initial_lambda", std::to_string(result.initial_lambda)}, {"final_lambda", std::to_string(result.final_lambda)},
      {"accepted_steps", std::to_string(result.accepted_steps)}, {"rejected_trials", std::to_string(result.rejected_trials)},
      {"objective_call_count", std::to_string(result.objective_call_count)}, {"normal_equation_call_count", std::to_string(result.normal_equation_call_count)},
      {"solve_call_count", std::to_string(result.solve_call_count)}, {"candidate_trial_count", std::to_string(result.candidate_trial_count)},
      {"linearization_count", std::to_string(result.linearizations.size())}, {"trial_count", std::to_string(result.trials.size())},
      {"initial_objective_ms", std::to_string(result.initial_objective_ms)}, {"total_objective_ms", std::to_string(result.total_objective_ms)},
      {"total_jacobian_ms", std::to_string(result.total_jacobian_ms)}, {"total_normal_equations_ms", std::to_string(result.total_normal_equations_ms)},
      {"total_solve_ms", std::to_string(result.total_solve_ms)}, {"total_control_ms", std::to_string(result.total_control_ms)}, {"total_lm_ms", std::to_string(result.total_lm_ms)},
    });
    std::cout << "POSE_LM_CPP_REFERENCE_WRITTEN status=" << result.status << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "POSE_LM_CPP_REFERENCE_ERROR: " << error.what() << '\n'; return 1;
  }
}
