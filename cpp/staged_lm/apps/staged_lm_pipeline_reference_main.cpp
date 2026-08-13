#include "staged_lm/binary_io.h"
#include "staged_lm/pipeline.h"
#include "staged_lm/performance.h"
#include "staged_lm/forward_reuse_cache.h"
#include "staged_lm/shape.h"

#include <Eigen/Core>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
int optional_integer(
    int argc, char** argv, const std::string& requested, int fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == requested) return std::stoi(argv[index + 1]);
  }
  return fallback;
}
std::string optional_string(
    int argc, char** argv, const std::string& requested,
    const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == requested) return argv[index + 1];
  }
  return fallback;
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
    // Native/vectorized builds are the production configuration.  O4 is the
    // only implicit production policy; O6 remains available only by an
    // explicit experimental/diagnostic --kernel-policy selection.
    const std::string default_kernel_policy =
        STAGED_LM_EXPECT_EIGEN_VECTORIZE && STAGED_LM_EXPECT_MARCH_NATIVE
            ? "optimized_native_stable_p"
            : "reference";
    const std::string kernel_policy = optional_string(
        argc, argv, "--kernel-policy", default_kernel_policy);
    if (kernel_policy != "reference" && kernel_policy != "optimized_native" &&
        kernel_policy != "optimized_native_stable_p" &&
        kernel_policy != "optimized_native_stable_p_transposed" &&
        kernel_policy != "experimental_stable_p_block_m512" &&
        kernel_policy != "optimized_native_stable_p_forward_reuse" &&
        kernel_policy != "diagnostic_forward_reuse_lin_to_obj" &&
        kernel_policy != "diagnostic_forward_reuse_obj_to_lin" &&
        kernel_policy != "diagnostic_forward_reuse_obj_to_lin_shadow_warmup" &&
        kernel_policy != "diagnostic_forward_reuse_retain_drop" &&
        kernel_policy != "optimized_native_pre_gemv" &&
        kernel_policy != "value_gemv_shadow" &&
        kernel_policy != "value_gemv_shadow_vectorized" &&
        kernel_policy != "value_gemv_shadow_native") {
      throw std::runtime_error("INVALID_KERNEL_POLICY");
    }
    const bool optimized = kernel_policy == "optimized_native" ||
        kernel_policy == "optimized_native_stable_p" ||
        kernel_policy == "optimized_native_stable_p_transposed" ||
        kernel_policy == "experimental_stable_p_block_m512" ||
        kernel_policy == "optimized_native_stable_p_forward_reuse" ||
        kernel_policy.rfind("diagnostic_forward_reuse_", 0) == 0 ||
        kernel_policy == "optimized_native_pre_gemv";
    const bool value_gemv_shadow = kernel_policy.rfind("value_gemv_shadow", 0) == 0;
    const bool value_gemv_optimized = kernel_policy == "optimized_native" ||
        kernel_policy == "optimized_native_stable_p" ||
        kernel_policy == "optimized_native_stable_p_transposed" ||
        kernel_policy == "experimental_stable_p_block_m512" ||
        kernel_policy == "optimized_native_stable_p_forward_reuse" ||
        kernel_policy.rfind("diagnostic_forward_reuse_", 0) == 0;
    const bool blocked_pose_offset =
        kernel_policy == "experimental_stable_p_block_m512";
    const bool stable_posedirs = kernel_policy == "optimized_native_stable_p" ||
        kernel_policy == "optimized_native_stable_p_transposed" ||
        kernel_policy == "experimental_stable_p_block_m512" ||
        kernel_policy == "optimized_native_stable_p_forward_reuse" ||
        kernel_policy.rfind("diagnostic_forward_reuse_", 0) == 0;
    const bool transposed_pose_offset =
        kernel_policy == "optimized_native_stable_p_transposed";
    const bool vectorized_shadow = kernel_policy == "value_gemv_shadow_vectorized";
    const bool native_shadow = kernel_policy == "value_gemv_shadow_native";
    if (kEigenVectorize != static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE)) {
      throw std::runtime_error("EIGEN_VECTORIZE_BUILD_POLICY_MISMATCH");
    }
    if (optimized && (!kEigenVectorize || !STAGED_LM_EXPECT_MARCH_NATIVE)) {
      throw std::runtime_error("OPTIMIZED_NATIVE_BUILD_POLICY_MISMATCH");
    }
    if (!optimized && !vectorized_shadow && !native_shadow &&
        (kEigenVectorize || STAGED_LM_EXPECT_MARCH_NATIVE)) {
      throw std::runtime_error("REFERENCE_BUILD_POLICY_MISMATCH");
    }
    if (vectorized_shadow && (!kEigenVectorize || STAGED_LM_EXPECT_MARCH_NATIVE))
      throw std::runtime_error("VECTORIZED_SHADOW_BUILD_POLICY_MISMATCH");
    if (native_shadow && (!kEigenVectorize || !STAGED_LM_EXPECT_MARCH_NATIVE))
      throw std::runtime_error("NATIVE_SHADOW_BUILD_POLICY_MISMATCH");
    if (Eigen::nbThreads() != 1) {
      throw std::runtime_error("EIGEN_THREAD_POLICY_MISMATCH");
    }
    set_experimental_pose_offset_gemm(optimized);
    set_experimental_pose_offset_transposed_gemm(transposed_pose_offset);
    set_experimental_pose_offset_blocking(blocked_pose_offset ? 512 : 0, 0);
    set_experimental_selected_transform_gemm(optimized);
    set_experimental_value_pose_offset_gemv(
        value_gemv_shadow || value_gemv_optimized);
    ForwardReuseMode reuse_mode = ForwardReuseMode::kOff;
    if (kernel_policy == "optimized_native_stable_p_forward_reuse")
      reuse_mode = ForwardReuseMode::kFull;
    else if (kernel_policy == "diagnostic_forward_reuse_lin_to_obj")
      reuse_mode = ForwardReuseMode::kLinearizationToObjectiveOnly;
    else if (kernel_policy == "diagnostic_forward_reuse_obj_to_lin")
      reuse_mode = ForwardReuseMode::kObjectiveToLinearizationOnly;
    else if (kernel_policy == "diagnostic_forward_reuse_obj_to_lin_shadow_warmup")
      reuse_mode = ForwardReuseMode::kObjectiveToLinearizationShadowWarmup;
    else if (kernel_policy == "diagnostic_forward_reuse_retain_drop")
      reuse_mode = ForwardReuseMode::kRetainPayloadDropBeforeLinearization;
    set_forward_reuse_mode(reuse_mode);
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
    auto enable_stable_posedirs = [](StagedPipelineProblem& target) {
      promote_selected_posedirs_to_stable_owner(target.pose_shape.cam2.pose);
      promote_selected_posedirs_to_stable_owner(target.pose_shape.cam4.pose);
      target.pose.cam2.stable_selected_posedirs =
          target.pose_shape.cam2.pose.stable_selected_posedirs;
      target.pose.cam4.stable_selected_posedirs =
          target.pose_shape.cam4.pose.stable_selected_posedirs;
      target.pose.cam2.selected_posedirs.clear();
      target.pose.cam4.selected_posedirs.clear();
    };

    const bool benchmark = std::filesystem::path(argv[0]).filename().string().find(
        "benchmark") != std::string::npos;
    if (benchmark) {
      const bool derivative_attribution = optional_integer(
          argc, argv, "--derivative-attribution", 0) != 0;
      const bool forward_reuse_attribution = optional_integer(
          argc, argv, "--forward-reuse-attribution", 0) != 0;
      const int warmup = optional_integer(argc, argv, "--warmup", 1);
      const int repeat = optional_integer(argc, argv, "--repeat", 5);
      for (int index = 0; index < warmup; ++index) {
        StagedPipelineProblem sample_problem = problem;
        if (stable_posedirs) enable_stable_posedirs(sample_problem);
        reset_kernel_dispatch_counters();
        (void)run_staged_pipeline(sample_problem);
      }
      std::filesystem::create_directories(output);
      std::ofstream runs(output / "pipeline_runs.csv");
      runs << "run,compute_ms,wall_ms,peak_rss_kib,stage1_ms,pose_ms,pose_shape_ms,"
              "stage1_jacobian_ms,stage1_objective_ms,stage1_ne_ms,stage1_solve_ms,stage1_control_ms,"
              "pose_jacobian_ms,pose_objective_ms,pose_ne_ms,pose_solve_ms,pose_control_ms,"
              "pose_shape_jacobian_ms,pose_shape_objective_ms,pose_shape_ne_ms,pose_shape_solve_ms,pose_shape_control_ms,final_checksum\n";
      std::ofstream kernels(output / "kernel_samples.csv");
      kernels << "run,stage,camera,kernel,point_count,milliseconds\n";
      if (forward_reuse_attribution) {
        StagedPipelineProblem attribution_problem = problem;
        if (stable_posedirs) enable_stable_posedirs(attribution_problem);
        reset_performance_samples();
        set_forward_reuse_attribution(true);
        set_performance_recording(true);
        (void)run_staged_pipeline(attribution_problem);
        set_performance_recording(false);
        set_forward_reuse_attribution(false);
        auto hash_parameters = [](const std::vector<float>& values) {
          std::uint64_t hash = 1469598103934665603ULL;
          const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
          for (std::size_t i = 0; i < values.size() * sizeof(float); ++i) {
            hash ^= bytes[i]; hash *= 1099511628211ULL;
          }
          return hash;
        };
        std::ofstream timeline(output / "lm_state_calls_raw.csv");
        timeline << "stage,chronological_index,outer,trial,call_kind,disposition,"
                    "parameter_dimension,parameter_hash\n";
        for (const auto& call : lm_state_call_samples()) {
          timeline << call.stage << ',' << call.chronological_index << ','
                   << call.outer << ',' << call.trial << ',' << call.call_kind
                   << ',' << call.disposition << ',' << call.parameters.size()
                   << ',' << std::hex << hash_parameters(call.parameters)
                   << std::dec << '\n';
        }
        std::ofstream forwards(output / "forward_value_calls_raw.csv");
        forwards << "stage,call_index,camera,N,parameter_hash,rotations_ms,"
                    "pose_offsets_ms,fk_ms,lbs_ms,erp_ms,total_ms,payload_bytes,";
        for (int field = 0; field < 10; ++field)
          forwards << "field" << field << "_hash" << (field == 9 ? '\n' : ',');
        int forward_index = 0;
        for (const auto& value : forward_value_samples()) {
          std::size_t bytes = 0; for (auto count : value.field_bytes) bytes += count;
          forwards << value.stage << ',' << forward_index++ << ',' << value.camera
                   << ',' << value.point_count << ',' << std::hex
                   << hash_parameters(value.parameters) << std::dec << ','
                   << value.rotations_ms << ',' << value.pose_offsets_ms << ','
                   << value.fk_ms << ',' << value.lbs_ms << ',' << value.erp_ms
                   << ',' << value.total_ms << ',' << bytes << ',';
          for (std::size_t field = 0; field < value.field_hashes.size(); ++field)
            forwards << std::hex << value.field_hashes[field] << std::dec
                     << (field + 1 == value.field_hashes.size() ? '\n' : ',');
        }
        std::ofstream attribution_kernels(output / "forward_component_calls_raw.csv");
        attribution_kernels << "stage,camera,kernel,N,milliseconds\n";
        for (const auto& sample : performance_samples())
          attribution_kernels << sample.stage << ',' << sample.camera << ','
                              << sample.kernel << ',' << sample.point_count << ','
                              << sample.milliseconds << '\n';
        std::ofstream shape_calls(output / "shape_forward_calls_raw.csv");
        shape_calls << "stage,call_index,parameter_hash,shape_hash,cam2_dynamic_hash,"
                       "cam4_dynamic_hash,payload_bytes\n";
        int shape_index=0;
        for(const auto& value:shape_forward_samples())
          shape_calls<<value.stage<<','<<shape_index++<<','<<std::hex
                     <<hash_parameters(value.parameters)<<','<<value.shape_hash<<','
                     <<value.cam2_dynamic_hash<<','<<value.cam4_dynamic_hash<<std::dec
                     <<','<<value.payload_bytes<<'\n';
      }
      std::vector<float> baseline;
      for (int index = 0; index < repeat; ++index) {
        StagedPipelineProblem sample_problem = problem;
        if (stable_posedirs) enable_stable_posedirs(sample_problem);
        reset_performance_samples();
        reset_kernel_dispatch_counters();
        set_pose_offset_derivative_attribution(
            derivative_attribution, derivative_attribution && index == 0);
        set_performance_recording(true);
        const auto wall_started = std::chrono::steady_clock::now();
        const StagedPipelineResult measured = run_staged_pipeline(sample_problem);
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_started).count();
        set_performance_recording(false);
        set_pose_offset_derivative_attribution(false, false);
        if (index == 0) baseline = measured.pose_shape.final_parameters;
        if (measured.pose_shape.final_parameters != baseline) {
          throw std::runtime_error("BENCHMARK_REPETITION_STATE_DRIFT");
        }
        float checksum = 0.0F;
        for (float value : measured.pose_shape.final_parameters) checksum += value;
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        runs << index << ',' << measured.compute_ms << ',' << wall_ms << ','
             << usage.ru_maxrss << ','
             << measured.stage1.total_lm_ms << ',' << measured.pose.total_lm_ms << ','
             << measured.pose_shape.total_lm_ms << ','
             << measured.stage1.total_jacobian_ms << ',' << measured.stage1.total_objective_ms << ',' << measured.stage1.total_normal_equations_ms << ',' << measured.stage1.total_solve_ms << ',' << measured.stage1.total_control_ms << ','
             << measured.pose.total_jacobian_ms << ',' << measured.pose.total_objective_ms << ',' << measured.pose.total_normal_equations_ms << ',' << measured.pose.total_solve_ms << ',' << measured.pose.total_control_ms << ','
             << measured.pose_shape.total_jacobian_ms << ',' << measured.pose_shape.total_objective_ms << ',' << measured.pose_shape.total_normal_equations_ms << ',' << measured.pose_shape.total_solve_ms << ',' << measured.pose_shape.total_control_ms << ','
             << checksum << '\n';
        for (const auto& sample : performance_samples()) {
          kernels << index << ',' << sample.stage << ',' << sample.camera << ','
                  << sample.kernel << ',' << sample.point_count << ','
                  << sample.milliseconds << '\n';
        }
        if (derivative_attribution) {
          std::ofstream fine(
              output / ("derivative_attribution_" + std::to_string(index) + ".csv"));
          fine << "stage,camera,call_index,N,Q,total_ms,d_pose_feature_ms,"
                  "setup_ms,output_prepare_ms,gemm_core_ms,postprocess_ms,other_ms,"
                  "output_size,capacity_before,capacity_after,pointer_before,"
                  "pointer_after,selected_posedirs_pointer\n";
          int call_index = 0;
          for (const auto& sample : pose_offset_derivative_attribution_samples()) {
            fine << sample.stage << ',' << sample.camera << ',' << call_index++ << ','
                 << sample.point_count << ',' << sample.parameter_count << ','
                 << sample.total_ms << ',' << sample.d_pose_feature_ms << ','
                 << sample.setup_ms << ',' << sample.output_preparation_ms << ','
                 << sample.gemm_core_ms << ',' << sample.postprocess_ms << ','
                 << sample.other_ms << ',' << sample.output_size << ','
                 << sample.capacity_before << ',' << sample.capacity_after << ','
                 << sample.pointer_before << ',' << sample.pointer_after << ','
                 << sample.selected_posedirs_pointer << '\n';
          }
          if (index == 0) {
            const auto corpus = output / "derivative_input_corpus";
            std::filesystem::create_directories(corpus);
            std::ofstream manifest(corpus / "manifest.csv");
            manifest << "call_index,stage,camera,N,Q,selected_posedirs_pointer,"
                        "selected_posedirs_file,d_pose_feature_file\n";
            int input_index = 0;
            for (const auto& state : pose_offset_derivative_input_states()) {
              const std::string posedirs = "selected_posedirs_" +
                  std::to_string(input_index) + ".f32";
              const std::string feature = "d_pose_feature_" +
                  std::to_string(input_index) + ".f32";
              write_float32_file(corpus / posedirs, state.selected_posedirs);
              write_float32_file(corpus / feature, state.d_pose_feature);
              manifest << input_index << ',' << state.stage << ',' << state.camera
                       << ',' << state.point_count << ",66,"
                       << state.selected_posedirs_pointer << ',' << posedirs << ','
                       << feature << '\n';
              ++input_index;
            }
          }
        }
        if (index == 0) {
          std::ofstream manifest(output / "value_pose_offset_call_manifest.csv");
          manifest << "call_index,stage,camera,N,M,K,feature_file\n";
          int state_index = 0;
          for (const auto& state : value_pose_offset_states()) {
            const std::string filename = "value_pose_feature_" +
                std::to_string(state_index) + ".f32";
            write_float32_file(output / filename, state.pose_feature);
            manifest << state_index << ',' << state.stage << ',' << state.camera
                     << ',' << state.point_count << ',' << state.point_count * 3
                     << ",486," << filename << '\n';
            ++state_index;
          }
        }
        const auto& counters = kernel_dispatch_counters();
        const auto& copy_audit = selected_posedirs_copy_audit();
        std::ofstream dispatch(output / ("dispatch_" + std::to_string(index) + ".txt"));
        dispatch << "kernel_policy=" << kernel_policy << '\n'
                 << "pose_offset_scalar_calls=" << counters.pose_offset_scalar_calls << '\n'
                 << "pose_offset_optimized_calls=" << counters.pose_offset_optimized_calls << '\n'
                 << "pose_offset_transposed_calls=" << counters.pose_offset_transposed_calls << '\n'
                 << "lbs_selected_transform_scalar_calls=" << counters.lbs_selected_transform_scalar_calls << '\n'
                 << "lbs_selected_transform_optimized_calls=" << counters.lbs_selected_transform_optimized_calls << '\n'
                 << "value_pose_offset_scalar_calls=" << counters.value_pose_offset_scalar_calls << '\n'
                 << "value_pose_offset_gemv_calls=" << counters.value_pose_offset_gemv_calls << '\n'
                 << "value_pose_offset_gemv_fallback_count=" << counters.value_pose_offset_gemv_fallback_count << '\n'
                 << "optimized_fallback_count=" << counters.optimized_fallback_count << '\n';
        dispatch << "selected_posedirs_policy="
                 << (stable_posedirs ? "stable_reference" : "owning_copy") << '\n'
                 << "derivative_call_posedirs_deep_copy_count="
                 << copy_audit.derivative_call_deep_copy_count << '\n'
                 << "derivative_call_posedirs_deep_copy_bytes="
                 << copy_audit.derivative_call_deep_copy_bytes << '\n'
                 << "derivative_call_posedirs_deep_copy_ms="
                 << copy_audit.derivative_call_deep_copy_ms << '\n'
                 << "cam2_posedirs_pointer="
                 << reinterpret_cast<std::uintptr_t>(
                        selected_posedirs_view(sample_problem.pose_shape.cam2.pose).data()) << '\n'
                 << "cam4_posedirs_pointer="
                 << reinterpret_cast<std::uintptr_t>(
                        selected_posedirs_view(sample_problem.pose_shape.cam4.pose).data()) << '\n';
        std::ofstream cache_file(output/("cache_events_"+std::to_string(index)+".csv"));
        cache_file<<"stage,index,consumer_kind,producer_kind,state_hash,lookup_result,cache_action,key_copy_bytes,payload_bytes,camera_forwards_skipped,cumulative_overhead_ms\n";
        int cache_index=0;for(const auto&e:forward_cache_events())cache_file<<e.stage<<','<<cache_index++<<','<<e.consumer_kind<<','<<e.producer_kind<<','<<std::hex<<e.state_hash<<std::dec<<','<<e.lookup_result<<','<<e.cache_action<<','<<e.key_copy_bytes<<','<<e.payload_bytes<<','<<e.camera_forwards_skipped<<','<<e.cumulative_overhead_ms<<'\n';
      }
      write_key_value_file(output / "build_policy.txt", {
          {"kernel_policy", kernel_policy},
          {"selected_posedirs_policy", stable_posedirs ? "stable_reference" : "owning_copy"},
          {"stable_owner_type", stable_posedirs ? "shared_ptr<const_vector_float>" : "none"},
          {"eigen_vectorize", kEigenVectorize ? "ON" : "OFF"},
          {"march_native", STAGED_LM_EXPECT_MARCH_NATIVE ? "ON" : "OFF"},
          {"eigen_threads", std::to_string(Eigen::nbThreads())},
          {"pose_offset_kernel", optimized ? "EIGEN_GEMM_NATIVE" : "SCALAR"},
          {"lbs_selected_transform_kernel", optimized ? "EIGEN_GEMM_NATIVE" : "SCALAR"},
          {"value_pose_offset_kernel", (value_gemv_shadow || value_gemv_optimized) ? "EIGEN_GEMV_NATIVE" : "SCALAR"},
          {"fast_math", "OFF"}, {"ffp_contract", "OFF"}, {"lto", "OFF"},
      });
      std::cout << "STAGED_LM_PIPELINE_BENCHMARK_WRITTEN\n";
      return 0;
    }

    if (stable_posedirs) enable_stable_posedirs(problem);
    reset_kernel_dispatch_counters();
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
    write_trajectory(output / "stage1_trajectory.csv", result.stage1);
    write_trajectory(output / "pose_trajectory.csv", result.pose);
    write_trajectory(output / "pose_shape_trajectory.csv", result.pose_shape);
    const auto& counters = kernel_dispatch_counters();
    write_key_value_file(output / "kernel_dispatch.txt", {
        {"kernel_policy", kernel_policy},
        {"pose_offset_scalar_calls", std::to_string(counters.pose_offset_scalar_calls)},
        {"pose_offset_optimized_calls", std::to_string(counters.pose_offset_optimized_calls)},
        {"pose_offset_transposed_calls", std::to_string(counters.pose_offset_transposed_calls)},
        {"lbs_selected_transform_scalar_calls", std::to_string(counters.lbs_selected_transform_scalar_calls)},
        {"lbs_selected_transform_optimized_calls", std::to_string(counters.lbs_selected_transform_optimized_calls)},
        {"value_pose_offset_scalar_calls", std::to_string(counters.value_pose_offset_scalar_calls)},
        {"value_pose_offset_gemv_calls", std::to_string(counters.value_pose_offset_gemv_calls)},
        {"value_pose_offset_gemv_fallback_count", std::to_string(counters.value_pose_offset_gemv_fallback_count)},
        {"optimized_fallback_count", std::to_string(counters.optimized_fallback_count)},
    });
    write_key_value_file(output / "build_policy.txt", {
        {"kernel_policy", kernel_policy},
        {"eigen_vectorize", kEigenVectorize ? "ON" : "OFF"},
        {"march_native", STAGED_LM_EXPECT_MARCH_NATIVE ? "ON" : "OFF"},
        {"eigen_threads", std::to_string(Eigen::nbThreads())},
        {"pose_offset_kernel", optimized ? "EIGEN_GEMM_NATIVE" : "SCALAR"},
        {"lbs_selected_transform_kernel", optimized ? "EIGEN_GEMM_NATIVE" : "SCALAR"},
        {"value_pose_offset_kernel", (value_gemv_shadow || value_gemv_optimized) ? "EIGEN_GEMV_NATIVE" : "SCALAR"},
        {"fast_math", "OFF"}, {"ffp_contract", "OFF"}, {"lto", "OFF"},
    });
    std::cout << "STAGED_LM_PIPELINE_CPP_WRITTEN\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "STAGED_LM_PIPELINE_CPP_ERROR: " << error.what() << '\n';
    return 1;
  }
}
