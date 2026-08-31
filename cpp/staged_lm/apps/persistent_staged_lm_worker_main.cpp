#include "staged_lm/binary_io.h"
#include "staged_lm/densekp_output.h"
#include "staged_lm/multiframe.h"
#include "staged_lm/runtime_preparation.h"
#include "staged_lm/shape.h"

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace staged_lm;
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kRequestMagic = 0x31514d4c;   // LMQ1
constexpr std::uint32_t kResponseMagic = 0x31524d4c;  // LMR1
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kSolve = 1;
constexpr std::uint32_t kShutdown = 2;
constexpr std::uint32_t kMaxPayload = 16 * 1024 * 1024;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string argument(int argc, char** argv, const std::string& requested) {
  for (int index = 1; index + 1 < argc; ++index)
    if (argv[index] == requested) return argv[index + 1];
  throw std::runtime_error("missing " + requested);
}

int integer(const std::map<std::string, std::string>& values,
            const std::string& key) {
  return std::stoi(values.at(key));
}

float real(const std::map<std::string, std::string>& values,
           const std::string& key) {
  return std::stof(values.at(key));
}

template <typename T>
void append(std::vector<std::uint8_t>& output, const T& value) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  output.insert(output.end(), bytes, bytes + sizeof(T));
}

template <typename T>
void append_vector(std::vector<std::uint8_t>& output,
                   const std::vector<T>& values) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
  output.insert(output.end(), bytes, bytes + values.size() * sizeof(T));
}

void append_string(std::vector<std::uint8_t>& output, const std::string& value) {
  append(output, static_cast<std::uint32_t>(value.size()));
  output.insert(output.end(), value.begin(), value.end());
}

class Reader {
 public:
  explicit Reader(const std::vector<std::uint8_t>& data) : data_(data) {}
  template <typename T> T scalar() {
    if (offset_ + sizeof(T) > data_.size()) throw std::runtime_error("REQUEST_TRUNCATED");
    T value;
    std::memcpy(&value, data_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }
  template <typename T> std::vector<T> vector(std::size_t count) {
    if (count > (data_.size() - offset_) / sizeof(T))
      throw std::runtime_error("REQUEST_ARRAY_TRUNCATED");
    std::vector<T> values(count);
    std::memcpy(values.data(), data_.data() + offset_, count * sizeof(T));
    offset_ += count * sizeof(T);
    return values;
  }
  void require_end() const {
    if (offset_ != data_.size()) throw std::runtime_error("REQUEST_TRAILING_BYTES");
  }
 private:
  const std::vector<std::uint8_t>& data_;
  std::size_t offset_ = 0;
};

bool read_exact(void* destination, std::size_t bytes) {
  std::cin.read(reinterpret_cast<char*>(destination), bytes);
  if (std::cin.gcount() == 0 && std::cin.eof()) return false;
  if (static_cast<std::size_t>(std::cin.gcount()) != bytes)
    throw std::runtime_error("PROTOCOL_TRUNCATED");
  return true;
}

void send_response(std::uint32_t status, const std::vector<std::uint8_t>& payload) {
  const std::uint32_t header[] = {
      kResponseMagic, kVersion, status, static_cast<std::uint32_t>(payload.size())};
  std::cout.write(reinterpret_cast<const char*>(header), sizeof(header));
  if (!payload.empty())
    std::cout.write(reinterpret_cast<const char*>(payload.data()), payload.size());
  std::cout.flush();
  if (!std::cout) throw std::runtime_error("PROTOCOL_WRITE_FAILED");
}

struct DynamicInput {
  std::vector<float> frame_parameters, beta;
  std::vector<float> cam2_left, cam2_right, cam4_left, cam4_right;
  float cam2_scale = 0.0F, cam4_scale = 0.0F;
  std::vector<std::int32_t> ids;
  std::vector<float> cam2_targets, cam4_targets;
};

DynamicInput decode_request(const std::vector<std::uint8_t>& payload) {
  Reader reader(payload);
  const auto count = reader.scalar<std::uint32_t>();
  if (count == 0 || count > 138) throw std::runtime_error("INVALID_POINT_COUNT");
  DynamicInput input;
  input.frame_parameters = reader.vector<float>(75);
  input.beta = reader.vector<float>(10);
  input.cam2_left = reader.vector<float>(45);
  input.cam2_right = reader.vector<float>(45);
  input.cam4_left = reader.vector<float>(45);
  input.cam4_right = reader.vector<float>(45);
  input.cam2_scale = reader.scalar<float>();
  input.cam4_scale = reader.scalar<float>();
  input.ids = reader.vector<std::int32_t>(count);
  input.cam2_targets = reader.vector<float>(count * 2);
  input.cam4_targets = reader.vector<float>(count * 2);
  reader.require_end();
  for (std::int32_t id : input.ids)
    if (id < 0 || id >= 138) throw std::runtime_error("INVALID_DENSEKP_ID");
  return input;
}

struct StaticCache {
  std::filesystem::path fixture;
  std::map<std::string, std::string> pipeline_meta, stage_meta, model_meta;
  std::shared_ptr<const DenseKPRegressor> regressor;
  std::vector<float> base_joints, joint_basis;
  std::vector<std::int64_t> parents;
  PoseShapeCameraInput cam2, cam4;
  int static_load_count = 0;

  explicit StaticCache(const std::filesystem::path& path) : fixture(path) {
    pipeline_meta = read_key_value_file(path / "pipeline_meta.txt");
    stage_meta = read_key_value_file(path / "stage1_fixture_meta.txt");
    model_meta = read_key_value_file(path / "model_fixture_meta.txt");
    auto dense = std::make_shared<DenseKPRegressor>();
    dense->support_count = integer(model_meta, "densekp_support_count");
    dense->densekp_count = integer(model_meta, "densekp_row_count");
    dense->support_vertex_ids = read_int32_file(
        path / "densekp_support_vertex_ids.i32", dense->support_count);
    auto weights = read_float32_file(
        path / "densekp_regressor_138x204.f32",
        dense->support_count * dense->densekp_count);
    build_densekp_sparse_representation(*dense, weights);
    regressor = std::move(dense);
    base_joints = read_float32_file(path / "model_base_joints.f32", 165);
    joint_basis = read_float32_file(path / "model_d_joints_d_beta.f32", 1650);
    parents = read_int64_file(path / "model_parents.i64", 55);
    cam2 = load_static_camera("cam2");
    cam4 = load_static_camera("cam4");
    ++static_load_count;
  }

  PoseShapeCameraInput load_static_camera(const std::string& camera) {
    PoseShapeCameraInput result;
    const int count = integer(model_meta, camera + "_support_count");
    const std::string prefix = "model_" + camera + "_";
    result.pose.camera = camera;
    result.pose.point_count = count;
    result.pose.selected_posedirs = read_float32_file(
        fixture / (prefix + "selected_posedirs.f32"), 486 * count * 3);
    result.pose.selected_lbs_weights = read_float32_file(
        fixture / (prefix + "selected_lbs_weights.f32"), count * 55);
    result.pose.image_width = integer(model_meta, "image_width");
    result.pose.image_height = integer(model_meta, "image_height");
    result.selected_base_v_shaped = read_float32_file(
        fixture / (prefix + "selected_base_v_shaped.f32"), count * 3);
    result.selected_beta_shapedirs = read_float32_file(
        fixture / (prefix + "selected_beta_shapedirs.f32"), count * 3 * 10);
    promote_selected_posedirs_to_stable_owner(result.pose);
    return result;
  }
};

PoseShapeCameraInput dynamic_camera(
    const PoseShapeCameraInput& cached, const DynamicInput& input,
    const std::shared_ptr<const DenseKPRegressor>& regressor,
    const std::string& camera) {
  PoseShapeCameraInput result = cached;
  const bool cam2 = camera == "cam2";
  result.pose.left_hand_pose = cam2 ? input.cam2_left : input.cam4_left;
  result.pose.right_hand_pose = cam2 ? input.cam2_right : input.cam4_right;
  result.pose.scale = cam2 ? input.cam2_scale : input.cam4_scale;
  auto observation = std::make_shared<DenseKPObservation>();
  observation->regressor = regressor;
  observation->densekp_ids = input.ids;
  observation->targets = cam2 ? input.cam2_targets : input.cam4_targets;
  result.pose.densekp_observation = std::move(observation);
  return result;
}

struct BuiltProblem {
  MultiFrameStagedLMProblem problem;
  double runtime_preparation_ms = 0.0;
};

BuiltProblem build_problem(const StaticCache& cache, const DynamicInput& input) {
  BuiltProblem built;
  auto& problem = built.problem;
  problem.base_joints = cache.base_joints;
  problem.joint_basis = cache.joint_basis;
  problem.beta_anchor = input.beta;
  problem.parents = cache.parents;
  problem.beta_limit = real(cache.model_meta, "beta_abs_limit");
  const float alpha = 1.0F / static_cast<float>(input.ids.size());
  problem.pose_shape_normal_config = {
      real(cache.model_meta, "charbonnier_eps"), alpha, alpha,
      real(cache.model_meta, "pose_anchor_scale"),
      real(cache.model_meta, "shape_anchor_scale")};
  problem.pose_shape_lm_config = {
      85, integer(cache.pipeline_meta, "pose_shape_max_accepted_steps"),
      integer(cache.pipeline_meta, "lm_max_trials"),
      std::stod(cache.pipeline_meta.at("rel_objective_tolerance")),
      std::stod(cache.pipeline_meta.at("scaled_gradient_tolerance")),
      std::stod(cache.pipeline_meta.at("step_tolerance"))};
  problem.evaluator_backend = EvaluatorBackendKind::kCpu;

  StagedLMFrameProblem frame;
  const std::vector<float> initial(input.frame_parameters.begin(),
                                   input.frame_parameters.begin() + 12);
  const LMConfig stage_config{
      12, integer(cache.pipeline_meta, "stage1_max_accepted_steps"),
      integer(cache.pipeline_meta, "lm_max_trials"),
      std::stod(cache.pipeline_meta.at("rel_objective_tolerance")),
      std::stod(cache.pipeline_meta.at("scaled_gradient_tolerance")),
      std::stod(cache.pipeline_meta.at("step_tolerance"))};
  auto stage_camera = [&](const std::string& camera,
                          const std::vector<float>& targets) {
    Stage1CameraInput value;
    value.camera = camera;
    const int offset = camera == "cam2" ? 0 : 6;
    for (int index = 0; index < 3; ++index) {
      value.orientation(index) = initial[offset + index];
      value.translation(index) = initial[offset + 3 + index];
    }
    value.targets = targets;
    value.point_count = static_cast<int>(input.ids.size());
    value.image_width = integer(cache.stage_meta, "image_width");
    value.image_height = integer(cache.stage_meta, "image_height");
    value.irls_epsilon = real(cache.stage_meta, "irls_epsilon");
    value.alpha_per_point = alpha;
    return value;
  };
  frame.stage1.cam2 = stage_camera("cam2", input.cam2_targets);
  frame.stage1.cam4 = stage_camera("cam4", input.cam4_targets);
  frame.stage1.lm_config = stage_config;
  frame.body_pose_anchor.assign(input.frame_parameters.begin() + 12,
                                input.frame_parameters.end());
  frame.pose_shape_cam2 = dynamic_camera(
      cache.cam2, input, cache.regressor, "cam2");
  frame.pose_shape_cam4 = dynamic_camera(
      cache.cam4, input, cache.regressor, "cam4");
  std::vector<float> anchor_joints = problem.base_joints;
  for (int coordinate = 0; coordinate < 165; ++coordinate)
    for (int beta = 0; beta < 10; ++beta)
      anchor_joints[coordinate] += problem.joint_basis[coordinate * 10 + beta]
          * problem.beta_anchor[beta];
  frame.pose.model = {anchor_joints, problem.parents};
  frame.pose.body_pose_anchor = frame.body_pose_anchor;
  frame.pose.normal_config = {
      problem.pose_shape_normal_config.charbonnier_eps, alpha, alpha,
      problem.pose_shape_normal_config.pose_anchor_scale};
  frame.pose.lm_config = {
      75, integer(cache.pipeline_meta, "pose_max_accepted_steps"),
      stage_config.max_trials, stage_config.rel_objective_tolerance,
      stage_config.scaled_gradient_tolerance, stage_config.step_tolerance};
  auto pose_camera = [&](const PoseShapeCameraInput& source) {
    PoseCameraInput camera = source.pose;
    camera.selected_v_shaped = evaluate_dynamic_vertices(
        source.selected_base_v_shaped, source.selected_beta_shapedirs,
        problem.beta_anchor, source.pose.point_count);
    return camera;
  };
  frame.pose.cam2 = pose_camera(frame.pose_shape_cam2);
  frame.pose.cam4 = pose_camera(frame.pose_shape_cam4);
  const auto prepared = prepare_stage1_geometry(
      input.frame_parameters, problem.beta_anchor, problem.base_joints,
      problem.joint_basis, problem.parents, frame.pose_shape_cam2,
      frame.pose_shape_cam4);
  auto install = [](Stage1CameraInput& destination,
                    const PoseShapeCameraInput& source,
                    const PreparedStage1CameraGeometry& geometry) {
    apply_densekp_operator(*source.pose.densekp_observation,
                           geometry.local_vertices, 3, 1,
                           destination.local_vertices);
    destination.rotation_center = geometry.rotation_center;
  };
  install(frame.stage1.cam2, frame.pose_shape_cam2, prepared.cam2);
  install(frame.stage1.cam4, frame.pose_shape_cam4, prepared.cam4);
  built.runtime_preparation_ms = prepared.preparation_ms;
  problem.frames.push_back(std::move(frame));
  return built;
}

std::vector<std::uint8_t> solve(
    const StaticCache& cache, const std::vector<std::uint8_t>& payload) {
  const auto request_started = Clock::now();
  const auto decode_started = Clock::now();
  DynamicInput input = decode_request(payload);
  const double decode_ms = elapsed_ms(decode_started);
  const auto setup_started = Clock::now();
  BuiltProblem built = build_problem(cache, input);
  const double setup_including_prep_ms = elapsed_ms(setup_started);
  const double dynamic_setup_ms =
      std::max(0.0, setup_including_prep_ms - built.runtime_preparation_ms);
  Eigen::setNbThreads(1);
  MultiFrameStagedLMResult result = run_multiframe_staged_pipeline(built.problem);
  const auto optimized_started = Clock::now();
  PoseShapeLMProblem output_problem;
  output_problem.base_joints = built.problem.base_joints;
  output_problem.joint_basis = built.problem.joint_basis;
  output_problem.beta_anchor = built.problem.beta_anchor;
  output_problem.body_pose_anchor = built.problem.frames[0].body_pose_anchor;
  output_problem.parents = built.problem.parents;
  output_problem.cam2 = built.problem.frames[0].pose_shape_cam2;
  output_problem.cam4 = built.problem.frames[0].pose_shape_cam4;
  output_problem.beta_limit = built.problem.beta_limit;
  std::vector<float> final85 = result.frames[0].final_parameters;
  final85.insert(final85.end(), result.beta_latent.begin(), result.beta_latent.end());
  auto optimized = evaluate_optimized_densekp(output_problem, final85);
  const double optimized_ms = elapsed_ms(optimized_started);

  const auto encode_started = Clock::now();
  std::vector<std::uint8_t> response;
  append(response, static_cast<std::uint32_t>(cache.static_load_count));
  append(response, decode_ms);
  append(response, dynamic_setup_ms);
  append(response, built.runtime_preparation_ms);
  append(response, result.stage1_total_ms);
  append(response, result.pose_total_ms);
  append(response, result.pose_shape_total_ms);
  append(response, optimized_ms);
  append(response, result.total_ms);
  append(response, elapsed_ms(request_started));
  const auto& stage = result.frames[0].stage1;
  const auto& pose = result.frames[0].pose;
  const auto& shape = result.pose_shape;
  append(response, stage.initial_objective);
  append(response, shape.final_objective);
  for (const LMRunResult* run : {&stage, &pose, &shape}) {
    append(response, static_cast<std::int32_t>(run->accepted_steps));
    append(response, static_cast<std::int32_t>(run->rejected_trials));
    append(response, static_cast<std::int32_t>(run->trials.size()));
    append_string(response, run->status);
    append_string(response, run->convergence_branch);
  }
  append_vector(response, result.frames[0].final_parameters);
  append_vector(response, result.physical_beta);
  append(response, static_cast<std::uint32_t>(optimized.point_count));
  append_vector(response, optimized.pairs);
  append_vector(response, optimized.valid);
  const double encode_ms = elapsed_ms(encode_started);
  append(response, encode_ms);
  return response;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto startup = Clock::now();
    const auto static_load_started = Clock::now();
    StaticCache cache(argument(argc, argv, "--static-fixture"));
    const double static_model_load_ms = elapsed_ms(static_load_started);
    std::vector<std::uint8_t> ready;
    append(ready, static_cast<std::uint32_t>(cache.static_load_count));
    append(ready, kVersion);
    append_string(ready, "cpu");
    append(ready, static_cast<std::uint32_t>(75));
    append(ready, static_cast<std::uint32_t>(10));
    append(ready, static_cast<std::uint32_t>(cache.regressor->densekp_count));
    append_string(ready, std::filesystem::weakly_canonical(cache.fixture).string());
    append(ready, elapsed_ms(startup));
    append(ready, static_model_load_ms);
    send_response(0, ready);
    std::cerr << "PERSISTENT_STAGED_LM_WORKER_READY static_load_count="
              << cache.static_load_count << " static_model_load_ms="
              << static_model_load_ms << '\n';
    while (true) {
      std::uint32_t header[4];
      if (!read_exact(header, sizeof(header))) break;
      if (header[0] != kRequestMagic || header[1] != kVersion)
        throw std::runtime_error("PROTOCOL_HEADER_INVALID");
      if (header[3] > kMaxPayload) throw std::runtime_error("PROTOCOL_PAYLOAD_TOO_LARGE");
      std::vector<std::uint8_t> payload(header[3]);
      if (!payload.empty() && !read_exact(payload.data(), payload.size()))
        throw std::runtime_error("PROTOCOL_PAYLOAD_TRUNCATED");
      if (header[2] == kShutdown) {
        send_response(0, {});
        break;
      }
      if (header[2] != kSolve) {
        std::vector<std::uint8_t> error;
        append_string(error, "UNKNOWN_COMMAND");
        send_response(1, error);
        continue;
      }
      try {
        send_response(0, solve(cache, payload));
      } catch (const std::exception& error) {
        std::vector<std::uint8_t> response;
        append_string(response, error.what());
        send_response(1, response);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PERSISTENT_STAGED_LM_WORKER_ERROR: " << error.what() << '\n';
    return 1;
  }
}
