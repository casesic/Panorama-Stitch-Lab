#include "staged_lm/densekp_output.h"

#include "staged_lm/shape.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace staged_lm {
namespace {

PoseCameraInput full_densekp_camera(
    const PoseShapeCameraInput& source,
    const std::vector<float>& frame_parameters,
    const std::vector<float>& beta) {
  PoseCameraInput camera = source.pose;
  if (!camera.densekp_observation ||
      !camera.densekp_observation->regressor) {
    throw std::runtime_error("DENSEKP_OUTPUT_REQUIRES_DENSEKP_INPUT");
  }
  camera.parameters = frame_parameters;
  camera.selected_v_shaped = evaluate_dynamic_vertices(
      source.selected_base_v_shaped, source.selected_beta_shapedirs, beta,
      camera.point_count);
  auto observation = std::make_shared<DenseKPObservation>(
      *camera.densekp_observation);
  observation->densekp_ids.resize(observation->regressor->densekp_count);
  std::iota(
      observation->densekp_ids.begin(), observation->densekp_ids.end(), 0);
  observation->targets.assign(observation->densekp_ids.size() * 2, 0.0F);
  camera.densekp_observation = std::move(observation);
  return camera;
}

std::vector<std::uint8_t> input_valid_ids(
    const PoseCameraInput& camera, int point_count) {
  std::vector<std::uint8_t> valid(point_count, 0);
  for (std::int32_t id : camera.densekp_observation->densekp_ids) {
    if (id >= 0 && id < point_count) valid[id] = 1;
  }
  return valid;
}

}  // namespace

OptimizedDenseKPObservation evaluate_optimized_densekp(
    const PoseShapeLMProblem& problem,
    const std::vector<float>& final_parameters) {
  if (final_parameters.size() != 85) {
    throw std::runtime_error("DENSEKP_OUTPUT_PARAMETER_DIMENSION_MISMATCH");
  }
  const std::vector<float> frame_parameters(
      final_parameters.begin(), final_parameters.begin() + 75);
  const std::vector<float> latent(
      final_parameters.begin() + 75, final_parameters.end());
  const ShapeValueResult shape = evaluate_shape_value(
      latent, problem.beta_limit, problem.base_joints, problem.joint_basis, 55);
  const PoseModelInput model{shape.joints, problem.parents};

  const int count = problem.cam2.pose.densekp_observation &&
          problem.cam2.pose.densekp_observation->regressor
      ? problem.cam2.pose.densekp_observation->regressor->densekp_count
      : 0;
  if (count <= 0 || !problem.cam4.pose.densekp_observation ||
      problem.cam4.pose.densekp_observation->regressor->densekp_count != count) {
    throw std::runtime_error("DENSEKP_OUTPUT_REGRESSOR_MISMATCH");
  }

  PoseCameraInput cam2 = full_densekp_camera(
      problem.cam2, frame_parameters, shape.beta);
  PoseCameraInput cam4 = full_densekp_camera(
      problem.cam4, frame_parameters, shape.beta);
  const PoseCameraResult value2 = evaluate_pose_camera_value(model, cam2);
  const PoseCameraResult value4 = evaluate_pose_camera_value(model, cam4);

  OptimizedDenseKPObservation output;
  output.point_count = count;
  output.pairs.resize(static_cast<std::size_t>(count) * 4);
  auto valid2 = input_valid_ids(problem.cam2.pose, count);
  auto valid4 = input_valid_ids(problem.cam4.pose, count);
  output.valid.resize(count);
  for (int point = 0; point < count; ++point) {
    output.pairs[point * 4] = value2.uv[point * 2];
    output.pairs[point * 4 + 1] = value2.uv[point * 2 + 1];
    output.pairs[point * 4 + 2] = value4.uv[point * 2];
    output.pairs[point * 4 + 3] = value4.uv[point * 2 + 1];
    output.valid[point] = valid2[point] && valid4[point];
    for (int component = 0; component < 4; ++component) {
      if (!std::isfinite(output.pairs[point * 4 + component])) {
        throw std::runtime_error("DENSEKP_OUTPUT_NONFINITE_COORDINATE");
      }
    }
  }
  return output;
}

void write_optimized_densekp_json(
    const std::filesystem::path& path,
    const OptimizedDenseKPObservation& observation) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("DENSEKP_OUTPUT_OPEN_FAILED");
  output << std::setprecision(9)
         << "{\n  \"coord_space\": \"erp_rotate\",\n  \"pairs\": [\n";
  for (int point = 0; point < observation.point_count; ++point) {
    output << "    [";
    for (int component = 0; component < 4; ++component) {
      if (component) output << ", ";
      output << observation.pairs[point * 4 + component];
    }
    output << "]" << (point + 1 == observation.point_count ? "\n" : ",\n");
  }
  output << "  ],\n  \"valid\": [";
  for (int point = 0; point < observation.point_count; ++point) {
    if (point) output << ", ";
    output << static_cast<int>(observation.valid[point]);
  }
  output << "]\n}\n";
  if (!output) throw std::runtime_error("DENSEKP_OUTPUT_WRITE_FAILED");
}

}  // namespace staged_lm
