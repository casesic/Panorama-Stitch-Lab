#include "staged_lm/runtime_preparation.h"

#include "staged_lm/shape.h"

#include <chrono>
#include <stdexcept>

namespace staged_lm {
namespace {

PreparedStage1CameraGeometry prepare_camera(
    const std::string& camera_name,
    const std::vector<float>& frame_parameters,
    const ShapeValueResult& shape,
    const std::vector<std::int64_t>& parents,
    const PoseShapeCameraInput& source,
    const std::vector<float>& physical_beta) {
  PoseCameraInput camera = source.pose;
  camera.parameters = frame_parameters;
  const int offset = camera_name == "cam2" ? 0 : 6;
  for (int component = 0; component < 6; ++component) {
    camera.parameters[offset + component] = 0.0F;
  }
  camera.selected_v_shaped = evaluate_dynamic_vertices(
      source.selected_base_v_shaped, source.selected_beta_shapedirs,
      physical_beta, source.pose.point_count);
  // Preparation is geometry-only; residual targets and DenseKP regression are
  // not consumed, but evaluate_pose_camera_value validates their ownership.
  PoseModelInput model{shape.joints, parents};
  const PoseCameraResult value = evaluate_pose_camera_value(model, camera);
  PreparedStage1CameraGeometry output;
  output.selected_vertices = value.vertices;
  output.rotation_center = Eigen::Vector3f(
      shape.joints[0] * camera.scale,
      shape.joints[1] * camera.scale,
      shape.joints[2] * camera.scale);
  output.local_vertices.resize(value.vertices.size());
  for (int vertex = 0; vertex < camera.point_count; ++vertex) {
    for (int xyz = 0; xyz < 3; ++xyz) {
      output.local_vertices[vertex * 3 + xyz] =
          value.vertices[vertex * 3 + xyz] - output.rotation_center(xyz);
    }
  }
  return output;
}

}  // namespace

PreparedStage1Geometry prepare_stage1_geometry(
    const std::vector<float>& frame_parameters,
    const std::vector<float>& physical_beta,
    const std::vector<float>& base_joints,
    const std::vector<float>& joint_basis,
    const std::vector<std::int64_t>& parents,
    const PoseShapeCameraInput& cam2,
    const PoseShapeCameraInput& cam4) {
  const auto started = std::chrono::steady_clock::now();
  if (frame_parameters.size() != 75 || physical_beta.size() != 10
      || base_joints.size() != 165 || joint_basis.size() != 1650
      || parents.size() != 55) {
    throw std::runtime_error("invalid runtime Stage1 preparation input");
  }
  // The shape API accepts latent beta; runtime input is explicitly physical
  // beta, so compute its linear joint update directly without tanh.
  ShapeValueResult shape;
  shape.beta = physical_beta;
  shape.joints = base_joints;
  for (int coordinate = 0; coordinate < 165; ++coordinate) {
    for (int beta = 0; beta < 10; ++beta) {
      shape.joints[coordinate] +=
          joint_basis[coordinate * 10 + beta] * physical_beta[beta];
    }
  }
  PreparedStage1Geometry output;
  output.cam2 = prepare_camera(
      "cam2", frame_parameters, shape, parents, cam2, physical_beta);
  output.cam4 = prepare_camera(
      "cam4", frame_parameters, shape, parents, cam4, physical_beta);
  output.preparation_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return output;
}

}  // namespace staged_lm
