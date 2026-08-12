#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

namespace staged_lm {

constexpr int kStage1ParameterDimension = 12;
constexpr int kCam2Offset = 0;
constexpr int kCam4Offset = 6;

struct Stage1CameraInput {
  std::string camera;
  Eigen::Vector3f orientation;
  Eigen::Vector3f translation;
  std::vector<float> local_vertices;
  Eigen::Vector3f rotation_center;
  std::vector<float> targets;
  int point_count = 0;
  int image_width = 0;
  int image_height = 0;
  float irls_epsilon = 0.0F;
  float alpha_per_point = 0.0F;
};

struct Stage1CameraResult {
  Eigen::Matrix3f rotation;
  Eigen::Matrix3f right_jacobian;
  std::vector<float> points;
  std::vector<float> uv;
  std::vector<float> projection_jacobian;
  std::vector<float> point_orientation_jacobian;
  std::vector<float> orientation_jacobian;
  std::vector<float> translation_jacobian;
  std::vector<float> frame_jacobian;
  std::vector<float> raw_residual;
  std::vector<float> point_weights;
  std::vector<float> row_scales;
  std::vector<float> weighted_residual;
  std::vector<float> weighted_jacobian;
  double so3_ms = 0.0;
  double point_transform_ms = 0.0;
  double erp_ms = 0.0;
  double rigid_jacobian_ms = 0.0;
  double irls_ms = 0.0;
};

struct Stage1FrameResult {
  Stage1CameraResult cam2;
  Stage1CameraResult cam4;
  Eigen::Matrix<double, 12, 12> hessian;
  Eigen::Matrix<double, 12, 1> gradient;
  double normal_equations_ms = 0.0;
};

Stage1CameraResult evaluate_stage1_camera(const Stage1CameraInput& input);
Stage1FrameResult evaluate_stage1_frame(
    const Stage1CameraInput& cam2, const Stage1CameraInput& cam4);

}  // namespace staged_lm
