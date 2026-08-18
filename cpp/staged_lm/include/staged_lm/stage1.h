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
  std::vector<float> local_vertices;       // point_count * 3
  Eigen::Vector3f rotation_center;
  std::vector<float> targets;              // point_count * 2
  int point_count = 0;
  int image_width = 0;
  int image_height = 0;
  float irls_epsilon = 0.0F;
  float alpha_per_point = 0.0F;
};

struct Stage1CameraResult {
  Eigen::Matrix3f rotation;
  Eigen::Matrix3f right_jacobian;
  std::vector<float> points;                       // point_count * 3
  std::vector<float> uv;                           // point_count * 2
  std::vector<float> projection_jacobian;          // point_count * 6
  std::vector<float> point_orientation_jacobian;   // point_count * 9
  std::vector<float> orientation_jacobian;         // point_count * 6
  std::vector<float> translation_jacobian;         // point_count * 6
  std::vector<float> frame_jacobian;               // point_count * 2 * 12
  std::vector<float> raw_residual;                 // point_count * 2
  std::vector<float> point_weights;                // point_count
  std::vector<float> row_scales;                   // point_count * 2
  std::vector<float> weighted_residual;            // point_count * 2
  std::vector<float> weighted_jacobian;            // point_count * 2 * 12
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
