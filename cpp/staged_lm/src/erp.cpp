#include "staged_lm/erp.h"

#include <algorithm>
#include <cmath>

namespace staged_lm {
namespace {

// ERP utilities project 3-D camera points and expose the projection Jacobian.
constexpr float kPi = 3.14159265358979323846F;

Eigen::Matrix3f fixed_rotation() {
  Eigen::Matrix3f rotation;
  rotation << 1.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
      0.0F, -1.0F, 0.0F;
  return rotation;
}

}  // namespace

ErpProjectionResult project_unified_erp_with_jacobian(
    const Eigen::Vector3f& point, int image_width, int image_height) {
  const Eigen::Matrix3f rotation = fixed_rotation();
  const Eigen::Vector3f camera_point = rotation * point;
  const float x = camera_point.x();
  const float y = camera_point.y();
  const float z = camera_point.z();
  const float horizontal2 = x * x + z * z;
  const float horizontal = std::sqrt(std::max(horizontal2, 1.0e-12F));
  const float radius2 = horizontal2 + y * y;
  const float radius = std::max(camera_point.norm(), 1.0e-6F);
  const float longitude = std::atan2(x, z);
  const float raw_latitude_argument = -y / radius;
  const float latitude_argument = std::clamp(
      raw_latitude_argument, -1.0F + 1.0e-6F, 1.0F - 1.0e-6F);
  const float latitude = std::asin(latitude_argument);

  ErpProjectionResult result;
  result.uv.x() = (longitude / (2.0F * kPi) + 0.5F)
      * static_cast<float>(image_width);
  result.uv.y() = (0.5F - latitude / kPi)
      * static_cast<float>(image_height);

  const float safe_horizontal2 = std::max(horizontal2, 1.0e-12F);
  const float safe_radius2 = std::max(radius2, 1.0e-12F);
  const float du_scale = static_cast<float>(image_width) / (2.0F * kPi);
  const float dv_scale = static_cast<float>(image_height) / kPi;
  Eigen::Matrix<float, 2, 3> camera_jacobian;
  camera_jacobian.row(0) <<
      du_scale * z / safe_horizontal2,
      0.0F,
      -du_scale * x / safe_horizontal2;
  if (radius2 > 1.0e-12F && raw_latitude_argument > -1.0F + 1.0e-6F
      && raw_latitude_argument < 1.0F - 1.0e-6F) {
    camera_jacobian.row(1) <<
        -dv_scale * y * x / (horizontal * safe_radius2),
        dv_scale * horizontal / safe_radius2,
        -dv_scale * y * z / (horizontal * safe_radius2);
  } else {
    camera_jacobian.row(1).setZero();
  }
  result.jacobian = camera_jacobian * rotation;
  return result;
}

Eigen::Vector2f project_unified_erp(
    const Eigen::Vector3f& point, int image_width, int image_height) {
  return project_unified_erp_with_jacobian(point, image_width, image_height).uv;
}

}  // namespace staged_lm
