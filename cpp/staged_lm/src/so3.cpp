#include "staged_lm/so3.h"

#include <cmath>

namespace staged_lm {
namespace {

Eigen::Matrix3f multiply3(const Eigen::Matrix3f& left, const Eigen::Matrix3f& right) {
  Eigen::Matrix3f result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      float value = 0.0F;
      for (int inner = 0; inner < 3; ++inner) {
        value = std::fma(left(row, inner), right(inner, column), value);
      }
      result(row, column) = value;
    }
  }
  return result;
}

}  // namespace

Eigen::Matrix3f skew(const Eigen::Vector3f& vector) {
  Eigen::Matrix3f result;
  result << 0.0F, -vector.z(), vector.y(),
      vector.z(), 0.0F, -vector.x(),
      -vector.y(), vector.x(), 0.0F;
  return result;
}

Eigen::Matrix3f axis_angle_to_matrix(const Eigen::Vector3f& axis_angle) {
  const float theta2 = (axis_angle.x() * axis_angle.x()
      + axis_angle.y() * axis_angle.y()) + axis_angle.z() * axis_angle.z();
  const float theta = std::sqrt(theta2);
  float a;
  float b;
  if (theta2 < 1.0e-8F) {
    a = 1.0F - theta2 / 6.0F + theta2 * theta2 / 120.0F;
    b = 0.5F - theta2 / 24.0F + theta2 * theta2 / 720.0F;
  } else {
    a = std::sin(theta) / theta;
    b = (1.0F - std::cos(theta)) / theta2;
  }
  const Eigen::Matrix3f phi_hat = skew(axis_angle);
  const Eigen::Matrix3f phi_hat_squared = multiply3(phi_hat, phi_hat);
  Eigen::Matrix3f result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const float identity = row == column ? 1.0F : 0.0F;
      result(row, column) = (identity + a * phi_hat(row, column))
          + b * phi_hat_squared(row, column);
    }
  }
  return result;
}

Eigen::Matrix3f so3_right_jacobian(const Eigen::Vector3f& axis_angle) {
  const float theta2 = (axis_angle.x() * axis_angle.x()
      + axis_angle.y() * axis_angle.y()) + axis_angle.z() * axis_angle.z();
  const float theta = std::sqrt(theta2);
  float a;
  float b;
  if (theta2 < 1.0e-8F) {
    a = 0.5F - theta2 / 24.0F + theta2 * theta2 / 720.0F;
    b = 1.0F / 6.0F - theta2 / 120.0F + theta2 * theta2 / 5040.0F;
  } else {
    a = (1.0F - std::cos(theta)) / theta2;
    b = (theta - std::sin(theta)) / (theta2 * theta);
  }
  const Eigen::Matrix3f phi_hat = skew(axis_angle);
  const Eigen::Matrix3f phi_hat_squared = multiply3(phi_hat, phi_hat);
  Eigen::Matrix3f result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const float identity = row == column ? 1.0F : 0.0F;
      result(row, column) = (identity - a * phi_hat(row, column))
          + b * phi_hat_squared(row, column);
    }
  }
  return result;
}

Eigen::Matrix3f rotated_point_jacobian(
    const Eigen::Vector3f& axis_angle, const Eigen::Vector3f& local_point) {
  const Eigen::Matrix3f first = multiply3(
      axis_angle_to_matrix(axis_angle), skew(local_point));
  return -multiply3(first, so3_right_jacobian(axis_angle));
}

}  // namespace staged_lm
