#include "staged_lm/erp.h"
#include "staged_lm/so3.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <array>
#include <cmath>
#include <iostream>

int main() {
  const std::array<Eigen::Vector3f, 3> rotations = {
      Eigen::Vector3f::Zero(),
      Eigen::Vector3f(1.0e-6F, -2.0e-6F, 3.0e-6F),
      Eigen::Vector3f(0.4F, -0.7F, 1.1F),
  };
  for (const auto& axis_angle : rotations) {
    const Eigen::Matrix3f rotation = staged_lm::axis_angle_to_matrix(axis_angle);
    const Eigen::Matrix3f right_jacobian = staged_lm::so3_right_jacobian(axis_angle);
    if (!rotation.allFinite() || !right_jacobian.allFinite()
        || (rotation.transpose() * rotation - Eigen::Matrix3f::Identity()).norm() > 2.0e-5F
        || std::abs(rotation.determinant() - 1.0F) > 2.0e-5F) {
      std::cerr << "SO3_SELF_TEST_FAILED\n";
      return 1;
    }
  }
  const std::array<Eigen::Vector3f, 3> points = {
      Eigen::Vector3f(0.2F, -0.3F, 2.0F),
      Eigen::Vector3f(-1.0F, 0.5F, 3.0F),
      Eigen::Vector3f(0.01F, 1.0F, -0.5F),
  };
  for (const auto& point : points) {
    const auto projection = staged_lm::project_unified_erp_with_jacobian(point, 5760, 2880);
    if (!projection.uv.allFinite() || !projection.jacobian.allFinite()) {
      std::cerr << "ERP_SELF_TEST_FAILED\n";
      return 1;
    }
  }
  std::cout << "STAGE1_CPP_SELF_TEST_PASS\n";
  return 0;
}
