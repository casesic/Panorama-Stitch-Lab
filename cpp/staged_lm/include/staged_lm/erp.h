#pragma once

#include <Eigen/Core>

namespace staged_lm {

struct ErpProjectionResult {
  Eigen::Vector2f uv;
  Eigen::Matrix<float, 2, 3> jacobian;
};

Eigen::Vector2f project_unified_erp(
    const Eigen::Vector3f& point, int image_width, int image_height);
ErpProjectionResult project_unified_erp_with_jacobian(
    const Eigen::Vector3f& point, int image_width, int image_height);

}  // namespace staged_lm
