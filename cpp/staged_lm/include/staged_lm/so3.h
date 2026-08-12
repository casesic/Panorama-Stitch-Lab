#pragma once

#include <Eigen/Core>

namespace staged_lm {

Eigen::Matrix3f skew(const Eigen::Vector3f& vector);
Eigen::Matrix3f axis_angle_to_matrix(const Eigen::Vector3f& axis_angle);
Eigen::Matrix3f so3_right_jacobian(const Eigen::Vector3f& axis_angle);
Eigen::Matrix3f rotated_point_jacobian(
    const Eigen::Vector3f& axis_angle, const Eigen::Vector3f& local_point);

}  // namespace staged_lm
