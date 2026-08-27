#pragma once

#include "staged_lm/pose_shape_forward.h"

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace staged_lm {

struct PreparedStage1CameraGeometry {
  std::vector<float> selected_vertices;
  std::vector<float> local_vertices;
  Eigen::Vector3f rotation_center = Eigen::Vector3f::Zero();
};

struct PreparedStage1Geometry {
  PreparedStage1CameraGeometry cam2;
  PreparedStage1CameraGeometry cam4;
  double preparation_ms = 0.0;
};

// Rebuild initialization-dependent Stage1 geometry from normalized staged-LM
// state. Static model data and observations remain externally owned.
PreparedStage1Geometry prepare_stage1_geometry(
    const std::vector<float>& frame_parameters,
    const std::vector<float>& physical_beta,
    const std::vector<float>& base_joints,
    const std::vector<float>& joint_basis,
    const std::vector<std::int64_t>& parents,
    const PoseShapeCameraInput& cam2,
    const PoseShapeCameraInput& cam4);

}  // namespace staged_lm
