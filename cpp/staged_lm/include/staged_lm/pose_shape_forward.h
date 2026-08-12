#pragma once
#include "staged_lm/pose_forward.h"
#include "staged_lm/shape.h"
namespace staged_lm {
struct PoseShapeCameraInput { PoseCameraInput pose; std::vector<float> selected_base_v_shaped,selected_beta_shapedirs; };
struct PoseShapeValueResult { ShapeValueResult shape; std::vector<float> cam2_dynamic_v_shaped,cam4_dynamic_v_shaped; PoseCameraResult cam2,cam4; double cam2_shape_ms=0,cam4_shape_ms=0,total_ms=0; double objective_primary=0,objective_pose_anchor=0,objective_shape_anchor=0,objective_total=0,objective_ms=0; };
PoseShapeValueResult evaluate_pose_shape_value(const std::vector<float>& parameters,const std::vector<float>& base_joints,const std::vector<float>& joint_basis,const std::vector<float>& beta_anchor,const std::vector<float>& body_pose_anchor,float beta_limit,float pose_scale,float shape_scale,float eps,float cam2_alpha,float cam4_alpha,const std::vector<std::int64_t>& parents,const PoseShapeCameraInput& cam2,const PoseShapeCameraInput& cam4);
}
