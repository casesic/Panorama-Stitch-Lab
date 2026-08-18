#pragma once
#include "staged_lm/lm.h"
#include "staged_lm/pose_shape_forward.h"
#include "staged_lm/pose_shape_normal_equations.h"
namespace staged_lm {
    struct PoseShapeLMProblem
    {
        std::vector<float> initial_parameters,
            base_joints, joint_basis, beta_anchor, body_pose_anchor;
        std::vector<std::int64_t> parents;
        PoseShapeCameraInput cam2, cam4;
        float beta_limit = 2.5F;
        PoseShapeNormalEquationConfig normal_config;
        LMConfig lm_config;
    };
    LMRunResult run_pose_shape_lm(const PoseShapeLMProblem &);
}
