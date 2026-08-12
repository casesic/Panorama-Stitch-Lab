#pragma once

#include "staged_lm/lm.h"
#include "staged_lm/pose_forward.h"
#include "staged_lm/pose_normal_equations.h"

namespace staged_lm {

struct PoseLMProblem {
  PoseModelInput model;
  PoseCameraInput cam2;
  PoseCameraInput cam4;
  std::vector<float> body_pose_anchor;
  PoseNormalEquationConfig normal_config;
  LMConfig lm_config;
};

LMRunResult run_pose_lm(const PoseLMProblem& problem);

}  // namespace staged_lm
