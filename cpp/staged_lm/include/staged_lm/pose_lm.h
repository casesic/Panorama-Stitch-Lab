#pragma once

#include "staged_lm/lm.h"
#include "staged_lm/evaluator.h"
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
  EvaluatorBackendKind evaluator_backend = EvaluatorBackendKind::kCpu;
};

LMRunResult run_pose_lm(const PoseLMProblem& problem);
std::vector<LMRunResult> run_multiframe_pose_lm_cuda(
    const std::vector<PoseLMProblem>& problems);

}  // namespace staged_lm
