#pragma once

#include "staged_lm/pose_lm.h"
#include "staged_lm/pose_shape_lm.h"
#include "staged_lm/stage1_lm.h"

namespace staged_lm {

struct StagedPipelineProblem {
  Stage1LMProblem stage1;
  PoseLMProblem pose;
  PoseShapeLMProblem pose_shape;
};

struct StagedPipelineResult {
  LMRunResult stage1;
  std::vector<float> pose_initial_parameters;
  LMRunResult pose;
  std::vector<float> pose_shape_initial_parameters;
  LMRunResult pose_shape;
  std::vector<float> final_physical_beta;
  double stage1_to_pose_ms = 0.0;
  double pose_to_pose_shape_ms = 0.0;
  double final_output_ms = 0.0;
  double compute_ms = 0.0;
};

StagedPipelineResult run_staged_pipeline(const StagedPipelineProblem& problem);

}  // namespace staged_lm
