#pragma once

#include "staged_lm/lm.h"
#include "staged_lm/stage1.h"

namespace staged_lm {

struct Stage1LMProblem {
  Stage1CameraInput cam2;
  Stage1CameraInput cam4;
  LMConfig lm_config;
};

LMRunResult run_stage1_lm(const Stage1LMProblem& problem);

}  // namespace staged_lm
