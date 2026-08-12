#include "staged_lm/pipeline.h"

#include <iostream>
#include <stdexcept>

int main() {
  staged_lm::StagedPipelineProblem problem;
  problem.stage1.cam2.camera = "cam2";
  problem.stage1.cam4.camera = "cam4";
  problem.stage1.lm_config = {12, 1, 1, 1e-7, 1e-5, 1e-5};
  try {
    (void)staged_lm::run_staged_pipeline(problem);
  } catch (const std::runtime_error&) {
    std::cout << "STAGED_LM_PIPELINE_CPP_SELF_TEST=PASS\n";
    return 0;
  }
  return 1;
}
