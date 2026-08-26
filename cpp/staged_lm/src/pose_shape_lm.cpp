#include "staged_lm/pose_shape_lm.h"

#include "staged_lm/cpu_evaluators.h"
#include "staged_lm/cuda_evaluators.h"

namespace staged_lm {

LMRunResult run_pose_shape_lm(const PoseShapeLMProblem& problem) {
  if (problem.evaluator_backend == EvaluatorBackendKind::kCuda) {
    CudaPoseShapeEvaluator evaluator(problem);
    return run_lm(problem.initial_parameters, problem.lm_config, evaluator);
  }
  CpuPoseShapeEvaluator evaluator(problem);
  return run_lm(problem.initial_parameters, problem.lm_config, evaluator);
}

}  // namespace staged_lm
