#include "staged_lm/pose_lm.h"

#include "staged_lm/cpu_evaluators.h"
#include "staged_lm/cuda_evaluators.h"

namespace staged_lm {

LMRunResult run_pose_lm(const PoseLMProblem& problem) {
  if (problem.evaluator_backend == EvaluatorBackendKind::kCuda) {
    CudaPoseEvaluator evaluator(problem);
    return run_lm(problem.cam2.parameters, problem.lm_config, evaluator);
  }
  CpuPoseEvaluator evaluator(problem);
  return run_lm(problem.cam2.parameters, problem.lm_config, evaluator);
}

std::vector<LMRunResult> run_multiframe_pose_lm_cuda(
    const std::vector<PoseLMProblem>& problems) {
  if (problems.empty()) return {};
  // Preserve the established single-frame CUDA path exactly.
  if (problems.size() == 1) return {run_pose_lm(problems.front())};
  CudaMultiFramePoseEvaluator evaluator(problems);
  std::vector<std::vector<float>> initial;
  std::vector<LMConfig> configs;
  initial.reserve(problems.size());
  configs.reserve(problems.size());
  for (const auto& problem : problems) {
    initial.push_back(problem.cam2.parameters);
    configs.push_back(problem.lm_config);
  }
  auto results = run_lm_batch(
      initial, configs,
      [&evaluator](const std::vector<int>& indices,
                   const std::vector<std::vector<float>>& parameters) {
        return evaluator.linearize_and_build(indices, parameters);
      },
      [&evaluator](const std::vector<int>& indices,
                   const std::vector<std::vector<float>>& parameters) {
        return evaluator.forward_and_evaluate(indices, parameters);
      });
  const CudaBatchTransferStats transfers = evaluator.transfer_stats();
  for (auto& result : results) {
    result.batch_h2d_count = transfers.h2d_count;
    result.batch_h2d_bytes = transfers.h2d_bytes;
    result.batch_d2h_count = transfers.d2h_count;
    result.batch_d2h_bytes = transfers.d2h_bytes;
  }
  return results;
}

}  // namespace staged_lm
