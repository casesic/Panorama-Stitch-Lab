#include "staged_lm/pipeline.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace staged_lm {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace

StagedPipelineResult run_staged_pipeline(const StagedPipelineProblem& problem) {
  const auto total_started = Clock::now();
  StagedPipelineResult result;
  result.stage1 = run_stage1_lm(problem.stage1);
  if (result.stage1.final_parameters.size() != 12) {
    throw std::runtime_error("STAGE1_OUTPUT_DIMENSION_MISMATCH");
  }

  auto started = Clock::now();
  result.pose_initial_parameters = result.stage1.final_parameters;
  result.pose_initial_parameters.insert(
      result.pose_initial_parameters.end(), problem.pose.body_pose_anchor.begin(),
      problem.pose.body_pose_anchor.end());
  result.stage1_to_pose_ms = elapsed_ms(started);
  if (result.pose_initial_parameters.size() != 75) {
    throw std::runtime_error("STAGE1_TO_POSE_HANDOFF_DIMENSION_MISMATCH");
  }

  PoseLMProblem pose_problem = problem.pose;
  pose_problem.cam2.parameters = result.pose_initial_parameters;
  pose_problem.cam4.parameters = result.pose_initial_parameters;
  result.pose = run_pose_lm(pose_problem);

  started = Clock::now();
  result.pose_shape_initial_parameters = result.pose.final_parameters;
  const float beta_limit = problem.pose_shape.beta_limit;
  for (float beta : problem.pose_shape.beta_anchor) {
    result.pose_shape_initial_parameters.push_back(std::atanh(beta / beta_limit));
  }
  result.pose_to_pose_shape_ms = elapsed_ms(started);
  if (result.pose_shape_initial_parameters.size() != 85) {
    throw std::runtime_error("POSE_TO_POSE_SHAPE_HANDOFF_DIMENSION_MISMATCH");
  }

  PoseShapeLMProblem pose_shape_problem = problem.pose_shape;
  pose_shape_problem.initial_parameters = result.pose_shape_initial_parameters;
  result.pose_shape = run_pose_shape_lm(pose_shape_problem);

  started = Clock::now();
  result.final_physical_beta.resize(10);
  for (int beta = 0; beta < 10; ++beta) {
    result.final_physical_beta[beta] = beta_limit * std::tanh(
        result.pose_shape.final_parameters[75 + beta]);
  }
  result.final_output_ms = elapsed_ms(started);
  result.compute_ms = elapsed_ms(total_started);
  return result;
}

}  // namespace staged_lm
