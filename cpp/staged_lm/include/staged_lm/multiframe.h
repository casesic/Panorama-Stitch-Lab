#pragma once

#include "staged_lm/pipeline.h"

#include <cstddef>
#include <memory>

namespace staged_lm {

constexpr int kStagedLMFrameDimension = 75;
constexpr int kStagedLMBetaDimension = 10;

struct MultiFramePoseShapeFrameProblem {
  PoseShapeCameraInput cam2;
  PoseShapeCameraInput cam4;
  std::vector<float> body_pose_anchor;
  std::vector<float> initial_parameters;
};

struct MultiFramePoseShapeLMProblem {
  std::vector<MultiFramePoseShapeFrameProblem> frames;
  std::vector<float> base_joints;
  std::vector<float> joint_basis;
  std::vector<float> beta_anchor;
  std::vector<std::int64_t> parents;
  float beta_limit = 2.5F;
  PoseShapeNormalEquationConfig normal_config;
  LMConfig lm_config;
  EvaluatorBackendKind evaluator_backend = EvaluatorBackendKind::kCpu;
};

struct CudaMultiFramePoseShapeWorkspace;

class CudaMultiFramePoseShapeEvaluator final : public EvaluatorBackend {
 public:
  explicit CudaMultiFramePoseShapeEvaluator(
      const MultiFramePoseShapeLMProblem& problem);
  ~CudaMultiFramePoseShapeEvaluator() override;
  void forward(const std::vector<float>& parameters) override;
  double evaluate_residual() override;
  void linearize(const std::vector<float>& parameters) override;
 DenseNormalEquations build_normal_equations() override;
 private:
  std::unique_ptr<CudaMultiFramePoseShapeWorkspace> workspace_;
};

std::size_t multiframe_parameter_dimension(std::size_t frame_count);
std::size_t multiframe_frame_offset(std::size_t frame_index);
std::size_t multiframe_beta_offset(std::size_t frame_count);

LMRunResult run_multiframe_pose_shape_lm(
    const MultiFramePoseShapeLMProblem& problem);

struct StagedLMFrameProblem {
  Stage1LMProblem stage1;
  PoseLMProblem pose;
  PoseShapeCameraInput pose_shape_cam2;
  PoseShapeCameraInput pose_shape_cam4;
  std::vector<float> body_pose_anchor;
};

struct MultiFrameStagedLMProblem {
  std::vector<StagedLMFrameProblem> frames;
  std::vector<float> base_joints;
  std::vector<float> joint_basis;
  std::vector<float> beta_anchor;
  std::vector<std::int64_t> parents;
  float beta_limit = 2.5F;
  PoseShapeNormalEquationConfig pose_shape_normal_config;
  LMConfig pose_shape_lm_config;
  EvaluatorBackendKind evaluator_backend = EvaluatorBackendKind::kCpu;
  bool batch_cuda_pose = true;
};

struct MultiFrameStagedLMFrameResult {
  LMRunResult stage1;
  std::vector<float> pose_initial_parameters;
  LMRunResult pose;
  std::vector<float> final_parameters;
};

struct MultiFrameStagedLMResult {
  std::vector<MultiFrameStagedLMFrameResult> frames;
  LMRunResult pose_shape;
  std::vector<float> beta_latent;
  std::vector<float> physical_beta;
  double stage1_total_ms = 0.0;
  double pose_total_ms = 0.0;
  double pose_shape_total_ms = 0.0;
  double total_ms = 0.0;
  double ms_per_frame = 0.0;
};

MultiFrameStagedLMResult run_multiframe_staged_pipeline(
    const MultiFrameStagedLMProblem& problem);

}  // namespace staged_lm
