#pragma once

#include "staged_lm/evaluator.h"
#include "staged_lm/pose_lm.h"
#include "staged_lm/pose_shape_lm.h"

#include <memory>

namespace staged_lm {

class CpuPoseEvaluator final : public EvaluatorBackend {
 public:
  explicit CpuPoseEvaluator(const PoseLMProblem& problem);
  ~CpuPoseEvaluator() override;

  void forward(const std::vector<float>& parameters) override;
  double evaluate_residual() override;
  void linearize(const std::vector<float>& parameters) override;
  DenseNormalEquations build_normal_equations() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class CpuPoseShapeEvaluator final : public EvaluatorBackend {
 public:
  explicit CpuPoseShapeEvaluator(const PoseShapeLMProblem& problem);
  ~CpuPoseShapeEvaluator() override;

  void forward(const std::vector<float>& parameters) override;
  double evaluate_residual() override;
  void linearize(const std::vector<float>& parameters) override;
  DenseNormalEquations build_normal_equations() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace staged_lm
