#pragma once

#include "staged_lm/evaluator.h"
#include "staged_lm/pose_jacobian.h"
#include "staged_lm/pose_normal_equations.h"
#include "staged_lm/pose_lm.h"
#include "staged_lm/pose_shape_lm.h"
#include "staged_lm/pose_shape_normal_equations.h"

#include <memory>
#include <string>

namespace staged_lm {

bool cuda_evaluator_available(std::string* reason = nullptr);
void begin_cuda_deferred_execution();
void end_cuda_deferred_execution();

struct CudaPoseWorkspace;
struct CudaMultiFramePoseBatchWorkspace;
struct CudaBatchTransferStats {
  std::size_t h2d_count = 0;
  std::size_t h2d_bytes = 0;
  std::size_t d2h_count = 0;
  std::size_t d2h_bytes = 0;
};
CudaPoseWorkspace* create_cuda_pose_workspace(
    const PoseModelInput& model, const PoseCameraInput& input);
void destroy_cuda_pose_workspace(CudaPoseWorkspace* workspace);
void reserve_cuda_pose_workspace(
    CudaPoseWorkspace* workspace, const PoseCameraInput& input,
    bool pose_shape_linearization);
void prepare_cuda_pose_shape_workspace(
    CudaPoseWorkspace* workspace, const PoseShapeCameraInput& input);

CudaMultiFramePoseBatchWorkspace* create_cuda_multiframe_pose_batch_workspace(
    const std::vector<CudaPoseWorkspace*>& cam2_workspaces,
    const std::vector<CudaPoseWorkspace*>& cam4_workspaces,
    const std::vector<float>& body_pose_anchors);
void destroy_cuda_multiframe_pose_batch_workspace(
    CudaMultiFramePoseBatchWorkspace* workspace);
void upload_cuda_multiframe_pose_batch(
    CudaMultiFramePoseBatchWorkspace* workspace,
    const std::vector<int>& frame_indices,
    const std::vector<std::vector<float>>& parameters,
    const std::vector<float>& packed_camera_parameters);
void activate_cuda_multiframe_pose_batch_camera(
    CudaMultiFramePoseBatchWorkspace* workspace, int batch_item, int camera);
void deactivate_cuda_multiframe_pose_batch_camera();
std::vector<DenseNormalEquations> build_cuda_multiframe_pose_normal_equations(
    CudaMultiFramePoseBatchWorkspace* workspace,
    const std::vector<int>& frame_indices,
    const PoseNormalEquationConfig& config,
    const std::vector<double>& jacobian_ms);
CudaBatchTransferStats cuda_multiframe_pose_batch_transfer_stats(
    const CudaMultiFramePoseBatchWorkspace* workspace);

struct CudaPoseShapeValueWorkspace;
CudaPoseShapeValueWorkspace* create_cuda_pose_shape_value_workspace(
    const PoseShapeLMProblem& problem);
void destroy_cuda_pose_shape_value_workspace(
    CudaPoseShapeValueWorkspace* workspace);

struct CudaPoseNormalEquationResult {
  DenseNormalEquations equations;
  std::vector<double> cam2_hessian_active;
  std::vector<double> cam2_gradient_active;
  std::vector<double> cam4_hessian_active;
  std::vector<double> cam4_gradient_active;
  double gpu_accumulation_ms = 0.0;
  double hg_d2h_ms = 0.0;
};

CudaPoseNormalEquationResult build_cuda_pose_normal_equations(
    CudaPoseWorkspace* cam2_workspace,
    CudaPoseWorkspace* cam4_workspace,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const PoseNormalEquationConfig& config);

class CudaPoseNormalEquationBuilder {
 public:
  static CudaPoseNormalEquationResult build(
      CudaPoseWorkspace* cam2_workspace,
      CudaPoseWorkspace* cam4_workspace,
      const std::vector<float>& parameters,
      const std::vector<float>& body_pose_anchor,
      const PoseNormalEquationConfig& config);
};

struct CudaPoseShapeNormalEquationResult {
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> g_pose;
  std::vector<double> g_beta;
  double gpu_accumulation_ms = 0.0;
  double blocks_d2h_ms = 0.0;
  double total_ms = 0.0;
};

struct CudaMultiFramePoseShapeNormalEquationWorkspace;
CudaMultiFramePoseShapeNormalEquationWorkspace*
create_cuda_multiframe_pose_shape_ne_workspace(int frame_count);
void destroy_cuda_multiframe_pose_shape_ne_workspace(
    CudaMultiFramePoseShapeNormalEquationWorkspace* workspace);

struct CudaMultiFramePoseShapeNormalEquationResult {
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> g_pose;
  std::vector<double> g_beta;
  double gpu_accumulation_ms = 0.0;
  double final_blocks_d2h_ms = 0.0;
  double total_ms = 0.0;
};

CudaMultiFramePoseShapeNormalEquationResult
build_cuda_multiframe_pose_shape_normal_equations(
    CudaMultiFramePoseShapeNormalEquationWorkspace* workspace,
    const std::vector<CudaPoseWorkspace*>& cam2_workspaces,
    const std::vector<CudaPoseWorkspace*>& cam4_workspaces,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchors,
    const std::vector<float>& beta_anchor,
    float beta_limit,
    const PoseShapeNormalEquationConfig& config);

CudaPoseShapeNormalEquationResult build_cuda_pose_shape_normal_equations(
    CudaPoseWorkspace* cam2_workspace,
    CudaPoseWorkspace* cam4_workspace,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const std::vector<float>& beta_anchor,
    const PoseShapeNormalEquationConfig& config);

PoseCameraResult evaluate_pose_camera_value_cuda(
    const PoseModelInput& model, const PoseCameraInput& input);
PoseCameraJacobianResult evaluate_pose_camera_with_jacobian_cuda(
    const PoseModelInput& model, const PoseCameraInput& input);
PoseCameraResult evaluate_pose_camera_value_cuda_with_workspace(
    CudaPoseWorkspace* workspace, const PoseModelInput& model,
    const PoseCameraInput& input);
PoseCameraJacobianResult evaluate_pose_camera_with_jacobian_cuda_with_workspace(
    CudaPoseWorkspace* workspace, const PoseModelInput& model,
    const PoseCameraInput& input);
void update_cuda_pose_workspace_shape(
    CudaPoseWorkspace* workspace,
    const std::vector<float>& joints,
    const std::vector<float>& selected_v_shaped);
std::vector<float> evaluate_pose_shape_beta_jacobian_cuda_with_workspace(
    CudaPoseWorkspace* workspace,
    const PoseShapeCameraInput& input,
    const std::vector<float>& d_joints_beta,
    const std::vector<float>& d_beta_d_latent,
    bool download);
PoseShapeValueResult evaluate_pose_shape_value_cuda(
    const std::vector<float>& parameters,
    const PoseShapeLMProblem& problem);
PoseShapeValueResult evaluate_pose_shape_value_cuda_with_workspace(
    CudaPoseShapeValueWorkspace* shape_workspace,
    CudaPoseWorkspace* cam2_workspace,
    CudaPoseWorkspace* cam4_workspace,
    const std::vector<float>& parameters,
    const PoseShapeLMProblem& problem);

class CudaPoseEvaluator final : public EvaluatorBackend {
 public:
  explicit CudaPoseEvaluator(const PoseLMProblem& problem);
  ~CudaPoseEvaluator() override;
  void forward(const std::vector<float>& parameters) override;
  double evaluate_residual() override;
  void linearize(const std::vector<float>& parameters) override;
  DenseNormalEquations build_normal_equations() override;
  void forward_batched(
      const std::vector<float>& parameters,
      CudaMultiFramePoseBatchWorkspace* workspace, int batch_item);
  void linearize_batched(
      const std::vector<float>& parameters,
      CudaMultiFramePoseBatchWorkspace* workspace, int batch_item);

  const PoseCameraResult& cam2_result() const;
  const PoseCameraResult& cam4_result() const;
  const PoseCameraJacobianResult& cam2_jacobian_result() const;
  const PoseCameraJacobianResult& cam4_jacobian_result() const;
  CudaPoseWorkspace* cam2_workspace() const;
  CudaPoseWorkspace* cam4_workspace() const;
  const PoseLMProblem& problem() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Multi-instance facade used by the CPU batch LM controller.  Frames remain
// independent LM systems; evaluator work is submitted through one batch API.
class CudaMultiFramePoseEvaluator {
 public:
  explicit CudaMultiFramePoseEvaluator(
      const std::vector<PoseLMProblem>& problems);
  ~CudaMultiFramePoseEvaluator();
  std::vector<DenseNormalEquations> linearize_and_build(
      const std::vector<int>& frame_indices,
      const std::vector<std::vector<float>>& parameters);
  std::vector<double> forward_and_evaluate(
      const std::vector<int>& frame_indices,
      const std::vector<std::vector<float>>& parameters);
  CudaBatchTransferStats transfer_stats() const;

 private:
  void upload_batch(
      const std::vector<int>& frame_indices,
      const std::vector<std::vector<float>>& parameters);
  std::vector<std::unique_ptr<CudaPoseEvaluator>> evaluators_;
  std::unique_ptr<CudaMultiFramePoseBatchWorkspace,
      void (*)(CudaMultiFramePoseBatchWorkspace*)> batch_workspace_{
          nullptr, destroy_cuda_multiframe_pose_batch_workspace};
};

class CudaPoseShapeEvaluator final : public EvaluatorBackend {
 public:
  explicit CudaPoseShapeEvaluator(const PoseShapeLMProblem& problem);
  ~CudaPoseShapeEvaluator() override;
  void forward(const std::vector<float>& parameters) override;
  double evaluate_residual() override;
  void linearize(const std::vector<float>& parameters) override;
  DenseNormalEquations build_normal_equations() override;

  const PoseShapeValueResult& value_result() const;
  CudaPoseWorkspace* cam2_workspace() const;
  CudaPoseWorkspace* cam4_workspace() const;
  double last_jacobian_ms() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace staged_lm
