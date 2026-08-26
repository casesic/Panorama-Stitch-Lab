#include "staged_lm/cuda_evaluators.h"

#include <stdexcept>

namespace staged_lm {

bool cuda_evaluator_available(std::string* reason) {
  if (reason) *reason = "staged_lm was built without CUDA";
  return false;
}
void begin_cuda_deferred_execution() {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
void end_cuda_deferred_execution() {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}

struct CudaPoseWorkspace {};
struct CudaMultiFramePoseBatchWorkspace {};
struct CudaPoseShapeValueWorkspace {};
struct CudaMultiFramePoseShapeNormalEquationWorkspace {};
CudaPoseWorkspace* create_cuda_pose_workspace(
    const PoseModelInput&, const PoseCameraInput&) { return new CudaPoseWorkspace; }
void destroy_cuda_pose_workspace(CudaPoseWorkspace* workspace) { delete workspace; }
void reserve_cuda_pose_workspace(
    CudaPoseWorkspace*, const PoseCameraInput&, bool) {}
void prepare_cuda_pose_shape_workspace(
    CudaPoseWorkspace*, const PoseShapeCameraInput&) {}
CudaMultiFramePoseBatchWorkspace* create_cuda_multiframe_pose_batch_workspace(
    const std::vector<CudaPoseWorkspace*>&,
    const std::vector<CudaPoseWorkspace*>&, const std::vector<float>&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
void destroy_cuda_multiframe_pose_batch_workspace(
    CudaMultiFramePoseBatchWorkspace* workspace) { delete workspace; }
void upload_cuda_multiframe_pose_batch(
    CudaMultiFramePoseBatchWorkspace*, const std::vector<int>&,
    const std::vector<std::vector<float>>&, const std::vector<float>&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
void activate_cuda_multiframe_pose_batch_camera(
    CudaMultiFramePoseBatchWorkspace*, int, int) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
void deactivate_cuda_multiframe_pose_batch_camera() {}
std::vector<DenseNormalEquations> build_cuda_multiframe_pose_normal_equations(
    CudaMultiFramePoseBatchWorkspace*, const std::vector<int>&,
    const PoseNormalEquationConfig&, const std::vector<double>&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
CudaBatchTransferStats cuda_multiframe_pose_batch_transfer_stats(
    const CudaMultiFramePoseBatchWorkspace*) { return {}; }
CudaPoseShapeValueWorkspace* create_cuda_pose_shape_value_workspace(
    const PoseShapeLMProblem&) { return new CudaPoseShapeValueWorkspace; }
void destroy_cuda_pose_shape_value_workspace(
    CudaPoseShapeValueWorkspace* workspace) { delete workspace; }
CudaMultiFramePoseShapeNormalEquationWorkspace*
create_cuda_multiframe_pose_shape_ne_workspace(int) {
  return new CudaMultiFramePoseShapeNormalEquationWorkspace;
}
void destroy_cuda_multiframe_pose_shape_ne_workspace(
    CudaMultiFramePoseShapeNormalEquationWorkspace* workspace) {
  delete workspace;
}
CudaMultiFramePoseShapeNormalEquationResult
build_cuda_multiframe_pose_shape_normal_equations(
    CudaMultiFramePoseShapeNormalEquationWorkspace*,
    const std::vector<CudaPoseWorkspace*>&,
    const std::vector<CudaPoseWorkspace*>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, float,
    const PoseShapeNormalEquationConfig&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
CudaPoseNormalEquationResult build_cuda_pose_normal_equations(
    CudaPoseWorkspace*, CudaPoseWorkspace*, const std::vector<float>&,
    const std::vector<float>&, const PoseNormalEquationConfig&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
CudaPoseNormalEquationResult CudaPoseNormalEquationBuilder::build(
    CudaPoseWorkspace* cam2, CudaPoseWorkspace* cam4,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const PoseNormalEquationConfig& config) {
  return build_cuda_pose_normal_equations(
      cam2, cam4, parameters, body_pose_anchor, config);
}
CudaPoseShapeNormalEquationResult build_cuda_pose_shape_normal_equations(
    CudaPoseWorkspace*, CudaPoseWorkspace*, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const PoseShapeNormalEquationConfig&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}

PoseCameraResult evaluate_pose_camera_value_cuda(
    const PoseModelInput&, const PoseCameraInput&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}

PoseCameraJacobianResult evaluate_pose_camera_with_jacobian_cuda(
    const PoseModelInput&, const PoseCameraInput&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
PoseCameraResult evaluate_pose_camera_value_cuda_with_workspace(
    CudaPoseWorkspace*, const PoseModelInput&, const PoseCameraInput&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
PoseCameraJacobianResult evaluate_pose_camera_with_jacobian_cuda_with_workspace(
    CudaPoseWorkspace*, const PoseModelInput&, const PoseCameraInput&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
void update_cuda_pose_workspace_shape(
    CudaPoseWorkspace*, const std::vector<float>&,
    const std::vector<float>&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
std::vector<float> evaluate_pose_shape_beta_jacobian_cuda_with_workspace(
    CudaPoseWorkspace*, const PoseShapeCameraInput&,
    const std::vector<float>&, const std::vector<float>&, bool) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}

PoseShapeValueResult evaluate_pose_shape_value_cuda(
    const std::vector<float>&, const PoseShapeLMProblem&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}
PoseShapeValueResult evaluate_pose_shape_value_cuda_with_workspace(
    CudaPoseShapeValueWorkspace*, CudaPoseWorkspace*, CudaPoseWorkspace*,
    const std::vector<float>&, const PoseShapeLMProblem&) {
  throw std::runtime_error("CUDA_EVALUATOR_NOT_BUILT");
}

}  // namespace staged_lm
