#include "staged_lm/cuda_evaluators.h"
#include "staged_lm/pose_normal_equations.h"
#include "staged_lm/pose_shape_schur.h"
#include "staged_lm/shape.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace staged_lm {
namespace {

PoseCameraInput camera_at(
    const PoseCameraInput& source, const std::vector<float>& parameters) {
  PoseCameraInput camera = source;
  camera.parameters = parameters;
  return camera;
}

double pose_objective(
    const PoseCameraResult& cam2, const PoseCameraResult& cam4,
    const std::vector<float>& parameters, const PoseLMProblem& problem) {
  double value = 0.0;
  const PoseCameraResult* cameras[] = {&cam2, &cam4};
  const float alpha[] = {problem.normal_config.cam2_alpha_per_point,
                         problem.normal_config.cam4_alpha_per_point};
  for (int camera = 0; camera < 2; ++camera) {
    const auto& residual = cameras[camera]->raw_residual;
    for (std::size_t point = 0; point < residual.size() / 2; ++point) {
      const float u = residual[point * 2];
      const float v = residual[point * 2 + 1];
      const float norm = std::sqrt(u * u + v * v);
      value += static_cast<double>(alpha[camera] *
          (std::sqrt(norm * norm + problem.normal_config.charbonnier_eps *
              problem.normal_config.charbonnier_eps) -
           problem.normal_config.charbonnier_eps));
    }
  }
  float anchor_sum = 0.0F;
  for (int body = 0; body < 63; ++body) {
    const float residual =
        (parameters[12 + body] - problem.body_pose_anchor[body]) *
        problem.normal_config.pose_anchor_scale;
    anchor_sum += residual * residual;
  }
  return static_cast<double>(static_cast<float>(value) + 0.5F * anchor_sum);
}

}  // namespace

struct CudaPoseEvaluator::Impl {
  explicit Impl(const PoseLMProblem& source) : problem(source),
      cam2_workspace(create_cuda_pose_workspace(source.model, source.cam2)),
      cam4_workspace(create_cuda_pose_workspace(source.model, source.cam4)) {
    // Establish the CUDA context, allocate every linearization buffer and
    // upload immutable per-camera assets before run_lm starts timing.
    evaluate_pose_camera_with_jacobian_cuda_with_workspace(
        cam2_workspace, problem.model,
        camera_at(problem.cam2, problem.cam2.parameters));
    evaluate_pose_camera_with_jacobian_cuda_with_workspace(
        cam4_workspace, problem.model,
        camera_at(problem.cam4, problem.cam4.parameters));
  }
  ~Impl() {
    destroy_cuda_pose_workspace(cam2_workspace);
    destroy_cuda_pose_workspace(cam4_workspace);
  }
  const PoseLMProblem& problem;
  std::vector<float> parameters;
  PoseCameraResult cam2;
  PoseCameraResult cam4;
  PoseCameraJacobianResult jacobian_cam2;
  PoseCameraJacobianResult jacobian_cam4;
  std::vector<float> linearization_parameters;
  bool ready = false;
  bool linearization_ready = false;
  CudaPoseWorkspace* cam2_workspace = nullptr;
  CudaPoseWorkspace* cam4_workspace = nullptr;
};

CudaPoseEvaluator::CudaPoseEvaluator(const PoseLMProblem& problem)
    : impl_(std::make_unique<Impl>(problem)) {}
CudaPoseEvaluator::~CudaPoseEvaluator() = default;

void CudaPoseEvaluator::forward(const std::vector<float>& parameters) {
  impl_->parameters = parameters;
  impl_->cam2 = evaluate_pose_camera_value_cuda_with_workspace(
      impl_->cam2_workspace, impl_->problem.model,
      camera_at(impl_->problem.cam2, parameters));
  impl_->cam4 = evaluate_pose_camera_value_cuda_with_workspace(
      impl_->cam4_workspace, impl_->problem.model,
      camera_at(impl_->problem.cam4, parameters));
  impl_->ready = true;
}

double CudaPoseEvaluator::evaluate_residual() {
  if (!impl_->ready) throw std::runtime_error("CUDA_POSE_FORWARD_REQUIRED");
  return pose_objective(
      impl_->cam2, impl_->cam4, impl_->parameters, impl_->problem);
}

void CudaPoseEvaluator::linearize(const std::vector<float>& parameters) {
  impl_->linearization_parameters = parameters;
  impl_->jacobian_cam2 = evaluate_pose_camera_with_jacobian_cuda_with_workspace(
      impl_->cam2_workspace, impl_->problem.model,
      camera_at(impl_->problem.cam2, parameters));
  impl_->jacobian_cam4 = evaluate_pose_camera_with_jacobian_cuda_with_workspace(
      impl_->cam4_workspace, impl_->problem.model,
      camera_at(impl_->problem.cam4, parameters));
  impl_->linearization_ready = true;
}
void CudaPoseEvaluator::forward_batched(
    const std::vector<float>& parameters,
    CudaMultiFramePoseBatchWorkspace* workspace, int batch_item) {
  impl_->parameters = parameters;
  activate_cuda_multiframe_pose_batch_camera(workspace, batch_item, 0);
  try {
    impl_->cam2 = evaluate_pose_camera_value_cuda_with_workspace(
        impl_->cam2_workspace, impl_->problem.model,
        camera_at(impl_->problem.cam2, parameters));
    deactivate_cuda_multiframe_pose_batch_camera();
    activate_cuda_multiframe_pose_batch_camera(workspace, batch_item, 1);
    impl_->cam4 = evaluate_pose_camera_value_cuda_with_workspace(
        impl_->cam4_workspace, impl_->problem.model,
        camera_at(impl_->problem.cam4, parameters));
    deactivate_cuda_multiframe_pose_batch_camera();
  } catch (...) {
    deactivate_cuda_multiframe_pose_batch_camera();
    throw;
  }
  impl_->ready = true;
}

void CudaPoseEvaluator::linearize_batched(
    const std::vector<float>& parameters,
    CudaMultiFramePoseBatchWorkspace* workspace, int batch_item) {
  impl_->linearization_parameters = parameters;
  activate_cuda_multiframe_pose_batch_camera(workspace, batch_item, 0);
  try {
    impl_->jacobian_cam2 =
        evaluate_pose_camera_with_jacobian_cuda_with_workspace(
            impl_->cam2_workspace, impl_->problem.model,
            camera_at(impl_->problem.cam2, parameters));
    deactivate_cuda_multiframe_pose_batch_camera();
    activate_cuda_multiframe_pose_batch_camera(workspace, batch_item, 1);
    impl_->jacobian_cam4 =
        evaluate_pose_camera_with_jacobian_cuda_with_workspace(
            impl_->cam4_workspace, impl_->problem.model,
            camera_at(impl_->problem.cam4, parameters));
    deactivate_cuda_multiframe_pose_batch_camera();
  } catch (...) {
    deactivate_cuda_multiframe_pose_batch_camera();
    throw;
  }
  impl_->linearization_ready = true;
}
DenseNormalEquations CudaPoseEvaluator::build_normal_equations() {
  if (!impl_->linearization_ready) {
    throw std::runtime_error("CUDA_POSE_LINEARIZATION_REQUIRED");
  }
  CudaPoseNormalEquationResult built = CudaPoseNormalEquationBuilder::build(
      impl_->cam2_workspace, impl_->cam4_workspace,
      impl_->linearization_parameters, impl_->problem.body_pose_anchor,
      impl_->problem.normal_config);
  DenseNormalEquations result = std::move(built.equations);
  result.jacobian_ms = impl_->jacobian_cam2.total_jacobian_ms +
      impl_->jacobian_cam4.total_jacobian_ms;
  return result;
}
const PoseCameraResult& CudaPoseEvaluator::cam2_result() const {
  return impl_->cam2;
}
const PoseCameraResult& CudaPoseEvaluator::cam4_result() const {
  return impl_->cam4;
}
const PoseCameraJacobianResult& CudaPoseEvaluator::cam2_jacobian_result() const {
  return impl_->jacobian_cam2;
}
const PoseCameraJacobianResult& CudaPoseEvaluator::cam4_jacobian_result() const {
  return impl_->jacobian_cam4;
}
CudaPoseWorkspace* CudaPoseEvaluator::cam2_workspace() const {
  return impl_->cam2_workspace;
}
CudaPoseWorkspace* CudaPoseEvaluator::cam4_workspace() const {
  return impl_->cam4_workspace;
}
const PoseLMProblem& CudaPoseEvaluator::problem() const {
  return impl_->problem;
}

CudaMultiFramePoseEvaluator::CudaMultiFramePoseEvaluator(
    const std::vector<PoseLMProblem>& problems) {
  evaluators_.reserve(problems.size());
  for (const auto& problem : problems) {
    evaluators_.push_back(std::make_unique<CudaPoseEvaluator>(problem));
  }
  std::vector<CudaPoseWorkspace*> cam2, cam4;
  std::vector<float> anchors;
  cam2.reserve(evaluators_.size());
  cam4.reserve(evaluators_.size());
  anchors.reserve(evaluators_.size() * 63);
  for (const auto& evaluator : evaluators_) {
    cam2.push_back(evaluator->cam2_workspace());
    cam4.push_back(evaluator->cam4_workspace());
    const auto& anchor = evaluator->problem().body_pose_anchor;
    anchors.insert(anchors.end(), anchor.begin(), anchor.end());
  }
  batch_workspace_.reset(create_cuda_multiframe_pose_batch_workspace(
      cam2, cam4, anchors));
}
CudaMultiFramePoseEvaluator::~CudaMultiFramePoseEvaluator() = default;

void CudaMultiFramePoseEvaluator::upload_batch(
    const std::vector<int>& frame_indices,
    const std::vector<std::vector<float>>& parameters) {
  constexpr int camera_stride = 168;
  constexpr int packed_stride = 75 + 2 * camera_stride;
  std::vector<float> packed(frame_indices.size() * packed_stride, 0.0F);
  for (std::size_t item = 0; item < frame_indices.size(); ++item) {
    const auto& parameter = parameters[item];
    const auto& problem = evaluators_.at(frame_indices[item])->problem();
    float* destination = packed.data() + item * packed_stride;
    std::copy(parameter.begin(), parameter.end(), destination);
    const PoseCameraInput* cameras[] = {&problem.cam2, &problem.cam4};
    for (int camera = 0; camera < 2; ++camera) {
      float* camera_data = destination + 75 + camera * camera_stride;
      const int camera_offset = camera == 0 ? 0 : 6;
      for (int xyz = 0; xyz < 3; ++xyz) {
        camera_data[xyz] = parameter[camera_offset + xyz];
        camera_data[165 + xyz] = parameter[camera_offset + 3 + xyz];
      }
      for (int component = 0; component < 63; ++component) {
        camera_data[3 + component] = parameter[12 + component];
      }
      std::copy(cameras[camera]->left_hand_pose.begin(),
                cameras[camera]->left_hand_pose.end(), camera_data + 75);
      std::copy(cameras[camera]->right_hand_pose.begin(),
                cameras[camera]->right_hand_pose.end(), camera_data + 120);
    }
  }
  upload_cuda_multiframe_pose_batch(
      batch_workspace_.get(), frame_indices, parameters, packed);
}

std::vector<DenseNormalEquations>
CudaMultiFramePoseEvaluator::linearize_and_build(
    const std::vector<int>& frame_indices,
    const std::vector<std::vector<float>>& parameters) {
  if (frame_indices.size() != parameters.size()) {
    throw std::runtime_error("CUDA_MULTIFRAME_POSE_BATCH_SIZE_MISMATCH");
  }
  upload_batch(frame_indices, parameters);
  begin_cuda_deferred_execution();
  try {
    for (std::size_t item = 0; item < frame_indices.size(); ++item) {
      evaluators_.at(frame_indices[item])->linearize_batched(
          parameters[item], batch_workspace_.get(), static_cast<int>(item));
    }
  } catch (...) {
    try { end_cuda_deferred_execution(); } catch (...) {}
    throw;
  }
  end_cuda_deferred_execution();
  std::vector<double> jacobian_ms;
  jacobian_ms.reserve(frame_indices.size());
  for (int frame : frame_indices) {
    const auto& evaluator = *evaluators_.at(frame);
    jacobian_ms.push_back(
        evaluator.cam2_jacobian_result().total_jacobian_ms +
        evaluator.cam4_jacobian_result().total_jacobian_ms);
  }
  return build_cuda_multiframe_pose_normal_equations(
      batch_workspace_.get(), frame_indices,
      evaluators_.at(frame_indices.front())->problem().normal_config,
      jacobian_ms);
}

std::vector<double> CudaMultiFramePoseEvaluator::forward_and_evaluate(
    const std::vector<int>& frame_indices,
    const std::vector<std::vector<float>>& parameters) {
  if (frame_indices.size() != parameters.size()) {
    throw std::runtime_error("CUDA_MULTIFRAME_POSE_BATCH_SIZE_MISMATCH");
  }
  upload_batch(frame_indices, parameters);
  for (std::size_t item = 0; item < frame_indices.size(); ++item) {
    evaluators_.at(frame_indices[item])->forward_batched(
        parameters[item], batch_workspace_.get(), static_cast<int>(item));
  }
  std::vector<double> result;
  result.reserve(frame_indices.size());
  for (int frame : frame_indices) {
    result.push_back(evaluators_.at(frame)->evaluate_residual());
  }
  return result;
}

CudaBatchTransferStats CudaMultiFramePoseEvaluator::transfer_stats() const {
  return cuda_multiframe_pose_batch_transfer_stats(batch_workspace_.get());
}

struct CudaPoseShapeEvaluator::Impl {
  explicit Impl(const PoseShapeLMProblem& source) : problem(source) {
    std::vector<float> latent(source.initial_parameters.begin() + 75,
                              source.initial_parameters.end());
    const ShapeValueResult shape = evaluate_shape_value(
        latent, source.beta_limit, source.base_joints, source.joint_basis, 55);
    PoseCameraInput cam2 = source.cam2.pose;
    PoseCameraInput cam4 = source.cam4.pose;
    cam2.selected_v_shaped = evaluate_dynamic_vertices(
        source.cam2.selected_base_v_shaped,
        source.cam2.selected_beta_shapedirs, shape.beta, cam2.point_count);
    cam4.selected_v_shaped = evaluate_dynamic_vertices(
        source.cam4.selected_base_v_shaped,
        source.cam4.selected_beta_shapedirs, shape.beta, cam4.point_count);
    PoseModelInput model{shape.joints, source.parents};
    cam2_workspace = create_cuda_pose_workspace(model, cam2);
    cam4_workspace = create_cuda_pose_workspace(model, cam4);
    cam2_value_workspace = create_cuda_pose_workspace(model, cam2);
    cam4_value_workspace = create_cuda_pose_workspace(model, cam4);
    shape_value_workspace = create_cuda_pose_shape_value_workspace(source);
    reserve_cuda_pose_workspace(cam2_value_workspace, cam2, false);
    reserve_cuda_pose_workspace(cam4_value_workspace, cam4, false);
    reserve_cuda_pose_workspace(cam2_workspace, cam2, true);
    reserve_cuda_pose_workspace(cam4_workspace, cam4, true);
    prepare_cuda_pose_shape_workspace(cam2_workspace, source.cam2);
    prepare_cuda_pose_shape_workspace(cam4_workspace, source.cam4);
  }
  ~Impl() {
    destroy_cuda_pose_workspace(cam2_workspace);
    destroy_cuda_pose_workspace(cam4_workspace);
    destroy_cuda_pose_workspace(cam2_value_workspace);
    destroy_cuda_pose_workspace(cam4_value_workspace);
    destroy_cuda_pose_shape_value_workspace(shape_value_workspace);
  }
  const PoseShapeLMProblem& problem;
  std::vector<float> parameters;
  PoseShapeValueResult value;
  bool ready = false;
  bool linearization_ready = false;
  std::vector<float> linearization_parameters;
  PoseCameraJacobianResult pose_cam2;
  PoseCameraJacobianResult pose_cam4;
  std::vector<float> beta_jacobian_cam2;
  std::vector<float> beta_jacobian_cam4;
  CudaPoseWorkspace* cam2_workspace = nullptr;
  CudaPoseWorkspace* cam4_workspace = nullptr;
  CudaPoseWorkspace* cam2_value_workspace = nullptr;
  CudaPoseWorkspace* cam4_value_workspace = nullptr;
  CudaPoseShapeValueWorkspace* shape_value_workspace = nullptr;
};

CudaPoseShapeEvaluator::CudaPoseShapeEvaluator(
    const PoseShapeLMProblem& problem)
    : impl_(std::make_unique<Impl>(problem)) {}
CudaPoseShapeEvaluator::~CudaPoseShapeEvaluator() = default;

void CudaPoseShapeEvaluator::forward(
    const std::vector<float>& parameters) {
  impl_->parameters = parameters;
  impl_->value = evaluate_pose_shape_value_cuda_with_workspace(
      impl_->shape_value_workspace, impl_->cam2_value_workspace,
      impl_->cam4_value_workspace, parameters, impl_->problem);
  impl_->ready = true;
}

double CudaPoseShapeEvaluator::evaluate_residual() {
  if (!impl_->ready) {
    throw std::runtime_error("CUDA_POSE_SHAPE_FORWARD_REQUIRED");
  }
  return evaluate_pose_shape_objective_from_common(
      impl_->parameters, impl_->value, impl_->problem.beta_anchor,
      impl_->problem.body_pose_anchor,
      impl_->problem.normal_config.pose_anchor_scale,
      impl_->problem.normal_config.shape_anchor_scale,
      impl_->problem.normal_config.charbonnier_eps,
      impl_->problem.normal_config.cam2_alpha_per_point,
      impl_->problem.normal_config.cam4_alpha_per_point);
}

void CudaPoseShapeEvaluator::linearize(const std::vector<float>& parameters) {
  impl_->linearization_parameters = parameters;
  const PoseShapeValueResult shape_value =
      evaluate_pose_shape_value_cuda_with_workspace(
          impl_->shape_value_workspace, impl_->cam2_value_workspace,
          impl_->cam4_value_workspace, parameters, impl_->problem);
  std::vector<float> d_beta(10), d_joints(165 * 10);
  for (int beta = 0; beta < 10; ++beta) {
    d_beta[beta] = impl_->problem.beta_limit *
        (1.0F - shape_value.shape.tanh_latent[beta] *
                    shape_value.shape.tanh_latent[beta]);
  }
  for (int joint = 0; joint < 165; ++joint)
    for (int beta = 0; beta < 10; ++beta)
      d_joints[joint * 10 + beta] =
          impl_->problem.joint_basis[joint * 10 + beta] * d_beta[beta];
  update_cuda_pose_workspace_shape(
      impl_->cam2_workspace, shape_value.shape.joints,
      shape_value.cam2_dynamic_v_shaped);
  update_cuda_pose_workspace_shape(
      impl_->cam4_workspace, shape_value.shape.joints,
      shape_value.cam4_dynamic_v_shaped);
  PoseCameraInput cam2 = impl_->problem.cam2.pose;
  PoseCameraInput cam4 = impl_->problem.cam4.pose;
  cam2.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam4.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam2.selected_v_shaped = shape_value.cam2_dynamic_v_shaped;
  cam4.selected_v_shaped = shape_value.cam4_dynamic_v_shaped;
  PoseModelInput model{shape_value.shape.joints, impl_->problem.parents};
  impl_->pose_cam2 = evaluate_pose_camera_with_jacobian_cuda_with_workspace(
      impl_->cam2_workspace, model, cam2);
  impl_->beta_jacobian_cam2 =
      evaluate_pose_shape_beta_jacobian_cuda_with_workspace(
          impl_->cam2_workspace, impl_->problem.cam2, d_joints, d_beta, false);
  impl_->pose_cam4 = evaluate_pose_camera_with_jacobian_cuda_with_workspace(
      impl_->cam4_workspace, model, cam4);
  impl_->beta_jacobian_cam4 =
      evaluate_pose_shape_beta_jacobian_cuda_with_workspace(
          impl_->cam4_workspace, impl_->problem.cam4, d_joints, d_beta, false);
  impl_->linearization_ready = true;
}
DenseNormalEquations CudaPoseShapeEvaluator::build_normal_equations() {
  if (!impl_->linearization_ready) {
    throw std::runtime_error("CUDA_POSE_SHAPE_LINEARIZATION_REQUIRED");
  }
  const CudaPoseShapeNormalEquationResult blocks =
      build_cuda_pose_shape_normal_equations(
          impl_->cam2_workspace, impl_->cam4_workspace,
          impl_->linearization_parameters, impl_->problem.body_pose_anchor,
          impl_->problem.beta_anchor, impl_->problem.normal_config);
  DenseNormalEquations output;
  output.hessian.assign(85 * 85, 0.0);
  output.gradient.resize(85);
  for (int frame = 0; frame < 75; ++frame) {
    output.gradient[frame] = blocks.g_pose[frame];
    for (int other = 0; other < 75; ++other)
      output.hessian[frame * 85 + other] = blocks.a[frame * 75 + other];
    for (int beta = 0; beta < 10; ++beta) {
      output.hessian[frame * 85 + 75 + beta] = blocks.b[frame * 10 + beta];
      output.hessian[(75 + beta) * 85 + frame] = blocks.b[frame * 10 + beta];
    }
  }
  for (int beta = 0; beta < 10; ++beta) {
    output.gradient[75 + beta] = blocks.g_beta[beta];
    for (int other = 0; other < 10; ++other)
      output.hessian[(75 + beta) * 85 + 75 + other] =
          blocks.c[beta * 10 + other];
  }
  output.jacobian_ms = impl_->pose_cam2.total_jacobian_ms +
      impl_->pose_cam4.total_jacobian_ms;
  output.normal_equations_ms = blocks.total_ms;
  auto system = std::make_shared<SchurSystem>(SchurSystem{
      1, 75, 10, blocks.a, blocks.b, blocks.c,
      blocks.g_pose, blocks.g_beta});
  output.solve_override = [system](double damping,
      std::vector<double>& delta, std::string& backend) {
    const PoseShapeSchurResult solved =
        solve_pose_shape_schur(*system, damping, false);
    if (solved.success) {
      delta = solved.full_delta;
      backend = "schur";
      return true;
    }
    backend = "dense_fallback";
    return solve_schur_dense_fallback(*system, damping, delta);
  };
  output.quadratic_form_override = [system](const std::vector<double>& delta) {
    return schur_quadratic_form(*system, delta);
  };
  return output;
}
const PoseShapeValueResult& CudaPoseShapeEvaluator::value_result() const {
  return impl_->value;
}
CudaPoseWorkspace* CudaPoseShapeEvaluator::cam2_workspace() const {
  return impl_->cam2_workspace;
}
CudaPoseWorkspace* CudaPoseShapeEvaluator::cam4_workspace() const {
  return impl_->cam4_workspace;
}
double CudaPoseShapeEvaluator::last_jacobian_ms() const {
  return impl_->pose_cam2.total_jacobian_ms +
      impl_->pose_cam4.total_jacobian_ms;
}

}  // namespace staged_lm
