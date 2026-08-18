#include "staged_lm/multiframe.h"
#include "staged_lm/pose_shape_schur.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

double g_schur_dense_max_error = 0.0;

staged_lm::PoseShapeCameraInput empty_camera(const char* name) {
  staged_lm::PoseShapeCameraInput camera;
  camera.pose.camera = name;
  camera.pose.parameters.assign(75, 0.0F);
  camera.pose.left_hand_pose.assign(45, 0.0F);
  camera.pose.right_hand_pose.assign(45, 0.0F);
  camera.pose.point_count = 0;
  camera.pose.image_width = 8192;
  camera.pose.image_height = 4096;
  return camera;
}

staged_lm::MultiFramePoseShapeLMProblem problem(int frame_count) {
  staged_lm::MultiFramePoseShapeLMProblem result;
  result.base_joints.assign(165, 0.0F);
  result.joint_basis.assign(1650, 0.0F);
  result.beta_anchor.assign(10, 0.0F);
  result.parents.resize(55);
  result.parents[0] = -1;
  for (int joint = 1; joint < 55; ++joint) result.parents[joint] = joint - 1;
  result.normal_config.charbonnier_eps = 1.0F;
  result.normal_config.cam2_alpha_per_point = 1.0F;
  result.normal_config.cam4_alpha_per_point = 1.0F;
  result.normal_config.pose_anchor_scale = 1.0F;
  result.normal_config.shape_anchor_scale = 1.0F;
  result.lm_config = {0, 1, 1, 1e-12, 1e-12, 1e-12};
  for (int frame = 0; frame < frame_count; ++frame) {
    staged_lm::MultiFramePoseShapeFrameProblem item;
    item.cam2 = empty_camera("cam2");
    item.cam4 = empty_camera("cam4");
    item.body_pose_anchor.assign(63, 0.0F);
    item.initial_parameters.assign(75, 0.0F);
    result.frames.push_back(std::move(item));
  }
  return result;
}

bool t1_equivalence() {
  using namespace staged_lm;
  auto multi_problem = problem(1);
  PoseShapeLMProblem single;
  single.initial_parameters.assign(85, 0.0F);
  single.base_joints = multi_problem.base_joints;
  single.joint_basis = multi_problem.joint_basis;
  single.beta_anchor = multi_problem.beta_anchor;
  single.body_pose_anchor = multi_problem.frames[0].body_pose_anchor;
  single.parents = multi_problem.parents;
  single.cam2 = multi_problem.frames[0].cam2;
  single.cam4 = multi_problem.frames[0].cam4;
  single.normal_config = multi_problem.normal_config;
  single.lm_config = multi_problem.lm_config;
  single.lm_config.parameter_dimension = 85;
  const LMRunResult old_result = run_pose_shape_lm(single);
  const LMRunResult new_result = run_multiframe_pose_shape_lm(multi_problem);
  if (old_result.final_parameters != new_result.final_parameters ||
      old_result.initial_objective != new_result.initial_objective ||
      old_result.final_objective != new_result.final_objective ||
      old_result.status != new_result.status ||
      old_result.trials.size() != new_result.trials.size() ||
      old_result.linearizations.size() != new_result.linearizations.size()) {
    return false;
  }
  for (std::size_t index = 0; index < old_result.trials.size(); ++index) {
    if (old_result.trials[index].candidate !=
            new_result.trials[index].candidate ||
        old_result.trials[index].accepted !=
            new_result.trials[index].accepted) {
      return false;
    }
  }
  return true;
}

bool layout_and_finite(int frame_count) {
  using namespace staged_lm;
  const LMRunResult result = run_multiframe_pose_shape_lm(problem(frame_count));
  const std::size_t expected = multiframe_parameter_dimension(frame_count);
  if (result.initial_parameters.size() != expected ||
      result.final_parameters.size() != expected ||
      multiframe_beta_offset(frame_count) != std::size_t(75 * frame_count) ||
      !std::isfinite(result.initial_objective) ||
      !std::isfinite(result.final_objective) || result.linearizations.empty()) {
    return false;
  }
  const auto& hessian = result.linearizations.front().hessian;
  for (int first = 0; first < frame_count; ++first) {
    for (int second = 0; second < frame_count; ++second) {
      if (first == second) continue;
      for (int row = 0; row < 75; ++row) {
        for (int column = 0; column < 75; ++column) {
          if (hessian[(first * 75 + row) * expected + second * 75 + column]
              != 0.0) return false;
        }
      }
    }
  }
  return true;
}

bool schur_dense_and_arrowhead() {
  using namespace staged_lm;
  SchurSystem system;
  system.frame_count = 2;
  system.frame_dim = 75;
  system.beta_dim = 10;
  system.a.assign(2 * 75 * 75, 0.0);
  system.b.assign(2 * 75 * 10, 0.0);
  system.c.assign(10 * 10, 0.0);
  system.g_frame.resize(2 * 75);
  system.g_beta.resize(10);
  for (int frame = 0; frame < 2; ++frame) {
    for (int index = 0; index < 75; ++index) {
      system.a[(frame * 75 + index) * 75 + index] = 2.0 + index * 0.01;
      system.g_frame[frame * 75 + index] = 0.001 * (frame + index + 1);
      for (int beta = 0; beta < 10; ++beta) {
        system.b[(frame * 75 + index) * 10 + beta] =
            1e-4 * (1 + frame + index + beta);
      }
    }
  }
  for (int beta = 0; beta < 10; ++beta) {
    system.c[beta * 10 + beta] = 4.0 + beta * 0.1;
    system.g_beta[beta] = -0.01 * (beta + 1);
  }
  const PoseShapeSchurResult solved =
      solve_pose_shape_schur(system, 0.1, true);
  if (!solved.success || solved.full_delta.size() != 160 ||
      solved.dense_delta.size() != 160) return false;
  double maximum_error = 0.0;
  for (std::size_t index = 0; index < solved.full_delta.size(); ++index) {
    maximum_error = std::max(
        maximum_error,
        std::abs(solved.full_delta[index] - solved.dense_delta[index]));
  }
  g_schur_dense_max_error = maximum_error;
  return maximum_error <= 1e-10;
}

}  // namespace

int main() {
  if (staged_lm::multiframe_parameter_dimension(1) != 85 ||
      staged_lm::multiframe_parameter_dimension(2) != 160 ||
      staged_lm::multiframe_parameter_dimension(4) != 310) return 1;
  if (!t1_equivalence()) return 2;
  if (!layout_and_finite(2)) return 3;
  if (!layout_and_finite(4)) return 4;
  if (!schur_dense_and_arrowhead()) return 5;
  std::cout << "T1_EQUIVALENCE=PASS\n"
            << "TRAJECTORY_T1_EXACT=YES\n"
            << "SHARED_BETA_BLOCK_COUNT=1\n"
            << "MULTIFRAME_SCHUR_DENSE_PARITY=PASS\n"
            << "MULTIFRAME_SCHUR_DENSE_MAX_ABS="
            << g_schur_dense_max_error << '\n'
            << "CROSS_FRAME_POSE_BLOCKS_ZERO=PASS\n"
            << "MULTI_FRAME_STAGED_LM_READY=PASS\n";
  return 0;
}
