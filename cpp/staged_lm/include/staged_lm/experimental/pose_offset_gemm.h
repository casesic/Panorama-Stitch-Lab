#pragma once

#include <vector>

namespace staged_lm::experimental {

struct PoseOffsetGemmTiming {
  double output_preparation_ms = 0.0;
  double view_setup_ms = 0.0;
  double input_pack_ms = 0.0;
  double gemm_core_ms = 0.0;
  double output_unpack_ms = 0.0;
  double total_ms = 0.0;
  std::size_t capacity_before = 0;
  std::size_t capacity_after = 0;
  const float* pointer_before = nullptr;
  const float* pointer_after = nullptr;
};

void compute_pose_offset_derivative_eigen_gemm(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    std::vector<float>& output,
    PoseOffsetGemmTiming* timing = nullptr);

// Experimental zero-copy orientation: Y^T = D^T P^T.  Inputs and the
// externally visible row-major Y layout are unchanged.
void compute_pose_offset_derivative_transposed_eigen_gemm(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    std::vector<float>& output,
    PoseOffsetGemmTiming* timing = nullptr);

// Experimental explicit blocking of the production P * D expression.  A zero
// block size means that dimension is left whole.  Blocks are Eigen views: no
// candidate-specific input/output temporary or packing buffer is created.
void compute_pose_offset_derivative_blocked_eigen_gemm(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature,
    int point_count,
    int row_block_size,
    int column_block_size,
    std::vector<float>& output,
    PoseOffsetGemmTiming* timing = nullptr);

}  // namespace staged_lm::experimental
