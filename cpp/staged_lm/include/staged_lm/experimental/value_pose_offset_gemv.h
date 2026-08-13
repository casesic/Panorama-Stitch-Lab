#pragma once

#include <vector>

namespace staged_lm::experimental {

struct ValuePoseOffsetGemvTiming {
  double input_pack_ms = 0.0;
  double gemv_core_ms = 0.0;
  double output_unpack_ms = 0.0;
  double total_ms = 0.0;
};

void compute_value_pose_offset_eigen_gemv(
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& pose_feature,
    int point_count,
    std::vector<float>& output,
    ValuePoseOffsetGemvTiming* timing = nullptr);

}  // namespace staged_lm::experimental
