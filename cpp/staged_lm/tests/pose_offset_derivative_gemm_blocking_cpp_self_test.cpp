#include "staged_lm/experimental/pose_offset_gemm.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  for (const int points : {1023, 927, 71}) {
    std::vector<float> p(static_cast<std::size_t>(points) * 3 * 486);
    std::vector<float> d(486 * 66);
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = float(int(i % 31) - 15) * 1e-4F;
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = float(int(i % 17) - 8) * 2e-4F;
    std::vector<float> baseline, blocked;
    staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p, d, points, baseline);
    const float* p_pointer = p.data();
    for (const auto blocks : {std::pair{512, 0}, std::pair{0, 48}}) {
      staged_lm::experimental::compute_pose_offset_derivative_blocked_eigen_gemm(
          p, d, points, blocks.first, blocks.second, blocked);
      if (blocked.size() != baseline.size() || p.data() != p_pointer)
        throw std::runtime_error("BLOCKING_LAYOUT_OR_STABLE_P_FAILURE");
      for (std::size_t i = 0; i < blocked.size(); ++i)
        if (!std::isfinite(blocked[i]) || std::abs(blocked[i] - baseline[i]) > 2e-8F)
          throw std::runtime_error("BLOCKING_PARITY_FAILURE");
    }
  }
  std::cout << "POSE_OFFSET_DERIVATIVE_GEMM_BLOCKING_CPP_SELF_TEST_PASS\n";
}
