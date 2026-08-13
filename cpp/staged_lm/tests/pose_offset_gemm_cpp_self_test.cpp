#include "staged_lm/experimental/pose_offset_gemm.h"
#include "staged_lm/pose_jacobian.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  constexpr int n = 3;
  std::vector<float> posedirs(486 * n * 3);
  std::vector<float> derivative(486 * 66);
  for (std::size_t index = 0; index < posedirs.size(); ++index)
    posedirs[index] = std::sin(static_cast<float>(index) * 0.01F);
  for (std::size_t index = 0; index < derivative.size(); ++index)
    derivative[index] = std::cos(static_cast<float>(index) * 0.003F);
  std::vector<float> scalar, gemm, transposed;
  staged_lm::compute_pose_offset_derivative_scalar(
      posedirs, derivative, n, scalar);
  staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(
      posedirs, derivative, n, gemm);
  staged_lm::experimental::compute_pose_offset_derivative_transposed_eigen_gemm(
      posedirs, derivative, n, transposed);
  float maximum = 0.0F;
  for (std::size_t index = 0; index < scalar.size(); ++index)
    maximum = std::max(maximum, std::abs(scalar[index] - gemm[index]));
  if (maximum > 1e-4F) return 1;
  if (transposed != gemm) return 2;
  std::cout << "POSE_OFFSET_GEMM_CPP_SELF_TEST=PASS\n";
  return 0;
}
