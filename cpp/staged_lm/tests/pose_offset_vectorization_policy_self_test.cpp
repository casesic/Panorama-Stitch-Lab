#include <Eigen/Core>

#include <iostream>

int main() {
#ifdef EIGEN_VECTORIZE
  constexpr bool vectorized = true;
#else
  constexpr bool vectorized = false;
#endif
  if (vectorized != static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE)) {
    std::cerr << "POSE_OFFSET_VECTORIZE_POLICY_MISMATCH\n";
    return 1;
  }
  if (STAGED_LM_EXPECT_MARCH_NATIVE && !vectorized) {
    std::cerr << "POSE_OFFSET_NATIVE_WITHOUT_VECTORIZE\n";
    return 1;
  }
  std::cout << "POSE_OFFSET_VECTORISATION_POLICY_SELF_TEST=PASS\n";
  return 0;
}
