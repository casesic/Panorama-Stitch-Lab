#include <Eigen/Core>

#include <iostream>

#ifdef EIGEN_VECTORIZE
constexpr bool kEigenVectorize = true;
#else
constexpr bool kEigenVectorize = false;
#endif

int main() {
  if (kEigenVectorize != static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE)) {
    return 1;
  }
  if (Eigen::nbThreads() != 1) {
    return 1;
  }
  if (STAGED_LM_EXPECT_MARCH_NATIVE && !kEigenVectorize) {
    return 1;
  }
  std::cout << "LBS_SELECTED_TRANSFORM_VECTORISATION_CPP_SELF_TEST=PASS\n";
  return 0;
}
