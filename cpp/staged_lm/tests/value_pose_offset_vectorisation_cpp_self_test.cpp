#include <Eigen/Core>

#include <iostream>
#include <stdexcept>

#ifdef EIGEN_VECTORIZE
constexpr bool kEigenVectorize = true;
#else
constexpr bool kEigenVectorize = false;
#endif

int main() {
  if (kEigenVectorize != static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE))
    throw std::runtime_error("EIGEN_VECTORIZE_BUILD_POLICY_MISMATCH");
  if (STAGED_LM_EXPECT_MARCH_NATIVE && !kEigenVectorize)
    throw std::runtime_error("NATIVE_REQUIRES_EIGEN_VECTORIZE");
  if (Eigen::nbThreads() != 1)
    throw std::runtime_error("EIGEN_THREAD_POLICY_MISMATCH");
  std::cout << "VALUE_POSE_OFFSET_VECTORISATION_CPP_SELF_TEST_PASS\n";
}
