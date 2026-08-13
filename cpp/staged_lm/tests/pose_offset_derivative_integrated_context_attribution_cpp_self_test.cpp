#include "staged_lm/experimental/pose_offset_gemm.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(){const int n=7;std::vector<float> p(n*3*486,.001F),d(486*66,.002F),a,b;staged_lm::experimental::PoseOffsetGemmTiming t;staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p,d,n,a,&t);staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p,d,n,b);if(a!=b||a.size()!=static_cast<std::size_t>(n*3*66)||!std::isfinite(t.total_ms)||t.total_ms+1e-6<t.output_preparation_ms+t.view_setup_ms+t.gemm_core_ms||t.capacity_after<a.size())throw std::runtime_error("ATTRIBUTION_SELF_TEST_FAILED");std::cout<<"POSE_OFFSET_DERIVATIVE_INTEGRATED_CONTEXT_ATTRIBUTION_CPP_SELF_TEST_PASS\n";}
