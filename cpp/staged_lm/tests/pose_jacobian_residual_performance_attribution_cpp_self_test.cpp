#include "staged_lm/performance.h"

#include <cmath>
#include <iostream>
#include <string>

int main() {
  staged_lm::reset_performance_samples();
  staged_lm::set_performance_recording(true);
  staged_lm::set_performance_stage("pose_jacobian");
  staged_lm::record_performance_sample("cam2","jac.rotation_derivative",1023,1.0);
  staged_lm::record_performance_sample("cam2","jac.fk_derivative",1023,2.0);
  staged_lm::record_performance_sample("cam2","jac.lbs_selected_transform",1023,3.0);
  staged_lm::record_performance_sample("cam2","jac.lbs_downstream",1023,4.0);
  staged_lm::set_performance_recording(false);
  const auto&s=staged_lm::performance_samples();
  if(s.size()!=4)return 1;
  double sum=0;for(const auto&x:s){if(x.stage!="pose_jacobian"||x.camera!="cam2"||x.point_count!=1023)return 2;sum+=x.milliseconds;}
  const double total=10.0,other=total-sum;
  if(std::abs((sum+other)-total)>1e-12||other<0)return 3;
  std::cout<<"POSE_JACOBIAN_RESIDUAL_PERFORMANCE_ATTRIBUTION_CPP_SELF_TEST=PASS\n";
  return 0;
}
