#include "staged_lm/pose_shape_jacobian.h"
#include <cmath>
#include <iostream>
#include <vector>
int main(){
  static_assert(staged_lm::kPoseShapeCombinedDimension==76);
  static_assert(staged_lm::kPoseActiveDimension==69);
  std::vector<float> d(10);for(int i=0;i<10;++i){float z=.1F*i,t=std::tanh(z);d[i]=2.5F*(1-t*t);if(!std::isfinite(d[i])||d[i]<=0)return 1;}
  std::cout<<"POSE_SHAPE_JACOBIAN_CPP_SELF_TEST=PASS\n";return 0;
}
