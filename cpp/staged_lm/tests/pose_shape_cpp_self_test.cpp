#include "staged_lm/shape.h"
#include <cmath>
#include <iostream>
int main(){std::vector<float> z(10,0),base(55*3,.5F),basis(55*3*10,.01F);auto s=staged_lm::evaluate_shape_value(z,2.5F,base,basis,55);if(s.beta!=z||s.joints!=base){std::cerr<<"BETA_ZERO_FAILED\n";return 1;}z.assign(10,100.F);s=staged_lm::evaluate_shape_value(z,2.5F,base,basis,55);for(float b:s.beta)if(!std::isfinite(b)||std::abs(b)>2.5F){std::cerr<<"BETA_LIMIT_FAILED\n";return 1;}auto v=staged_lm::evaluate_dynamic_vertices({1,2,3},std::vector<float>(30,0),s.beta,1);for(float x:v)if(!std::isfinite(x)){std::cerr<<"SHAPE_FINITE_FAILED\n";return 1;}std::cout<<"POSE_SHAPE_CPP_SELF_TEST_PASS\n";return 0;}
