#include "staged_lm/experimental/value_pose_offset_gemv.h"
#include "staged_lm/pose_forward.h"
#include <cmath>
#include <iostream>
#include <vector>
bool check(int n,int hot){std::vector<float>p(486*n*3),x(486),a,b;for(size_t i=0;i<p.size();++i)p[i]=float(int(i%31)-15)*1e-3F;if(hot>=0)x[hot]=1;else for(int i=0;i<486;++i)x[i]=float(i%7)*1e-3F;staged_lm::compute_value_pose_offset_scalar(p,x,n,a);staged_lm::experimental::compute_value_pose_offset_eigen_gemv(p,x,n,b);for(size_t i=0;i<a.size();++i)if(!std::isfinite(b[i])||std::abs(a[i]-b[i])>2e-5F)return false;if(hot>=0)for(int r=0;r<n*3;++r)if(b[r]!=p[hot*n*3+r])return false;return true;}
int main(){if(!check(2,-1)||!check(1023,-1)||!check(927,-1)||!check(3,137))return 1;std::vector<float>p(486*9,1),z(486),o;staged_lm::experimental::compute_value_pose_offset_eigen_gemv(p,z,3,o);for(float x:o)if(x!=0)return 1;bool rejected=false;try{staged_lm::experimental::compute_value_pose_offset_eigen_gemv(p,z,3,z);}catch(...){rejected=true;}if(!rejected)return 1;std::cout<<"POSE_OBJECTIVE_POSE_OFFSET_GEMV_CPP_SELF_TEST=PASS\n";}
