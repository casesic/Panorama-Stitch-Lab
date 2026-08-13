#include "staged_lm/experimental/selected_transform_gemm.h"
#include "staged_lm/pose_jacobian.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
bool check(int q,bool zero,bool onehot){int n=3;std::vector<float>w(n*55),da(55*16*q);for(int i=0;i<n;++i)for(int j=0;j<55;++j)w[i*55+j]=zero?0:(onehot?(j==7?1:0):float((i+j)%9)/99);for(std::size_t i=0;i<da.size();++i)da[i]=float(int(i%17)-8)*1e-3F;std::vector<float>a,b;staged_lm::compute_selected_transform_derivative_scalar(w,da,n,q,a);staged_lm::experimental::compute_selected_transform_derivative_eigen_gemm(w,da,n,q,b);for(std::size_t i=0;i<a.size();++i)if(!std::isfinite(b[i])||std::abs(a[i]-b[i])>2e-6F)return false;if(onehot)for(int i=0;i<n;++i)for(int e=0;e<16;++e)for(int x=0;x<q;++x)if(b[(i*16+e)*q+x]!=da[(7*16+e)*q+x])return false;return true;}
int main(){if(!check(5,false,false)||!check(66,false,false)||!check(76,false,false)||!check(66,true,false)||!check(76,false,true))return 1;int q=7,n=2,j=9,e=11,x=4;std::vector<float>w(n*55),da(55*16*q),out;w[j]=.25F;w[55+j]=.75F;da[(j*16+e)*q+x]=1;staged_lm::experimental::compute_selected_transform_derivative_eigen_gemm(w,da,n,q,out);if(out[(e*q)+x]!=.25F||out[((16+e)*q)+x]!=.75F)return 1;bool rejected=false;try{staged_lm::experimental::compute_selected_transform_derivative_eigen_gemm(w,da,n,q,w);}catch(const std::runtime_error&){rejected=true;}if(!rejected)return 1;std::cout<<"SELECTED_TRANSFORM_GEMM_CPP_SELF_TEST=PASS\n";return 0;}
