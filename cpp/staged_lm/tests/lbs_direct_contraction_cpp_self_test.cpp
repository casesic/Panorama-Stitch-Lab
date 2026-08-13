#include "staged_lm/experimental/lbs_direct.h"
#include "staged_lm/pose_jacobian.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
bool check(int n,int q,bool zero_weights,bool one_hot,bool zero_da,bool zero_dh){
  std::vector<float>w(n*55),da(55*16*q),t(n*16),v(n*3),dh(n*3*q);
  for(int i=0;i<n;++i)for(int j=0;j<55;++j)w[i*55+j]=zero_weights?0.0F:(one_hot?(j==7?1.0F:0.0F):static_cast<float>((i+j)%7+1)/1540.0F);
  for(std::size_t i=0;i<da.size();++i)da[i]=zero_da?0.0F:static_cast<float>(static_cast<int>(i%17)-8)*1e-4F;
  for(std::size_t i=0;i<t.size();++i)t[i]=static_cast<float>(static_cast<int>(i%11)-5)*0.02F;
  for(std::size_t i=0;i<v.size();++i)v[i]=static_cast<float>(i%5)*0.1F;
  for(std::size_t i=0;i<dh.size();++i)dh[i]=zero_dh?0.0F:static_cast<float>(static_cast<int>(i%13)-6)*2e-4F;
  staged_lm::LbsDerivativeMaterializedResult a;
  std::vector<float>b;staged_lm::experimental::LbsDirectDebugTerms d;
  staged_lm::compute_lbs_derivative_materialized(w,da,t,v,dh,n,q,a);
  staged_lm::experimental::compute_lbs_derivative_direct(w,da,t,v,dh,n,q,b,&d);
  if(b.size()!=static_cast<std::size_t>(n*3*q))return false;
  for(std::size_t i=0;i<b.size();++i)if(!std::isfinite(b[i])||std::abs(a.skinned_derivatives[i]-b[i])>2e-6F||std::abs(a.transform_term[i]-d.transform_term[i])>2e-6F||a.vertex_term[i]!=d.vertex_term[i])return false;
  if(zero_weights&&zero_dh)for(float x:b)if(x!=0)return false;
  return true;
}}
int main(){
  if(!check(2,5,false,false,false,false)||!check(2,66,false,false,false,false)||
     !check(2,76,false,false,false,false)||!check(2,66,true,false,false,false)||
     !check(2,66,false,true,false,true)||!check(2,66,false,false,false,true)||
     !check(2,66,false,false,true,false)||!check(2,66,true,false,true,true)){
    std::cerr<<"LBS_DIRECT_CONTRACTION_SELF_TEST_FAILED\n";return 1;}
  std::vector<float> alias(55, 0.0F), da(55 * 16, 0.0F), transforms(16, 0.0F),
      posed(3, 0.0F), dh(3, 0.0F);
  bool rejected_alias = false;
  try {
    staged_lm::experimental::compute_lbs_derivative_direct(
        alias, da, transforms, posed, dh, 1, 1, alias);
  } catch (const std::runtime_error&) { rejected_alias = true; }
  if (!rejected_alias) { std::cerr<<"LBS_DIRECT_ALIAS_TEST_FAILED\n";return 1; }
  std::cout<<"LBS_DIRECT_CONTRACTION_CPP_SELF_TEST=PASS\n";return 0;
}
