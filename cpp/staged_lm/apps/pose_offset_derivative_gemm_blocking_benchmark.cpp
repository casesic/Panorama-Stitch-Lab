#include "staged_lm/binary_io.h"
#include "staged_lm/experimental/pose_offset_gemm.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Candidate { std::string name, type; int bm, bn; };
struct Call { int id, n; std::string stage, camera; std::vector<float> p, d; };
struct Run { double total=0, checksum=0; std::vector<double> call_ms; };
double ms(Clock::time_point s) { return std::chrono::duration<double,std::milli>(Clock::now()-s).count(); }
std::filesystem::path required(int n,char**v,const std::string&k){for(int i=1;i+1<n;++i)if(v[i]==k)return v[i+1];throw std::runtime_error("missing "+k);}
int option(int n,char**v,const std::string&k,int d){for(int i=1;i+1<n;++i)if(v[i]==k)return std::stoi(v[i+1]);return d;}
std::vector<std::string> split(const std::string&s){std::vector<std::string>r;std::stringstream q(s);std::string x;while(std::getline(q,x,','))r.push_back(x);return r;}
double checksum(const std::vector<float>&x){double s=0;for(std::size_t i=0;i<x.size();i+=997)s+=x[i];return s;}
Run trajectory(const Candidate& candidate,const std::vector<Call>& calls,
               const std::vector<float>& p2,const std::vector<float>& p4,
               std::vector<std::vector<float>>* outputs=nullptr) {
  Run r; if(outputs) outputs->resize(calls.size()); r.call_ms.reserve(calls.size());
  const auto started=Clock::now();
  for(std::size_t i=0;i<calls.size();++i){
    const auto& c=calls[i]; const auto& p=c.camera=="cam2"?p2:p4;
    std::vector<float> scratch; auto& y=outputs?(*outputs)[i]:scratch;
    staged_lm::experimental::PoseOffsetGemmTiming timing;
    if(candidate.type=="A") staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p,c.d,c.n,y,&timing);
    else staged_lm::experimental::compute_pose_offset_derivative_blocked_eigen_gemm(p,c.d,c.n,candidate.bm,candidate.bn,y,&timing);
    r.call_ms.push_back(timing.total_ms); r.checksum+=checksum(y);
  }
  r.total=ms(started); return r;
}
}

int main(int argc,char**argv){try{
  const auto fixture=required(argc,argv,"--fixture"), output=required(argc,argv,"--output");
  const int warmup=option(argc,argv,"--warmup",3), repeats=option(argc,argv,"--repeat",10), seed=option(argc,argv,"--seed",51366);
  if(Eigen::nbThreads()!=1) throw std::runtime_error("EIGEN_THREAD_POLICY_MISMATCH");
  std::filesystem::create_directories(output);
  std::ifstream in(fixture/"manifest.csv"); std::string line; std::getline(in,line); std::vector<Call> calls;
  while(std::getline(in,line)){auto f=split(line);const int n=std::stoi(f[3]);calls.push_back({std::stoi(f[0]),n,f[1],f[2],staged_lm::read_float32_file(fixture/f[6],static_cast<std::size_t>(n)*3*486),staged_lm::read_float32_file(fixture/f[7],486*66)});}
  if(calls.size()!=16)throw std::runtime_error("EXPECTED_16_CALL_CORPUS");
  const auto p2=calls[0].p; auto cam4=std::find_if(calls.begin(),calls.end(),[](const Call&c){return c.camera=="cam4";}); if(cam4==calls.end())throw std::runtime_error("CAM4_MISSING"); const auto p4=cam4->p;
  for(const auto&c:calls)if(c.p!=(c.camera=="cam2"?p2:p4))throw std::runtime_error("POSEDIRS_CONTENT_NOT_STATIC_BY_CAMERA");
  const std::vector<Candidate> candidates={{"A_CURRENT_EIGEN_GEMM","A",0,0},{"B_M192","M",192,0},{"B_M256","M",256,0},{"B_M384","M",384,0},{"B_M512","M",512,0},{"C_N16","N",0,16},{"C_N24","N",0,24},{"C_N32","N",0,32},{"C_N48","N",0,48}};
  for(int w=0;w<warmup;++w)for(const auto&c:candidates)trajectory(c,calls,p2,p4);
  std::ofstream samples(output/"samples.csv"); samples<<"candidate,sample,trajectory_ms,pose_cam2_ms,pose_cam4_ms,pose_shape_cam2_ms,pose_shape_cam4_ms,checksum\n"<<std::setprecision(12);
  std::ofstream per(output/"per_call.csv"); per<<"candidate,sample,stage,camera,call_index,M,Q,gemm_ms\n"<<std::setprecision(12);
  std::mt19937 rng(seed); std::vector<int> order(candidates.size()); for(std::size_t i=0;i<order.size();++i)order[i]=static_cast<int>(i);
  for(int sample=0;sample<repeats;++sample){std::shuffle(order.begin(),order.end(),rng);for(int index:order){const auto&r=trajectory(candidates[index],calls,p2,p4);double pc2=0,pc4=0,sc2=0,sc4=0;for(std::size_t i=0;i<calls.size();++i){const auto&c=calls[i];const double t=r.call_ms[i];if(c.stage=="pose_jacobian")(c.camera=="cam2"?pc2:pc4)+=t;else if(c.stage=="pose_shape_jacobian")(c.camera=="cam2"?sc2:sc4)+=t;else throw std::runtime_error("UNKNOWN_CORPUS_STAGE");per<<candidates[index].name<<','<<sample<<','<<c.stage<<','<<c.camera<<','<<i<<','<<c.n*3<<",66,"<<t<<'\n';}samples<<candidates[index].name<<','<<sample<<','<<r.total<<','<<pc2<<','<<pc4<<','<<sc2<<','<<sc4<<','<<r.checksum<<'\n';}}
  std::vector<std::vector<float>> baseline; trajectory(candidates[0],calls,p2,p4,&baseline);
  std::ofstream parity(output/"parity.csv");parity<<"candidate,call_index,stage,camera,max_abs,mean_abs,p99_abs,rel_frob,worst_row,worst_q\n"<<std::setprecision(12);
  for(std::size_t ci=1;ci<candidates.size();++ci){std::vector<std::vector<float>> got;trajectory(candidates[ci],calls,p2,p4,&got);for(std::size_t i=0;i<calls.size();++i){double ma=0,mean=0,ss=0,rr=0;std::size_t wi=0;std::vector<double> diffs;diffs.reserve(got[i].size());for(std::size_t j=0;j<got[i].size();++j){const double d=std::abs(double(got[i][j])-baseline[i][j]);diffs.push_back(d);mean+=d;ss+=d*d;rr+=double(baseline[i][j])*baseline[i][j];if(d>ma){ma=d;wi=j;}}std::sort(diffs.begin(),diffs.end());parity<<candidates[ci].name<<','<<i<<','<<calls[i].stage<<','<<calls[i].camera<<','<<ma<<','<<mean/diffs.size()<<','<<diffs[static_cast<std::size_t>(.99*(diffs.size()-1))]<<','<<std::sqrt(ss/std::max(rr,1e-300))<<','<<wi/66<<','<<wi%66<<'\n';}}
  std::ofstream manifest(output/"candidate_manifest.csv");manifest<<"candidate,blocking_type,BM,BN,M,K,N,block_count_cam2,block_count_cam4,tail_cam2,tail_cam4,user_temp_bytes,user_allocations,complexity\n";for(const auto&c:candidates){const int bm=c.bm?c.bm:3069,bn=c.bn?c.bn:66;manifest<<c.name<<','<<c.type<<','<<c.bm<<','<<c.bn<<",3069/2781,486,66,"<<(3069+bm-1)/bm<<','<<(2781+bm-1)/bm<<','<<3069%bm<<','<<2781%bm<<",0,0,LOW\n";}
  std::cout<<"POSE_OFFSET_DERIVATIVE_GEMM_BLOCKING_BENCHMARK_WRITTEN\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
