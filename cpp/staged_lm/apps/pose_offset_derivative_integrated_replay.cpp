#include "staged_lm/binary_io.h"
#include "staged_lm/experimental/pose_offset_gemm.h"

#include <Eigen/Core>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point s){return std::chrono::duration<double,std::milli>(Clock::now()-s).count();}
std::filesystem::path arg(int n,char**v,const std::string&k){for(int i=1;i+1<n;++i)if(v[i]==k)return v[i+1];throw std::runtime_error("missing "+k);}
int opt(int n,char**v,const std::string&k,int d){for(int i=1;i+1<n;++i)if(v[i]==k)return std::stoi(v[i+1]);return d;}
std::vector<std::string> split(const std::string&s){std::vector<std::string>r;std::stringstream q(s);std::string x;while(std::getline(q,x,','))r.push_back(x);return r;}
float checksum(const std::vector<float>&x){float s=0;for(std::size_t i=0;i<x.size();i+=997)s+=x[i];return s;}
}
int main(int argc,char**argv){try{
 const auto fixture=arg(argc,argv,"--fixture"),output=arg(argc,argv,"--output");const int warm=opt(argc,argv,"--warmup",5),repeat=opt(argc,argv,"--repeat",50);
 if(Eigen::nbThreads()!=1)throw std::runtime_error("EIGEN_THREAD_POLICY_MISMATCH");std::filesystem::create_directories(output);std::ifstream in(fixture/"manifest.csv");std::string line;std::getline(in,line);std::ofstream out(output/"isolated_samples.csv");out<<"call_index,stage,camera,iteration,mode,total_ms,core_ms,checksum\n";std::vector<float> disturbance(16*1024*1024,1.0F);struct State{int id,n;std::string stage,camera;std::vector<float>p,d,y;};std::vector<State> states;
 while(std::getline(in,line)){auto f=split(line);const int id=std::stoi(f[0]),n=std::stoi(f[3]);states.push_back({id,n,f[1],f[2],staged_lm::read_float32_file(fixture/f[6],static_cast<std::size_t>(n)*3*486),staged_lm::read_float32_file(fixture/f[7],486*66),{}});}
 for(auto& state:states){const int id=state.id,n=state.n;auto&p=state.p;auto&d=state.d;auto&y=state.y;
  for(int i=0;i<warm;++i)staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p,d,n,y);
  for(int i=0;i<repeat;++i){
   staged_lm::experimental::PoseOffsetGemmTiming t;staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p,d,n,y,&t);out<<id<<','<<state.stage<<','<<state.camera<<','<<i<<",warm_core,"<<t.total_ms<<','<<t.gemm_core_ms<<','<<checksum(y)<<'\n';
   std::vector<float> fresh;staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p,d,n,fresh,&t);out<<id<<','<<state.stage<<','<<state.camera<<','<<i<<",call_equivalent,"<<t.total_ms<<','<<t.gemm_core_ms<<','<<checksum(fresh)<<'\n';
   volatile float sink=0;for(std::size_t z=0;z<disturbance.size();z+=16)sink+=disturbance[z];(void)sink;staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(p,d,n,y,&t);out<<id<<','<<state.stage<<','<<state.camera<<','<<i<<",cache_disturbed,"<<t.total_ms<<','<<t.gemm_core_ms<<','<<checksum(y)<<'\n';
  }
 }
 for(int i=0;i<repeat;++i)for(auto& state:states){staged_lm::experimental::PoseOffsetGemmTiming t;staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(state.p,state.d,state.n,state.y,&t);out<<state.id<<','<<state.stage<<','<<state.camera<<','<<i<<",trajectory_order,"<<t.total_ms<<','<<t.gemm_core_ms<<','<<checksum(state.y)<<'\n';}
 staged_lm::write_key_value_file(output/"environment.txt",{{"same_gemm_implementation","YES"},{"same_build_policy","YES"},{"same_input_layout","YES"},{"same_output_layout","YES"},{"same_map_policy","YES"},{"eigen_threads",std::to_string(Eigen::nbThreads())},{"warmup",std::to_string(warm)},{"repeat",std::to_string(repeat)}});std::cout<<"POSE_OFFSET_DERIVATIVE_INTEGRATED_REPLAY_WRITTEN\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
