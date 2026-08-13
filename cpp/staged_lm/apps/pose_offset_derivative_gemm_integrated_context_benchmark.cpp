#include "staged_lm/binary_io.h"
#include "staged_lm/experimental/pose_offset_gemm.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point s) { return std::chrono::duration<double,std::milli>(Clock::now()-s).count(); }
std::filesystem::path required(int n,char**v,const std::string&k){for(int i=1;i+1<n;++i)if(v[i]==k)return v[i+1];throw std::runtime_error("missing "+k);}
int option(int n,char**v,const std::string&k,int d){for(int i=1;i+1<n;++i)if(v[i]==k)return std::stoi(v[i+1]);return d;}
std::vector<std::string> split(const std::string&s){std::vector<std::string>r;std::stringstream q(s);std::string x;while(std::getline(q,x,','))r.push_back(x);return r;}
double checksum(const std::vector<float>&x){double s=0;for(std::size_t i=0;i<x.size();i+=997)s+=x[i];return s;}
struct Call {int id,n;std::string stage,camera;std::vector<float> p,d;};
struct Run {double total=0,copy=0,core=0,prepare=0,sum=0;std::vector<double> call_core;std::vector<const float*> p_pointers;std::vector<std::vector<float>> outputs;};
Run trajectory(const std::string&candidate,const std::vector<Call>&calls,const std::vector<float>&p2,const std::vector<float>&p4,bool keep_outputs){
  Run r; if(keep_outputs)r.outputs.resize(calls.size()); const auto started=Clock::now();
  for(std::size_t i=0;i<calls.size();++i){const auto&c=calls[i];const std::vector<float>*p=&(c.camera=="cam2"?p2:p4);std::vector<float> temporary;
    if(candidate=="A_CURRENT"){const auto cs=Clock::now();temporary=c.p;r.copy+=elapsed(cs);p=&temporary;}
    std::vector<float> scratch;auto&y=keep_outputs?r.outputs[i]:scratch;staged_lm::experimental::PoseOffsetGemmTiming t;
    if(candidate=="C_STABLE_P_TRANSPOSED")staged_lm::experimental::compute_pose_offset_derivative_transposed_eigen_gemm(*p,c.d,c.n,y,&t);
    else staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(*p,c.d,c.n,y,&t);
    r.core+=t.gemm_core_ms;r.prepare+=t.output_preparation_ms;r.sum+=checksum(y);r.call_core.push_back(t.gemm_core_ms);r.p_pointers.push_back(p->data());
  }r.total=elapsed(started);return r;
}
}
int main(int argc,char**argv){try{
 const auto fixture=required(argc,argv,"--fixture"),output=required(argc,argv,"--output");const int warm=option(argc,argv,"--warmup",3),repeat=option(argc,argv,"--repeat",30);
 if(Eigen::nbThreads()!=1)throw std::runtime_error("EIGEN_THREAD_POLICY_MISMATCH");std::filesystem::create_directories(output);
 std::ifstream in(fixture/"manifest.csv");std::string line;std::getline(in,line);std::vector<Call> calls;
 while(std::getline(in,line)){auto f=split(line);int n=std::stoi(f[3]);calls.push_back({std::stoi(f[0]),n,f[1],f[2],staged_lm::read_float32_file(fixture/f[6],static_cast<std::size_t>(n)*3*486),staged_lm::read_float32_file(fixture/f[7],486*66)});}
 if(calls.size()!=16)throw std::runtime_error("EXPECTED_16_CALL_CORPUS");const auto p2=calls[0].p,p4=calls[1].p;
 for(const auto&c:calls){const auto&stable=c.camera=="cam2"?p2:p4;if(c.p!=stable)throw std::runtime_error("POSEDIRS_CONTENT_NOT_STATIC_BY_CAMERA");}
 const std::vector<std::string> candidates={"A_CURRENT","B_STABLE_P","C_STABLE_P_TRANSPOSED"};
 for(int i=0;i<warm;++i)for(const auto&c:candidates)trajectory(c,calls,p2,p4,false);
 std::ofstream samples(output/"trajectory_samples.csv");samples<<"round,sequence,candidate,total_ms,copy_ms,output_prepare_ms,gemm_core_ms,checksum\n"<<std::setprecision(12);
 std::ofstream per_call(output/"per_call_breakdown.csv");per_call<<"candidate,sample,stage,camera,call_index,p_pointer,gemm_ms,camera_switch,first_use\n"<<std::setprecision(12);
 const std::vector<std::string> order={"A_CURRENT","B_STABLE_P","C_STABLE_P_TRANSPOSED","C_STABLE_P_TRANSPOSED","B_STABLE_P","A_CURRENT"};
 for(int round=0;round<repeat/2;++round)for(std::size_t seq=0;seq<order.size();++seq){auto r=trajectory(order[seq],calls,p2,p4,false);samples<<round<<','<<seq<<','<<order[seq]<<','<<r.total<<','<<r.copy<<','<<r.prepare<<','<<r.core<<','<<r.sum<<'\n';for(std::size_t i=0;i<calls.size();++i)per_call<<order[seq]<<','<<round*2+(seq>=3)<<','<<calls[i].stage<<','<<calls[i].camera<<','<<i<<','<<reinterpret_cast<std::uintptr_t>(r.p_pointers[i])<<','<<r.call_core[i]<<','<<(i>0&&calls[i-1].camera!=calls[i].camera)<<','<<(i<2)<<'\n';}
 std::ofstream calls_out(output/"call_parity.csv");calls_out<<"call_index,stage,camera,candidate,max_abs,rel_frob,p_pointer,output_file\n"<<std::setprecision(12);
 auto a=trajectory("A_CURRENT",calls,p2,p4,true);for(const auto&candidate:{std::string("B_STABLE_P"),std::string("C_STABLE_P_TRANSPOSED")}){auto r=trajectory(candidate,calls,p2,p4,true);for(std::size_t i=0;i<calls.size();++i){double md=0,dd=0,rr=0;for(std::size_t j=0;j<a.outputs[i].size();++j){double d=double(r.outputs[i][j])-a.outputs[i][j];md=std::max(md,std::abs(d));dd+=d*d;rr+=double(a.outputs[i][j])*a.outputs[i][j];}auto file=candidate+"_call_"+std::to_string(i)+".f32";staged_lm::write_float32_file(output/file,r.outputs[i]);const auto&p=calls[i].camera=="cam2"?p2:p4;calls_out<<i<<','<<calls[i].stage<<','<<calls[i].camera<<','<<candidate<<','<<md<<','<<std::sqrt(dd/std::max(rr,1e-300))<<','<<reinterpret_cast<std::uintptr_t>(p.data())<<','<<file<<'\n';}}
 staged_lm::write_key_value_file(output/"environment.txt",{{"candidate_a","per-call selected_posedirs value copy + current GEMM"},{"candidate_b","immutable per-camera selected_posedirs + current GEMM"},{"candidate_c","immutable per-camera selected_posedirs + zero-copy transposed expression"},{"candidate_d","NOT_RUN_NOT_NEEDED_FOR_PRIMARY_ABC"},{"calls","16"},{"warmup_trajectories",std::to_string(warm)},{"measured_trajectories_per_candidate",std::to_string(repeat)},{"interleave","A_B_C_C_B_A"},{"eigen_threads",std::to_string(Eigen::nbThreads())},{"cam2_p_bytes",std::to_string(p2.size()*sizeof(float))},{"cam4_p_bytes",std::to_string(p4.size()*sizeof(float))},{"a_copy_count_per_trajectory","16"},{"b_copy_count_per_trajectory","0"},{"c_copy_count_per_trajectory","0"}});
 std::cout<<"POSE_OFFSET_DERIVATIVE_GEMM_INTEGRATED_CONTEXT_BAKEOFF_WRITTEN\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
