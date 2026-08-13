#include "staged_lm/binary_io.h"
#include "staged_lm/experimental/value_pose_offset_gemv.h"
#include "staged_lm/pose_forward.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point s){return std::chrono::duration<double,std::milli>(Clock::now()-s).count();}
std::filesystem::path arg(int n,char**v,const std::string&k){for(int i=1;i+1<n;++i)if(v[i]==k)return v[i+1];throw std::runtime_error("missing "+k);}
int opt(int n,char**v,const std::string&k,int d){for(int i=1;i+1<n;++i)if(v[i]==k)return std::stoi(v[i+1]);return d;}
std::vector<std::string> split(const std::string&s){std::vector<std::string>r;std::stringstream x(s);std::string v;while(std::getline(x,v,',')){if(!v.empty()&&v.back()=='\r')v.pop_back();r.push_back(v);}return r;}
float sum(const std::vector<float>&v){float x=0;for(size_t i=0;i<v.size();i+=97)x+=v[i];return x;}
std::string enabled(bool value){return value?"ON":"OFF";}
std::string alignment(const void* pointer){const auto x=reinterpret_cast<std::uintptr_t>(pointer);return "mod16="+std::to_string(x%16)+";mod32="+std::to_string(x%32)+";mod64="+std::to_string(x%64);}
#ifdef EIGEN_VECTORIZE
constexpr bool kEigenVectorize=true;
#else
constexpr bool kEigenVectorize=false;
#endif
#ifdef EIGEN_VECTORIZE_SSE
constexpr bool kEigenSse=true;
#else
constexpr bool kEigenSse=false;
#endif
#ifdef EIGEN_VECTORIZE_AVX
constexpr bool kEigenAvx=true;
#else
constexpr bool kEigenAvx=false;
#endif
#ifdef EIGEN_VECTORIZE_AVX2
constexpr bool kEigenAvx2=true;
#else
constexpr bool kEigenAvx2=false;
#endif
#ifdef EIGEN_VECTORIZE_AVX512
constexpr bool kEigenAvx512=true;
#else
constexpr bool kEigenAvx512=false;
#endif
}
int main(int argc,char**argv){try{
 auto fixture=arg(argc,argv,"--fixture"),output=arg(argc,argv,"--output");int warm=opt(argc,argv,"--warmup",5),repeat=opt(argc,argv,"--repeat",100);
 if(kEigenVectorize!=static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE))throw std::runtime_error("EIGEN_VECTORIZE_BUILD_POLICY_MISMATCH");
 if(Eigen::nbThreads()!=1)throw std::runtime_error("EIGEN_THREADS_MUST_BE_ONE");
 std::filesystem::create_directories(output);std::ifstream in(fixture/"call_manifest.csv");std::string line;std::getline(in,line);std::ofstream samples(output/"benchmark_samples.csv");samples<<"call_index,stage,caller,camera,iteration,candidate,total_ms,pack_ms,core_ms,unpack_ms,checksum\n";
 std::vector<double> empty_timer_samples(10000);for(auto& value:empty_timer_samples){auto s=Clock::now();value=ms(s);}std::sort(empty_timer_samples.begin(),empty_timer_samples.end());const double empty_timer_median_ms=empty_timer_samples[empty_timer_samples.size()/2];std::map<std::string,std::string> alignments;while(std::getline(in,line)){auto f=split(line);int id=std::stoi(f[0]),n=std::stoi(f[7]);auto posedirs=staged_lm::read_float32_file(fixture/(f[3]+"_selected_posedirs.f32"),486*n*3);auto feature=staged_lm::read_float32_file(fixture/f[10],486);std::vector<float>a,b;
  alignments["call_"+std::to_string(id)+"_posedirs_alignment"]=alignment(posedirs.data());alignments["call_"+std::to_string(id)+"_feature_alignment"]=alignment(feature.data());
  for(int i=0;i<warm;++i){staged_lm::compute_value_pose_offset_scalar(posedirs,feature,n,a);staged_lm::experimental::compute_value_pose_offset_eigen_gemv(posedirs,feature,n,b);}
  alignments["call_"+std::to_string(id)+"_output_alignment"]=alignment(b.data());
  for(int i=0;i<repeat;++i){auto scalar=[&]{auto s=Clock::now();staged_lm::compute_value_pose_offset_scalar(posedirs,feature,n,a);double t=ms(s);samples<<id<<','<<f[1]<<','<<f[2]<<','<<f[3]<<','<<i<<",scalar,"<<t<<",0,"<<t<<",0,"<<sum(a)<<'\n';};auto gemv=[&]{staged_lm::experimental::ValuePoseOffsetGemvTiming t;staged_lm::experimental::compute_value_pose_offset_eigen_gemv(posedirs,feature,n,b,&t);samples<<id<<','<<f[1]<<','<<f[2]<<','<<f[3]<<','<<i<<",gemv,"<<t.total_ms<<','<<t.input_pack_ms<<','<<t.gemv_core_ms<<','<<t.output_unpack_ms<<','<<sum(b)<<'\n';};if(i%2){gemv();scalar();}else{scalar();gemv();}}
  staged_lm::write_float32_file(output/("scalar_"+std::to_string(id)+".f32"),a);staged_lm::write_float32_file(output/("gemv_"+std::to_string(id)+".f32"),b);
 }
 std::map<std::string,std::string> environment={{"eigen_vectorize",enabled(kEigenVectorize)},{"eigen_vectorize_sse",enabled(kEigenSse)},{"eigen_vectorize_avx",enabled(kEigenAvx)},{"eigen_vectorize_avx2",enabled(kEigenAvx2)},{"eigen_vectorize_avx512",enabled(kEigenAvx512)},{"eigen_max_align_bytes",std::to_string(EIGEN_MAX_ALIGN_BYTES)},{"march_native",enabled(STAGED_LM_EXPECT_MARCH_NATIVE)},{"fast_math","OFF"},{"lto","OFF"},{"ffp_contract","OFF"},{"eigen_threads",std::to_string(Eigen::nbThreads())},{"empty_timer_median_ms",std::to_string(empty_timer_median_ms)},{"input_pack_required","NO"},{"output_unpack_required","NO"}};environment.insert(alignments.begin(),alignments.end());staged_lm::write_key_value_file(output/"environment.txt",environment);
 std::cout<<"VALUE_POSE_OFFSET_GEMV_BENCHMARK_WRITTEN\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
