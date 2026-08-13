#include "staged_lm/binary_io.h"
#include "staged_lm/experimental/lbs_direct.h"
#include "staged_lm/pose_jacobian.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
std::filesystem::path arg(int argc,char**argv,const std::string&key){for(int i=1;i+1<argc;++i)if(argv[i]==key)return argv[i+1];throw std::runtime_error("missing "+key);}
int integer(int argc,char**argv,const std::string&key,int fallback){for(int i=1;i+1<argc;++i)if(argv[i]==key)return std::stoi(argv[i+1]);return fallback;}
float checksum(const std::vector<float>& values){float result=0;for(std::size_t i=0;i<values.size();i+=997)result+=values[i];return result;}
}

int main(int argc,char**argv){try{
  const auto fixture=arg(argc,argv,"--fixture"),output=arg(argc,argv,"--output");
  const int warmup=integer(argc,argv,"--warmup",3),repeat=integer(argc,argv,"--repeat",20);
  std::filesystem::create_directories(output);
  std::ofstream csv(output/"benchmark_samples.csv");
  csv<<"context,iteration,candidate,total_ms,checksum\n";
  struct Context{const char*name;int n,q;};
  const Context contexts[]={{"pose_cam2_q66",1023,66},{"pose_cam4_q66",927,66},{"pose_shape_base_cam2_q66",1023,66},{"pose_shape_base_cam4_q66",927,66},{"pose_shape_combined_cam2_q76",1023,76},{"pose_shape_combined_cam4_q76",927,76}};
  for(const auto& c:contexts){const auto d=fixture/c.name;
    const auto weights=staged_lm::read_float32_file(d/"weights.f32",c.n*55);
    const auto da=staged_lm::read_float32_file(d/"relative_transform_derivatives.f32",55*16*c.q);
    const auto transforms=staged_lm::read_float32_file(d/"selected_transforms.f32",c.n*16);
    const auto posed=staged_lm::read_float32_file(d/"posed_vertices.f32",c.n*3);
    const auto dh=staged_lm::read_float32_file(d/"posed_vertex_derivatives.f32",c.n*3*c.q);
    staged_lm::LbsDerivativeMaterializedResult current;
    std::vector<float> direct; staged_lm::experimental::LbsDirectDebugTerms debug;
    for(int i=0;i<warmup;++i){staged_lm::compute_lbs_derivative_materialized(weights,da,transforms,posed,dh,c.n,c.q,current);staged_lm::experimental::compute_lbs_derivative_direct(weights,da,transforms,posed,dh,c.n,c.q,direct);}
    for(int i=0;i<repeat;++i){
      auto run_current=[&]{auto s=Clock::now();staged_lm::compute_lbs_derivative_materialized(weights,da,transforms,posed,dh,c.n,c.q,current);double t=ms(s);csv<<c.name<<','<<i<<",current,"<<t<<','<<checksum(current.skinned_derivatives)<<'\n';};
      auto run_direct=[&]{auto s=Clock::now();staged_lm::experimental::compute_lbs_derivative_direct(weights,da,transforms,posed,dh,c.n,c.q,direct);double t=ms(s);csv<<c.name<<','<<i<<",direct,"<<t<<','<<checksum(direct)<<'\n';};
      if(i%2==0){run_current();run_direct();}else{run_direct();run_current();}
    }
    staged_lm::experimental::compute_lbs_derivative_direct(weights,da,transforms,posed,dh,c.n,c.q,direct,&debug);
    staged_lm::write_float32_file(output/(std::string(c.name)+"_current.f32"),current.skinned_derivatives);
    staged_lm::write_float32_file(output/(std::string(c.name)+"_direct.f32"),direct);
    staged_lm::write_float32_file(output/(std::string(c.name)+"_current_transform.f32"),current.transform_term);
    staged_lm::write_float32_file(output/(std::string(c.name)+"_direct_transform.f32"),debug.transform_term);
    staged_lm::write_float32_file(output/(std::string(c.name)+"_current_vertex.f32"),current.vertex_term);
    staged_lm::write_float32_file(output/(std::string(c.name)+"_direct_vertex.f32"),debug.vertex_term);
  }
  staged_lm::write_key_value_file(output/"environment.txt",{{"compile_flags","-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off"},{"eigen_threads","1"},{"benchmark_debug_capture","OFF"},{"march_native","OFF"},{"fast_math","OFF"}});
  std::cout<<"LBS_DIRECT_CONTRACTION_BENCHMARK_WRITTEN\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
