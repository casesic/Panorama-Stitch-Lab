#include "staged_lm/binary_io.h"
#include "staged_lm/experimental/pose_offset_gemm.h"
#include "staged_lm/pose_jacobian.h"

#include <Eigen/Core>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <cstdint>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::filesystem::path argument(int argc, char** argv, const std::string& key) {
  for (int index = 1; index + 1 < argc; ++index)
    if (argv[index] == key) return argv[index + 1];
  throw std::runtime_error("missing " + key);
}
int optional_integer(int argc, char** argv, const std::string& key, int fallback) {
  for (int index = 1; index + 1 < argc; ++index)
    if (argv[index] == key) return std::stoi(argv[index + 1]);
  return fallback;
}
float checksum(const std::vector<float>& values) {
  float result = 0.0F;
  for (std::size_t index = 0; index < values.size(); index += 997)
    result += values[index];
  return result;
}
std::string enabled(bool value) { return value ? "ON" : "OFF"; }
std::string pointer_alignment(const void* pointer) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return "mod16=" + std::to_string(address % 16) +
      ";mod32=" + std::to_string(address % 32) +
      ";mod64=" + std::to_string(address % 64);
}
#ifdef EIGEN_VECTORIZE
constexpr bool kEigenVectorize = true;
#else
constexpr bool kEigenVectorize = false;
#endif
#ifdef EIGEN_VECTORIZE_SSE
constexpr bool kEigenSse = true;
#else
constexpr bool kEigenSse = false;
#endif
#ifdef EIGEN_VECTORIZE_AVX
constexpr bool kEigenAvx = true;
#else
constexpr bool kEigenAvx = false;
#endif
#ifdef EIGEN_VECTORIZE_AVX2
constexpr bool kEigenAvx2 = true;
#else
constexpr bool kEigenAvx2 = false;
#endif
#ifdef EIGEN_VECTORIZE_AVX512
constexpr bool kEigenAvx512 = true;
#else
constexpr bool kEigenAvx512 = false;
#endif
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto fixture = argument(argc, argv, "--fixture");
    const auto output = argument(argc, argv, "--output");
    const int warmup = optional_integer(argc, argv, "--warmup", 3);
    const int repeat = optional_integer(argc, argv, "--repeat", 20);
    if (kEigenVectorize != static_cast<bool>(STAGED_LM_EXPECT_EIGEN_VECTORIZE))
      throw std::runtime_error("EIGEN_VECTORIZE_BUILD_POLICY_MISMATCH");
    std::filesystem::create_directories(output);
    std::ofstream samples(output / "benchmark_samples.csv");
    samples << "context,iteration,candidate,total_ms,pack_ms,core_ms,unpack_ms,checksum\n";
    const std::pair<const char*, int> contexts[] = {
        {"pose_cam2", 1023}, {"pose_cam4", 927},
        {"pose_shape_cam2", 1023}, {"pose_shape_cam4", 927},
    };
    std::map<std::string, std::string> alignments;
    for (const auto& [name, count] : contexts) {
      const auto directory = fixture / name;
      const auto posedirs = staged_lm::read_float32_file(
          directory / "selected_posedirs.f32", 486 * count * 3);
      const auto derivative = staged_lm::read_float32_file(
          directory / "d_pose_feature.f32", 486 * 66);
      std::vector<float> scalar, gemm;
      gemm.resize(static_cast<std::size_t>(count) * 3 * 66);
      alignments.emplace(std::string(name) + "_posedirs_alignment",
                         pointer_alignment(posedirs.data()));
      alignments.emplace(std::string(name) + "_derivative_alignment",
                         pointer_alignment(derivative.data()));
      alignments.emplace(std::string(name) + "_output_alignment",
                         pointer_alignment(gemm.data()));
      for (int index = 0; index < warmup; ++index) {
        staged_lm::compute_pose_offset_derivative_scalar(
            posedirs, derivative, count, scalar);
        staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(
            posedirs, derivative, count, gemm);
      }
      for (int index = 0; index < repeat; ++index) {
        auto run_scalar = [&] {
          const auto started = Clock::now();
          staged_lm::compute_pose_offset_derivative_scalar(
              posedirs, derivative, count, scalar);
          const double total = elapsed_ms(started);
          samples << name << ',' << index << ",scalar," << total
                  << ",0," << total << ",0," << checksum(scalar) << '\n';
        };
        auto run_gemm = [&] {
          staged_lm::experimental::PoseOffsetGemmTiming timing;
          staged_lm::experimental::compute_pose_offset_derivative_eigen_gemm(
              posedirs, derivative, count, gemm, &timing);
          samples << name << ',' << index << ",eigen_gemm," << timing.total_ms
                  << ',' << timing.input_pack_ms << ',' << timing.gemm_core_ms
                  << ',' << timing.output_unpack_ms << ',' << checksum(gemm) << '\n';
        };
        if (index % 2 == 0) { run_scalar(); run_gemm(); }
        else { run_gemm(); run_scalar(); }
      }
      staged_lm::write_float32_file(output / (std::string(name) + "_scalar.f32"), scalar);
      staged_lm::write_float32_file(output / (std::string(name) + "_gemm.f32"), gemm);
    }
    std::map<std::string, std::string> environment = {
        {"eigen_nb_threads", std::to_string(Eigen::nbThreads())},
        {"eigen_vectorize", enabled(kEigenVectorize)},
        {"eigen_vectorize_sse", enabled(kEigenSse)},
        {"eigen_vectorize_avx", enabled(kEigenAvx)},
        {"eigen_vectorize_avx2", enabled(kEigenAvx2)},
        {"eigen_vectorize_avx512", enabled(kEigenAvx512)},
        {"eigen_max_align_bytes", std::to_string(EIGEN_MAX_ALIGN_BYTES)},
        {"march_native", enabled(STAGED_LM_EXPECT_MARCH_NATIVE)},
        {"fast_math", "OFF"}, {"ffp_contract", "OFF"},
        {"input_pack_required", "NO"}, {"output_unpack_required", "NO"},
    };
    environment.insert(alignments.begin(), alignments.end());
    staged_lm::write_key_value_file(output / "environment.txt", environment);
    std::cout << "POSE_OFFSET_DERIVATIVE_GEMM_BENCHMARK_WRITTEN\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
