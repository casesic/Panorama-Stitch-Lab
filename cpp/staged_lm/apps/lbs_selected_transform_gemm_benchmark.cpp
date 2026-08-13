#include "staged_lm/binary_io.h"
#include "staged_lm/experimental/selected_transform_gemm.h"
#include "staged_lm/pose_jacobian.h"

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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
    if (Eigen::nbThreads() != 1)
      throw std::runtime_error("EIGEN_THREAD_POLICY_MISMATCH");

    std::filesystem::create_directories(output);
    std::ofstream samples(output / "benchmark_samples.csv");
    samples << "context,iteration,candidate,total_ms,pack_ms,core_ms,unpack_ms,"
               "dt_checksum,dskinned_checksum\n";
    struct Context { const char* name; int points; int q; };
    const Context contexts[] = {
        {"pose_cam2_q66", 1023, 66}, {"pose_cam4_q66", 927, 66},
        {"pose_shape_base_cam2_q66", 1023, 66},
        {"pose_shape_base_cam4_q66", 927, 66},
        {"pose_shape_combined_cam2_q76", 1023, 76},
        {"pose_shape_combined_cam4_q76", 927, 76},
    };
    std::map<std::string, std::string> alignments;
    for (const auto& context : contexts) {
      const auto directory = fixture / context.name;
      const auto weights = staged_lm::read_float32_file(
          directory / "weights.f32", context.points * 55);
      const auto derivative = staged_lm::read_float32_file(
          directory / "relative_transform_derivatives.f32",
          55 * 16 * context.q);
      const auto transforms = staged_lm::read_float32_file(
          directory / "selected_transforms.f32", context.points * 16);
      const auto vertices = staged_lm::read_float32_file(
          directory / "posed_vertices.f32", context.points * 3);
      const auto vertex_derivative = staged_lm::read_float32_file(
          directory / "posed_vertex_derivatives.f32",
          context.points * 3 * context.q);
      std::vector<float> scalar_dt, gemm_dt, scalar_transform, scalar_vertex,
          scalar_skinned, gemm_transform, gemm_vertex, gemm_skinned;
      gemm_dt.resize(static_cast<std::size_t>(context.points) * 16 * context.q);
      alignments.emplace(std::string(context.name) + "_weights_alignment",
                         pointer_alignment(weights.data()));
      alignments.emplace(std::string(context.name) + "_dA_alignment",
                         pointer_alignment(derivative.data()));
      alignments.emplace(std::string(context.name) + "_dT_alignment",
                         pointer_alignment(gemm_dt.data()));

      for (int index = 0; index < warmup; ++index) {
        staged_lm::compute_selected_transform_derivative_scalar(
            weights, derivative, context.points, context.q, scalar_dt);
        staged_lm::experimental::compute_selected_transform_derivative_eigen_gemm(
            weights, derivative, context.points, context.q, gemm_dt);
        staged_lm::compute_lbs_derivative_from_selected_transforms(
            scalar_dt, transforms, vertices, vertex_derivative,
            context.points, context.q, scalar_transform, scalar_vertex,
            scalar_skinned);
        staged_lm::compute_lbs_derivative_from_selected_transforms(
            gemm_dt, transforms, vertices, vertex_derivative,
            context.points, context.q, gemm_transform, gemm_vertex, gemm_skinned);
      }
      for (int index = 0; index < repeat; ++index) {
        auto run_gemm_dt = [&] {
          staged_lm::experimental::SelectedTransformGemmTiming timing;
          staged_lm::experimental::compute_selected_transform_derivative_eigen_gemm(
              weights, derivative, context.points, context.q, gemm_dt, &timing);
          samples << context.name << ',' << index << ",gemm_dt," << timing.total_ms
                  << ',' << timing.input_pack_ms << ',' << timing.gemm_core_ms
                  << ',' << timing.output_unpack_ms << ',' << checksum(gemm_dt)
                  << ",0\n";
        };
        auto run_gemm_full = [&] {
          const auto started = Clock::now();
          staged_lm::experimental::compute_selected_transform_derivative_eigen_gemm(
              weights, derivative, context.points, context.q, gemm_dt);
          staged_lm::compute_lbs_derivative_from_selected_transforms(
              gemm_dt, transforms, vertices, vertex_derivative,
              context.points, context.q, gemm_transform, gemm_vertex, gemm_skinned);
          samples << context.name << ',' << index << ",gemm_full,"
                  << elapsed_ms(started) << ",0,0,0," << checksum(gemm_dt) << ','
                  << checksum(gemm_skinned) << '\n';
        };
        if (index % 2 == 0) { run_gemm_dt(); run_gemm_full(); }
        else { run_gemm_full(); run_gemm_dt(); }
      }
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_scalar_dt.f32"), scalar_dt);
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_gemm_dt.f32"), gemm_dt);
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_current_transform.f32"),
          scalar_transform);
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_gemm_transform.f32"),
          gemm_transform);
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_current_vertex.f32"), scalar_vertex);
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_gemm_vertex.f32"), gemm_vertex);
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_current_skinned.f32"),
          scalar_skinned);
      staged_lm::write_float32_file(
          output / (std::string(context.name) + "_gemm_skinned.f32"), gemm_skinned);
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
    std::cout << "LBS_SELECTED_TRANSFORM_GEMM_BENCHMARK_WRITTEN\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
