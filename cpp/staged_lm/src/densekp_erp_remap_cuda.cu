#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
struct Workspace {
  int input_height, input_width, output_height, output_width;
  float *map_x2, *map_y2, *map_x4, *map_y4;
  std::uint8_t *source2, *source4, *output2, *output4;
};

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}

__device__ __forceinline__ std::uint8_t pixel(
    const std::uint8_t* source, int height, int width,
    int y, int x, int channel) {
  return x >= 0 && x < width && y >= 0 && y < height
      ? source[(y * width + x) * 3 + channel] : 0;
}

__global__ void remap_linear_constant_zero(
    const std::uint8_t* source, int input_height, int input_width,
    const float* map_x, const float* map_y,
    std::uint8_t* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  // Match OpenCV INTER_LINEAR: 5-bit fractional map coordinates and
  // 15-bit integer interpolation coefficients.
  const int fixed_x = __float2int_rn(map_x[index] * 32.0F);
  const int fixed_y = __float2int_rn(map_y[index] * 32.0F);
  const int x = fixed_x >> 5;
  const int y = fixed_y >> 5;
  const int fraction_x = fixed_x & 31;
  const int fraction_y = fixed_y & 31;
  const int weights[4] = {
      (32 - fraction_x) * (32 - fraction_y) * 32,
      fraction_x * (32 - fraction_y) * 32,
      (32 - fraction_x) * fraction_y * 32,
      fraction_x * fraction_y * 32};
  for (int channel = 0; channel < 3; ++channel) {
    const int value =
        pixel(source, input_height, input_width, y, x, channel) * weights[0] +
        pixel(source, input_height, input_width, y, x + 1, channel) * weights[1] +
        pixel(source, input_height, input_width, y + 1, x, channel) * weights[2] +
        pixel(source, input_height, input_width, y + 1, x + 1, channel) * weights[3];
    output[index * 3 + channel] =
        static_cast<std::uint8_t>((value + (1 << 14)) >> 15);
  }
}
}  // namespace

extern "C" void* densekp_erp_remap_create(
    int input_height, int input_width, int output_height, int output_width,
    const float* map_x2, const float* map_y2,
    const float* map_x4, const float* map_y4) {
  try {
    auto* workspace = new Workspace{
        input_height, input_width, output_height, output_width};
    const std::size_t map_bytes =
        std::size_t(output_height) * output_width * sizeof(float);
    const std::size_t source_bytes =
        std::size_t(input_height) * input_width * 3;
    const std::size_t output_bytes =
        std::size_t(output_height) * output_width * 3;
    check(cudaMalloc(&workspace->map_x2, map_bytes), "cudaMalloc map_x2");
    check(cudaMalloc(&workspace->map_y2, map_bytes), "cudaMalloc map_y2");
    check(cudaMalloc(&workspace->map_x4, map_bytes), "cudaMalloc map_x4");
    check(cudaMalloc(&workspace->map_y4, map_bytes), "cudaMalloc map_y4");
    check(cudaMalloc(&workspace->source2, source_bytes), "cudaMalloc source2");
    check(cudaMalloc(&workspace->source4, source_bytes), "cudaMalloc source4");
    check(cudaMalloc(&workspace->output2, output_bytes), "cudaMalloc output2");
    check(cudaMalloc(&workspace->output4, output_bytes), "cudaMalloc output4");
    check(cudaMemcpy(workspace->map_x2, map_x2, map_bytes, cudaMemcpyHostToDevice), "copy map_x2");
    check(cudaMemcpy(workspace->map_y2, map_y2, map_bytes, cudaMemcpyHostToDevice), "copy map_y2");
    check(cudaMemcpy(workspace->map_x4, map_x4, map_bytes, cudaMemcpyHostToDevice), "copy map_x4");
    check(cudaMemcpy(workspace->map_y4, map_y4, map_bytes, cudaMemcpyHostToDevice), "copy map_y4");
    return workspace;
  } catch (...) {
    return nullptr;
  }
}

extern "C" int densekp_erp_remap_run(
    void* opaque, const std::uint8_t* source2, const std::uint8_t* source4,
    std::uint8_t* output2, std::uint8_t* output4) {
  if (!opaque) return 1;
  try {
    auto* workspace = static_cast<Workspace*>(opaque);
    const std::size_t source_bytes = std::size_t(workspace->input_height) * workspace->input_width * 3;
    const std::size_t output_bytes = std::size_t(workspace->output_height) * workspace->output_width * 3;
    check(cudaMemcpy(workspace->source2, source2, source_bytes, cudaMemcpyHostToDevice), "copy source2");
    check(cudaMemcpy(workspace->source4, source4, source_bytes, cudaMemcpyHostToDevice), "copy source4");
    const int count = workspace->output_height * workspace->output_width;
    remap_linear_constant_zero<<<(count + 255) / 256, 256>>>(workspace->source2,workspace->input_height,workspace->input_width,workspace->map_x2,workspace->map_y2,workspace->output2,count);
    remap_linear_constant_zero<<<(count + 255) / 256, 256>>>(workspace->source4,workspace->input_height,workspace->input_width,workspace->map_x4,workspace->map_y4,workspace->output4,count);
    check(cudaGetLastError(), "remap kernel");
    check(cudaMemcpy(output2, workspace->output2, output_bytes, cudaMemcpyDeviceToHost), "copy output2");
    check(cudaMemcpy(output4, workspace->output4, output_bytes, cudaMemcpyDeviceToHost), "copy output4");
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" void densekp_erp_remap_destroy(void* opaque) {
  auto* workspace = static_cast<Workspace*>(opaque);
  if (!workspace) return;
  cudaFree(workspace->map_x2); cudaFree(workspace->map_y2);
  cudaFree(workspace->map_x4); cudaFree(workspace->map_y4);
  cudaFree(workspace->source2); cudaFree(workspace->source4);
  cudaFree(workspace->output2); cudaFree(workspace->output4);
  delete workspace;
}
