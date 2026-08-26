#include "staged_lm/cuda_evaluators.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace staged_lm {

struct CudaPoseWorkspace {
  struct Block { void* pointer = nullptr; std::size_t bytes = 0; };
  std::vector<Block> blocks;
  std::size_t cursor = 0;
  float* posedirs = nullptr;
  float* shaped = nullptr;
  float* joints = nullptr;
  float* lbs_weights = nullptr;
  long long* parents = nullptr;
  float* targets = nullptr;
  int* densekp_ids = nullptr;
  int* densekp_row_offsets = nullptr;
  int* densekp_support_indices = nullptr;
  float* densekp_weights = nullptr;
  float* beta_shapedirs = nullptr;
  const float* last_residual = nullptr;
  const float* last_observation_points = nullptr;
  const float* last_active_jacobian = nullptr;
  const float* last_rotations = nullptr;
  const float* last_posed_vertices = nullptr;
  const float* last_relative_transforms = nullptr;
  const float* last_beta_jacobian = nullptr;
  int last_row_count = 0;
  double* normal_hessian = nullptr;
  double* normal_gradient = nullptr;
  double* normal_cross = nullptr;
  double* normal_beta_hessian = nullptr;
  double* normal_beta_gradient = nullptr;
  ~CudaPoseWorkspace() {
    for (auto& block : blocks) if (block.pointer) cudaFree(block.pointer);
    void* persistent[] = {posedirs, shaped, joints, lbs_weights, parents,
        targets, densekp_ids, densekp_row_offsets,
        densekp_support_indices, densekp_weights};
    for (void* pointer : persistent) if (pointer) cudaFree(pointer);
    if (normal_hessian) cudaFree(normal_hessian);
    if (normal_gradient) cudaFree(normal_gradient);
    if (normal_cross) cudaFree(normal_cross);
    if (normal_beta_hessian) cudaFree(normal_beta_hessian);
    if (normal_beta_gradient) cudaFree(normal_beta_gradient);
    if (beta_shapedirs) cudaFree(beta_shapedirs);
  }
  void reset() { cursor = 0; }
  Block* acquire(std::size_t bytes) {
    if (cursor == blocks.size()) blocks.push_back({});
    Block& block = blocks[cursor++];
    if (block.bytes < bytes) {
      if (block.pointer) check_cuda_free(block.pointer);
      cudaError_t status = cudaMalloc(&block.pointer, bytes);
      if (status != cudaSuccess) throw std::runtime_error(
          std::string("cudaMalloc: ") + cudaGetErrorString(status));
      block.bytes = bytes;
    }
    return &block;
  }
 private:
  static void check_cuda_free(void* pointer) {
    const cudaError_t status = cudaFree(pointer);
    if (status != cudaSuccess) throw std::runtime_error(
        std::string("cudaFree: ") + cudaGetErrorString(status));
  }
};

struct CudaPoseShapeValueWorkspace {
  float* latent = nullptr;
  float* base_joints = nullptr;
  float* joint_basis = nullptr;
  float* tanh_latent = nullptr;
  float* beta = nullptr;
  float* joints = nullptr;
  float* cam2_base = nullptr;
  float* cam2_basis = nullptr;
  float* cam2_dynamic = nullptr;
  float* cam4_base = nullptr;
  float* cam4_basis = nullptr;
  float* cam4_dynamic = nullptr;
  ~CudaPoseShapeValueWorkspace() {
    void* pointers[] = {latent, base_joints, joint_basis, tanh_latent, beta,
        joints, cam2_base, cam2_basis, cam2_dynamic, cam4_base, cam4_basis,
        cam4_dynamic};
    for (void* pointer : pointers) if (pointer) cudaFree(pointer);
  }
};

struct CudaMultiFramePoseBatchWorkspace {
  int frame_count = 0;
  int batch_count = 0;
  // Per batch item: parameters[75], cam2 axis[165]+translation[3],
  // cam4 axis[165]+translation[3].
  float* packed = nullptr;
  int* frame_indices = nullptr;
  float* body_pose_anchors = nullptr;
  double** cam2_hessians = nullptr;
  double** cam2_gradients = nullptr;
  double** cam4_hessians = nullptr;
  double** cam4_gradients = nullptr;
  double* output = nullptr;
  double* hessians = nullptr;  // alias into output
  double* gradients = nullptr; // alias into output
  std::vector<CudaPoseWorkspace*> cam2_workspaces;
  std::vector<CudaPoseWorkspace*> cam4_workspaces;
  CudaBatchTransferStats transfer_stats;
  ~CudaMultiFramePoseBatchWorkspace() {
    void* pointers[] = {packed, frame_indices, body_pose_anchors,
        cam2_hessians, cam2_gradients, cam4_hessians, cam4_gradients,
        output};
    for (void* pointer : pointers) if (pointer) cudaFree(pointer);
  }
};

struct CudaMultiFramePoseShapeNormalEquationWorkspace {
  int frame_count = 0;
  double* a = nullptr;
  double* b = nullptr;
  double* frame_c = nullptr;
  double* g_pose = nullptr;
  double* frame_g_beta = nullptr;
  double* shared_c = nullptr;
  double* shared_g_beta = nullptr;
  double* pose_prior_gradient = nullptr;
  double* shape_prior_hessian = nullptr;
  double* shape_prior_gradient = nullptr;
  ~CudaMultiFramePoseShapeNormalEquationWorkspace() {
    void* pointers[] = {a, b, frame_c, g_pose, frame_g_beta, shared_c,
        shared_g_beta, pose_prior_gradient, shape_prior_hessian,
        shape_prior_gradient};
    for (void* pointer : pointers) if (pointer) cudaFree(pointer);
  }
};

namespace {

constexpr int kJointCount = 55;
constexpr int kFeatureCount = 486;
constexpr int kBetaCount = 10;
constexpr int kRotationParameterCount = 66;
constexpr int kActiveParameterCount = 69;
constexpr float kPi = 3.14159265358979323846F;
using Clock = std::chrono::steady_clock;
struct ExecutionStats {
  double allocation_ms = 0.0;
  double memcpy_ms = 0.0;
  double h2d_ms = 0.0;
  double d2h_ms = 0.0;
  double kernel_ms = 0.0;
};
thread_local CudaPoseWorkspace* active_workspace = nullptr;
thread_local ExecutionStats* active_stats = nullptr;
thread_local bool compact_linearization = false;
thread_local int deferred_execution_depth = 0;
thread_local const float* batched_axis = nullptr;
thread_local const float* batched_translation = nullptr;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ != 0) {
      const auto started = Clock::now();
      if (active_workspace) {
        auto* block = active_workspace->acquire(count_ * sizeof(T));
        data_ = static_cast<T*>(block->pointer);
        pooled_ = true;
      } else {
        check(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
      }
      if (active_stats) active_stats->allocation_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    }
  }
  ~DeviceBuffer() { if (data_ && !pooled_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* data() { return data_; }
  const T* data() const { return data_; }
  void upload(const std::vector<T>& host) {
    if (host.size() != count_) throw std::runtime_error("CUDA_UPLOAD_SIZE");
    const auto started = Clock::now();
    check(cudaMemcpy(data_, host.data(), count_ * sizeof(T),
                     cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    if (active_stats) {
      const double duration = std::chrono::duration<double, std::milli>(
          Clock::now() - started).count();
      active_stats->memcpy_ms += duration;
      active_stats->h2d_ms += duration;
    }
  }
  void download(std::vector<T>& host) const {
    host.resize(count_);
    const auto started = Clock::now();
    check(cudaMemcpy(host.data(), data_, count_ * sizeof(T),
                     cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    if (active_stats) {
      const double duration = std::chrono::duration<double, std::milli>(
          Clock::now() - started).count();
      active_stats->memcpy_ms += duration;
      active_stats->d2h_ms += duration;
    }
  }
 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
  bool pooled_ = false;
};

__device__ void multiply4_device(
    const float* left, const float* right, float* output) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      float value = 0.0F;
      for (int inner = 0; inner < 4; ++inner) {
        value = fmaf(left[row * 4 + inner], right[inner * 4 + column], value);
      }
      output[row * 4 + column] = value;
    }
  }
}

__global__ void rotations_kernel(const float* axis, float* rotations) {
  const int joint = blockIdx.x * blockDim.x + threadIdx.x;
  if (joint >= kJointCount) return;
  const float x = axis[joint * 3];
  const float y = axis[joint * 3 + 1];
  const float z = axis[joint * 3 + 2];
  const float theta2 = (x * x + y * y) + z * z;
  const float theta = sqrtf(theta2);
  float a;
  float b;
  if (theta2 < 1.0e-8F) {
    a = 1.0F - theta2 / 6.0F + theta2 * theta2 / 120.0F;
    b = 0.5F - theta2 / 24.0F + theta2 * theta2 / 720.0F;
  } else {
    a = sinf(theta) / theta;
    b = (1.0F - cosf(theta)) / theta2;
  }
  const float skew[9] = {0, -z, y, z, 0, -x, -y, x, 0};
  float squared[9];
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      float value = 0.0F;
      for (int inner = 0; inner < 3; ++inner) {
        value = fmaf(skew[row * 3 + inner], skew[inner * 3 + column], value);
      }
      squared[row * 3 + column] = value;
    }
  }
  for (int index = 0; index < 9; ++index) {
    const int row = index / 3;
    const int column = index % 3;
    rotations[joint * 9 + index] =
        ((row == column ? 1.0F : 0.0F) + a * skew[index]) +
        b * squared[index];
  }
}

__global__ void feature_kernel(const float* rotations, float* feature) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kFeatureCount) return;
  const int joint = index / 9 + 1;
  const int matrix = index % 9;
  feature[index] = rotations[joint * 9 + matrix] -
      (matrix / 3 == matrix % 3 ? 1.0F : 0.0F);
}

__global__ void pose_offset_kernel(
    const float* posedirs, const float* feature, int element_count,
    float* offsets) {
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= element_count) return;
  float value = 0.0F;
  for (int feature_index = 0; feature_index < kFeatureCount; ++feature_index) {
    value = fmaf(
        posedirs[feature_index * element_count + element],
        feature[feature_index], value);
  }
  offsets[element] = value;
}

__global__ void add_vertices_kernel(
    const float* shaped, const float* offsets, int count, float* posed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) posed[index] = shaped[index] + offsets[index];
}

__global__ void fk_kernel(
    const float* rotations, const float* joints, const long long* parents,
    float* relative_transforms) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float global[kJointCount * 16];
  for (int joint = 0; joint < kJointCount; ++joint) {
    float local[16] = {};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        local[row * 4 + column] = rotations[joint * 9 + row * 3 + column];
      }
      const int parent = static_cast<int>(parents[joint]);
      local[row * 4 + 3] = joint == 0 ? joints[row] :
          joints[joint * 3 + row] - joints[parent * 3 + row];
    }
    local[15] = 1.0F;
    if (joint == 0) {
      for (int index = 0; index < 16; ++index) global[index] = local[index];
    } else {
      multiply4_device(
          global + parents[joint] * 16, local, global + joint * 16);
    }
  }
  for (int joint = 0; joint < kJointCount; ++joint) {
    const float* transform = global + joint * 16;
    float transformed[4] = {};
    for (int row = 0; row < 4; ++row) {
      for (int xyz = 0; xyz < 3; ++xyz) {
        transformed[row] = fmaf(
            transform[row * 4 + xyz], joints[joint * 3 + xyz],
            transformed[row]);
      }
    }
    for (int index = 0; index < 16; ++index) {
      relative_transforms[joint * 16 + index] = transform[index];
    }
    for (int row = 0; row < 4; ++row) {
      relative_transforms[joint * 16 + row * 4 + 3] =
          transform[row * 4 + 3] - transformed[row];
    }
  }
}

__global__ void lbs_kernel(
    const float* posed, const float* weights,
    const float* relative_transforms, int point_count, float scale,
    const float* translation, float* vertices) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= point_count) return;
  const float homogeneous[4] = {
      posed[vertex * 3], posed[vertex * 3 + 1],
      posed[vertex * 3 + 2], 1.0F};
  for (int xyz = 0; xyz < 3; ++xyz) {
    float value = 0.0F;
    for (int component = 0; component < 4; ++component) {
      float selected = 0.0F;
      for (int joint = 0; joint < kJointCount; ++joint) {
        selected = fmaf(weights[vertex * kJointCount + joint],
                        relative_transforms[joint * 16 + xyz * 4 + component],
                        selected);
      }
      value = fmaf(selected, homogeneous[component], value);
    }
    vertices[vertex * 3 + xyz] = value * scale + translation[xyz];
  }
}

__global__ void densekp_kernel(
    const float* vertices, const int* ids, const int* row_offsets,
    const int* support_indices, const float* weights, int observation_count,
    float* points) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= observation_count * 3) return;
  const int observation = index / 3;
  const int xyz = index % 3;
  const int row = ids[observation];
  float value = 0.0F;
  for (int nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
    value = fmaf(weights[nz], vertices[support_indices[nz] * 3 + xyz], value);
  }
  points[index] = value;
}

__global__ void erp_kernel(
    const float* points, const float* targets, int count,
    int width, int height, float* uv, float* residual) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= count) return;
  const float x = points[point * 3];
  const float y = points[point * 3 + 2];
  const float z = -points[point * 3 + 1];
  const float radius = fmaxf(sqrtf((x * x + y * y) + z * z), 1.0e-6F);
  const float longitude = atan2f(x, z);
  const float argument = fminf(fmaxf(-y / radius, -1.0F + 1.0e-6F),
                               1.0F - 1.0e-6F);
  const float latitude = asinf(argument);
  const float u = (longitude / (2.0F * kPi) + 0.5F) * width;
  const float v = (0.5F - latitude / kPi) * height;
  uv[point * 2] = u;
  uv[point * 2 + 1] = v;
  residual[point * 2] = u - targets[point * 2];
  residual[point * 2 + 1] = v - targets[point * 2 + 1];
}

__global__ void shape_kernel(
    const float* latent, float limit, const float* base_joints,
    const float* joint_basis, float* tanh_latent, float* beta,
    float* joints) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kBetaCount) {
    tanh_latent[index] = tanhf(latent[index]);
    beta[index] = limit * tanh_latent[index];
  }
  __syncthreads();
  if (index < kJointCount * 3) {
    float value = base_joints[index];
    for (int component = 0; component < kBetaCount; ++component) {
      value = fmaf(joint_basis[index * kBetaCount + component],
                   beta[component], value);
    }
    joints[index] = value;
  }
}

__global__ void dynamic_vertices_kernel(
    const float* base, const float* basis, const float* beta,
    int element_count, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) return;
  float value = base[index];
  for (int component = 0; component < kBetaCount; ++component) {
    value = fmaf(basis[index * kBetaCount + component], beta[component], value);
  }
  output[index] = value;
}

__device__ void right_jacobian_device(const float* phi, float* right) {
  const float x = phi[0], y = phi[1], z = phi[2];
  const float theta2 = (x * x + y * y) + z * z;
  const float theta = sqrtf(theta2);
  float a, b;
  if (theta2 < 1.0e-8F) {
    a = 0.5F - theta2 / 24.0F + theta2 * theta2 / 720.0F;
    b = 1.0F / 6.0F - theta2 / 120.0F + theta2 * theta2 / 5040.0F;
  } else {
    a = (1.0F - cosf(theta)) / theta2;
    b = (theta - sinf(theta)) / (theta2 * theta);
  }
  const float skew[9] = {0, -z, y, z, 0, -x, -y, x, 0};
  float squared[9];
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
    float v = 0.0F;
    for (int k = 0; k < 3; ++k) v = fmaf(skew[r * 3 + k], skew[k * 3 + c], v);
    squared[r * 3 + c] = v;
  }
  for (int i = 0; i < 9; ++i) {
    right[i] = ((i / 3 == i % 3) ? 1.0F : 0.0F) - a * skew[i] + b * squared[i];
  }
}

__global__ void pose_derivative_core_kernel(
    const float* axis, const float* rotations, const float* joints,
    const long long* parents, float* d_rotation, float* d_relative) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kRotationParameterCount) return;
  const int variable_joint = q / 3;
  const int component = q % 3;
  float right[9];
  right_jacobian_device(axis + variable_joint * 3, right);
  float basis[9] = {};
  const float cx = right[component], cy = right[3 + component], cz = right[6 + component];
  basis[1] = -cz; basis[2] = cy; basis[3] = cz;
  basis[5] = -cx; basis[6] = -cy; basis[7] = cx;
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
    float v = 0.0F;
    for (int k = 0; k < 3; ++k) v = fmaf(
        rotations[variable_joint * 9 + r * 3 + k], basis[k * 3 + c], v);
    d_rotation[((variable_joint * 9 + r * 3 + c) * kRotationParameterCount) + q] = v;
  }
  float global[kJointCount * 16];
  float dglobal[kJointCount * 16];
  for (int joint = 0; joint < kJointCount; ++joint) {
    float local[16] = {}, dlocal[16] = {};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        local[r * 4 + c] = rotations[joint * 9 + r * 3 + c];
        dlocal[r * 4 + c] = d_rotation[((joint * 9 + r * 3 + c) * kRotationParameterCount) + q];
      }
      const int parent = static_cast<int>(parents[joint]);
      local[r * 4 + 3] = joint == 0 ? joints[r] : joints[joint * 3 + r] - joints[parent * 3 + r];
    }
    local[15] = 1.0F;
    if (joint == 0) {
      for (int i = 0; i < 16; ++i) { global[i] = local[i]; dglobal[i] = dlocal[i]; }
    } else {
      const float* pg = global + parents[joint] * 16;
      const float* dpg = dglobal + parents[joint] * 16;
      for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        float v = 0.0F, dv = 0.0F;
        for (int k = 0; k < 4; ++k) {
          v = fmaf(pg[r * 4 + k], local[k * 4 + c], v);
          dv += fmaf(dpg[r * 4 + k], local[k * 4 + c],
                     pg[r * 4 + k] * dlocal[k * 4 + c]);
        }
        global[joint * 16 + r * 4 + c] = v;
        dglobal[joint * 16 + r * 4 + c] = dv;
      }
    }
  }
  for (int joint = 0; joint < kJointCount; ++joint) for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      float v = dglobal[joint * 16 + r * 4 + c];
      if (c == 3) for (int xyz = 0; xyz < 3; ++xyz) {
        v = fmaf(-dglobal[joint * 16 + r * 4 + xyz], joints[joint * 3 + xyz], v);
      }
      d_relative[((joint * 16 + r * 4 + c) * kRotationParameterCount) + q] = v;
    }
}

__global__ void pose_vertex_derivative_kernel(
    const float* posedirs, const float* weights, const float* relative,
    const float* d_relative, const float* rotations, const float* d_rotation,
    const float* posed, int point_count, float scale, float* d_vertices) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = point_count * 3 * kRotationParameterCount;
  if (index >= total) return;
  const int q = index % kRotationParameterCount;
  const int element = index / kRotationParameterCount;
  const int vertex = element / 3;
  const int out_xyz = element % 3;
  float dposed[3] = {};
  for (int xyz = 0; xyz < 3; ++xyz) {
    const int e = vertex * 3 + xyz;
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      const int joint = 1 + feature / 9;
      const int matrix = feature % 9;
      dposed[xyz] = fmaf(posedirs[feature * point_count * 3 + e],
          d_rotation[((joint * 9 + matrix) * kRotationParameterCount) + q], dposed[xyz]);
    }
  }
  const float homogeneous[4] = {posed[vertex * 3], posed[vertex * 3 + 1], posed[vertex * 3 + 2], 1.0F};
  float derivative = 0.0F;
  for (int c = 0; c < 4; ++c) {
    float selected = 0.0F, dselected = 0.0F;
    for (int joint = 0; joint < kJointCount; ++joint) {
      const float w = weights[vertex * kJointCount + joint];
      selected = fmaf(w, relative[joint * 16 + out_xyz * 4 + c], selected);
      dselected = fmaf(w, d_relative[((joint * 16 + out_xyz * 4 + c) * kRotationParameterCount) + q], dselected);
    }
    derivative = fmaf(dselected, homogeneous[c], derivative);
    if (c < 3) derivative = fmaf(selected, dposed[c], derivative);
  }
  d_vertices[index] = derivative * scale;
}

__global__ void densekp_derivative_kernel(
    const float* derivatives, const int* ids, const int* row_offsets,
    const int* support_indices, const float* weights, int observations,
    float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = observations * 3 * kRotationParameterCount;
  if (index >= total) return;
  const int q = index % kRotationParameterCount;
  const int e = index / kRotationParameterCount;
  const int obs = e / 3, xyz = e % 3, row = ids[obs];
  float v = 0.0F;
  for (int nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
    v = fmaf(weights[nz], derivatives[((support_indices[nz] * 3 + xyz) * kRotationParameterCount) + q], v);
  }
  output[index] = v;
}

__global__ void beta_fk_derivative_kernel(
    const float* rotations, const float* joints, const long long* parents,
    const float* d_joints, float* d_relative) {
  const int beta = blockIdx.x * blockDim.x + threadIdx.x;
  if (beta >= kBetaCount) return;
  float global[kJointCount * 16];
  float dglobal[kJointCount * 16];
  for (int joint = 0; joint < kJointCount; ++joint) {
    float local[16] = {}, dlocal[16] = {};
    const int parent = static_cast<int>(parents[joint]);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c)
        local[r * 4 + c] = rotations[joint * 9 + r * 3 + c];
      local[r * 4 + 3] = joint == 0 ? joints[r] :
          joints[joint * 3 + r] - joints[parent * 3 + r];
      dlocal[r * 4 + 3] = d_joints[(joint * 3 + r) * kBetaCount + beta] -
          (joint == 0 ? 0.0F : d_joints[(parent * 3 + r) * kBetaCount + beta]);
    }
    local[15] = 1.0F;
    if (joint == 0) {
      for (int i = 0; i < 16; ++i) { global[i] = local[i]; dglobal[i] = dlocal[i]; }
    } else {
      const float* pg = global + parent * 16;
      const float* dpg = dglobal + parent * 16;
      for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        float value = 0.0F, derivative = 0.0F;
        for (int k = 0; k < 4; ++k) {
          value = fmaf(pg[r * 4 + k], local[k * 4 + c], value);
          derivative += fmaf(dpg[r * 4 + k], local[k * 4 + c],
                             pg[r * 4 + k] * dlocal[k * 4 + c]);
        }
        global[joint * 16 + r * 4 + c] = value;
        dglobal[joint * 16 + r * 4 + c] = derivative;
      }
    }
  }
  for (int joint = 0; joint < kJointCount; ++joint) for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      float value = dglobal[joint * 16 + r * 4 + c];
      if (c == 3) {
        for (int xyz = 0; xyz < 3; ++xyz) {
          value = fmaf(-dglobal[joint * 16 + r * 4 + xyz],
                       joints[joint * 3 + xyz], value);
          value = fmaf(-global[joint * 16 + r * 4 + xyz],
                       d_joints[(joint * 3 + xyz) * kBetaCount + beta], value);
        }
      }
      d_relative[((joint * 16 + r * 4 + c) * kBetaCount) + beta] = value;
    }
}

__global__ void beta_vertex_derivative_kernel(
    const float* beta_shapedirs, const float* d_beta,
    const float* lbs_weights, const float* relative,
    const float* d_relative, const float* posed, int point_count,
    float scale, float* derivatives) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = point_count * 3 * kBetaCount;
  if (index >= total) return;
  const int beta = index % kBetaCount;
  const int element = index / kBetaCount;
  const int vertex = element / 3, out_xyz = element % 3;
  float dposed[3];
  for (int xyz = 0; xyz < 3; ++xyz) {
    dposed[xyz] = beta_shapedirs[(vertex * 3 + xyz) * kBetaCount + beta] *
        d_beta[beta];
  }
  const float homogeneous[4] = {posed[vertex * 3], posed[vertex * 3 + 1],
                                posed[vertex * 3 + 2], 1.0F};
  float derivative = 0.0F;
  for (int c = 0; c < 4; ++c) {
    float selected = 0.0F, dselected = 0.0F;
    for (int joint = 0; joint < kJointCount; ++joint) {
      const float weight = lbs_weights[vertex * kJointCount + joint];
      selected = fmaf(weight, relative[joint * 16 + out_xyz * 4 + c], selected);
      dselected = fmaf(weight,
          d_relative[((joint * 16 + out_xyz * 4 + c) * kBetaCount) + beta],
          dselected);
    }
    derivative = fmaf(dselected, homogeneous[c], derivative);
    if (c < 3) derivative = fmaf(selected, dposed[c], derivative);
  }
  derivatives[index] = derivative * scale;
}

__global__ void densekp_beta_derivative_kernel(
    const float* derivatives, const int* ids, const int* row_offsets,
    const int* support_indices, const float* weights, int observations,
    float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = observations * 3 * kBetaCount;
  if (index >= total) return;
  const int beta = index % kBetaCount;
  const int element = index / kBetaCount;
  const int observation = element / 3, xyz = element % 3;
  const int row = ids[observation];
  float value = 0.0F;
  for (int nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
    value = fmaf(weights[nz],
        derivatives[((support_indices[nz] * 3 + xyz) * kBetaCount) + beta], value);
  }
  output[index] = value;
}

__global__ void beta_erp_chain_kernel(
    const float* points, const float* dpoints, int count, int width, int height,
    float* beta_jacobian) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = count * 2 * kBetaCount;
  if (index >= total) return;
  const int beta = index % kBetaCount;
  const int row = index / kBetaCount, point = row / 2, uv = row % 2;
  const float px = points[point * 3], py = points[point * 3 + 1], pz = points[point * 3 + 2];
  const float x = px, y = pz, z = -py;
  const float h2 = x * x + z * z, h = sqrtf(fmaxf(h2, 1.0e-12F));
  const float r2 = h2 + y * y;
  float camera_j[3] = {};
  if (uv == 0) {
    const float s = width / (2.0F * kPi);
    camera_j[0] = s * z / fmaxf(h2, 1.0e-12F);
    camera_j[2] = -s * x / fmaxf(h2, 1.0e-12F);
  } else {
    const float radius = fmaxf(sqrtf(r2), 1.0e-6F), raw = -y / radius;
    if (r2 > 1.0e-12F && raw > -1.0F + 1.0e-6F && raw < 1.0F - 1.0e-6F) {
      const float s = height / kPi;
      camera_j[0] = -s * y * x / (h * fmaxf(r2, 1.0e-12F));
      camera_j[1] = s * h / fmaxf(r2, 1.0e-12F);
      camera_j[2] = -s * y * z / (h * fmaxf(r2, 1.0e-12F));
    }
  }
  const float j[3] = {camera_j[0], -camera_j[2], camera_j[1]};
  float value = 0.0F;
  for (int xyz = 0; xyz < 3; ++xyz)
    value = fmaf(j[xyz], dpoints[((point * 3 + xyz) * kBetaCount) + beta], value);
  beta_jacobian[index] = value;
}

__global__ void erp_jacobian_kernel(
    const float* points, const float* dpoints, int count, int width, int height,
    float* projection, float* uv_pose, float* active) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count * 2) return;
  const int point = row / 2, uv = row % 2;
  const float px = points[point * 3], py = points[point * 3 + 1], pz = points[point * 3 + 2];
  const float x = px, y = pz, z = -py;
  const float h2 = x * x + z * z, h = sqrtf(fmaxf(h2, 1.0e-12F));
  const float r2 = h2 + y * y;
  float camera_j[3] = {};
  if (uv == 0) {
    const float s = width / (2.0F * kPi);
    camera_j[0] = s * z / fmaxf(h2, 1.0e-12F);
    camera_j[2] = -s * x / fmaxf(h2, 1.0e-12F);
  } else {
    const float radius = fmaxf(sqrtf(r2), 1.0e-6F);
    const float raw = -y / radius;
    if (r2 > 1.0e-12F && raw > -1.0F + 1.0e-6F && raw < 1.0F - 1.0e-6F) {
      const float s = height / kPi;
      camera_j[0] = -s * y * x / (h * fmaxf(r2, 1.0e-12F));
      camera_j[1] = s * h / fmaxf(r2, 1.0e-12F);
      camera_j[2] = -s * y * z / (h * fmaxf(r2, 1.0e-12F));
    }
  }
  const float j[3] = {camera_j[0], -camera_j[2], camera_j[1]};
  for (int xyz = 0; xyz < 3; ++xyz) projection[point * 6 + uv * 3 + xyz] = j[xyz];
  for (int q = 0; q < kRotationParameterCount; ++q) {
    float v = 0.0F;
    for (int xyz = 0; xyz < 3; ++xyz) v = fmaf(j[xyz], dpoints[((point * 3 + xyz) * kRotationParameterCount) + q], v);
    uv_pose[row * kRotationParameterCount + q] = v;
    if (q < 3) active[row * kActiveParameterCount + q] = v;
    else active[row * kActiveParameterCount + 3 + q] = v;
  }
  for (int xyz = 0; xyz < 3; ++xyz) active[row * kActiveParameterCount + 3 + xyz] = j[xyz];
}

__global__ void pose_normal_equations_kernel(
    const float* residual, const float* jacobian, int row_count,
    float alpha, float eps, double* hessian, double* gradient) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int hessian_count = kActiveParameterCount * kActiveParameterCount;
  if (index < hessian_count) {
    const int a = index / kActiveParameterCount;
    const int b = index % kActiveParameterCount;
    double sum = 0.0;
    for (int row = 0; row < row_count; ++row) {
      const int point = row / 2;
      const float u = residual[point * 2];
      const float v = residual[point * 2 + 1];
      const float weight = alpha / sqrtf(u * u + v * v + eps * eps);
      const float scale = sqrtf(weight);
      const float ja = jacobian[row * kActiveParameterCount + a] * scale;
      const float jb = jacobian[row * kActiveParameterCount + b] * scale;
      sum += static_cast<double>(ja) * static_cast<double>(jb);
    }
    hessian[index] = sum;
  } else if (index < hessian_count + kActiveParameterCount) {
    const int column = index - hessian_count;
    double sum = 0.0;
    for (int row = 0; row < row_count; ++row) {
      const int point = row / 2;
      const float u = residual[point * 2];
      const float v = residual[point * 2 + 1];
      const float weight = alpha / sqrtf(u * u + v * v + eps * eps);
      const float scale = sqrtf(weight);
      const float weighted_j =
          jacobian[row * kActiveParameterCount + column] * scale;
      const float weighted_r = residual[row] * scale;
      sum += static_cast<double>(weighted_j) * static_cast<double>(weighted_r);
    }
    gradient[column] = sum;
  }
}

__global__ void pose_shape_beta_blocks_kernel(
    const float* residual, const float* active_jacobian,
    const float* beta_jacobian, int row_count, float alpha, float eps,
    double* cross, double* beta_hessian, double* beta_gradient) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int cross_count = kActiveParameterCount * kBetaCount;
  const int hessian_count = kBetaCount * kBetaCount;
  if (index < cross_count) {
    const int a = index / kBetaCount, b = index % kBetaCount;
    double sum = 0.0;
    for (int row = 0; row < row_count; ++row) {
      const int point = row / 2;
      const float u = residual[point * 2], v = residual[point * 2 + 1];
      const float scale = sqrtf(alpha / sqrtf(u * u + v * v + eps * eps));
      const float ja = active_jacobian[row * kActiveParameterCount + a] * scale;
      const float jb = beta_jacobian[row * kBetaCount + b] * scale;
      sum += static_cast<double>(ja) * static_cast<double>(jb);
    }
    cross[index] = sum;
  } else if (index < cross_count + hessian_count) {
    const int local = index - cross_count;
    const int a = local / kBetaCount, b = local % kBetaCount;
    double sum = 0.0;
    for (int row = 0; row < row_count; ++row) {
      const int point = row / 2;
      const float u = residual[point * 2], v = residual[point * 2 + 1];
      const float scale = sqrtf(alpha / sqrtf(u * u + v * v + eps * eps));
      const float ja = beta_jacobian[row * kBetaCount + a] * scale;
      const float jb = beta_jacobian[row * kBetaCount + b] * scale;
      sum += static_cast<double>(ja) * static_cast<double>(jb);
    }
    beta_hessian[local] = sum;
  } else if (index < cross_count + hessian_count + kBetaCount) {
    const int beta = index - cross_count - hessian_count;
    double sum = 0.0;
    for (int row = 0; row < row_count; ++row) {
      const int point = row / 2;
      const float u = residual[point * 2], v = residual[point * 2 + 1];
      const float scale = sqrtf(alpha / sqrtf(u * u + v * v + eps * eps));
      const float jb = beta_jacobian[row * kBetaCount + beta] * scale;
      const float wr = residual[row] * scale;
      sum += static_cast<double>(jb) * static_cast<double>(wr);
    }
    beta_gradient[beta] = sum;
  }
}

__device__ int frame_parameter_to_active(int parameter, bool cam4) {
  if (parameter >= 12) return 6 + parameter - 12;
  if (!cam4 && parameter < 6) return parameter;
  if (cam4 && parameter >= 6) return parameter - 6;
  return -1;
}

__global__ void consolidate_pose_frame_blocks_kernel(
    double* const* cam2_h, double* const* cam2_g,
    double* const* cam4_h, double* const* cam4_g,
    const int* frame_indices, const float* packed,
    const float* body_pose_anchors, int batch_count,
    double pose_prior_diagonal, double* hessians, double* gradients) {
  const int item = blockIdx.y;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int hessian_count = 75 * 75;
  constexpr int output_count = hessian_count + 75;
  constexpr int packed_stride = 75 + 2 * 168;
  if (item >= batch_count || index >= output_count) return;
  const int frame = frame_indices[item];
  if (index < hessian_count) {
    const int row = index / 75, column = index % 75;
    const int r2 = frame_parameter_to_active(row, false);
    const int c2 = frame_parameter_to_active(column, false);
    const int r4 = frame_parameter_to_active(row, true);
    const int c4 = frame_parameter_to_active(column, true);
    double value = 0.0;
    if (r2 >= 0 && c2 >= 0) value +=
        cam2_h[frame][r2 * kActiveParameterCount + c2];
    if (r4 >= 0 && c4 >= 0) value +=
        cam4_h[frame][r4 * kActiveParameterCount + c4];
    if (row == column && row >= 12) value += pose_prior_diagonal;
    hessians[item * hessian_count + index] = value;
  } else {
    const int row = index - hessian_count;
    const int r2 = frame_parameter_to_active(row, false);
    const int r4 = frame_parameter_to_active(row, true);
    double value = 0.0;
    if (r2 >= 0) value += cam2_g[frame][r2];
    if (r4 >= 0) value += cam4_g[frame][r4];
    if (row >= 12) {
      const float delta = packed[item * packed_stride + row] -
          body_pose_anchors[frame * 63 + row - 12];
      value += static_cast<double>(delta) * pose_prior_diagonal;
    }
    gradients[item * 75 + row] = value;
  }
}

__global__ void consolidate_pose_shape_frame_blocks_kernel(
    const double* cam2_h, const double* cam2_g,
    const double* cam2_b, const double* cam2_c, const double* cam2_g_beta,
    const double* cam4_h, const double* cam4_g,
    const double* cam4_b, const double* cam4_c, const double* cam4_g_beta,
    const double* pose_prior_gradient, double pose_prior_diagonal,
    int frame, double* a, double* b, double* frame_c,
    double* g_pose, double* frame_g_beta) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int a_count = 75 * 75;
  constexpr int b_count = 75 * kBetaCount;
  constexpr int c_count = kBetaCount * kBetaCount;
  constexpr int g_pose_count = 75;
  constexpr int total = a_count + b_count + c_count + g_pose_count + kBetaCount;
  if (index >= total) return;
  if (index < a_count) {
    const int row = index / 75, column = index % 75;
    const int r2 = frame_parameter_to_active(row, false);
    const int c2 = frame_parameter_to_active(column, false);
    const int r4 = frame_parameter_to_active(row, true);
    const int c4 = frame_parameter_to_active(column, true);
    double value = 0.0;
    if (r2 >= 0 && c2 >= 0) value += cam2_h[r2 * kActiveParameterCount + c2];
    if (r4 >= 0 && c4 >= 0) value += cam4_h[r4 * kActiveParameterCount + c4];
    if (row == column && row >= 12) value += pose_prior_diagonal;
    a[frame * a_count + index] = value;
  } else if (index < a_count + b_count) {
    const int local = index - a_count;
    const int row = local / kBetaCount, beta = local % kBetaCount;
    const int r2 = frame_parameter_to_active(row, false);
    const int r4 = frame_parameter_to_active(row, true);
    double value = 0.0;
    if (r2 >= 0) value += cam2_b[r2 * kBetaCount + beta];
    if (r4 >= 0) value += cam4_b[r4 * kBetaCount + beta];
    b[frame * b_count + local] = value;
  } else if (index < a_count + b_count + c_count) {
    const int local = index - a_count - b_count;
    frame_c[frame * c_count + local] = cam2_c[local] + cam4_c[local];
  } else if (index < a_count + b_count + c_count + g_pose_count) {
    const int row = index - a_count - b_count - c_count;
    const int r2 = frame_parameter_to_active(row, false);
    const int r4 = frame_parameter_to_active(row, true);
    double value = 0.0;
    if (r2 >= 0) value += cam2_g[r2];
    if (r4 >= 0) value += cam4_g[r4];
    if (row >= 12) value += pose_prior_gradient[frame * 63 + row - 12];
    g_pose[frame * g_pose_count + row] = value;
  } else {
    const int beta = index - a_count - b_count - c_count - g_pose_count;
    frame_g_beta[frame * kBetaCount + beta] =
        cam2_g_beta[beta] + cam4_g_beta[beta];
  }
}

__global__ void reduce_shared_beta_blocks_kernel(
    const double* frame_c, const double* frame_g_beta, int frame_count,
    const double* prior_hessian, const double* prior_gradient,
    double* shared_c, double* shared_g_beta) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kBetaCount * kBetaCount) {
    double value = 0.0;
    const bool diagonal = index / kBetaCount == index % kBetaCount;
    for (int frame = 0; frame < frame_count; ++frame) {
      double local = frame_c[frame * kBetaCount * kBetaCount + index];
      if (diagonal) local += prior_hessian[index / kBetaCount];
      value += local;
    }
    if (diagonal && frame_count > 1) {
      value -= static_cast<double>(frame_count - 1) *
          prior_hessian[index / kBetaCount];
    }
    shared_c[index] = value;
  } else if (index < kBetaCount * kBetaCount + kBetaCount) {
    const int beta = index - kBetaCount * kBetaCount;
    double value = 0.0;
    for (int frame = 0; frame < frame_count; ++frame) {
      value += frame_g_beta[frame * kBetaCount + beta] + prior_gradient[beta];
    }
    if (frame_count > 1) {
      value -= static_cast<double>(frame_count - 1) * prior_gradient[beta];
    }
    shared_g_beta[beta] = value;
  }
}

void finish_kernel(const char* name) {
  const auto started = Clock::now();
  check(cudaGetLastError(), name);
  if (deferred_execution_depth == 0) check(cudaDeviceSynchronize(), name);
  if (active_stats) active_stats->kernel_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

void begin_cuda_deferred_execution() { ++deferred_execution_depth; }
void end_cuda_deferred_execution() {
  if (deferred_execution_depth <= 0) {
    throw std::runtime_error("CUDA_DEFERRED_EXECUTION_SCOPE_MISMATCH");
  }
  --deferred_execution_depth;
  if (deferred_execution_depth == 0) {
    check(cudaDeviceSynchronize(), "multi-frame CUDA batch");
  }
}

CudaPoseWorkspace* create_cuda_pose_workspace(
    const PoseModelInput& model, const PoseCameraInput& input) {
  auto workspace = std::make_unique<CudaPoseWorkspace>();
  auto upload = [](auto*& pointer, const auto& values, const char* name) {
    using T = std::remove_pointer_t<std::decay_t<decltype(pointer)>>;
    if (values.empty()) return;
    check(cudaMalloc(&pointer, values.size() * sizeof(T)), name);
    check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                     cudaMemcpyHostToDevice), name);
  };
  upload(workspace->posedirs, selected_posedirs_view(input), "workspace posedirs");
  upload(workspace->shaped, input.selected_v_shaped, "workspace shaped");
  upload(workspace->joints, model.joints, "workspace joints");
  upload(workspace->lbs_weights, input.selected_lbs_weights, "workspace lbs");
  std::vector<long long> parent_values(model.parents.begin(), model.parents.end());
  upload(workspace->parents, parent_values, "workspace parents");
  const std::vector<float>* targets = &input.targets;
  if (input.densekp_observation) targets = &input.densekp_observation->targets;
  upload(workspace->targets, *targets, "workspace targets");
  if (input.densekp_observation) {
    const auto& observation = *input.densekp_observation;
    const auto& regressor = *observation.regressor;
    std::vector<int> ids(observation.densekp_ids.begin(), observation.densekp_ids.end());
    std::vector<int> offsets(regressor.row_offsets.begin(), regressor.row_offsets.end());
    std::vector<int> indices(regressor.support_indices.begin(), regressor.support_indices.end());
    upload(workspace->densekp_ids, ids, "workspace densekp ids");
    upload(workspace->densekp_row_offsets, offsets, "workspace densekp offsets");
    upload(workspace->densekp_support_indices, indices, "workspace densekp indices");
    upload(workspace->densekp_weights, regressor.sparse_weights, "workspace densekp weights");
  }
  check(cudaMalloc(&workspace->normal_hessian,
                   kActiveParameterCount * kActiveParameterCount * sizeof(double)),
        "workspace normal hessian");
  check(cudaMalloc(&workspace->normal_gradient,
                   kActiveParameterCount * sizeof(double)),
        "workspace normal gradient");
  check(cudaMalloc(&workspace->normal_cross,
                   kActiveParameterCount * kBetaCount * sizeof(double)),
        "workspace normal cross");
  check(cudaMalloc(&workspace->normal_beta_hessian,
                   kBetaCount * kBetaCount * sizeof(double)),
        "workspace normal beta hessian");
  check(cudaMalloc(&workspace->normal_beta_gradient,
                   kBetaCount * sizeof(double)),
        "workspace normal beta gradient");
  return workspace.release();
}

CudaPoseShapeValueWorkspace* create_cuda_pose_shape_value_workspace(
    const PoseShapeLMProblem& problem) {
  auto workspace = std::make_unique<CudaPoseShapeValueWorkspace>();
  auto allocate = [](float*& pointer, std::size_t count, const char* name) {
    check(cudaMalloc(&pointer, count * sizeof(float)), name);
  };
  auto upload = [&](float*& pointer, const std::vector<float>& values,
                    const char* name) {
    allocate(pointer, values.size(), name);
    check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(float),
                     cudaMemcpyHostToDevice), name);
  };
  allocate(workspace->latent, kBetaCount, "shape workspace latent");
  upload(workspace->base_joints, problem.base_joints,
         "shape workspace base joints");
  upload(workspace->joint_basis, problem.joint_basis,
         "shape workspace joint basis");
  allocate(workspace->tanh_latent, kBetaCount, "shape workspace tanh");
  allocate(workspace->beta, kBetaCount, "shape workspace beta");
  allocate(workspace->joints, kJointCount * 3, "shape workspace joints");
  upload(workspace->cam2_base, problem.cam2.selected_base_v_shaped,
         "shape workspace cam2 base");
  upload(workspace->cam2_basis, problem.cam2.selected_beta_shapedirs,
         "shape workspace cam2 basis");
  allocate(workspace->cam2_dynamic,
      problem.cam2.pose.point_count * 3, "shape workspace cam2 dynamic");
  upload(workspace->cam4_base, problem.cam4.selected_base_v_shaped,
         "shape workspace cam4 base");
  upload(workspace->cam4_basis, problem.cam4.selected_beta_shapedirs,
         "shape workspace cam4 basis");
  allocate(workspace->cam4_dynamic,
      problem.cam4.pose.point_count * 3, "shape workspace cam4 dynamic");
  return workspace.release();
}

void destroy_cuda_pose_shape_value_workspace(
    CudaPoseShapeValueWorkspace* workspace) {
  delete workspace;
}

CudaMultiFramePoseShapeNormalEquationWorkspace*
create_cuda_multiframe_pose_shape_ne_workspace(int frame_count) {
  if (frame_count <= 0) {
    throw std::runtime_error("CUDA_MULTIFRAME_NE_WORKSPACE_INPUT");
  }
  auto workspace =
      std::make_unique<CudaMultiFramePoseShapeNormalEquationWorkspace>();
  workspace->frame_count = frame_count;
  auto allocate = [](double*& pointer, std::size_t count, const char* name) {
    check(cudaMalloc(&pointer, count * sizeof(double)), name);
  };
  allocate(workspace->a, static_cast<std::size_t>(frame_count) * 75 * 75,
           "multiframe workspace A");
  allocate(workspace->b, static_cast<std::size_t>(frame_count) * 75 * kBetaCount,
           "multiframe workspace B");
  allocate(workspace->frame_c,
      static_cast<std::size_t>(frame_count) * kBetaCount * kBetaCount,
      "multiframe workspace frame C");
  allocate(workspace->g_pose, static_cast<std::size_t>(frame_count) * 75,
           "multiframe workspace g pose");
  allocate(workspace->frame_g_beta,
      static_cast<std::size_t>(frame_count) * kBetaCount,
      "multiframe workspace frame g beta");
  allocate(workspace->shared_c, kBetaCount * kBetaCount,
           "multiframe workspace shared C");
  allocate(workspace->shared_g_beta, kBetaCount,
           "multiframe workspace shared g beta");
  allocate(workspace->pose_prior_gradient,
      static_cast<std::size_t>(frame_count) * 63,
      "multiframe workspace pose prior gradient");
  allocate(workspace->shape_prior_hessian, kBetaCount,
           "multiframe workspace shape prior H");
  allocate(workspace->shape_prior_gradient, kBetaCount,
           "multiframe workspace shape prior g");
  return workspace.release();
}

void destroy_cuda_multiframe_pose_shape_ne_workspace(
    CudaMultiFramePoseShapeNormalEquationWorkspace* workspace) {
  delete workspace;
}

void destroy_cuda_pose_workspace(CudaPoseWorkspace* workspace) {
  delete workspace;
}

CudaMultiFramePoseBatchWorkspace* create_cuda_multiframe_pose_batch_workspace(
    const std::vector<CudaPoseWorkspace*>& cam2_workspaces,
    const std::vector<CudaPoseWorkspace*>& cam4_workspaces,
    const std::vector<float>& body_pose_anchors) {
  const int frame_count = static_cast<int>(cam2_workspaces.size());
  if (frame_count <= 0 || cam4_workspaces.size() != cam2_workspaces.size() ||
      body_pose_anchors.size() != static_cast<std::size_t>(frame_count * 63)) {
    throw std::runtime_error("CUDA_MULTIFRAME_POSE_WORKSPACE_INPUT");
  }
  auto workspace = std::make_unique<CudaMultiFramePoseBatchWorkspace>();
  workspace->frame_count = frame_count;
  workspace->cam2_workspaces = cam2_workspaces;
  workspace->cam4_workspaces = cam4_workspaces;
  constexpr int packed_stride = 75 + 2 * 168;
  check(cudaMalloc(&workspace->packed,
      static_cast<std::size_t>(frame_count) * packed_stride * sizeof(float)),
      "multiframe pose packed parameters");
  check(cudaMalloc(&workspace->frame_indices,
      static_cast<std::size_t>(frame_count) * sizeof(int)),
      "multiframe pose frame indices");
  check(cudaMalloc(&workspace->body_pose_anchors,
      body_pose_anchors.size() * sizeof(float)),
      "multiframe pose anchors");
  check(cudaMemcpy(workspace->body_pose_anchors, body_pose_anchors.data(),
      body_pose_anchors.size() * sizeof(float), cudaMemcpyHostToDevice),
      "multiframe pose anchors H2D");
  auto allocate_pointer_array = [&](double*** output,
                                    const std::vector<double*>& pointers,
                                    const char* name) {
    check(cudaMalloc(output, pointers.size() * sizeof(double*)), name);
    check(cudaMemcpy(*output, pointers.data(), pointers.size() * sizeof(double*),
                     cudaMemcpyHostToDevice), name);
  };
  std::vector<double*> cam2_h(frame_count), cam2_g(frame_count);
  std::vector<double*> cam4_h(frame_count), cam4_g(frame_count);
  for (int frame = 0; frame < frame_count; ++frame) {
    cam2_h[frame] = cam2_workspaces[frame]->normal_hessian;
    cam2_g[frame] = cam2_workspaces[frame]->normal_gradient;
    cam4_h[frame] = cam4_workspaces[frame]->normal_hessian;
    cam4_g[frame] = cam4_workspaces[frame]->normal_gradient;
  }
  allocate_pointer_array(&workspace->cam2_hessians, cam2_h,
                         "multiframe pose cam2 H pointers");
  allocate_pointer_array(&workspace->cam2_gradients, cam2_g,
                         "multiframe pose cam2 g pointers");
  allocate_pointer_array(&workspace->cam4_hessians, cam4_h,
                         "multiframe pose cam4 H pointers");
  allocate_pointer_array(&workspace->cam4_gradients, cam4_g,
                         "multiframe pose cam4 g pointers");
  const std::size_t hessian_values =
      static_cast<std::size_t>(frame_count) * 75 * 75;
  const std::size_t gradient_values = static_cast<std::size_t>(frame_count) * 75;
  check(cudaMalloc(&workspace->output,
      (hessian_values + gradient_values) * sizeof(double)),
      "multiframe pose H/g output");
  workspace->hessians = workspace->output;
  workspace->gradients = workspace->output + hessian_values;
  return workspace.release();
}

void destroy_cuda_multiframe_pose_batch_workspace(
    CudaMultiFramePoseBatchWorkspace* workspace) {
  delete workspace;
}

void upload_cuda_multiframe_pose_batch(
    CudaMultiFramePoseBatchWorkspace* workspace,
    const std::vector<int>& frame_indices,
    const std::vector<std::vector<float>>& parameters,
    const std::vector<float>& packed_camera_parameters) {
  constexpr int packed_stride = 75 + 2 * 168;
  if (!workspace || frame_indices.empty() ||
      frame_indices.size() != parameters.size() ||
      frame_indices.size() > static_cast<std::size_t>(workspace->frame_count) ||
      packed_camera_parameters.size() != frame_indices.size() * packed_stride) {
    throw std::runtime_error("CUDA_MULTIFRAME_POSE_UPLOAD_INPUT");
  }
  for (std::size_t item = 0; item < frame_indices.size(); ++item) {
    if (frame_indices[item] < 0 || frame_indices[item] >= workspace->frame_count ||
        parameters[item].size() != 75) {
      throw std::runtime_error("CUDA_MULTIFRAME_POSE_UPLOAD_INPUT");
    }
  }
  workspace->batch_count = static_cast<int>(frame_indices.size());
  check(cudaMemcpy(workspace->packed, packed_camera_parameters.data(),
      packed_camera_parameters.size() * sizeof(float), cudaMemcpyHostToDevice),
      "multiframe pose packed H2D");
  ++workspace->transfer_stats.h2d_count;
  workspace->transfer_stats.h2d_bytes +=
      packed_camera_parameters.size() * sizeof(float);
  check(cudaMemcpy(workspace->frame_indices, frame_indices.data(),
      frame_indices.size() * sizeof(int), cudaMemcpyHostToDevice),
      "multiframe pose indices H2D");
  ++workspace->transfer_stats.h2d_count;
  workspace->transfer_stats.h2d_bytes += frame_indices.size() * sizeof(int);
}

void activate_cuda_multiframe_pose_batch_camera(
    CudaMultiFramePoseBatchWorkspace* workspace, int batch_item, int camera) {
  constexpr int packed_stride = 75 + 2 * 168;
  if (!workspace || batch_item < 0 || batch_item >= workspace->batch_count ||
      (camera != 0 && camera != 1)) {
    throw std::runtime_error("CUDA_MULTIFRAME_POSE_ACTIVATE_INPUT");
  }
  const float* camera_data = workspace->packed +
      static_cast<std::size_t>(batch_item) * packed_stride + 75 + camera * 168;
  batched_axis = camera_data;
  batched_translation = camera_data + 165;
}

void deactivate_cuda_multiframe_pose_batch_camera() {
  batched_axis = nullptr;
  batched_translation = nullptr;
}

std::vector<DenseNormalEquations> build_cuda_multiframe_pose_normal_equations(
    CudaMultiFramePoseBatchWorkspace* workspace,
    const std::vector<int>& frame_indices,
    const PoseNormalEquationConfig& config,
    const std::vector<double>& jacobian_ms) {
  if (!workspace || frame_indices.size() != jacobian_ms.size() ||
      static_cast<int>(frame_indices.size()) != workspace->batch_count) {
    throw std::runtime_error("CUDA_MULTIFRAME_POSE_NE_INPUT");
  }
  const auto started = Clock::now();
  constexpr int active_count =
      kActiveParameterCount * kActiveParameterCount + kActiveParameterCount;
  for (int frame : frame_indices) {
    CudaPoseWorkspace* cameras[] = {workspace->cam2_workspaces.at(frame),
                                    workspace->cam4_workspaces.at(frame)};
    const float alphas[] = {config.cam2_alpha_per_point,
                            config.cam4_alpha_per_point};
    for (int camera = 0; camera < 2; ++camera) {
      auto* source = cameras[camera];
      pose_normal_equations_kernel<<<(active_count + 255) / 256, 256>>>(
          source->last_residual, source->last_active_jacobian,
          source->last_row_count, alphas[camera], config.charbonnier_eps,
          source->normal_hessian, source->normal_gradient);
      check(cudaGetLastError(), "multiframe pose normal equations kernel");
    }
  }
  constexpr int output_count = 75 * 75 + 75;
  const dim3 grid((output_count + 255) / 256,
                  static_cast<unsigned>(frame_indices.size()));
  const double prior = static_cast<double>(config.pose_anchor_scale) *
      static_cast<double>(config.pose_anchor_scale);
  consolidate_pose_frame_blocks_kernel<<<grid, 256>>>(
      workspace->cam2_hessians, workspace->cam2_gradients,
      workspace->cam4_hessians, workspace->cam4_gradients,
      workspace->frame_indices, workspace->packed,
      workspace->body_pose_anchors, workspace->batch_count, prior,
      workspace->hessians, workspace->gradients);
  check(cudaGetLastError(), "multiframe pose consolidate kernel");
  check(cudaDeviceSynchronize(), "multiframe pose normal equations");
  const std::size_t hessian_values = frame_indices.size() * 75 * 75;
  const std::size_t gradient_values = frame_indices.size() * 75;
  std::vector<double> host_output(hessian_values + gradient_values);
  check(cudaMemcpy(host_output.data(), workspace->output,
      host_output.size() * sizeof(double), cudaMemcpyDeviceToHost),
      "multiframe pose H/g D2H");
  ++workspace->transfer_stats.d2h_count;
  workspace->transfer_stats.d2h_bytes += host_output.size() * sizeof(double);
  const auto gradients = host_output.begin() + hessian_values;
  const double duration = elapsed_ms(started);
  std::vector<DenseNormalEquations> result(frame_indices.size());
  for (std::size_t item = 0; item < result.size(); ++item) {
    result[item].hessian.assign(
        host_output.begin() + item * 75 * 75,
        host_output.begin() + (item + 1) * 75 * 75);
    result[item].gradient.assign(
        gradients + item * 75,
        gradients + (item + 1) * 75);
    result[item].jacobian_ms = jacobian_ms[item];
    result[item].normal_equations_ms = duration / result.size();
  }
  return result;
}

CudaBatchTransferStats cuda_multiframe_pose_batch_transfer_stats(
    const CudaMultiFramePoseBatchWorkspace* workspace) {
  return workspace ? workspace->transfer_stats : CudaBatchTransferStats{};
}

void reserve_cuda_pose_workspace(
    CudaPoseWorkspace* workspace, const PoseCameraInput& input,
    bool pose_shape_linearization) {
  if (!workspace) throw std::runtime_error("CUDA_WORKSPACE_RESERVE_INPUT");
  const std::size_t n3 = static_cast<std::size_t>(input.point_count) * 3;
  const std::size_t observations = pose_observation_count(input);
  auto reserve_float = [&](std::size_t count) {
    workspace->acquire(count * sizeof(float));
  };
  auto reserve_value = [&]() {
    reserve_float(165);
    reserve_float(495);
    reserve_float(486);
    reserve_float(n3);
    reserve_float(n3);
    reserve_float(kJointCount * 16);
    reserve_float(3);
    reserve_float(n3);
    reserve_float(observations * 3);
    reserve_float(observations * 2);
    reserve_float(observations * 2);
  };
  workspace->reset();
  reserve_value();
  if (pose_shape_linearization) {
    reserve_float(165);
    reserve_float(495);
    reserve_float(486);
    reserve_float(n3);
    reserve_float(n3);
    reserve_float(kJointCount * 16);
    reserve_float(kJointCount * 9 * kRotationParameterCount);
    reserve_float(kJointCount * 16 * kRotationParameterCount);
    reserve_float(n3 * kRotationParameterCount);
    reserve_float(observations * 3 * kRotationParameterCount);
    reserve_float(observations * 6);
    reserve_float(observations * 2 * kRotationParameterCount);
    reserve_float(observations * 2 * kActiveParameterCount);
    reserve_float(kJointCount * 3 * kBetaCount);
    reserve_float(kBetaCount);
    reserve_float(kJointCount * 16 * kBetaCount);
    reserve_float(n3 * kBetaCount);
    reserve_float(observations * 3 * kBetaCount);
    reserve_float(observations * 2 * kBetaCount);
  }
  workspace->reset();
}

void prepare_cuda_pose_shape_workspace(
    CudaPoseWorkspace* workspace, const PoseShapeCameraInput& input) {
  if (!workspace || input.selected_beta_shapedirs.empty()) {
    throw std::runtime_error("CUDA_POSE_SHAPE_WORKSPACE_INPUT");
  }
  if (!workspace->beta_shapedirs) {
    check(cudaMalloc(&workspace->beta_shapedirs,
        input.selected_beta_shapedirs.size() * sizeof(float)),
        "workspace beta shapedirs");
    check(cudaMemcpy(workspace->beta_shapedirs,
        input.selected_beta_shapedirs.data(),
        input.selected_beta_shapedirs.size() * sizeof(float),
        cudaMemcpyHostToDevice), "workspace beta shapedirs H2D");
  }
}

namespace {
class WorkspaceScope {
 public:
  WorkspaceScope(CudaPoseWorkspace* workspace, ExecutionStats* stats,
                 bool compact = false, bool reset = true)
      : previous_workspace_(active_workspace), previous_stats_(active_stats),
        previous_compact_(compact_linearization) {
    active_workspace = workspace;
    active_stats = stats;
    compact_linearization = compact;
    if (workspace && reset) workspace->reset();
  }
  ~WorkspaceScope() {
    active_workspace = previous_workspace_;
    active_stats = previous_stats_;
    compact_linearization = previous_compact_;
  }
 private:
  CudaPoseWorkspace* previous_workspace_;
  ExecutionStats* previous_stats_;
  bool previous_compact_;
};
}  // namespace

bool cuda_evaluator_available(std::string* reason) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    if (reason) {
      *reason = status == cudaSuccess ? "no CUDA device" : cudaGetErrorString(status);
    }
    return false;
  }
  if (reason) reason->clear();
  return true;
}

PoseCameraResult evaluate_pose_camera_value_cuda(
    const PoseModelInput& model, const PoseCameraInput& input) {
  std::string reason;
  if (!cuda_evaluator_available(&reason)) {
    throw std::runtime_error("CUDA_EVALUATOR_UNAVAILABLE: " + reason);
  }
  if (input.parameters.size() != 75 || model.joints.size() != 165 ||
      model.parents.size() != 55 || input.point_count <= 0) {
    throw std::runtime_error("invalid CUDA Pose value input");
  }
  const auto started = Clock::now();
  const int n = input.point_count;
  const int camera_offset = input.camera == "cam2" ? 0 :
      input.camera == "cam4" ? 6 : -1;
  if (camera_offset < 0) throw std::runtime_error("unsupported CUDA camera");
  PoseCameraResult result;
  result.axis_angles.assign(165, 0.0F);
  for (int xyz = 0; xyz < 3; ++xyz) {
    result.axis_angles[xyz] = input.parameters[camera_offset + xyz];
  }
  for (int component = 0; component < 63; ++component) {
    result.axis_angles[3 + component] = input.parameters[12 + component];
  }
  for (int component = 0; component < 45; ++component) {
    result.axis_angles[75 + component] = input.left_hand_pose[component];
    result.axis_angles[120 + component] = input.right_hand_pose[component];
  }

  DeviceBuffer<float> axis(165), rotations(495), feature(486);
  DeviceBuffer<float> posedirs(active_workspace ? 0 : selected_posedirs_view(input).size());
  DeviceBuffer<float> shaped(active_workspace ? 0 : n * 3), offsets(n * 3), posed(n * 3);
  DeviceBuffer<float> joints(active_workspace ? 0 : 165);
  DeviceBuffer<float> lbs_weights(active_workspace ? 0 : input.selected_lbs_weights.size());
  DeviceBuffer<long long> parents(active_workspace ? 0 : 55);
  DeviceBuffer<float> transforms(55 * 16), translation(3), vertices(n * 3);
  if (!batched_axis) axis.upload(result.axis_angles);
  if (!active_workspace) {
    posedirs.upload(selected_posedirs_view(input));
    shaped.upload(input.selected_v_shaped);
    joints.upload(model.joints);
    lbs_weights.upload(input.selected_lbs_weights);
  }
  std::vector<long long> parent_values(model.parents.begin(), model.parents.end());
  if (!active_workspace) parents.upload(parent_values);
  std::vector<float> translation_values(
      input.parameters.begin() + camera_offset + 3,
      input.parameters.begin() + camera_offset + 6);
  if (!batched_translation) translation.upload(translation_values);

  const float* axis_data = batched_axis ? batched_axis : axis.data();
  const float* translation_data =
      batched_translation ? batched_translation : translation.data();

  rotations_kernel<<<1, 64>>>(axis_data, rotations.data());
  finish_kernel("rotations_kernel");
  feature_kernel<<<2, 256>>>(rotations.data(), feature.data());
  finish_kernel("feature_kernel");
  pose_offset_kernel<<<(n * 3 + 255) / 256, 256>>>(
      active_workspace ? active_workspace->posedirs : posedirs.data(),
      feature.data(), n * 3, offsets.data());
  finish_kernel("pose_offset_kernel");
  add_vertices_kernel<<<(n * 3 + 255) / 256, 256>>>(
      active_workspace ? active_workspace->shaped : shaped.data(),
      offsets.data(), n * 3, posed.data());
  finish_kernel("add_vertices_kernel");
  fk_kernel<<<1, 1>>>(rotations.data(),
                      active_workspace ? active_workspace->joints : joints.data(),
                      active_workspace ? active_workspace->parents : parents.data(),
                      transforms.data());
  finish_kernel("fk_kernel");
  lbs_kernel<<<(n + 255) / 256, 256>>>(
      posed.data(), active_workspace ? active_workspace->lbs_weights : lbs_weights.data(),
      transforms.data(), n, input.scale,
      translation_data, vertices.data());
  finish_kernel("lbs_kernel");
  if (!compact_linearization) vertices.download(result.vertices);

  const std::vector<float>* targets = &input.targets;
  const int observation_count = pose_observation_count(input);
  DeviceBuffer<float> observation_points(observation_count * 3);
  if (input.densekp_observation) {
    const auto& observation = *input.densekp_observation;
    const auto& regressor = *observation.regressor;
    DeviceBuffer<int> ids(active_workspace ? 0 : observation.densekp_ids.size());
    DeviceBuffer<int> offsets_device(active_workspace ? 0 : regressor.row_offsets.size());
    DeviceBuffer<int> indices(active_workspace ? 0 : regressor.support_indices.size());
    DeviceBuffer<float> sparse_weights(active_workspace ? 0 : regressor.sparse_weights.size());
    std::vector<int> id_values(observation.densekp_ids.begin(),
                               observation.densekp_ids.end());
    std::vector<int> offset_values(regressor.row_offsets.begin(),
                                   regressor.row_offsets.end());
    std::vector<int> index_values(regressor.support_indices.begin(),
                                  regressor.support_indices.end());
    if (!active_workspace) {
      ids.upload(id_values);
      offsets_device.upload(offset_values);
      indices.upload(index_values);
      sparse_weights.upload(regressor.sparse_weights);
    }
    densekp_kernel<<<(observation_count * 3 + 255) / 256, 256>>>(
        vertices.data(), active_workspace ? active_workspace->densekp_ids : ids.data(),
        active_workspace ? active_workspace->densekp_row_offsets : offsets_device.data(),
        active_workspace ? active_workspace->densekp_support_indices : indices.data(),
        active_workspace ? active_workspace->densekp_weights : sparse_weights.data(),
        observation_count, observation_points.data());
    finish_kernel("densekp_kernel");
    targets = &observation.targets;
  } else {
    check(cudaMemcpy(observation_points.data(), vertices.data(), n * 3 * sizeof(float),
                     cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
  }
  if (active_workspace) {
    active_workspace->last_observation_points = observation_points.data();
  }
  if (!compact_linearization) observation_points.download(result.observation_points);
  DeviceBuffer<float> targets_device(active_workspace ? 0 : targets->size());
  DeviceBuffer<float> uv(observation_count * 2), residual(observation_count * 2);
  if (!active_workspace) targets_device.upload(*targets);
  erp_kernel<<<(observation_count + 255) / 256, 256>>>(
      observation_points.data(), active_workspace ? active_workspace->targets : targets_device.data(), observation_count,
      input.image_width, input.image_height, uv.data(), residual.data());
  finish_kernel("erp_kernel");
  if (active_workspace) {
    active_workspace->last_residual = residual.data();
    active_workspace->last_row_count = observation_count * 2;
  }
  if (!compact_linearization) {
    uv.download(result.uv);
    residual.download(result.raw_residual);
  }
  result.total_value_forward_ms = elapsed_ms(started);
  return result;
}

PoseCameraJacobianResult evaluate_pose_camera_with_jacobian_cuda(
    const PoseModelInput& model, const PoseCameraInput& input) {
  const auto started = Clock::now();
  PoseCameraJacobianResult result;
  result.value = evaluate_pose_camera_value_cuda(model, input);
  const int n = input.point_count;
  const int observation_count = pose_observation_count(input);

  DeviceBuffer<float> axis(165), rotations(495), feature(486);
  DeviceBuffer<float> posedirs(active_workspace ? 0 : selected_posedirs_view(input).size());
  DeviceBuffer<float> shaped(active_workspace ? 0 : n * 3), offsets(n * 3), posed(n * 3);
  DeviceBuffer<float> joints(active_workspace ? 0 : 165);
  DeviceBuffer<float> lbs_weights(active_workspace ? 0 : input.selected_lbs_weights.size());
  DeviceBuffer<long long> parents(active_workspace ? 0 : 55);
  DeviceBuffer<float> relative(55 * 16);
  if (!batched_axis) axis.upload(result.value.axis_angles);
  if (!active_workspace) {
    posedirs.upload(selected_posedirs_view(input));
    shaped.upload(input.selected_v_shaped);
    joints.upload(model.joints);
    lbs_weights.upload(input.selected_lbs_weights);
  }
  std::vector<long long> parent_values(model.parents.begin(), model.parents.end());
  if (!active_workspace) parents.upload(parent_values);
  const float* axis_data = batched_axis ? batched_axis : axis.data();
  rotations_kernel<<<1, 64>>>(axis_data, rotations.data());
  finish_kernel("jacobian_rotations_kernel");
  feature_kernel<<<2, 256>>>(rotations.data(), feature.data());
  finish_kernel("jacobian_feature_kernel");
  pose_offset_kernel<<<(n * 3 + 255) / 256, 256>>>(
      active_workspace ? active_workspace->posedirs : posedirs.data(),
      feature.data(), n * 3, offsets.data());
  finish_kernel("jacobian_pose_offset_kernel");
  add_vertices_kernel<<<(n * 3 + 255) / 256, 256>>>(
      active_workspace ? active_workspace->shaped : shaped.data(),
      offsets.data(), n * 3, posed.data());
  finish_kernel("jacobian_add_vertices_kernel");
  fk_kernel<<<1, 1>>>(rotations.data(),
      active_workspace ? active_workspace->joints : joints.data(),
      active_workspace ? active_workspace->parents : parents.data(), relative.data());
  finish_kernel("jacobian_fk_kernel");
  if (active_workspace) {
    active_workspace->last_rotations = rotations.data();
    active_workspace->last_posed_vertices = posed.data();
    active_workspace->last_relative_transforms = relative.data();
  }

  DeviceBuffer<float> d_rotation(kJointCount * 9 * kRotationParameterCount);
  DeviceBuffer<float> d_relative(kJointCount * 16 * kRotationParameterCount);
  check(cudaMemset(d_rotation.data(), 0,
      kJointCount * 9 * kRotationParameterCount * sizeof(float)), "cudaMemset d_rotation");
  pose_derivative_core_kernel<<<1, 128>>>(
      axis_data, rotations.data(),
      active_workspace ? active_workspace->joints : joints.data(),
      active_workspace ? active_workspace->parents : parents.data(),
      d_rotation.data(), d_relative.data());
  finish_kernel("pose_derivative_core_kernel");

  DeviceBuffer<float> d_vertices(
      static_cast<std::size_t>(n) * 3 * kRotationParameterCount);
  const int derivative_count = n * 3 * kRotationParameterCount;
  pose_vertex_derivative_kernel<<<(derivative_count + 255) / 256, 256>>>(
      active_workspace ? active_workspace->posedirs : posedirs.data(),
      active_workspace ? active_workspace->lbs_weights : lbs_weights.data(),
      relative.data(), d_relative.data(),
      rotations.data(), d_rotation.data(), posed.data(), n, input.scale,
      d_vertices.data());
  finish_kernel("pose_vertex_derivative_kernel");

  DeviceBuffer<float> observation_points(active_workspace ? 0 : observation_count * 3);
  if (!active_workspace) observation_points.upload(result.value.observation_points);
  DeviceBuffer<float> observation_derivatives(
      static_cast<std::size_t>(observation_count) * 3 * kRotationParameterCount);
  if (input.densekp_observation) {
    const auto& observation = *input.densekp_observation;
    const auto& regressor = *observation.regressor;
    DeviceBuffer<int> ids(active_workspace ? 0 : observation.densekp_ids.size());
    DeviceBuffer<int> row_offsets(active_workspace ? 0 : regressor.row_offsets.size());
    DeviceBuffer<int> support_indices(active_workspace ? 0 : regressor.support_indices.size());
    DeviceBuffer<float> weights(active_workspace ? 0 : regressor.sparse_weights.size());
    if (!active_workspace) {
      ids.upload(std::vector<int>(observation.densekp_ids.begin(), observation.densekp_ids.end()));
      row_offsets.upload(std::vector<int>(regressor.row_offsets.begin(), regressor.row_offsets.end()));
      support_indices.upload(std::vector<int>(regressor.support_indices.begin(), regressor.support_indices.end()));
      weights.upload(regressor.sparse_weights);
    }
    const int count = observation_count * 3 * kRotationParameterCount;
    densekp_derivative_kernel<<<(count + 255) / 256, 256>>>(
        d_vertices.data(), active_workspace ? active_workspace->densekp_ids : ids.data(),
        active_workspace ? active_workspace->densekp_row_offsets : row_offsets.data(),
        active_workspace ? active_workspace->densekp_support_indices : support_indices.data(),
        active_workspace ? active_workspace->densekp_weights : weights.data(),
        observation_count, observation_derivatives.data());
    finish_kernel("densekp_derivative_kernel");
  } else {
    check(cudaMemcpy(observation_derivatives.data(), d_vertices.data(),
        static_cast<std::size_t>(derivative_count) * sizeof(float),
        cudaMemcpyDeviceToDevice), "cudaMemcpy derivative D2D");
  }

  DeviceBuffer<float> projection(observation_count * 6);
  DeviceBuffer<float> uv_pose(
      static_cast<std::size_t>(observation_count) * 2 * kRotationParameterCount);
  DeviceBuffer<float> active(
      static_cast<std::size_t>(observation_count) * 2 * kActiveParameterCount);
  check(cudaMemset(active.data(), 0,
      static_cast<std::size_t>(observation_count) * 2 * kActiveParameterCount * sizeof(float)),
      "cudaMemset active");
  erp_jacobian_kernel<<<(observation_count * 2 + 255) / 256, 256>>>(
      active_workspace ? active_workspace->last_observation_points : observation_points.data(),
      observation_derivatives.data(), observation_count,
      input.image_width, input.image_height, projection.data(), uv_pose.data(), active.data());
  finish_kernel("erp_jacobian_kernel");
  if (active_workspace) {
    active_workspace->last_active_jacobian = active.data();
    active_workspace->last_row_count = observation_count * 2;
  }
  // The evaluator/normal-equation boundary consumes only residual and the
  // active Jacobian. Preserve materialized intermediates for the standalone
  // reference API, but do not transfer them in the persistent production path.
  if (!active_workspace) {
    d_rotation.download(result.d_rotation);
    d_relative.download(result.d_relative_transforms);
    d_vertices.download(result.d_vertices_rotpose);
    observation_derivatives.download(result.observation_d_vertices_rotpose);
    projection.download(result.projection_jacobian);
    result.translation_jacobian = result.projection_jacobian;
    uv_pose.download(result.uv_rotpose_jacobian);
  }
  if (!active_workspace) active.download(result.active_jacobian);
  if (!active_workspace) {
    result.frame_jacobian.assign(
        static_cast<std::size_t>(observation_count) * 2 * kPoseParameterDimension, 0.0F);
    const int camera_column = input.camera == "cam2" ? 0 : 6;
    for (int row = 0; row < observation_count * 2; ++row) {
      for (int c = 0; c < 6; ++c) {
        result.frame_jacobian[row * kPoseParameterDimension + camera_column + c] =
            result.active_jacobian[row * kActiveParameterCount + c];
      }
      for (int body = 0; body < 63; ++body) {
        result.frame_jacobian[row * kPoseParameterDimension + 12 + body] =
            result.active_jacobian[row * kActiveParameterCount + 6 + body];
      }
    }
  }
  result.total_jacobian_ms = elapsed_ms(started);
  return result;
}

PoseCameraResult evaluate_pose_camera_value_cuda_with_workspace(
    CudaPoseWorkspace* workspace, const PoseModelInput& model,
    const PoseCameraInput& input) {
  ExecutionStats stats;
  WorkspaceScope scope(workspace, &stats);
  return evaluate_pose_camera_value_cuda(model, input);
}

PoseCameraJacobianResult evaluate_pose_camera_with_jacobian_cuda_with_workspace(
    CudaPoseWorkspace* workspace, const PoseModelInput& model,
    const PoseCameraInput& input) {
  ExecutionStats stats;
  WorkspaceScope scope(workspace, &stats, true);
  PoseCameraJacobianResult result =
      evaluate_pose_camera_with_jacobian_cuda(model, input);
  result.cuda_allocation_ms = stats.allocation_ms;
  result.cuda_memcpy_ms = stats.memcpy_ms;
  result.cuda_h2d_ms = stats.h2d_ms;
  result.cuda_d2h_ms = stats.d2h_ms;
  result.cuda_kernel_ms = stats.kernel_ms;
  return result;
}

void update_cuda_pose_workspace_shape(
    CudaPoseWorkspace* workspace, const std::vector<float>& joints,
    const std::vector<float>& selected_v_shaped) {
  if (!workspace || joints.size() != kJointCount * 3 ||
      selected_v_shaped.empty()) {
    throw std::runtime_error("CUDA_POSE_SHAPE_UPDATE_INPUT");
  }
  check(cudaMemcpy(workspace->joints, joints.data(),
                   joints.size() * sizeof(float), cudaMemcpyHostToDevice),
        "workspace dynamic joints H2D");
  check(cudaMemcpy(workspace->shaped, selected_v_shaped.data(),
                   selected_v_shaped.size() * sizeof(float), cudaMemcpyHostToDevice),
        "workspace dynamic vertices H2D");
}

std::vector<float> evaluate_pose_shape_beta_jacobian_cuda_with_workspace(
    CudaPoseWorkspace* workspace, const PoseShapeCameraInput& input,
    const std::vector<float>& d_joints_beta,
    const std::vector<float>& d_beta_d_latent, bool download) {
  if (!workspace || !workspace->last_rotations ||
      !workspace->last_posed_vertices || !workspace->last_relative_transforms ||
      d_joints_beta.size() != kJointCount * 3 * kBetaCount ||
      d_beta_d_latent.size() != kBetaCount) {
    throw std::runtime_error("CUDA_POSE_SHAPE_BETA_INPUT");
  }
  if (!workspace->beta_shapedirs) {
    check(cudaMalloc(&workspace->beta_shapedirs,
        input.selected_beta_shapedirs.size() * sizeof(float)),
        "workspace beta shapedirs");
    check(cudaMemcpy(workspace->beta_shapedirs,
        input.selected_beta_shapedirs.data(),
        input.selected_beta_shapedirs.size() * sizeof(float),
        cudaMemcpyHostToDevice), "workspace beta shapedirs H2D");
  }
  WorkspaceScope scope(workspace, nullptr, true, false);
  const int n = input.pose.point_count;
  const int observations = pose_observation_count(input.pose);
  DeviceBuffer<float> d_joints(d_joints_beta.size());
  DeviceBuffer<float> d_beta(kBetaCount);
  DeviceBuffer<float> d_relative(kJointCount * 16 * kBetaCount);
  DeviceBuffer<float> d_vertices(static_cast<std::size_t>(n) * 3 * kBetaCount);
  DeviceBuffer<float> observation_derivatives(
      static_cast<std::size_t>(observations) * 3 * kBetaCount);
  DeviceBuffer<float> beta_jacobian(
      static_cast<std::size_t>(observations) * 2 * kBetaCount);
  d_joints.upload(d_joints_beta);
  d_beta.upload(d_beta_d_latent);
  beta_fk_derivative_kernel<<<1, 32>>>(
      workspace->last_rotations, workspace->joints, workspace->parents,
      d_joints.data(), d_relative.data());
  finish_kernel("beta_fk_derivative_kernel");
  const int vertex_count = n * 3 * kBetaCount;
  beta_vertex_derivative_kernel<<<(vertex_count + 255) / 256, 256>>>(
      workspace->beta_shapedirs, d_beta.data(), workspace->lbs_weights,
      workspace->last_relative_transforms, d_relative.data(),
      workspace->last_posed_vertices, n, input.pose.scale, d_vertices.data());
  finish_kernel("beta_vertex_derivative_kernel");
  if (input.pose.densekp_observation) {
    const int count = observations * 3 * kBetaCount;
    densekp_beta_derivative_kernel<<<(count + 255) / 256, 256>>>(
        d_vertices.data(), workspace->densekp_ids,
        workspace->densekp_row_offsets, workspace->densekp_support_indices,
        workspace->densekp_weights, observations,
        observation_derivatives.data());
    finish_kernel("densekp_beta_derivative_kernel");
  } else {
    check(cudaMemcpy(observation_derivatives.data(), d_vertices.data(),
        static_cast<std::size_t>(vertex_count) * sizeof(float),
        cudaMemcpyDeviceToDevice), "beta derivative D2D");
  }
  const int uv_count = observations * 2 * kBetaCount;
  beta_erp_chain_kernel<<<(uv_count + 255) / 256, 256>>>(
      workspace->last_observation_points, observation_derivatives.data(),
      observations, input.pose.image_width, input.pose.image_height,
      beta_jacobian.data());
  finish_kernel("beta_erp_chain_kernel");
  workspace->last_beta_jacobian = beta_jacobian.data();
  std::vector<float> result;
  if (download) beta_jacobian.download(result);
  return result;
}

CudaPoseNormalEquationResult build_cuda_pose_normal_equations(
    CudaPoseWorkspace* cam2_workspace,
    CudaPoseWorkspace* cam4_workspace,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const PoseNormalEquationConfig& config) {
  if (!cam2_workspace || !cam4_workspace || parameters.size() != 75 ||
      body_pose_anchor.size() != 63) {
    throw std::runtime_error("CUDA_POSE_NORMAL_EQUATION_INPUT");
  }
  const auto total_started = Clock::now();
  CudaPoseNormalEquationResult output;
  auto build_camera = [&](CudaPoseWorkspace* workspace, float alpha,
                          std::vector<double>& hessian,
                          std::vector<double>& gradient) {
    if (!workspace->last_residual || !workspace->last_active_jacobian ||
        workspace->last_row_count <= 0) {
      throw std::runtime_error("CUDA_POSE_LINEARIZATION_REQUIRED");
    }
    const int count = kActiveParameterCount * kActiveParameterCount +
        kActiveParameterCount;
    const auto kernel_started = Clock::now();
    pose_normal_equations_kernel<<<(count + 255) / 256, 256>>>(
        workspace->last_residual, workspace->last_active_jacobian,
        workspace->last_row_count, alpha, config.charbonnier_eps,
        workspace->normal_hessian, workspace->normal_gradient);
    check(cudaGetLastError(), "pose_normal_equations_kernel");
    check(cudaDeviceSynchronize(), "pose_normal_equations_kernel");
    output.gpu_accumulation_ms += elapsed_ms(kernel_started);
    const auto copy_started = Clock::now();
    hessian.resize(kActiveParameterCount * kActiveParameterCount);
    gradient.resize(kActiveParameterCount);
    check(cudaMemcpy(hessian.data(), workspace->normal_hessian,
                     hessian.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "pose normal hessian D2H");
    check(cudaMemcpy(gradient.data(), workspace->normal_gradient,
                     gradient.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "pose normal gradient D2H");
    output.hg_d2h_ms += elapsed_ms(copy_started);
  };
  build_camera(cam2_workspace, config.cam2_alpha_per_point,
               output.cam2_hessian_active, output.cam2_gradient_active);
  build_camera(cam4_workspace, config.cam4_alpha_per_point,
               output.cam4_hessian_active, output.cam4_gradient_active);

  PoseCameraNormalEquations cam2_active;
  cam2_active.hessian_active = output.cam2_hessian_active;
  cam2_active.gradient_active = output.cam2_gradient_active;
  PoseCameraNormalEquations cam4_active;
  cam4_active.hessian_active = output.cam4_hessian_active;
  cam4_active.gradient_active = output.cam4_gradient_active;
  output.equations.hessian.assign(75 * 75, 0.0);
  output.equations.gradient.assign(75, 0.0);
  scatter_pose_active_to_frame(
      cam2_active, "cam2", output.equations.hessian, output.equations.gradient);
  scatter_pose_active_to_frame(
      cam4_active, "cam4", output.equations.hessian, output.equations.gradient);
  const double scale_squared = static_cast<double>(config.pose_anchor_scale) *
      static_cast<double>(config.pose_anchor_scale);
  for (int body = 0; body < 63; ++body) {
    const int index = 12 + body;
    const float delta = parameters[index] - body_pose_anchor[body];
    output.equations.hessian[index * 75 + index] += scale_squared;
    output.equations.gradient[index] +=
        static_cast<double>(delta) * scale_squared;
  }
  output.equations.normal_equations_ms = elapsed_ms(total_started);
  return output;
}

CudaPoseNormalEquationResult CudaPoseNormalEquationBuilder::build(
    CudaPoseWorkspace* cam2_workspace,
    CudaPoseWorkspace* cam4_workspace,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const PoseNormalEquationConfig& config) {
  return build_cuda_pose_normal_equations(
      cam2_workspace, cam4_workspace, parameters, body_pose_anchor, config);
}

CudaPoseShapeNormalEquationResult build_cuda_pose_shape_normal_equations(
    CudaPoseWorkspace* cam2_workspace,
    CudaPoseWorkspace* cam4_workspace,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchor,
    const std::vector<float>& beta_anchor,
    const PoseShapeNormalEquationConfig& config) {
  if (parameters.size() != 85 || beta_anchor.size() != kBetaCount) {
    throw std::runtime_error("CUDA_POSE_SHAPE_NE_INPUT");
  }
  const auto total_started = Clock::now();
  std::vector<float> frame(parameters.begin(), parameters.begin() + 75);
  CudaPoseNormalEquationResult pose = build_cuda_pose_normal_equations(
      cam2_workspace, cam4_workspace, frame, body_pose_anchor, config);
  CudaPoseShapeNormalEquationResult output;
  output.a = pose.equations.hessian;
  output.g_pose = pose.equations.gradient;
  output.b.assign(75 * kBetaCount, 0.0);
  output.c.assign(kBetaCount * kBetaCount, 0.0);
  output.g_beta.assign(kBetaCount, 0.0);
  auto camera = [&](CudaPoseWorkspace* workspace, float alpha,
                    const char* name) {
    if (!workspace->last_beta_jacobian) {
      throw std::runtime_error("CUDA_POSE_SHAPE_BETA_LINEARIZATION_REQUIRED");
    }
    const int count = kActiveParameterCount * kBetaCount +
        kBetaCount * kBetaCount + kBetaCount;
    const auto kernel_started = Clock::now();
    pose_shape_beta_blocks_kernel<<<(count + 255) / 256, 256>>>(
        workspace->last_residual, workspace->last_active_jacobian,
        workspace->last_beta_jacobian, workspace->last_row_count,
        alpha, config.charbonnier_eps, workspace->normal_cross,
        workspace->normal_beta_hessian, workspace->normal_beta_gradient);
    check(cudaGetLastError(), "pose_shape_beta_blocks_kernel");
    check(cudaDeviceSynchronize(), "pose_shape_beta_blocks_kernel");
    output.gpu_accumulation_ms += elapsed_ms(kernel_started);
    std::vector<double> cross(kActiveParameterCount * kBetaCount);
    std::vector<double> c(kBetaCount * kBetaCount), g(kBetaCount);
    const auto copy_started = Clock::now();
    check(cudaMemcpy(cross.data(), workspace->normal_cross,
        cross.size() * sizeof(double), cudaMemcpyDeviceToHost), "shape cross D2H");
    check(cudaMemcpy(c.data(), workspace->normal_beta_hessian,
        c.size() * sizeof(double), cudaMemcpyDeviceToHost), "shape C D2H");
    check(cudaMemcpy(g.data(), workspace->normal_beta_gradient,
        g.size() * sizeof(double), cudaMemcpyDeviceToHost), "shape g D2H");
    output.blocks_d2h_ms += elapsed_ms(copy_started);
    const int camera_start = std::string(name) == "cam2" ? 0 : 6;
    for (int active = 0; active < kActiveParameterCount; ++active) {
      const int frame_index = active < 6 ? camera_start + active : 12 + active - 6;
      for (int beta = 0; beta < kBetaCount; ++beta)
        output.b[frame_index * kBetaCount + beta] +=
            cross[active * kBetaCount + beta];
    }
    for (int i = 0; i < kBetaCount * kBetaCount; ++i) output.c[i] += c[i];
    for (int i = 0; i < kBetaCount; ++i) output.g_beta[i] += g[i];
  };
  camera(cam2_workspace, config.cam2_alpha_per_point, "cam2");
  camera(cam4_workspace, config.cam4_alpha_per_point, "cam4");
  for (int beta = 0; beta < kBetaCount; ++beta) {
    const float z = parameters[75 + beta];
    const float t = std::tanh(z);
    const float physical = 2.5F * t;
    const float derivative = 2.5F * (1.0F - t * t);
    const float diagonal = derivative * config.shape_anchor_scale;
    const float residual = (physical - beta_anchor[beta]) * config.shape_anchor_scale;
    output.c[beta * kBetaCount + beta] +=
        static_cast<double>(diagonal) * diagonal;
    output.g_beta[beta] += static_cast<double>(diagonal) * residual;
  }
  output.gpu_accumulation_ms += pose.gpu_accumulation_ms;
  output.blocks_d2h_ms += pose.hg_d2h_ms;
  output.total_ms = elapsed_ms(total_started);
  return output;
}

CudaMultiFramePoseShapeNormalEquationResult
build_cuda_multiframe_pose_shape_normal_equations(
    CudaMultiFramePoseShapeNormalEquationWorkspace* workspace,
    const std::vector<CudaPoseWorkspace*>& cam2_workspaces,
    const std::vector<CudaPoseWorkspace*>& cam4_workspaces,
    const std::vector<float>& parameters,
    const std::vector<float>& body_pose_anchors,
    const std::vector<float>& beta_anchor,
    float beta_limit,
    const PoseShapeNormalEquationConfig& config) {
  if (!workspace || cam2_workspaces.size() != cam4_workspaces.size() ||
      static_cast<int>(cam2_workspaces.size()) != workspace->frame_count ||
      parameters.size() != static_cast<std::size_t>(workspace->frame_count * 75 + 10) ||
      body_pose_anchors.size() !=
          static_cast<std::size_t>(workspace->frame_count * 63) ||
      beta_anchor.size() != kBetaCount) {
    throw std::runtime_error("CUDA_MULTIFRAME_POSE_SHAPE_NE_INPUT");
  }
  const auto total_started = Clock::now();
  const int frame_count = workspace->frame_count;
  std::vector<double> pose_prior_gradient(
      static_cast<std::size_t>(frame_count) * 63);
  const double pose_prior_diagonal =
      static_cast<double>(config.pose_anchor_scale) * config.pose_anchor_scale;
  for (int frame = 0; frame < frame_count; ++frame) {
    for (int body = 0; body < 63; ++body) {
      const float delta = parameters[frame * 75 + 12 + body] -
          body_pose_anchors[frame * 63 + body];
      pose_prior_gradient[frame * 63 + body] =
          static_cast<double>(delta) * pose_prior_diagonal;
    }
  }
  std::vector<double> shape_prior_hessian(kBetaCount);
  std::vector<double> shape_prior_gradient(kBetaCount);
  const std::size_t beta_offset = static_cast<std::size_t>(frame_count) * 75;
  for (int beta = 0; beta < kBetaCount; ++beta) {
    const float z = parameters[beta_offset + beta];
    const float t = std::tanh(z);
    const float physical = beta_limit * t;
    const float derivative = beta_limit * (1.0F - t * t);
    const float diagonal = derivative * config.shape_anchor_scale;
    const float residual =
        (physical - beta_anchor[beta]) * config.shape_anchor_scale;
    shape_prior_hessian[beta] = static_cast<double>(diagonal) * diagonal;
    shape_prior_gradient[beta] = static_cast<double>(diagonal) * residual;
  }
  check(cudaMemcpy(workspace->pose_prior_gradient, pose_prior_gradient.data(),
      pose_prior_gradient.size() * sizeof(double), cudaMemcpyHostToDevice),
      "multiframe pose prior H2D");
  check(cudaMemcpy(workspace->shape_prior_hessian, shape_prior_hessian.data(),
      shape_prior_hessian.size() * sizeof(double), cudaMemcpyHostToDevice),
      "multiframe shape prior H H2D");
  check(cudaMemcpy(workspace->shape_prior_gradient, shape_prior_gradient.data(),
      shape_prior_gradient.size() * sizeof(double), cudaMemcpyHostToDevice),
      "multiframe shape prior g H2D");

  const auto gpu_started = Clock::now();
  constexpr int active_ne_count =
      kActiveParameterCount * kActiveParameterCount + kActiveParameterCount;
  constexpr int beta_ne_count = kActiveParameterCount * kBetaCount +
      kBetaCount * kBetaCount + kBetaCount;
  constexpr int frame_block_count =
      75 * 75 + 75 * kBetaCount + kBetaCount * kBetaCount + 75 + kBetaCount;
  for (int frame = 0; frame < frame_count; ++frame) {
    CudaPoseWorkspace* cameras[] = {
        cam2_workspaces[frame], cam4_workspaces[frame]};
    const float alphas[] = {
        config.cam2_alpha_per_point, config.cam4_alpha_per_point};
    for (int camera = 0; camera < 2; ++camera) {
      CudaPoseWorkspace* current = cameras[camera];
      if (!current || !current->last_residual ||
          !current->last_active_jacobian || !current->last_beta_jacobian) {
        throw std::runtime_error("CUDA_MULTIFRAME_LINEARIZATION_REQUIRED");
      }
      pose_normal_equations_kernel<<<(active_ne_count + 255) / 256, 256>>>(
          current->last_residual, current->last_active_jacobian,
          current->last_row_count, alphas[camera], config.charbonnier_eps,
          current->normal_hessian, current->normal_gradient);
      check(cudaGetLastError(), "multiframe pose NE kernel");
      pose_shape_beta_blocks_kernel<<<(beta_ne_count + 255) / 256, 256>>>(
          current->last_residual, current->last_active_jacobian,
          current->last_beta_jacobian, current->last_row_count,
          alphas[camera], config.charbonnier_eps, current->normal_cross,
          current->normal_beta_hessian, current->normal_beta_gradient);
      check(cudaGetLastError(), "multiframe beta NE kernel");
    }
    consolidate_pose_shape_frame_blocks_kernel<<<
        (frame_block_count + 255) / 256, 256>>>(
        cameras[0]->normal_hessian, cameras[0]->normal_gradient,
        cameras[0]->normal_cross, cameras[0]->normal_beta_hessian,
        cameras[0]->normal_beta_gradient,
        cameras[1]->normal_hessian, cameras[1]->normal_gradient,
        cameras[1]->normal_cross, cameras[1]->normal_beta_hessian,
        cameras[1]->normal_beta_gradient, workspace->pose_prior_gradient,
        pose_prior_diagonal, frame, workspace->a, workspace->b,
        workspace->frame_c, workspace->g_pose, workspace->frame_g_beta);
    check(cudaGetLastError(), "multiframe frame consolidation kernel");
  }
  reduce_shared_beta_blocks_kernel<<<1, 128>>>(
      workspace->frame_c, workspace->frame_g_beta, frame_count,
      workspace->shape_prior_hessian, workspace->shape_prior_gradient,
      workspace->shared_c, workspace->shared_g_beta);
  check(cudaGetLastError(), "multiframe shared beta reduction kernel");
  check(cudaDeviceSynchronize(), "multiframe normal equations");

  CudaMultiFramePoseShapeNormalEquationResult output;
  output.gpu_accumulation_ms = elapsed_ms(gpu_started);
  output.a.resize(static_cast<std::size_t>(frame_count) * 75 * 75);
  output.b.resize(static_cast<std::size_t>(frame_count) * 75 * kBetaCount);
  output.c.resize(kBetaCount * kBetaCount);
  output.g_pose.resize(static_cast<std::size_t>(frame_count) * 75);
  output.g_beta.resize(kBetaCount);
  const auto copy_started = Clock::now();
  check(cudaMemcpy(output.a.data(), workspace->a,
      output.a.size() * sizeof(double), cudaMemcpyDeviceToHost),
      "multiframe A D2H");
  check(cudaMemcpy(output.b.data(), workspace->b,
      output.b.size() * sizeof(double), cudaMemcpyDeviceToHost),
      "multiframe B D2H");
  check(cudaMemcpy(output.c.data(), workspace->shared_c,
      output.c.size() * sizeof(double), cudaMemcpyDeviceToHost),
      "multiframe C D2H");
  check(cudaMemcpy(output.g_pose.data(), workspace->g_pose,
      output.g_pose.size() * sizeof(double), cudaMemcpyDeviceToHost),
      "multiframe g pose D2H");
  check(cudaMemcpy(output.g_beta.data(), workspace->shared_g_beta,
      output.g_beta.size() * sizeof(double), cudaMemcpyDeviceToHost),
      "multiframe g beta D2H");
  output.final_blocks_d2h_ms = elapsed_ms(copy_started);
  output.total_ms = elapsed_ms(total_started);
  return output;
}

PoseShapeValueResult evaluate_pose_shape_value_cuda(
    const std::vector<float>& parameters, const PoseShapeLMProblem& problem) {
  if (parameters.size() != 85) {
    throw std::runtime_error("invalid CUDA PoseShape parameters");
  }
  const auto started = Clock::now();
  PoseShapeValueResult result;
  std::vector<float> latent(parameters.begin() + 75, parameters.end());
  DeviceBuffer<float> latent_device(10), base_joints(problem.base_joints.size());
  DeviceBuffer<float> joint_basis(problem.joint_basis.size());
  DeviceBuffer<float> tanh_device(10), beta_device(10), joints_device(165);
  latent_device.upload(latent);
  base_joints.upload(problem.base_joints);
  joint_basis.upload(problem.joint_basis);
  shape_kernel<<<1, 256>>>(
      latent_device.data(), problem.beta_limit, base_joints.data(),
      joint_basis.data(), tanh_device.data(), beta_device.data(),
      joints_device.data());
  finish_kernel("shape_kernel");
  tanh_device.download(result.shape.tanh_latent);
  beta_device.download(result.shape.beta);
  joints_device.download(result.shape.joints);

  auto dynamic_vertices = [&](const PoseShapeCameraInput& camera,
                              std::vector<float>& output) {
    const int count = camera.pose.point_count * 3;
    DeviceBuffer<float> base(camera.selected_base_v_shaped.size());
    DeviceBuffer<float> basis(camera.selected_beta_shapedirs.size());
    DeviceBuffer<float> dynamic(count);
    base.upload(camera.selected_base_v_shaped);
    basis.upload(camera.selected_beta_shapedirs);
    dynamic_vertices_kernel<<<(count + 255) / 256, 256>>>(
        base.data(), basis.data(), beta_device.data(), count, dynamic.data());
    finish_kernel("dynamic_vertices_kernel");
    dynamic.download(output);
  };
  dynamic_vertices(problem.cam2, result.cam2_dynamic_v_shaped);
  dynamic_vertices(problem.cam4, result.cam4_dynamic_v_shaped);
  PoseModelInput model{result.shape.joints, problem.parents};
  PoseCameraInput cam2 = problem.cam2.pose;
  PoseCameraInput cam4 = problem.cam4.pose;
  cam2.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam4.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam2.selected_v_shaped = result.cam2_dynamic_v_shaped;
  cam4.selected_v_shaped = result.cam4_dynamic_v_shaped;
  result.cam2 = evaluate_pose_camera_value_cuda(model, cam2);
  result.cam4 = evaluate_pose_camera_value_cuda(model, cam4);
  result.objective_total = evaluate_pose_shape_objective_from_common(
      parameters, result, problem.beta_anchor, problem.body_pose_anchor,
      problem.normal_config.pose_anchor_scale,
      problem.normal_config.shape_anchor_scale,
      problem.normal_config.charbonnier_eps,
      problem.normal_config.cam2_alpha_per_point,
      problem.normal_config.cam4_alpha_per_point,
      &result.objective_primary, &result.objective_pose_anchor,
      &result.objective_shape_anchor);
  result.total_ms = elapsed_ms(started);
  return result;
}

PoseShapeValueResult evaluate_pose_shape_value_cuda_with_workspace(
    CudaPoseShapeValueWorkspace* workspace,
    CudaPoseWorkspace* cam2_workspace,
    CudaPoseWorkspace* cam4_workspace,
    const std::vector<float>& parameters, const PoseShapeLMProblem& problem) {
  if (!workspace || !cam2_workspace || !cam4_workspace ||
      parameters.size() != 85) {
    throw std::runtime_error("invalid persistent CUDA PoseShape input");
  }
  const auto started = Clock::now();
  PoseShapeValueResult result;
  check(cudaMemcpy(workspace->latent, parameters.data() + 75,
                   kBetaCount * sizeof(float), cudaMemcpyHostToDevice),
        "shape workspace latent H2D");
  shape_kernel<<<1, 256>>>(
      workspace->latent, problem.beta_limit, workspace->base_joints,
      workspace->joint_basis, workspace->tanh_latent, workspace->beta,
      workspace->joints);
  finish_kernel("shape_kernel");
  result.shape.tanh_latent.resize(kBetaCount);
  result.shape.beta.resize(kBetaCount);
  result.shape.joints.resize(kJointCount * 3);
  check(cudaMemcpy(result.shape.tanh_latent.data(), workspace->tanh_latent,
                   kBetaCount * sizeof(float), cudaMemcpyDeviceToHost),
        "shape workspace tanh D2H");
  check(cudaMemcpy(result.shape.beta.data(), workspace->beta,
                   kBetaCount * sizeof(float), cudaMemcpyDeviceToHost),
        "shape workspace beta D2H");
  check(cudaMemcpy(result.shape.joints.data(), workspace->joints,
                   kJointCount * 3 * sizeof(float), cudaMemcpyDeviceToHost),
        "shape workspace joints D2H");

  auto dynamic_vertices = [&](const PoseShapeCameraInput& camera,
                              const float* base, const float* basis,
                              float* dynamic, std::vector<float>& output) {
    const int count = camera.pose.point_count * 3;
    dynamic_vertices_kernel<<<(count + 255) / 256, 256>>>(
        base, basis, workspace->beta, count, dynamic);
    finish_kernel("dynamic_vertices_kernel");
    output.resize(count);
    check(cudaMemcpy(output.data(), dynamic, count * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "shape workspace dynamic D2H");
  };
  dynamic_vertices(problem.cam2, workspace->cam2_base, workspace->cam2_basis,
                   workspace->cam2_dynamic, result.cam2_dynamic_v_shaped);
  dynamic_vertices(problem.cam4, workspace->cam4_base, workspace->cam4_basis,
                   workspace->cam4_dynamic, result.cam4_dynamic_v_shaped);

  PoseModelInput model{result.shape.joints, problem.parents};
  PoseCameraInput cam2 = problem.cam2.pose;
  PoseCameraInput cam4 = problem.cam4.pose;
  cam2.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam4.parameters.assign(parameters.begin(), parameters.begin() + 75);
  cam2.selected_v_shaped = result.cam2_dynamic_v_shaped;
  cam4.selected_v_shaped = result.cam4_dynamic_v_shaped;
  update_cuda_pose_workspace_shape(
      cam2_workspace, result.shape.joints, result.cam2_dynamic_v_shaped);
  update_cuda_pose_workspace_shape(
      cam4_workspace, result.shape.joints, result.cam4_dynamic_v_shaped);
  result.cam2 = evaluate_pose_camera_value_cuda_with_workspace(
      cam2_workspace, model, cam2);
  result.cam4 = evaluate_pose_camera_value_cuda_with_workspace(
      cam4_workspace, model, cam4);
  result.objective_total = evaluate_pose_shape_objective_from_common(
      parameters, result, problem.beta_anchor, problem.body_pose_anchor,
      problem.normal_config.pose_anchor_scale,
      problem.normal_config.shape_anchor_scale,
      problem.normal_config.charbonnier_eps,
      problem.normal_config.cam2_alpha_per_point,
      problem.normal_config.cam4_alpha_per_point,
      &result.objective_primary, &result.objective_pose_anchor,
      &result.objective_shape_anchor);
  result.total_ms = elapsed_ms(started);
  return result;
}

}  // namespace staged_lm
