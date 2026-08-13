#pragma once

#include <string>
#include <cstdint>
#include <vector>

namespace staged_lm {

struct PoseCameraResult;
struct ShapeValueResult;

struct KernelTimingSample {
  std::string stage;
  std::string camera;
  std::string kernel;
  int point_count = 0;
  double milliseconds = 0.0;
};

struct KernelDispatchCounters {
  int pose_offset_scalar_calls = 0;
  int pose_offset_optimized_calls = 0;
  int pose_offset_transposed_calls = 0;
  int lbs_selected_transform_scalar_calls = 0;
  int lbs_selected_transform_optimized_calls = 0;
  int value_pose_offset_scalar_calls = 0;
  int value_pose_offset_gemv_calls = 0;
  int value_pose_offset_gemv_fallback_count = 0;
  int optimized_fallback_count = 0;
};

struct ValuePoseOffsetState {
  std::string stage;
  std::string camera;
  int point_count = 0;
  std::vector<float> pose_feature;
};

struct PoseOffsetDerivativeAttributionSample {
  std::string stage;
  std::string camera;
  int point_count = 0;
  int parameter_count = 66;
  double total_ms = 0.0;
  double d_pose_feature_ms = 0.0;
  double setup_ms = 0.0;
  double output_preparation_ms = 0.0;
  double gemm_core_ms = 0.0;
  double postprocess_ms = 0.0;
  double other_ms = 0.0;
  std::size_t output_size = 0;
  std::size_t capacity_before = 0;
  std::size_t capacity_after = 0;
  std::uintptr_t pointer_before = 0;
  std::uintptr_t pointer_after = 0;
  std::uintptr_t selected_posedirs_pointer = 0;
};

struct PoseOffsetDerivativeInputState {
  std::string stage;
  std::string camera;
  int point_count = 0;
  std::uintptr_t selected_posedirs_pointer = 0;
  std::vector<float> selected_posedirs;
  std::vector<float> d_pose_feature;
};

void reset_performance_samples();
void set_performance_recording(bool enabled);
void set_performance_stage(const std::string& stage);
void record_performance_sample(
    const std::string& camera, const std::string& kernel,
    int point_count, double milliseconds);
const std::vector<KernelTimingSample>& performance_samples();
void reset_kernel_dispatch_counters();
void record_pose_offset_dispatch(bool optimized);
void record_pose_offset_transposed_dispatch();
void record_lbs_selected_transform_dispatch(bool optimized);
void record_value_pose_offset_dispatch(bool gemv, bool fallback = false);
const KernelDispatchCounters& kernel_dispatch_counters();
void record_value_pose_offset_state(
    const std::string& camera, int point_count,
    const std::vector<float>& pose_feature);
const std::vector<ValuePoseOffsetState>& value_pose_offset_states();
void set_pose_offset_derivative_attribution(bool enabled, bool capture_inputs);
bool pose_offset_derivative_attribution_enabled();
void record_pose_offset_derivative_attribution(
    const PoseOffsetDerivativeAttributionSample& sample,
    const std::vector<float>& selected_posedirs,
    const std::vector<float>& d_pose_feature);
const std::vector<PoseOffsetDerivativeAttributionSample>&
pose_offset_derivative_attribution_samples();
const std::vector<PoseOffsetDerivativeInputState>&
pose_offset_derivative_input_states();

struct SelectedPosedirsCopyAudit {
  std::size_t derivative_call_deep_copy_count = 0;
  std::size_t derivative_call_deep_copy_bytes = 0;
  double derivative_call_deep_copy_ms = 0.0;
};
void record_selected_posedirs_derivative_copy(std::size_t bytes, double ms);
const SelectedPosedirsCopyAudit& selected_posedirs_copy_audit();

struct LMStateCallSample {
  std::string stage;
  int chronological_index = 0;
  int outer = -1;
  int trial = -1;
  std::string call_kind;
  std::string disposition;
  std::vector<float> parameters;
};

struct ForwardValueSample {
  std::string stage;
  std::string camera;
  int point_count = 0;
  std::vector<float> parameters;
  std::vector<std::uint64_t> field_hashes;
  std::vector<std::size_t> field_bytes;
  double rotations_ms = 0.0;
  double pose_offsets_ms = 0.0;
  double fk_ms = 0.0;
  double lbs_ms = 0.0;
  double erp_ms = 0.0;
  double total_ms = 0.0;
};
struct ShapeForwardSample {
  std::string stage;
  std::vector<float> parameters;
  std::uint64_t shape_hash = 0;
  std::uint64_t cam2_dynamic_hash = 0;
  std::uint64_t cam4_dynamic_hash = 0;
  std::size_t payload_bytes = 0;
};

void set_forward_reuse_attribution(bool enabled);
void set_lm_attribution_stage(const std::string& stage);
void record_lm_state_call(int outer, int trial, const std::string& call_kind,
                          const std::string& disposition,
                          const std::vector<float>& parameters);
void update_last_lm_objective_disposition(const std::string& disposition);
void record_forward_value_sample(const std::string& camera, int point_count,
                                 const std::vector<float>& parameters,
                                 const PoseCameraResult& result);
const std::vector<LMStateCallSample>& lm_state_call_samples();
const std::vector<ForwardValueSample>& forward_value_samples();
void record_shape_forward_sample(const std::vector<float>& parameters,
                                 const ShapeValueResult& shape,
                                 const std::vector<float>& cam2_dynamic,
                                 const std::vector<float>& cam4_dynamic);
const std::vector<ShapeForwardSample>& shape_forward_samples();
struct ForwardCacheEvent {
  std::string stage,consumer_kind,producer_kind,lookup_result,cache_action;
  std::uint64_t state_hash=0;std::size_t key_copy_bytes=0,payload_bytes=0;
  int camera_forwards_skipped=0;double cumulative_overhead_ms=0.0;
};
void record_forward_cache_event(const std::string& stage,const std::string& consumer,
    const std::string& producer,std::uint64_t hash,const std::string& lookup,
    const std::string& action,std::size_t key_bytes,std::size_t payload_bytes,
    int skipped,double overhead_ms);
const std::vector<ForwardCacheEvent>& forward_cache_events();

}  // namespace staged_lm
