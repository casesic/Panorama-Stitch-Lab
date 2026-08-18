#include "staged_lm/performance.h"
#include "staged_lm/forward_reuse_cache.h"
#include "staged_lm/pose_forward.h"
#include "staged_lm/shape.h"

#include <cstring>

namespace staged_lm
{
  namespace
  {
    bool enabled = false;
    std::string current_stage;
    std::vector<KernelTimingSample> samples;
    KernelDispatchCounters dispatch_counters;
    std::vector<ValuePoseOffsetState> value_states;
    bool derivative_attribution_enabled = false;
    bool capture_derivative_inputs = false;
    std::vector<PoseOffsetDerivativeAttributionSample> derivative_samples;
    std::vector<PoseOffsetDerivativeInputState> derivative_states;
    SelectedPosedirsCopyAudit posedirs_copy_audit;
    bool forward_reuse_attribution = false;
    std::string lm_attribution_stage;
    std::vector<LMStateCallSample> lm_calls;
    std::vector<ForwardValueSample> forward_values;
    std::vector<ShapeForwardSample> shape_values;
    std::vector<ForwardCacheEvent> cache_events;
    ForwardReuseMode reuse_mode = ForwardReuseMode::kOff;
    std::uint64_t context_counter = 0;

    std::uint64_t hash_bytes(const std::vector<float> &values)
    {
      std::uint64_t hash = 1469598103934665603ULL;
      const auto *bytes = reinterpret_cast<const unsigned char *>(values.data());
      for (std::size_t i = 0; i < values.size() * sizeof(float); ++i)
      {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
      }
      return hash;
    }
  } // namespace

  void reset_performance_samples()
  {
    samples.clear();
    value_states.clear();
    derivative_samples.clear();
    derivative_states.clear();
    posedirs_copy_audit = {};
    lm_calls.clear();
    forward_values.clear();
    shape_values.clear();
    cache_events.clear();
  }
  void set_performance_recording(bool value) { enabled = value; }
  void set_performance_stage(const std::string &stage) { current_stage = stage; }
  void record_performance_sample(
      const std::string &camera, const std::string &kernel,
      int point_count, double milliseconds)
  {
    if (enabled)
      samples.push_back(
          {current_stage, camera, kernel, point_count, milliseconds});
  }
  const std::vector<KernelTimingSample> &performance_samples() { return samples; }
  void reset_kernel_dispatch_counters() { dispatch_counters = {}; }
  void record_pose_offset_dispatch(bool optimized)
  {
    if (optimized)
      ++dispatch_counters.pose_offset_optimized_calls;
    else
      ++dispatch_counters.pose_offset_scalar_calls;
  }
  void record_pose_offset_transposed_dispatch()
  {
    ++dispatch_counters.pose_offset_transposed_calls;
  }
  void record_lbs_selected_transform_dispatch(bool optimized)
  {
    if (optimized)
      ++dispatch_counters.lbs_selected_transform_optimized_calls;
    else
      ++dispatch_counters.lbs_selected_transform_scalar_calls;
  }
  void record_value_pose_offset_dispatch(bool gemv, bool fallback)
  {
    if (gemv)
      ++dispatch_counters.value_pose_offset_gemv_calls;
    else
      ++dispatch_counters.value_pose_offset_scalar_calls;
    if (fallback)
      ++dispatch_counters.value_pose_offset_gemv_fallback_count;
  }
  const KernelDispatchCounters &kernel_dispatch_counters()
  {
    return dispatch_counters;
  }
  void record_value_pose_offset_state(
      const std::string &camera, int point_count,
      const std::vector<float> &pose_feature)
  {
    if (enabled)
      value_states.push_back(
          {current_stage, camera, point_count, pose_feature});
  }
  const std::vector<ValuePoseOffsetState> &value_pose_offset_states()
  {
    return value_states;
  }
  void set_pose_offset_derivative_attribution(bool value, bool capture_inputs)
  {
    derivative_attribution_enabled = value;
    capture_derivative_inputs = value && capture_inputs;
  }
  bool pose_offset_derivative_attribution_enabled()
  {
    return derivative_attribution_enabled;
  }
  void record_pose_offset_derivative_attribution(
      const PoseOffsetDerivativeAttributionSample &sample,
      const std::vector<float> &selected_posedirs,
      const std::vector<float> &d_pose_feature)
  {
    if (!derivative_attribution_enabled)
      return;
    PoseOffsetDerivativeAttributionSample recorded = sample;
    recorded.stage = current_stage;
    derivative_samples.push_back(recorded);
    if (capture_derivative_inputs)
      derivative_states.push_back({current_stage, sample.camera, sample.point_count,
                                   sample.selected_posedirs_pointer, selected_posedirs, d_pose_feature});
  }
  const std::vector<PoseOffsetDerivativeAttributionSample> &
  pose_offset_derivative_attribution_samples() { return derivative_samples; }
  const std::vector<PoseOffsetDerivativeInputState> &
  pose_offset_derivative_input_states() { return derivative_states; }
  void record_selected_posedirs_derivative_copy(std::size_t bytes, double ms)
  {
    ++posedirs_copy_audit.derivative_call_deep_copy_count;
    posedirs_copy_audit.derivative_call_deep_copy_bytes += bytes;
    posedirs_copy_audit.derivative_call_deep_copy_ms += ms;
  }
  const SelectedPosedirsCopyAudit &selected_posedirs_copy_audit()
  {
    return posedirs_copy_audit;
  }
  void set_forward_reuse_attribution(bool value) { forward_reuse_attribution = value; }
  void set_lm_attribution_stage(const std::string &stage) { lm_attribution_stage = stage; }
  void record_lm_state_call(int outer, int trial, const std::string &kind,
                            const std::string &disposition,
                            const std::vector<float> &parameters)
  {
    if (!forward_reuse_attribution)
      return;
    lm_calls.push_back({lm_attribution_stage, static_cast<int>(lm_calls.size()),
                        outer, trial, kind, disposition, parameters});
  }
  void update_last_lm_objective_disposition(const std::string &disposition)
  {
    if (!forward_reuse_attribution)
      return;
    for (auto it = lm_calls.rbegin(); it != lm_calls.rend(); ++it)
    {
      if (it->stage == lm_attribution_stage && it->call_kind == "objective" &&
          it->disposition == "candidate_pending")
      {
        it->disposition = disposition;
        return;
      }
    }
  }
  void record_forward_value_sample(const std::string &camera, int point_count,
                                   const std::vector<float> &parameters,
                                   const PoseCameraResult &r)
  {
    if (!forward_reuse_attribution)
      return;
    const std::vector<const std::vector<float> *> fields = {
        &r.rotations, &r.pose_feature, &r.pose_offsets, &r.v_posed, &r.global_transforms,
        &r.relative_transforms, &r.selected_transforms, &r.vertices, &r.uv, &r.raw_residual};
    ForwardValueSample s;
    s.stage = current_stage;
    s.camera = camera;
    s.point_count = point_count;
    s.parameters = parameters;
    s.rotations_ms = r.rotations_ms;
    s.pose_offsets_ms = r.pose_offsets_ms;
    s.fk_ms = r.fk_ms;
    s.lbs_ms = r.lbs_ms;
    s.erp_ms = r.erp_ms;
    s.total_ms = r.total_value_forward_ms;
    for (const auto *f : fields)
    {
      s.field_hashes.push_back(hash_bytes(*f));
      s.field_bytes.push_back(f->size() * sizeof(float));
    }
    forward_values.push_back(std::move(s));
  }
  const std::vector<LMStateCallSample> &lm_state_call_samples() { return lm_calls; }
  const std::vector<ForwardValueSample> &forward_value_samples() { return forward_values; }
  void record_shape_forward_sample(const std::vector<float> &parameters,
                                   const ShapeValueResult &shape,
                                   const std::vector<float> &cam2_dynamic,
                                   const std::vector<float> &cam4_dynamic)
  {
    if (!forward_reuse_attribution)
      return;
    std::vector<float> combined = shape.tanh_latent;
    combined.insert(combined.end(), shape.beta.begin(), shape.beta.end());
    combined.insert(combined.end(), shape.joints.begin(), shape.joints.end());
    shape_values.push_back({current_stage, parameters, hash_bytes(combined),
                            hash_bytes(cam2_dynamic), hash_bytes(cam4_dynamic),
                            (combined.size() + cam2_dynamic.size() + cam4_dynamic.size()) * sizeof(float)});
  }
  const std::vector<ShapeForwardSample> &shape_forward_samples() { return shape_values; }
  void record_forward_cache_event(const std::string &stage, const std::string &consumer,
                                  const std::string &producer, std::uint64_t hash, const std::string &lookup,
                                  const std::string &action, std::size_t key_bytes, std::size_t payload_bytes,
                                  int skipped, double overhead_ms) { cache_events.push_back({stage, consumer, producer, lookup, action, hash, key_bytes, payload_bytes, skipped, overhead_ms}); }
  const std::vector<ForwardCacheEvent> &forward_cache_events() { return cache_events; }
  void set_stage_forward_reuse(bool v) { reuse_mode = v ? ForwardReuseMode::kFull : ForwardReuseMode::kOff; }
  bool stage_forward_reuse_enabled() { return reuse_mode != ForwardReuseMode::kOff; }
  void set_forward_reuse_mode(ForwardReuseMode mode) { reuse_mode = mode; }
  ForwardReuseMode forward_reuse_mode() { return reuse_mode; }
  bool forward_reuse_consumes(ForwardReuseMode mode, ForwardRole consumer)
  {
    if (mode == ForwardReuseMode::kFull)
      return true;
    if (consumer == ForwardRole::kObjective)
      return mode == ForwardReuseMode::kLinearizationToObjectiveOnly;
    return mode == ForwardReuseMode::kObjectiveToLinearizationOnly ||
           mode == ForwardReuseMode::kObjectiveToLinearizationShadowWarmup;
  }
  bool forward_reuse_stores(ForwardReuseMode mode, ForwardRole producer)
  {
    if (mode == ForwardReuseMode::kFull)
      return true;
    if (producer == ForwardRole::kLinearization)
      return mode == ForwardReuseMode::kLinearizationToObjectiveOnly;
    return mode == ForwardReuseMode::kObjectiveToLinearizationOnly ||
           mode == ForwardReuseMode::kObjectiveToLinearizationShadowWarmup ||
           mode == ForwardReuseMode::kRetainPayloadDropBeforeLinearization;
  }
  std::uint64_t next_forward_reuse_context() { return ++context_counter; }
} // namespace staged_lm
