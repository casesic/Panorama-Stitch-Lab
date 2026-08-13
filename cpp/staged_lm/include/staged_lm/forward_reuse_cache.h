#pragma once

#include "staged_lm/performance.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace staged_lm {
enum class ForwardRole { kObjective, kLinearization };
enum class ForwardReuseMode {
  kOff,
  kFull,
  kLinearizationToObjectiveOnly,
  kObjectiveToLinearizationOnly,
  kObjectiveToLinearizationShadowWarmup,
  kRetainPayloadDropBeforeLinearization,
};
inline const char* forward_role_name(ForwardRole r) {
  return r == ForwardRole::kObjective ? "OBJECTIVE" : "LINEARIZATION";
}

template<class Payload> class ExactRoleSingleSlotCache {
 public:
  ExactRoleSingleSlotCache(std::string stage, std::uint64_t context)
      : stage_(std::move(stage)), context_(context) {}
  bool consume(const std::vector<float>& state, ForwardRole consumer,
               std::uint64_t context, Payload& output) {
    const auto started=std::chrono::steady_clock::now();
    std::string result="miss",action="none";
    const bool had_value = valid_;
    const ForwardRole recorded_producer = producer_;
    bool hit=false;
    if(valid_ && context!=context_){result="context_miss";++context_misses_;}
    else if(valid_ && producer_==consumer){result="role_miss";++role_misses_;}
    else if(valid_ && (state.size()!=key_.size() ||
             std::memcmp(state.data(),key_.data(),state.size()*sizeof(float))!=0)){
      result="state_miss";++state_misses_;
    } else if(valid_){output=std::move(payload_);valid_=false;consumed_=true;
      result="hit";action="consume";hit=true;++hits_;
      if(producer_==ForwardRole::kLinearization)++lin_to_obj_hits_;else ++obj_to_lin_hits_;
    } else ++misses_;
    overhead_ms_+=elapsed(started);
    record_forward_cache_event(stage_,forward_role_name(consumer),
        had_value?forward_role_name(recorded_producer):"NONE",hash(state),result,action,
        state.size()*sizeof(float),0,hit?2:0,overhead_ms_);
    return hit;
  }
  void store(const std::vector<float>& state,ForwardRole producer,
             std::uint64_t context,Payload payload,std::size_t payload_bytes){
    const auto started=std::chrono::steady_clock::now();
    const bool overwrite=valid_;if(overwrite)++overwrites_;
    key_=state;key_copy_bytes_+=state.size()*sizeof(float);producer_=producer;
    context_=context;payload_=std::move(payload);valid_=true;consumed_=false;
    peak_payload_bytes_=std::max(peak_payload_bytes_,payload_bytes);++move_stores_;
    overhead_ms_+=elapsed(started);
    record_forward_cache_event(stage_,forward_role_name(producer),
        forward_role_name(producer),hash(state),overwrite?"miss":"miss",
        overwrite?"overwrite":"store",state.size()*sizeof(float),payload_bytes,0,overhead_ms_);
  }
  bool discard(const std::vector<float>& state, ForwardRole consumer,
               std::uint64_t context) {
    const auto started = std::chrono::steady_clock::now();
    if (!valid_ || context != context_ || producer_ == consumer ||
        state.size() != key_.size() ||
        std::memcmp(state.data(), key_.data(), state.size() * sizeof(float)) != 0)
      return false;
    const ForwardRole producer = producer_;
    const std::size_t bytes = peak_payload_bytes_;
    payload_ = Payload{};
    key_.clear();
    valid_ = false;
    const double milliseconds = elapsed(started);
    overhead_ms_ += milliseconds;
    record_forward_cache_event(stage_, forward_role_name(consumer),
        forward_role_name(producer), hash(state), "hit", "drop",
        state.size() * sizeof(float), bytes, 0, overhead_ms_);
    return true;
  }
  int hits()const{return hits_;} int misses()const{return misses_;}
 private:
  static double elapsed(std::chrono::steady_clock::time_point s){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count();}
  static std::uint64_t hash(const std::vector<float>& v){std::uint64_t h=1469598103934665603ULL;auto*p=reinterpret_cast<const unsigned char*>(v.data());for(std::size_t i=0;i<v.size()*sizeof(float);++i){h^=p[i];h*=1099511628211ULL;}return h;}
  std::string stage_;std::uint64_t context_;bool valid_=false,consumed_=false;ForwardRole producer_=ForwardRole::kObjective;std::vector<float> key_;Payload payload_{};
  int hits_=0,misses_=0,role_misses_=0,state_misses_=0,context_misses_=0,overwrites_=0,lin_to_obj_hits_=0,obj_to_lin_hits_=0,move_stores_=0;std::size_t key_copy_bytes_=0,peak_payload_bytes_=0;double overhead_ms_=0;
};
void set_stage_forward_reuse(bool enabled);
bool stage_forward_reuse_enabled();
void set_forward_reuse_mode(ForwardReuseMode mode);
ForwardReuseMode forward_reuse_mode();
bool forward_reuse_consumes(ForwardReuseMode mode, ForwardRole consumer);
bool forward_reuse_stores(ForwardReuseMode mode, ForwardRole producer);
std::uint64_t next_forward_reuse_context();
}  // namespace staged_lm
