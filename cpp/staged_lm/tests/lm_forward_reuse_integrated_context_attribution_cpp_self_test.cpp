#include "staged_lm/forward_reuse_cache.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Payload {
  std::unique_ptr<int> value;
  Payload() = default;
  explicit Payload(int input) : value(std::make_unique<int>(input)) {}
  Payload(Payload&&) = default;
  Payload& operator=(Payload&&) = default;
  Payload(const Payload&) = delete;
  Payload& operator=(const Payload&) = delete;
};
void require(bool condition, const char* error) {
  if (!condition) throw std::runtime_error(error);
}
}

int main() {
  using namespace staged_lm;
  require(!forward_reuse_consumes(ForwardReuseMode::kOff, ForwardRole::kObjective), "D0");
  require(forward_reuse_consumes(ForwardReuseMode::kFull, ForwardRole::kObjective) &&
          forward_reuse_consumes(ForwardReuseMode::kFull, ForwardRole::kLinearization), "D1");
  require(forward_reuse_stores(ForwardReuseMode::kLinearizationToObjectiveOnly, ForwardRole::kLinearization) &&
          forward_reuse_consumes(ForwardReuseMode::kLinearizationToObjectiveOnly, ForwardRole::kObjective) &&
          !forward_reuse_consumes(ForwardReuseMode::kLinearizationToObjectiveOnly, ForwardRole::kLinearization), "D2");
  require(forward_reuse_stores(ForwardReuseMode::kObjectiveToLinearizationOnly, ForwardRole::kObjective) &&
          forward_reuse_consumes(ForwardReuseMode::kObjectiveToLinearizationOnly, ForwardRole::kLinearization) &&
          !forward_reuse_consumes(ForwardReuseMode::kObjectiveToLinearizationOnly, ForwardRole::kObjective), "D3");
  require(forward_reuse_consumes(ForwardReuseMode::kObjectiveToLinearizationShadowWarmup, ForwardRole::kLinearization), "D4");
  require(forward_reuse_stores(ForwardReuseMode::kRetainPayloadDropBeforeLinearization, ForwardRole::kObjective) &&
          !forward_reuse_consumes(ForwardReuseMode::kRetainPayloadDropBeforeLinearization, ForwardRole::kLinearization), "D5");

  const std::vector<float> state{1.0F, 2.0F};
  ExactRoleSingleSlotCache<Payload> cache("diagnostic", 7);
  cache.store(state, ForwardRole::kObjective, 7, Payload(11), sizeof(int));
  require(cache.discard(state, ForwardRole::kLinearization, 7), "D5_DROP");
  Payload output;
  require(!cache.consume(state, ForwardRole::kLinearization, 7, output), "D5_REUSED");
  return 0;
}
