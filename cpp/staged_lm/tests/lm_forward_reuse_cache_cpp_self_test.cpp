#include "staged_lm/forward_reuse_cache.h"
#include <iostream>
#include <memory>
#include <stdexcept>
struct Payload{std::unique_ptr<int> value;Payload()=default;explicit Payload(int v):value(new int(v)){}Payload(Payload&&)=default;Payload&operator=(Payload&&)=default;Payload(const Payload&)=delete;Payload&operator=(const Payload&)=delete;};
int main(){using namespace staged_lm;const std::vector<float>a{1,2},b{1,3};Payload out;
 {ExactRoleSingleSlotCache<Payload> c("pose",1);c.store(a,ForwardRole::kObjective,1,Payload(7),4);if(!c.consume(a,ForwardRole::kLinearization,1,out)||!out.value||*out.value!=7)throw std::runtime_error("EXACT_OPPOSITE_ROLE_HIT_FAILED");if(c.consume(a,ForwardRole::kLinearization,1,out))throw std::runtime_error("CONSUMED_ENTRY_REUSED");}
 {ExactRoleSingleSlotCache<Payload> c("pose",2);c.store(a,ForwardRole::kObjective,2,Payload(1),4);if(c.consume(b,ForwardRole::kLinearization,2,out))throw std::runtime_error("ONE_BYTE_STATE_FALSE_HIT");if(c.consume(a,ForwardRole::kObjective,2,out))throw std::runtime_error("SAME_ROLE_FALSE_HIT");if(c.consume(a,ForwardRole::kLinearization,3,out))throw std::runtime_error("WRONG_CONTEXT_FALSE_HIT");c.store(b,ForwardRole::kObjective,2,Payload(9),4);if(!c.consume(b,ForwardRole::kLinearization,2,out)||*out.value!=9)throw std::runtime_error("OVERWRITE_FAILED");}
 {ExactRoleSingleSlotCache<Payload> c("pose_shape",4);c.store(a,ForwardRole::kObjective,4,Payload(2),4);c.store(b,ForwardRole::kObjective,4,Payload(3),4);if(c.consume(a,ForwardRole::kLinearization,4,out)||!c.consume(b,ForwardRole::kLinearization,4,out))throw std::runtime_error("REJECTED_CANDIDATE_STALE_HIT");}
 std::cout<<"LM_FORWARD_REUSE_CACHE_CPP_SELF_TEST_PASS\n";}
