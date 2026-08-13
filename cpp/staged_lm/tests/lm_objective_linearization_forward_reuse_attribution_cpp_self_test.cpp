#include "staged_lm/performance.h"
#include "staged_lm/pose_forward.h"
#include "staged_lm/shape.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace staged_lm;
  reset_performance_samples(); set_forward_reuse_attribution(true);
  set_lm_attribution_stage("test");
  const std::vector<float> x{1,2,3};
  record_lm_state_call(-1,-1,"linearization","current",x);
  record_lm_state_call(-1,-1,"objective","initial_current",x);
  record_lm_state_call(0,0,"objective","candidate_pending",x);
  update_last_lm_objective_disposition("rejected");
  if(lm_state_call_samples().size()!=3 ||
     std::memcmp(lm_state_call_samples()[0].parameters.data(),
                 lm_state_call_samples()[1].parameters.data(),x.size()*sizeof(float))!=0 ||
     lm_state_call_samples()[2].disposition!="rejected")
    throw std::runtime_error("LM_STATE_RECORDER_FAILURE");
  PoseCameraResult pose;pose.rotations={1};pose.pose_feature={2};pose.pose_offsets={3};
  pose.v_posed={4};pose.global_transforms={5};pose.relative_transforms={6};
  pose.selected_transforms={7};pose.vertices={8};pose.uv={9};pose.raw_residual={10};
  record_forward_value_sample("cam2",1,x,pose);record_forward_value_sample("cam2",1,x,pose);
  if(forward_value_samples().size()!=2 ||
     forward_value_samples()[0].field_hashes!=forward_value_samples()[1].field_hashes)
    throw std::runtime_error("POSE_RESULT_COMPARATOR_FAILURE");
  ShapeValueResult shape;shape.tanh_latent={1};shape.beta={2};shape.joints={3};
  record_shape_forward_sample(x,shape,{4,5},{6,7});record_shape_forward_sample(x,shape,{4,5},{6,7});
  if(shape_forward_samples().size()!=2 || shape_forward_samples()[0].shape_hash!=shape_forward_samples()[1].shape_hash)
    throw std::runtime_error("SHAPE_RESULT_COMPARATOR_FAILURE");
  set_forward_reuse_attribution(false);
  std::cout<<"LM_OBJECTIVE_LINEARIZATION_FORWARD_REUSE_ATTRIBUTION_CPP_SELF_TEST_PASS\n";
}
