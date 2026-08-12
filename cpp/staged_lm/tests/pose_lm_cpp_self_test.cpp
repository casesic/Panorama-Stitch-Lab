#include "staged_lm/lm.h"

#include <cmath>
#include <iostream>

int main() {
  if (staged_lm::increase_lm_damping(1.0) != 10.0
      || staged_lm::increase_lm_damping(1e12) != 1e12
      || staged_lm::update_lm_damping_after_acceptance(3.0, 0.8) != 1.0
      || staged_lm::update_lm_damping_after_acceptance(1.0, 0.2) != 3.0
      || staged_lm::update_lm_damping_after_acceptance(1.0, 0.5) != 1.0) {
    std::cerr << "LM_DAMPING_POLICY_FAILED\n"; return 1;
  }
  staged_lm::LMConfig config;
  config.parameter_dimension = 1; config.max_accepted_steps = 1; config.max_trials = 2;
  config.scaled_gradient_tolerance = 1e-12;
  config.step_tolerance = 1e-12; config.rel_objective_tolerance = 1e-12;
  auto equations = [](const std::vector<float>& x) {
    staged_lm::DenseNormalEquations value;
    value.hessian = {1.0}; value.gradient = {static_cast<double>(x[0] - 2.0F)};
    return value;
  };
  auto objective = [](const std::vector<float>& x) {
    const double d = static_cast<double>(x[0]) - 2.0; return 0.5 * d * d;
  };
  const auto accepted = staged_lm::run_lm({0.0F}, config, equations, objective);
  if (accepted.accepted_steps != 1 || accepted.trials.size() != 1
      || !accepted.trials[0].accepted
      || accepted.final_lambda >= accepted.initial_lambda) {
    std::cerr << "LM_ACCEPT_HIGH_RHO_FAILED\n"; return 1;
  }
  auto worse = [](const std::vector<float>& x) { return static_cast<double>(x[0]); };
  const auto rejected = staged_lm::run_lm({0.0F}, config, equations, worse);
  if (rejected.accepted_steps != 0 || rejected.rejected_trials != 2
      || rejected.final_lambda <= rejected.initial_lambda
      || rejected.status != "LM_NO_ACCEPTED_TRIAL") {
    std::cerr << "LM_REJECTION_FAILED\n"; return 1;
  }
  if (staged_lm::check_lm_convergence(0.0, 0.0, 0.0, config)
      != "CONVERGED_SCALED_GRADIENT") {
    std::cerr << "LM_CONVERGENCE_ORDER_FAILED\n"; return 1;
  }
  if (staged_lm::classify_lm_no_accepted_trial(5, 9e-5)
      != "CONVERGED_SCALED_GRADIENT") {
    std::cerr << "LM_LEGACY_CLASSIFICATION_FAILED\n"; return 1;
  }
  std::cout << "POSE_LM_CPP_SELF_TEST_PASS\n"; return 0;
}
