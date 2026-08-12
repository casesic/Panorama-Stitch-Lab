#include "staged_lm/lm.h"

#include <iostream>

namespace {

bool run_structured_acceptance_test() {
  using namespace staged_lm;
  const LMConfig config{2, 1, 3, 1e-12, 1e-12, 1e-12};
  int linearizations = 0;
  int solves = 0;
  int objectives = 0;
  int quadratic_forms = 0;
  NormalEquationFunction normal_equations =
      [&](const std::vector<float>&) {
        ++linearizations;
        DenseNormalEquations equations;
        equations.hessian = {2.0, 0.0, 0.0, 3.0};
        equations.gradient = {-2.0, -3.0};
        equations.solve_override = [&solves](
            double, std::vector<double>& delta, std::string& backend) {
          ++solves;
          delta = {1.0, 1.0};
          backend = "schur";
          return true;
        };
        equations.quadratic_form_override = [&quadratic_forms](
            const std::vector<double>& delta) {
          ++quadratic_forms;
          return 2.0 * delta[0] * delta[0] +
                 3.0 * delta[1] * delta[1];
        };
        return equations;
      };
  ObjectiveFunction objective = [&](const std::vector<float>& parameters) {
    ++objectives;
    return static_cast<double>(
        (parameters[0] - 1.0F) * (parameters[0] - 1.0F) +
        (parameters[1] - 1.0F) * (parameters[1] - 1.0F));
  };
  const LMRunResult result =
      run_lm({0.0F, 0.0F}, config, normal_equations, objective);
  return result.accepted_steps == 1 &&
         result.trials[0].solve_backend == "schur" &&
         linearizations == 2 && objectives == 2 && solves == 1 &&
         quadratic_forms == 1 && result.final_parameters[0] == 1.0F &&
         result.final_parameters[1] == 1.0F;
}

bool run_fallback_dispatch_test() {
  using namespace staged_lm;
  const LMConfig config{1, 1, 1, 1e-12, 1e-12, 1e-12};
  NormalEquationFunction normal_equations = [](const std::vector<float>&) {
    DenseNormalEquations equations;
    equations.hessian = {1.0};
    equations.gradient = {-1.0};
    equations.solve_override = [](
        double, std::vector<double>& delta, std::string& backend) {
      delta = {1.0};
      backend = "dense_fallback";
      return true;
    };
    return equations;
  };
  ObjectiveFunction objective = [](const std::vector<float>& parameters) {
    return static_cast<double>(
        (parameters[0] - 1.0F) * (parameters[0] - 1.0F));
  };
  const LMRunResult result =
      run_lm({0.0F}, config, normal_equations, objective);
  return result.accepted_steps == 1 &&
         result.trials[0].solve_backend == "dense_fallback";
}

}  // namespace

int main() {
  if (!run_structured_acceptance_test()) return 1;
  if (!run_fallback_dispatch_test()) return 2;
  std::cout << "POSE_SHAPE_LM_CPP_SELF_TEST=PASS\n";
  return 0;
}
