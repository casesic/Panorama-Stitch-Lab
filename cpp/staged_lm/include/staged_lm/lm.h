#pragma once

#include <functional>
#include <string>
#include <vector>

namespace staged_lm {

class EvaluatorBackend;

struct DenseNormalEquations {
  std::vector<double> hessian;
  std::vector<double> gradient;
  double jacobian_ms = 0.0;
  double normal_equations_ms = 0.0;
  std::function<bool(double, std::vector<double>&, std::string&)> solve_override;
  std::function<double(const std::vector<double>&)> quadratic_form_override;
};

struct LMConfig {
  int parameter_dimension = 0;
  int max_accepted_steps = 0;
  int max_trials = 0;
  double rel_objective_tolerance = 0.0;
  double scaled_gradient_tolerance = 0.0;
  double step_tolerance = 0.0;
};

struct LMLinearizationRecord {
  int index = 0;
  int outer = 0;
  std::vector<float> parameters;
  std::vector<double> hessian;
  std::vector<double> gradient;
  double objective = 0.0;
  double lambda_before = 0.0;
  double max_diagonal = 0.0; // 对角线最大值, 决定初始阻尼
  double scaled_gradient = 0.0;
  double relative_objective_change = 0.0;
  double step_norm = 0.0;
  std::string convergence = "NONE";
};

struct LMTrialRecord {
  int index = 0;
  int outer = 0;
  int trial = 0;
  double lambda_before = 0.0;
  double lambda_after = 0.0;
  bool solve_success = false;
  std::string solve_backend;
  std::vector<double> delta_double;
  std::vector<float> delta_float;
  std::vector<float> candidate;
  double candidate_objective = 0.0;
  double predicted_reduction = 0.0;
  double actual_reduction = 0.0;
  double rho = 0.0;
  bool accepted = false;
  double delta_norm = 0.0;
  double min_diagonal = 0.0;
  double max_diagonal = 0.0;
  double gradient_norm = 0.0;
};

struct LMRunResult {
  std::vector<float> initial_parameters;
  std::vector<float> final_parameters;
  double initial_objective = 0.0;
  double initial_lambda = 0.0;
  double final_objective = 0.0;
  double final_lambda = 0.0;
  std::string status;
  std::string convergence_branch;
  int accepted_steps = 0;
  int rejected_trials = 0;
  int objective_call_count = 0;
  int normal_equation_call_count = 0;
  int solve_call_count = 0;
  int candidate_trial_count = 0;
  std::vector<LMLinearizationRecord> linearizations;
  std::vector<LMTrialRecord> trials;
  double initial_objective_ms = 0.0;
  double total_objective_ms = 0.0;
  double total_jacobian_ms = 0.0;
  double total_normal_equations_ms = 0.0;
  double total_solve_ms = 0.0;
  double total_control_ms = 0.0;
  double total_lm_ms = 0.0;
  std::size_t batch_h2d_count = 0;
  std::size_t batch_h2d_bytes = 0;
  std::size_t batch_d2h_count = 0;
  std::size_t batch_d2h_bytes = 0;
};

using NormalEquationFunction = std::function<DenseNormalEquations(const std::vector<float>&)>;
using ObjectiveFunction = std::function<double(const std::vector<float>&)>;
using BatchNormalEquationFunction = std::function<std::vector<DenseNormalEquations>(
    const std::vector<int>&, const std::vector<std::vector<float>>&) >;
using BatchObjectiveFunction = std::function<std::vector<double>(
    const std::vector<int>&, const std::vector<std::vector<float>>&) >;

double initialise_lm_damping(double max_diagonal);
double increase_lm_damping(double damping);
double update_lm_damping_after_acceptance(double damping, double rho);
std::string check_lm_convergence(
    double scaled_gradient, double step_norm, double relative_objective,
    const LMConfig& config);
std::string classify_lm_no_accepted_trial(int accepted_steps, double scaled_gradient);

LMRunResult run_lm(
    const std::vector<float>& initial_parameters, const LMConfig& config,
    const NormalEquationFunction& normal_equations,
    const ObjectiveFunction& true_objective);

LMRunResult run_lm(
    const std::vector<float>& initial_parameters, const LMConfig& config,
    EvaluatorBackend& evaluator);

// Run independent LM problems in lockstep.  The controller and each problem's
// damping/acceptance state remain on CPU; only evaluator calls are grouped.
std::vector<LMRunResult> run_lm_batch(
    const std::vector<std::vector<float>>& initial_parameters,
    const std::vector<LMConfig>& configs,
    const BatchNormalEquationFunction& normal_equations,
    const BatchObjectiveFunction& true_objective);

}  // namespace staged_lm
