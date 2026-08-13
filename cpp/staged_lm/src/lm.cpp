#include "staged_lm/lm.h"
#include "staged_lm/performance.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace staged_lm {
namespace {
// Generic LM controls damping, candidate evaluation, and accept/reject; stage
// wrappers provide the objective and normal-equation callbacks.
using Clock = std::chrono::steady_clock;
constexpr double kLambdaMinimum = 1e-10;
constexpr double kLambdaMaximum = 1e12;

double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double vector_norm(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values) sum += value * value;
  return std::sqrt(sum);
}

double hessian_norm(const std::vector<double>& values) { return vector_norm(values); }

double scaled_gradient(const DenseNormalEquations& equations) {
  return vector_norm(equations.gradient) / std::max(1.0, hessian_norm(equations.hessian));
}

bool solve_damped(
    const DenseNormalEquations& equations, int dimension, double damping,
    std::vector<double>& delta, std::string& backend) {
  if (equations.solve_override) {
    return equations.solve_override(damping, delta, backend);
  }
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const Matrix> hessian(equations.hessian.data(), dimension, dimension);
  Matrix matrix = hessian;
  matrix.diagonal().array() += damping;
  Eigen::Map<const Eigen::VectorXd> gradient(equations.gradient.data(), dimension);
  Eigen::LLT<Matrix> cholesky(matrix);
  if (cholesky.info() == Eigen::Success) {
    const Eigen::VectorXd solution = cholesky.solve(-gradient);
    if (cholesky.info() == Eigen::Success && solution.allFinite()) {
      delta.assign(solution.data(), solution.data() + dimension);
      backend = "cholesky";
      return true;
    }
  }
  const Eigen::PartialPivLU<Matrix> lu(matrix);
  const Eigen::VectorXd solution = lu.solve(-gradient);
  backend = "dense_solve";
  if (!solution.allFinite()) return false;
  delta.assign(solution.data(), solution.data() + dimension);
  return true;
}

}  // namespace

double initialise_lm_damping(double max_diagonal) {
  if (!std::isfinite(max_diagonal) || max_diagonal <= 0.0) {
    throw std::runtime_error("INVALID_INITIAL_LM_DAMPING");
  }
  return max_diagonal * 1e-3;
}
double increase_lm_damping(double damping) { return std::min(damping * 10.0, kLambdaMaximum); }
double update_lm_damping_after_acceptance(double damping, double rho) {
  if (rho > 0.75) return std::max(damping / 3.0, kLambdaMinimum);
  if (rho < 0.25) return std::min(damping * 3.0, kLambdaMaximum);
  return damping;
}
std::string check_lm_convergence(
    double gradient, double step, double relative, const LMConfig& config) {
  if (gradient < config.scaled_gradient_tolerance) return "CONVERGED_SCALED_GRADIENT";
  if (step < config.step_tolerance) return "CONVERGED_STEP";
  if (std::abs(relative) < config.rel_objective_tolerance) return "CONVERGED_REL_OBJECTIVE";
  return "NONE";
}
std::string classify_lm_no_accepted_trial(int accepted_steps, double gradient) {
  return accepted_steps > 4 && gradient < 1e-4
             ? "CONVERGED_SCALED_GRADIENT" : "LM_NO_ACCEPTED_TRIAL";
}

LMRunResult run_lm(
    const std::vector<float>& initial, const LMConfig& config,
    const NormalEquationFunction& normal_equations,
    const ObjectiveFunction& true_objective) {
  if (static_cast<int>(initial.size()) != config.parameter_dimension) {
    throw std::runtime_error("LM_PARAMETER_DIMENSION_MISMATCH");
  }
  const auto total_started = Clock::now();
  LMRunResult result;
  result.initial_parameters = initial;
  std::vector<float> x = initial;

  auto evaluate_equations = [&](int outer, double objective, double lambda) {
    record_lm_state_call(outer, -1, "linearization", "current", x);
    DenseNormalEquations equations = normal_equations(x);
    ++result.normal_equation_call_count;
    result.total_jacobian_ms += equations.jacobian_ms;
    result.total_normal_equations_ms += equations.normal_equations_ms;
    LMLinearizationRecord record;
    record.index = static_cast<int>(result.linearizations.size());
    record.outer = outer; record.parameters = x;
    record.hessian = equations.hessian; record.gradient = equations.gradient;
    record.objective = objective; record.lambda_before = lambda;
    double diagonal_max = -std::numeric_limits<double>::infinity();
    for (int index = 0; index < config.parameter_dimension; ++index) {
      diagonal_max = std::max(
          diagonal_max, equations.hessian[index * config.parameter_dimension + index]);
    }
    record.max_diagonal = diagonal_max;
    record.scaled_gradient = scaled_gradient(equations);
    result.linearizations.push_back(std::move(record));
    return equations;
  };

  DenseNormalEquations equations = evaluate_equations(-1, std::numeric_limits<double>::quiet_NaN(), 0.0);
  double damping = initialise_lm_damping(result.linearizations[0].max_diagonal);
  auto started = Clock::now();
  record_lm_state_call(-1, -1, "objective", "initial_current", x);
  double current_objective = true_objective(x);
  result.initial_objective_ms = ms(started);
  result.total_objective_ms += result.initial_objective_ms;
  ++result.objective_call_count;
  result.initial_objective = current_objective;
  result.initial_lambda = damping;
  result.linearizations[0].objective = current_objective;
  result.linearizations[0].lambda_before = damping;
  result.status = "RUNNING";
  result.convergence_branch = "NONE";

  for (int outer = 0; outer < config.max_accepted_steps; ++outer) {
    const double objective_before = current_objective;
    bool accepted_this = false;
    double last_delta_norm = 0.0;
    for (int trial = 0; trial < config.max_trials; ++trial) {
      LMTrialRecord record;
      record.index = static_cast<int>(result.trials.size());
      record.outer = outer; record.trial = trial; record.lambda_before = damping;
      for (int index = 0; index < config.parameter_dimension; ++index) {
        const double value = equations.hessian[index * config.parameter_dimension + index];
        record.min_diagonal = index == 0 ? value : std::min(record.min_diagonal, value);
        record.max_diagonal = index == 0 ? value : std::max(record.max_diagonal, value);
      }
      record.gradient_norm = vector_norm(equations.gradient);
      started = Clock::now();
      record.solve_success = solve_damped(
          equations, config.parameter_dimension, damping,
          record.delta_double, record.solve_backend);
      result.total_solve_ms += ms(started);
      ++result.solve_call_count;
      if (!record.solve_success) {
        damping = increase_lm_damping(damping);
        ++result.rejected_trials;
        record.lambda_after = damping;
        result.trials.push_back(std::move(record));
        continue;
      }
      ++result.candidate_trial_count;
      record.delta_float.resize(config.parameter_dimension);
      record.candidate.resize(config.parameter_dimension);
      for (int index = 0; index < config.parameter_dimension; ++index) {
        record.delta_float[index] = static_cast<float>(record.delta_double[index]);
        record.candidate[index] = x[index] + record.delta_float[index];
      }
      started = Clock::now();
      record_lm_state_call(outer, trial, "objective", "candidate_pending",
                           record.candidate);
      record.candidate_objective = true_objective(record.candidate);
      result.total_objective_ms += ms(started);
      ++result.objective_call_count;
      double linear = 0.0;
      for (int row = 0; row < config.parameter_dimension; ++row) {
        linear += equations.gradient[row] * record.delta_double[row];
      }
      double quadratic = 0.0;
      if (equations.quadratic_form_override) {
        quadratic = equations.quadratic_form_override(record.delta_double);
      } else {
        for (int row = 0; row < config.parameter_dimension; ++row) {
          double h_delta = 0.0;
          for (int column = 0; column < config.parameter_dimension; ++column) {
            h_delta += equations.hessian[row * config.parameter_dimension + column]
                       * record.delta_double[column];
          }
          quadratic += record.delta_double[row] * h_delta;
        }
      }
      record.predicted_reduction = -linear - 0.5 * quadratic;
      record.actual_reduction = objective_before - record.candidate_objective;
      record.rho = record.predicted_reduction > 0.0
          ? record.actual_reduction / std::max(record.predicted_reduction, 1e-12) : 0.0;
      record.accepted = record.predicted_reduction > 0.0
                        && record.actual_reduction > 0.0 && record.rho > 0.0;
      record.delta_norm = vector_norm(record.delta_double);
      last_delta_norm = record.delta_norm;
      if (record.accepted) {
        update_last_lm_objective_disposition("accepted");
        x = record.candidate; current_objective = record.candidate_objective;
        ++result.accepted_steps; accepted_this = true;
        damping = update_lm_damping_after_acceptance(damping, record.rho);
      } else {
        update_last_lm_objective_disposition("rejected");
        damping = increase_lm_damping(damping); ++result.rejected_trials;
      }
      record.lambda_after = damping;
      result.trials.push_back(std::move(record));
      if (accepted_this) break;
    }
    if (!accepted_this) {
      result.status = classify_lm_no_accepted_trial(
          result.accepted_steps, scaled_gradient(equations));
      result.convergence_branch = "NO_ACCEPTED_TRIAL";
      break;
    }
    DenseNormalEquations next = evaluate_equations(outer, current_objective, damping);
    const double relative = (objective_before - current_objective)
                            / std::max(std::abs(objective_before), 1e-16);
    const std::string convergence = check_lm_convergence(
        scaled_gradient(next), last_delta_norm, relative, config);
    result.linearizations.back().relative_objective_change = relative;
    result.linearizations.back().step_norm = last_delta_norm;
    result.linearizations.back().convergence = convergence;
    if (convergence != "NONE") {
      result.status = convergence; result.convergence_branch = convergence; break;
    }
    equations = std::move(next);
  }
  if (result.status == "RUNNING") {
    result.status = "MAX_ACCEPTED_STEPS";
    result.convergence_branch = result.status;
  }
  result.final_parameters = x; result.final_objective = current_objective;
  result.final_lambda = damping;
  result.total_lm_ms = ms(total_started);
  result.total_control_ms = std::max(
      0.0, result.total_lm_ms - result.total_objective_ms
             - result.total_jacobian_ms - result.total_normal_equations_ms
             - result.total_solve_ms);
  return result;
}

}  // namespace staged_lm
