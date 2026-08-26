#include "staged_lm/lm.h"

#include "staged_lm/evaluator.h"

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
        x = record.candidate; current_objective = record.candidate_objective;
        ++result.accepted_steps; accepted_this = true;
        damping = update_lm_damping_after_acceptance(damping, record.rho);
      } else {
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

LMRunResult run_lm(
    const std::vector<float>& initial, const LMConfig& config,
    EvaluatorBackend& evaluator) {
  NormalEquationFunction normal_equations =
      [&evaluator](const std::vector<float>& parameters) {
        evaluator.linearize(parameters);
        return evaluator.build_normal_equations();
      };
  ObjectiveFunction objective =
      [&evaluator](const std::vector<float>& parameters) {
        evaluator.forward(parameters);
        return evaluator.evaluate_residual();
      };
  return run_lm(initial, config, normal_equations, objective);
}

std::vector<LMRunResult> run_lm_batch(
    const std::vector<std::vector<float>>& initial,
    const std::vector<LMConfig>& configs,
    const BatchNormalEquationFunction& normal_equations,
    const BatchObjectiveFunction& true_objective) {
  const std::size_t count = initial.size();
  if (count == 0 || configs.size() != count) {
    throw std::runtime_error("LM_BATCH_SIZE_MISMATCH");
  }
  const auto total_started = Clock::now();
  std::vector<LMRunResult> results(count);
  std::vector<std::vector<float>> x = initial;
  std::vector<DenseNormalEquations> equations(count);
  std::vector<double> damping(count, 0.0), objectives(count, 0.0);
  std::vector<int> outer(count, 0);
  std::vector<bool> running(count, true);
  int maximum_trial_count = 0;
  for (const auto& config : configs) {
    maximum_trial_count = std::max(maximum_trial_count, config.max_trials);
  }

  auto append_linearization = [&](int index, DenseNormalEquations built,
                                  int outer_index, double objective,
                                  double lambda) {
    auto& result = results[index];
    const auto& config = configs[index];
    ++result.normal_equation_call_count;
    result.total_jacobian_ms += built.jacobian_ms;
    result.total_normal_equations_ms += built.normal_equations_ms;
    LMLinearizationRecord record;
    record.index = static_cast<int>(result.linearizations.size());
    record.outer = outer_index;
    record.parameters = x[index];
    record.hessian = built.hessian;
    record.gradient = built.gradient;
    record.objective = objective;
    record.lambda_before = lambda;
    record.max_diagonal = -std::numeric_limits<double>::infinity();
    for (int d = 0; d < config.parameter_dimension; ++d) {
      record.max_diagonal = std::max(
          record.max_diagonal,
          built.hessian[d * config.parameter_dimension + d]);
    }
    record.scaled_gradient = scaled_gradient(built);
    result.linearizations.push_back(std::move(record));
    equations[index] = std::move(built);
  };

  std::vector<int> all_indices(count);
  for (std::size_t i = 0; i < count; ++i) {
    all_indices[i] = static_cast<int>(i);
    if (static_cast<int>(initial[i].size()) != configs[i].parameter_dimension) {
      throw std::runtime_error("LM_PARAMETER_DIMENSION_MISMATCH");
    }
    results[i].initial_parameters = initial[i];
  }
  auto initial_equations = normal_equations(all_indices, x);
  if (initial_equations.size() != count) {
    throw std::runtime_error("LM_BATCH_NORMAL_EQUATION_SIZE_MISMATCH");
  }
  for (std::size_t i = 0; i < count; ++i) {
    append_linearization(
        static_cast<int>(i), std::move(initial_equations[i]), -1,
        std::numeric_limits<double>::quiet_NaN(), 0.0);
    damping[i] = initialise_lm_damping(results[i].linearizations[0].max_diagonal);
  }
  auto objective_started = Clock::now();
  objectives = true_objective(all_indices, x);
  const double initial_objective_total_ms = ms(objective_started);
  if (objectives.size() != count) {
    throw std::runtime_error("LM_BATCH_OBJECTIVE_SIZE_MISMATCH");
  }
  for (std::size_t i = 0; i < count; ++i) {
    auto& result = results[i];
    result.initial_objective_ms = initial_objective_total_ms / count;
    result.total_objective_ms += result.initial_objective_ms;
    ++result.objective_call_count;
    result.initial_objective = objectives[i];
    result.initial_lambda = damping[i];
    result.linearizations[0].objective = objectives[i];
    result.linearizations[0].lambda_before = damping[i];
    result.status = "RUNNING";
    result.convergence_branch = "NONE";
  }

  while (std::any_of(running.begin(), running.end(), [](bool value) { return value; })) {
    std::vector<double> objective_before = objectives;
    std::vector<double> last_delta_norm(count, 0.0);
    std::vector<bool> accepted(count, false);
    std::vector<bool> trial_active = running;
    for (int trial = 0; trial < maximum_trial_count; ++trial) {
      std::vector<int> candidate_indices;
      std::vector<std::vector<float>> candidates;
      std::vector<LMTrialRecord> records(count);
      for (std::size_t i = 0; i < count; ++i) {
        if (!trial_active[i] || trial >= configs[i].max_trials) continue;
        auto& result = results[i];
        const auto& config = configs[i];
        LMTrialRecord& record = records[i];
        record.index = static_cast<int>(result.trials.size());
        record.outer = outer[i];
        record.trial = trial;
        record.lambda_before = damping[i];
        for (int d = 0; d < config.parameter_dimension; ++d) {
          const double value = equations[i].hessian[d * config.parameter_dimension + d];
          record.min_diagonal = d == 0 ? value : std::min(record.min_diagonal, value);
          record.max_diagonal = d == 0 ? value : std::max(record.max_diagonal, value);
        }
        record.gradient_norm = vector_norm(equations[i].gradient);
        const auto solve_started = Clock::now();
        record.solve_success = solve_damped(
            equations[i], config.parameter_dimension, damping[i],
            record.delta_double, record.solve_backend);
        result.total_solve_ms += ms(solve_started);
        ++result.solve_call_count;
        if (!record.solve_success) {
          damping[i] = increase_lm_damping(damping[i]);
          ++result.rejected_trials;
          record.lambda_after = damping[i];
          result.trials.push_back(std::move(record));
          continue;
        }
        ++result.candidate_trial_count;
        record.delta_float.resize(config.parameter_dimension);
        record.candidate.resize(config.parameter_dimension);
        for (int d = 0; d < config.parameter_dimension; ++d) {
          record.delta_float[d] = static_cast<float>(record.delta_double[d]);
          record.candidate[d] = x[i][d] + record.delta_float[d];
        }
        candidate_indices.push_back(static_cast<int>(i));
        candidates.push_back(record.candidate);
      }
      if (!candidate_indices.empty()) {
        objective_started = Clock::now();
        const auto candidate_objectives = true_objective(candidate_indices, candidates);
        const double batch_ms = ms(objective_started);
        if (candidate_objectives.size() != candidate_indices.size()) {
          throw std::runtime_error("LM_BATCH_OBJECTIVE_SIZE_MISMATCH");
        }
        for (std::size_t k = 0; k < candidate_indices.size(); ++k) {
          const int i = candidate_indices[k];
          auto& result = results[i];
          auto& record = records[i];
          const auto& config = configs[i];
          result.total_objective_ms += batch_ms / candidate_indices.size();
          ++result.objective_call_count;
          record.candidate_objective = candidate_objectives[k];
          double linear = 0.0, quadratic = 0.0;
          for (int row = 0; row < config.parameter_dimension; ++row) {
            linear += equations[i].gradient[row] * record.delta_double[row];
          }
          if (equations[i].quadratic_form_override) {
            quadratic = equations[i].quadratic_form_override(record.delta_double);
          } else {
            for (int row = 0; row < config.parameter_dimension; ++row) {
              double h_delta = 0.0;
              for (int column = 0; column < config.parameter_dimension; ++column) {
                h_delta += equations[i].hessian[row * config.parameter_dimension + column]
                    * record.delta_double[column];
              }
              quadratic += record.delta_double[row] * h_delta;
            }
          }
          record.predicted_reduction = -linear - 0.5 * quadratic;
          record.actual_reduction = objective_before[i] - record.candidate_objective;
          record.rho = record.predicted_reduction > 0.0
              ? record.actual_reduction / std::max(record.predicted_reduction, 1e-12)
              : 0.0;
          record.accepted = record.predicted_reduction > 0.0 &&
              record.actual_reduction > 0.0 && record.rho > 0.0;
          record.delta_norm = vector_norm(record.delta_double);
          last_delta_norm[i] = record.delta_norm;
          if (record.accepted) {
            x[i] = record.candidate;
            objectives[i] = record.candidate_objective;
            ++result.accepted_steps;
            accepted[i] = true;
            trial_active[i] = false;
            damping[i] = update_lm_damping_after_acceptance(damping[i], record.rho);
          } else {
            damping[i] = increase_lm_damping(damping[i]);
            ++result.rejected_trials;
          }
          record.lambda_after = damping[i];
          result.trials.push_back(std::move(record));
        }
      }
    }

    std::vector<int> accepted_indices;
    std::vector<std::vector<float>> accepted_parameters;
    for (std::size_t i = 0; i < count; ++i) {
      if (!running[i]) continue;
      if (!accepted[i]) {
        results[i].status = classify_lm_no_accepted_trial(
            results[i].accepted_steps, scaled_gradient(equations[i]));
        results[i].convergence_branch = "NO_ACCEPTED_TRIAL";
        running[i] = false;
      } else {
        accepted_indices.push_back(static_cast<int>(i));
        accepted_parameters.push_back(x[i]);
      }
    }
    if (!accepted_indices.empty()) {
      auto next = normal_equations(accepted_indices, accepted_parameters);
      if (next.size() != accepted_indices.size()) {
        throw std::runtime_error("LM_BATCH_NORMAL_EQUATION_SIZE_MISMATCH");
      }
      for (std::size_t k = 0; k < accepted_indices.size(); ++k) {
        const int i = accepted_indices[k];
        DenseNormalEquations next_equations = std::move(next[k]);
        append_linearization(i, next_equations, outer[i], objectives[i], damping[i]);
        const double relative = (objective_before[i] - objectives[i]) /
            std::max(std::abs(objective_before[i]), 1e-16);
        const std::string convergence = check_lm_convergence(
            scaled_gradient(equations[i]), last_delta_norm[i], relative, configs[i]);
        auto& record = results[i].linearizations.back();
        record.relative_objective_change = relative;
        record.step_norm = last_delta_norm[i];
        record.convergence = convergence;
        ++outer[i];
        if (convergence != "NONE") {
          results[i].status = convergence;
          results[i].convergence_branch = convergence;
          running[i] = false;
        } else if (outer[i] >= configs[i].max_accepted_steps) {
          results[i].status = "MAX_ACCEPTED_STEPS";
          results[i].convergence_branch = results[i].status;
          running[i] = false;
        }
      }
    }
  }
  const double total_ms = ms(total_started);
  for (std::size_t i = 0; i < count; ++i) {
    auto& result = results[i];
    result.final_parameters = x[i];
    result.final_objective = objectives[i];
    result.final_lambda = damping[i];
    result.total_lm_ms = total_ms;
    result.total_control_ms = std::max(
        0.0, total_ms - result.total_objective_ms - result.total_jacobian_ms -
            result.total_normal_equations_ms - result.total_solve_ms);
  }
  return results;
}

}  // namespace staged_lm
