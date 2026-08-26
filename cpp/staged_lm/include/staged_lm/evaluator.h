#pragma once

#include "staged_lm/lm.h"

#include <vector>

namespace staged_lm {

enum class EvaluatorBackendKind { kCpu, kCuda };

const char* evaluator_backend_name(EvaluatorBackendKind backend);
EvaluatorBackendKind parse_evaluator_backend(const char* value);

// Device-independent boundary between the CPU LM controller and stage
// evaluation. A backend owns any value/linearization state needed between
// each paired call.
class EvaluatorBackend {
 public:
  virtual ~EvaluatorBackend() = default;

  virtual void forward(const std::vector<float>& parameters) = 0;
  virtual double evaluate_residual() = 0;
  virtual void linearize(const std::vector<float>& parameters) = 0;
  virtual DenseNormalEquations build_normal_equations() = 0;
};

}  // namespace staged_lm
