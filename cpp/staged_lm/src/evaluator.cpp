#include "staged_lm/evaluator.h"

#include <stdexcept>
#include <string>

namespace staged_lm {

const char* evaluator_backend_name(EvaluatorBackendKind backend) {
  return backend == EvaluatorBackendKind::kCuda ? "cuda" : "cpu";
}

EvaluatorBackendKind parse_evaluator_backend(const char* value) {
  const std::string name(value == nullptr ? "" : value);
  if (name == "cpu") return EvaluatorBackendKind::kCpu;
  if (name == "cuda") return EvaluatorBackendKind::kCuda;
  throw std::runtime_error("unsupported evaluator backend: " + name);
}

}  // namespace staged_lm
