#include "staged_lm/binary_io.h"

#include <fstream>
#include <stdexcept>

namespace staged_lm {

std::map<std::string, std::string> read_key_value_file(
    const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open metadata file: " + path.string());
  }
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos || separator == 0) {
      throw std::runtime_error("invalid metadata line: " + line);
    }
    values.emplace(line.substr(0, separator), line.substr(separator + 1));
  }
  return values;
}

std::vector<float> read_float32_file(
    const std::filesystem::path& path, std::size_t expected_count) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing fixture file: " + path.string());
  }
  const std::uintmax_t expected_bytes = expected_count * sizeof(float);
  const std::uintmax_t actual_bytes = std::filesystem::file_size(path);
  if (actual_bytes != expected_bytes) {
    throw std::runtime_error(
        "fixture byte count mismatch: " + path.string() + " expected="
        + std::to_string(expected_bytes) + " actual=" + std::to_string(actual_bytes));
  }
  std::vector<float> values(expected_count);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(expected_bytes))) {
    throw std::runtime_error("failed reading fixture file: " + path.string());
  }
  return values;
}

std::vector<std::int64_t> read_int64_file(
    const std::filesystem::path& path, std::size_t expected_count) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing fixture file: " + path.string());
  }
  const std::uintmax_t expected_bytes = expected_count * sizeof(std::int64_t);
  const std::uintmax_t actual_bytes = std::filesystem::file_size(path);
  if (actual_bytes != expected_bytes) {
    throw std::runtime_error(
        "fixture byte count mismatch: " + path.string() + " expected="
        + std::to_string(expected_bytes) + " actual=" + std::to_string(actual_bytes));
  }
  std::vector<std::int64_t> values(expected_count);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(expected_bytes))) {
    throw std::runtime_error("failed reading fixture file: " + path.string());
  }
  return values;
}

std::vector<std::int32_t> read_int32_file(
    const std::filesystem::path& path, std::size_t expected_count) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing fixture file: " + path.string());
  }
  const std::uintmax_t expected_bytes = expected_count * sizeof(std::int32_t);
  if (std::filesystem::file_size(path) != expected_bytes) {
    throw std::runtime_error("fixture byte count mismatch: " + path.string());
  }
  std::vector<std::int32_t> values(expected_count);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(expected_bytes))) {
    throw std::runtime_error("failed reading fixture file: " + path.string());
  }
  return values;
}

void write_float32_file(
    const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream) {
    throw std::runtime_error("failed writing output file: " + path.string());
  }
}

void write_float64_file(
    const std::filesystem::path& path, const std::vector<double>& values) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(double)));
  if (!stream) {
    throw std::runtime_error("failed writing output file: " + path.string());
  }
}

void write_key_value_file(
    const std::filesystem::path& path,
    const std::map<std::string, std::string>& values) {
  std::ofstream stream(path);
  for (const auto& [key, value] : values) {
    stream << key << '=' << value << '\n';
  }
  if (!stream) {
    throw std::runtime_error("failed writing metadata file: " + path.string());
  }
}

}  // namespace staged_lm
