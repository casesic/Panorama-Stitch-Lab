#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace staged_lm {

std::map<std::string, std::string> read_key_value_file(
    const std::filesystem::path& path);
std::vector<float> read_float32_file(
    const std::filesystem::path& path, std::size_t expected_count);
std::vector<std::int64_t> read_int64_file(
    const std::filesystem::path& path, std::size_t expected_count);
std::vector<std::int32_t> read_int32_file(
    const std::filesystem::path& path, std::size_t expected_count);
void write_float32_file(
    const std::filesystem::path& path, const std::vector<float>& values);
void write_float64_file(
    const std::filesystem::path& path, const std::vector<double>& values);
void write_key_value_file(
    const std::filesystem::path& path,
    const std::map<std::string, std::string>& values);

}  // namespace staged_lm
