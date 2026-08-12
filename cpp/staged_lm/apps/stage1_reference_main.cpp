#include "staged_lm/binary_io.h"
#include "staged_lm/stage1.h"

#include <Eigen/Core>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path argument_value(
    int argc, char** argv, const std::string& requested) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == requested) {
      return argv[index + 1];
    }
  }
  throw std::runtime_error("missing required argument: " + requested);
}

float float_value(
    const std::map<std::string, std::string>& metadata, const std::string& key) {
  return std::stof(metadata.at(key));
}

int int_value(
    const std::map<std::string, std::string>& metadata, const std::string& key) {
  return std::stoi(metadata.at(key));
}

Eigen::Vector3f vector3(const std::vector<float>& values) {
  return Eigen::Vector3f(values.at(0), values.at(1), values.at(2));
}

std::vector<float> matrix3_row_major(const Eigen::Matrix3f& matrix) {
  std::vector<float> values(9);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      values[row * 3 + column] = matrix(row, column);
    }
  }
  return values;
}

template <typename MatrixType>
std::vector<double> matrix12_row_major(const MatrixType& matrix) {
  std::vector<double> values(144);
  for (int row = 0; row < 12; ++row) {
    for (int column = 0; column < 12; ++column) {
      values[row * 12 + column] = matrix(row, column);
    }
  }
  return values;
}

staged_lm::Stage1CameraInput load_camera(
    const std::filesystem::path& fixture,
    const std::map<std::string, std::string>& metadata,
    const std::vector<float>& parameters,
    const std::string& camera) {
  const int count = int_value(metadata, camera + "_count");
  const int offset = camera == "cam2" ? staged_lm::kCam2Offset : staged_lm::kCam4Offset;
  staged_lm::Stage1CameraInput input;
  input.camera = camera;
  input.orientation = Eigen::Vector3f(
      parameters[offset], parameters[offset + 1], parameters[offset + 2]);
  input.translation = Eigen::Vector3f(
      parameters[offset + 3], parameters[offset + 4], parameters[offset + 5]);
  input.local_vertices = staged_lm::read_float32_file(
      fixture / (camera + "_local_vertices.f32"), count * 3);
  input.rotation_center = vector3(staged_lm::read_float32_file(
      fixture / (camera + "_rotation_center.f32"), 3));
  input.targets = staged_lm::read_float32_file(
      fixture / (camera + "_targets.f32"), count * 2);
  input.point_count = count;
  input.image_width = int_value(metadata, "image_width");
  input.image_height = int_value(metadata, "image_height");
  input.irls_epsilon = float_value(metadata, "irls_epsilon");
  input.alpha_per_point = float_value(metadata, camera + "_alpha_per_point");
  return input;
}

void write_camera(
    const std::filesystem::path& output, const std::string& camera,
    const staged_lm::Stage1CameraResult& result) {
  using staged_lm::write_float32_file;
  write_float32_file(output / (camera + "_R.f32"), matrix3_row_major(result.rotation));
  write_float32_file(output / (camera + "_Jr.f32"), matrix3_row_major(result.right_jacobian));
  write_float32_file(output / (camera + "_points.f32"), result.points);
  write_float32_file(output / (camera + "_uv.f32"), result.uv);
  write_float32_file(output / (camera + "_J_proj.f32"), result.projection_jacobian);
  write_float32_file(output / (camera + "_dpoint_dorient.f32"), result.point_orientation_jacobian);
  write_float32_file(output / (camera + "_J_orientation.f32"), result.orientation_jacobian);
  write_float32_file(output / (camera + "_J_translation.f32"), result.translation_jacobian);
  write_float32_file(output / (camera + "_J_frame.f32"), result.frame_jacobian);
  write_float32_file(output / (camera + "_raw_residual.f32"), result.raw_residual);
  write_float32_file(output / (camera + "_point_weights.f32"), result.point_weights);
  write_float32_file(output / (camera + "_row_scales.f32"), result.row_scales);
  write_float32_file(output / (camera + "_weighted_residual.f32"), result.weighted_residual);
  write_float32_file(output / (camera + "_weighted_J.f32"), result.weighted_jacobian);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto total_start = std::chrono::steady_clock::now();
    const std::filesystem::path fixture = argument_value(argc, argv, "--fixture");
    const std::filesystem::path output = argument_value(argc, argv, "--output");
    const auto metadata = staged_lm::read_key_value_file(fixture / "fixture_meta.txt");
    if (metadata.at("schema_version") != "1"
        || metadata.at("reprojection_objective") != "charbonnier_irls") {
      throw std::runtime_error("unsupported Stage1 fixture schema or objective");
    }
    const auto parameters = staged_lm::read_float32_file(
        fixture / "parameters.f32", staged_lm::kStage1ParameterDimension);
    const auto cam2 = load_camera(fixture, metadata, parameters, "cam2");
    const auto cam4 = load_camera(fixture, metadata, parameters, "cam4");
    const staged_lm::Stage1FrameResult result = staged_lm::evaluate_stage1_frame(cam2, cam4);

    std::filesystem::create_directories(output);
    write_camera(output, "cam2", result.cam2);
    write_camera(output, "cam4", result.cam4);
    staged_lm::write_float64_file(output / "H_frame.f64", matrix12_row_major(result.hessian));
    staged_lm::write_float64_file(
        output / "g_frame.f64",
        std::vector<double>(result.gradient.data(), result.gradient.data() + 12));

    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_start).count();
    std::map<std::string, std::string> output_metadata = {
        {"schema_version", "1"}, {"frame", metadata.at("frame")},
        {"cam2_count", metadata.at("cam2_count")},
        {"cam4_count", metadata.at("cam4_count")},
        {"parameter_dim", "12"}, {"float_intermediates", "float32"},
        {"normal_equations_dtype", "float64"},
        {"row_order", "cam2_then_cam4"},
        {"normal_equations_reduction", "concatenate_then_single_eigen_gemm"},
    };
    auto add_timing = [&output_metadata](const std::string& name, double value) {
      output_metadata[name] = std::to_string(value);
    };
    add_timing("cam2_so3_ms", result.cam2.so3_ms);
    add_timing("cam2_point_transform_ms", result.cam2.point_transform_ms);
    add_timing("cam2_erp_ms", result.cam2.erp_ms);
    add_timing("cam2_rigid_jacobian_ms", result.cam2.rigid_jacobian_ms);
    add_timing("cam2_irls_ms", result.cam2.irls_ms);
    add_timing("cam4_so3_ms", result.cam4.so3_ms);
    add_timing("cam4_point_transform_ms", result.cam4.point_transform_ms);
    add_timing("cam4_erp_ms", result.cam4.erp_ms);
    add_timing("cam4_rigid_jacobian_ms", result.cam4.rigid_jacobian_ms);
    add_timing("cam4_irls_ms", result.cam4.irls_ms);
    add_timing("normal_equations_ms", result.normal_equations_ms);
    add_timing("total_including_io_ms", total_ms);
    staged_lm::write_key_value_file(output / "cpp_output_meta.txt", output_metadata);
    std::cout << "STAGE1_CPP_REFERENCE_WRITTEN output=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "STAGE1_CPP_REFERENCE_ERROR: " << error.what() << '\n';
    return 1;
  }
}
