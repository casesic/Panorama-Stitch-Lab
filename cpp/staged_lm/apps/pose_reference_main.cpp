#include "staged_lm/binary_io.h"
#include "staged_lm/pose_forward.h"
#include "staged_lm/pose_jacobian.h"
#include "staged_lm/pose_normal_equations.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path argument_value(int argc, char** argv, const std::string& name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      return argv[index + 1];
    }
  }
  throw std::runtime_error("missing required argument: " + name);
}

int integer(const std::map<std::string, std::string>& meta, const std::string& key) {
  return std::stoi(meta.at(key));
}

float real(const std::map<std::string, std::string>& meta, const std::string& key) {
  return std::stof(meta.at(key));
}

staged_lm::PoseCameraInput load_camera(
    const std::filesystem::path& fixture,
    const std::map<std::string, std::string>& meta,
    const std::vector<float>& parameters,
    const std::string& camera) {
  const int count = integer(meta, camera + "_count");
  staged_lm::PoseCameraInput input;
  input.camera = camera;
  input.parameters = parameters;
  input.left_hand_pose = staged_lm::read_float32_file(
      fixture / (camera + "_left_hand_pose.f32"), 45);
  input.right_hand_pose = staged_lm::read_float32_file(
      fixture / (camera + "_right_hand_pose.f32"), 45);
  input.scale = staged_lm::read_float32_file(
      fixture / (camera + "_scale.f32"), 1).at(0);
  input.selected_v_shaped = staged_lm::read_float32_file(
      fixture / (camera + "_selected_v_shaped.f32"), count * 3);
  input.selected_posedirs = staged_lm::read_float32_file(
      fixture / (camera + "_selected_posedirs.f32"),
      staged_lm::kPoseFeatureDimension * count * 3);
  input.selected_lbs_weights = staged_lm::read_float32_file(
      fixture / (camera + "_selected_lbs_weights.f32"),
      count * staged_lm::kPoseJointCount);
  input.targets = staged_lm::read_float32_file(
      fixture / (camera + "_targets.f32"), count * 2);
  input.point_count = count;
  input.image_width = integer(meta, "image_width");
  input.image_height = integer(meta, "image_height");
  return input;
}

void write_camera(
    const std::filesystem::path& output, const std::string& camera,
    const staged_lm::PoseCameraJacobianResult& jacobian_result) {
  const staged_lm::PoseCameraResult& result = jacobian_result.value;
  const std::pair<const char*, const std::vector<float>*> values[] = {
      {"axis_angles", &result.axis_angles}, {"rotations", &result.rotations},
      {"pose_feature", &result.pose_feature}, {"pose_offsets", &result.pose_offsets},
      {"v_posed", &result.v_posed}, {"relative_joints", &result.relative_joints},
      {"local_transforms", &result.local_transforms},
      {"global_transforms", &result.global_transforms},
      {"rest_transforms", &result.rest_transforms},
      {"relative_transforms", &result.relative_transforms},
      {"selected_transforms", &result.selected_transforms},
      {"skinned_vertices", &result.skinned_vertices}, {"vertices", &result.vertices},
      {"uv", &result.uv}, {"raw_residual", &result.raw_residual},
  };
  for (const auto& [name, data] : values) {
    staged_lm::write_float32_file(output / (camera + "_" + name + ".f32"), *data);
  }
  const std::pair<const char*, const std::vector<float>*> derivatives[] = {
      {"variable_right_jacobians", &jacobian_result.variable_right_jacobians},
      {"d_rotation", &jacobian_result.d_rotation},
      {"d_pose_feature", &jacobian_result.d_pose_feature},
      {"d_pose_offsets", &jacobian_result.d_pose_offsets},
      {"d_v_posed", &jacobian_result.d_v_posed},
      {"d_local_transforms", &jacobian_result.d_local_transforms},
      {"d_global_transforms", &jacobian_result.d_global_transforms},
      {"d_relative_transforms", &jacobian_result.d_relative_transforms},
      {"d_selected_transforms", &jacobian_result.d_selected_transforms},
      {"d_skin_from_transform", &jacobian_result.d_skin_from_transform},
      {"d_skin_from_vertex", &jacobian_result.d_skin_from_vertex},
      {"d_skinned_vertices", &jacobian_result.d_skinned_vertices},
      {"d_vertices_rotpose", &jacobian_result.d_vertices_rotpose},
      {"J_proj", &jacobian_result.projection_jacobian},
      {"J_uv_rotpose", &jacobian_result.uv_rotpose_jacobian},
      {"J_translation", &jacobian_result.translation_jacobian},
      {"J_active", &jacobian_result.active_jacobian},
      {"J_frame", &jacobian_result.frame_jacobian},
  };
  for (const auto& [name, data] : derivatives) {
    staged_lm::write_float32_file(output / (camera + "_" + name + ".f32"), *data);
  }
}

void write_camera_normal_equations(
    const std::filesystem::path& output, const std::string& camera,
    const staged_lm::PoseCameraNormalEquations& result) {
  staged_lm::write_float32_file(output / (camera + "_point_weights.f32"), result.point_weights);
  staged_lm::write_float32_file(output / (camera + "_row_scales.f32"), result.row_scales);
  staged_lm::write_float32_file(
      output / (camera + "_weighted_residual.f32"), result.weighted_residual);
  staged_lm::write_float32_file(
      output / (camera + "_weighted_J_active.f32"), result.weighted_jacobian_active);
  staged_lm::write_float64_file(output / (camera + "_H_active.f64"), result.hessian_active);
  staged_lm::write_float64_file(output / (camera + "_g_active.f64"), result.gradient_active);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto fixture = argument_value(argc, argv, "--fixture");
    const auto output = argument_value(argc, argv, "--output");
    const auto meta = staged_lm::read_key_value_file(fixture / "fixture_meta.txt");
    if (meta.at("schema_version") != "1" || integer(meta, "parameter_dim") != 75
        || integer(meta, "joint_count") != staged_lm::kPoseJointCount) {
      throw std::runtime_error("unsupported Pose fixture schema");
    }
    staged_lm::PoseModelInput model;
    model.joints = staged_lm::read_float32_file(
        fixture / "joints.f32", staged_lm::kPoseJointCount * 3);
    model.parents = staged_lm::read_int64_file(
        fixture / "parents.i64", staged_lm::kPoseJointCount);
    const auto parameters = staged_lm::read_float32_file(
        fixture / "parameters.f32", staged_lm::kPoseParameterDimension);
    const auto cam2 = staged_lm::evaluate_pose_camera_with_jacobian(
        model, load_camera(fixture, meta, parameters, "cam2"));
    const auto cam4 = staged_lm::evaluate_pose_camera_with_jacobian(
        model, load_camera(fixture, meta, parameters, "cam4"));
    staged_lm::PoseNormalEquationConfig normal_config;
    normal_config.charbonnier_eps = real(meta, "charbonnier_eps");
    normal_config.cam2_alpha_per_point = real(meta, "cam2_alpha_per_point");
    normal_config.cam4_alpha_per_point = real(meta, "cam4_alpha_per_point");
    normal_config.pose_anchor_scale = real(meta, "pose_anchor_scale");
    const auto normal = staged_lm::build_pose_normal_equations(
        cam2, cam4, parameters,
        staged_lm::read_float32_file(fixture / "body_pose_anchor.f32", 63),
        normal_config);
    std::filesystem::create_directories(output);
    write_camera(output, "cam2", cam2);
    write_camera(output, "cam4", cam4);
    write_camera_normal_equations(output, "cam2", normal.cam2);
    write_camera_normal_equations(output, "cam4", normal.cam4);
    staged_lm::write_float64_file(output / "H_data_frame.f64", normal.hessian_data_frame);
    staged_lm::write_float64_file(output / "g_data_frame.f64", normal.gradient_data_frame);
    staged_lm::write_float32_file(output / "pose_anchor_delta.f32", normal.pose_anchor_delta);
    staged_lm::write_float32_file(
        output / "pose_anchor_residual.f32", normal.pose_anchor_residual);
    staged_lm::write_float64_file(
        output / "H_pose_anchor.f64", normal.hessian_pose_anchor);
    staged_lm::write_float64_file(
        output / "g_pose_anchor.f64", normal.gradient_pose_anchor);
    staged_lm::write_float64_file(output / "H_frame.f64", normal.hessian_frame);
    staged_lm::write_float64_file(output / "g_frame.f64", normal.gradient_frame);
    std::map<std::string, std::string> output_meta = {
        {"schema_version", "1"}, {"frame", meta.at("frame")},
        {"cam2_count", meta.at("cam2_count")}, {"cam4_count", meta.at("cam4_count")},
        {"parameter_dim", "75"}, {"joint_count", "55"},
        {"matrix_storage", "canonical_row_major"},
        {"float_dtype", "float32"},
    };
    auto timings = [&output_meta](const std::string& camera,
                                  const staged_lm::PoseCameraJacobianResult& result) {
      const staged_lm::PoseCameraResult& value = result.value;
      output_meta[camera + "_rotations_ms"] = std::to_string(value.rotations_ms);
      output_meta[camera + "_pose_offsets_ms"] = std::to_string(value.pose_offsets_ms);
      output_meta[camera + "_fk_ms"] = std::to_string(value.fk_ms);
      output_meta[camera + "_lbs_ms"] = std::to_string(value.lbs_ms);
      output_meta[camera + "_erp_ms"] = std::to_string(value.erp_ms);
      output_meta[camera + "_total_value_forward_ms"] =
          std::to_string(value.total_value_forward_ms);
      output_meta[camera + "_rotation_derivative_ms"] =
          std::to_string(result.rotation_derivative_ms);
      output_meta[camera + "_pose_feature_derivative_ms"] =
          std::to_string(result.pose_feature_derivative_ms);
      output_meta[camera + "_pose_offset_derivative_ms"] =
          std::to_string(result.pose_offset_derivative_ms);
      output_meta[camera + "_fk_derivative_ms"] = std::to_string(result.fk_derivative_ms);
      output_meta[camera + "_lbs_derivative_ms"] = std::to_string(result.lbs_derivative_ms);
      output_meta[camera + "_erp_jacobian_chain_ms"] =
          std::to_string(result.erp_jacobian_chain_ms);
      output_meta[camera + "_assemble_active_ms"] =
          std::to_string(result.assemble_active_ms);
      output_meta[camera + "_total_jacobian_ms"] = std::to_string(result.total_jacobian_ms);
    };
    timings("cam2", cam2);
    timings("cam4", cam4);
    auto ne_timings = [&output_meta](
        const std::string& camera,
        const staged_lm::PoseCameraNormalEquations& result) {
      output_meta[camera + "_ne_irls_ms"] = std::to_string(result.irls_ms);
      output_meta[camera + "_ne_row_scaling_ms"] = std::to_string(result.row_scaling_ms);
      output_meta[camera + "_ne_fp64_cast_ms"] = std::to_string(result.fp64_cast_ms);
      output_meta[camera + "_ne_active_hg_ms"] =
          std::to_string(result.active_normal_equations_ms);
    };
    ne_timings("cam2", normal.cam2);
    ne_timings("cam4", normal.cam4);
    output_meta["ne_scatter_ms"] = std::to_string(normal.scatter_ms);
    output_meta["ne_anchor_ms"] = std::to_string(normal.anchor_ms);
    output_meta["ne_total_ms"] = std::to_string(normal.total_normal_equations_ms);
    output_meta["ne_dtype_before_hg"] = "float32";
    output_meta["ne_dtype_after_cast"] = "float64";
    const std::map<std::string, std::string> shapes = {
        {"shape.variable_right_jacobians", "22,3,3"},
        {"axes.variable_right_jacobians", "joint,row,col"},
        {"shape.d_rotation", "55,3,3,66"},
        {"axes.d_rotation", "joint,row,col,param"},
        {"shape.d_pose_feature", "486,66"},
        {"axes.d_pose_feature", "feature,param"},
        {"shape.d_pose_offsets", "N,3,66"},
        {"axes.d_pose_offsets", "vertex,xyz,param"},
        {"shape.d_v_posed", "N,3,66"},
        {"axes.d_v_posed", "vertex,xyz,param"},
        {"shape.d_local_transforms", "55,4,4,66"},
        {"shape.d_global_transforms", "55,4,4,66"},
        {"shape.d_relative_transforms", "55,4,4,66"},
        {"axes.d_transforms", "joint,row,col,param"},
        {"shape.d_selected_transforms", "N,4,4,66"},
        {"axes.d_selected_transforms", "vertex,row,col,param"},
        {"shape.d_skin_from_transform", "N,3,66"},
        {"shape.d_skin_from_vertex", "N,3,66"},
        {"shape.d_skinned_vertices", "N,3,66"},
        {"shape.d_vertices_rotpose", "N,3,66"},
        {"axes.d_vertices", "vertex,xyz,param"},
        {"shape.J_proj", "N,2,3"},
        {"axes.J_proj", "vertex,uv,xyz"},
        {"shape.J_uv_rotpose", "N,2,66"},
        {"axes.J_uv_rotpose", "vertex,uv,param"},
        {"shape.J_translation", "N,2,3"},
        {"axes.J_translation", "vertex,uv,translation"},
        {"shape.J_active", "N,2,69"},
        {"axes.J_active", "vertex,uv,active_param"},
        {"shape.J_frame", "2N,75"},
        {"axes.J_frame", "flattened_point_uv,frame_param"},
        {"dtype.derivatives", "float32"},
        {"shape.point_weights", "N"}, {"shape.row_scales", "2N"},
        {"shape.weighted_residual", "2N"},
        {"shape.weighted_J_active", "2N,69"},
        {"shape.H_active", "69,69"}, {"shape.g_active", "69"},
        {"shape.H_frame", "75,75"}, {"shape.g_frame", "75"},
        {"dtype.weighted", "float32"}, {"dtype.normal_equations", "float64"},
    };
    output_meta.insert(shapes.begin(), shapes.end());
    staged_lm::write_key_value_file(output / "cpp_output_meta.txt", output_meta);
    std::cout << "POSE_VALUE_CPP_REFERENCE_WRITTEN output=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "POSE_VALUE_CPP_REFERENCE_ERROR: " << error.what() << '\n';
    return 1;
  }
}
