#include "staged_lm/binary_io.h"
#include "staged_lm/pose_shape_forward.h"
#include <filesystem>
#include <iostream>
#include <map>
namespace
{
    int iv(const std::map<std::string, std::string> &m, const std::string &k) { return std::stoi(m.at(k)); }
    float fv(const std::map<std::string, std::string> &m, const std::string &k) { return std::stof(m.at(k)); }
    std::filesystem::path arg(int n, char **v, const std::string &k)
    {
        for (int i = 1; i + 1 < n; ++i)
            if (v[i] == k)
                return v[i + 1];
        throw std::runtime_error("missing " + k);
    }
    staged_lm::PoseShapeCameraInput camera(const std::filesystem::path &f, const std::map<std::string, std::string> &m, const std::vector<float> &p, const std::string &c)
    {
        int n = iv(m, c + "_count");
        staged_lm::PoseShapeCameraInput x;
        x.pose.camera = c;
        x.pose.parameters = {p.begin(), p.begin() + 75};
        x.pose.point_count = n;
        x.pose.left_hand_pose = staged_lm::read_float32_file(f / (c + "_left_hand_pose.f32"), 45);
        x.pose.right_hand_pose = staged_lm::read_float32_file(f / (c + "_right_hand_pose.f32"), 45);
        x.pose.scale = staged_lm::read_float32_file(f / (c + "_scale.f32"), 1)[0];
        x.pose.selected_posedirs = staged_lm::read_float32_file(f / (c + "_selected_posedirs.f32"), 486 * n * 3);
        x.pose.selected_lbs_weights = staged_lm::read_float32_file(f / (c + "_selected_lbs_weights.f32"), n * 55);
        x.pose.targets = staged_lm::read_float32_file(f / (c + "_targets.f32"), n * 2);
        x.pose.image_width = iv(m, "image_width");
        x.pose.image_height = iv(m, "image_height");
        x.selected_base_v_shaped = staged_lm::read_float32_file(f / (c + "_selected_base_v_shaped.f32"), n * 3);
        x.selected_beta_shapedirs = staged_lm::read_float32_file(f / (c + "_selected_beta_shapedirs.f32"), n * 3 * 10);
        return x;
    }
    void wc(const std::filesystem::path &o, const std::string &c, const staged_lm::PoseCameraResult &r, const std::vector<float> &dyn)
    {
        std::pair<const char *, const std::vector<float> *> a[] = {{"dynamic_v_shaped", &dyn}, {"axis_angles", &r.axis_angles}, {"rotations", &r.rotations}, {"pose_feature", &r.pose_feature}, {"pose_offsets", &r.pose_offsets}, {"v_posed", &r.v_posed}, {"relative_joints", &r.relative_joints}, {"local_transforms", &r.local_transforms}, {"global_transforms", &r.global_transforms}, {"rest_transforms", &r.rest_transforms}, {"relative_transforms", &r.relative_transforms}, {"selected_transforms", &r.selected_transforms}, {"skinned_vertices", &r.skinned_vertices}, {"vertices", &r.vertices}, {"uv", &r.uv}, {"raw_residual", &r.raw_residual}};
        for (auto [n, v] : a)
            staged_lm::write_float32_file(o / (c + "_" + n + ".f32"), *v);
    }
}
int main(int argc, char **argv)
{
    try
    {
        auto f = arg(argc, argv, "--fixture"), o = arg(argc, argv, "--output");
        auto m = staged_lm::read_key_value_file(f / "fixture_meta.txt");
        auto p = staged_lm::read_float32_file(f / "parameters.f32", 85);
        auto r = staged_lm::evaluate_pose_shape_value(p, staged_lm::read_float32_file(f / "base_joints.f32", 165), staged_lm::read_float32_file(f / "d_joints_d_beta.f32", 1650), staged_lm::read_float32_file(f / "beta_anchor.f32", 10), staged_lm::read_float32_file(f / "body_pose_anchor.f32", 63), fv(m, "beta_abs_limit"), fv(m, "pose_anchor_scale"), fv(m, "shape_anchor_scale"), fv(m, "charbonnier_eps"), fv(m, "cam2_alpha"), fv(m, "cam4_alpha"), staged_lm::read_int64_file(f / "parents.i64", 55), camera(f, m, p, "cam2"), camera(f, m, p, "cam4"));
        std::filesystem::create_directories(o);
        staged_lm::write_float32_file(o / "tanh_beta_latent.f32", r.shape.tanh_latent);
        staged_lm::write_float32_file(o / "physical_beta.f32", r.shape.beta);
        staged_lm::write_float32_file(o / "dynamic_joints.f32", r.shape.joints);
        wc(o, "cam2", r.cam2, r.cam2_dynamic_v_shaped);
        wc(o, "cam4", r.cam4, r.cam4_dynamic_v_shaped);
        staged_lm::write_float64_file(o / "objective_primary.f64", {r.objective_primary});
        staged_lm::write_float64_file(o / "objective_pose_anchor.f64", {r.objective_pose_anchor});
        staged_lm::write_float64_file(o / "objective_shape_anchor.f64", {r.objective_shape_anchor});
        staged_lm::write_float64_file(o / "objective_total.f64", {r.objective_total});
        staged_lm::write_key_value_file(o / "cpp_output_meta.txt", {{"shared_beta_ms", std::to_string(r.shape.beta_ms)}, {"shared_dynamic_joints_ms", std::to_string(r.shape.dynamic_joints_ms)}, {"cam2_dynamic_shape_ms", std::to_string(r.cam2_shape_ms)}, {"cam4_dynamic_shape_ms", std::to_string(r.cam4_shape_ms)}, {"cam2_value_ms", std::to_string(r.cam2.total_value_forward_ms)}, {"cam4_value_ms", std::to_string(r.cam4.total_value_forward_ms)}, {"objective_ms", std::to_string(r.objective_ms)}, {"total_ms", std::to_string(r.total_ms)}});
        std::cout << "POSE_SHAPE_VALUE_CPP_REFERENCE_WRITTEN\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
