#pragma once
#include <vector>
namespace staged_lm {
struct ShapeValueResult { std::vector<float> tanh_latent, beta, joints; double beta_ms=0, dynamic_joints_ms=0; };
ShapeValueResult evaluate_shape_value(const std::vector<float>& latent,float limit,const std::vector<float>& base_joints,const std::vector<float>& joint_basis,int joint_count);
std::vector<float> evaluate_dynamic_vertices(const std::vector<float>& base,const std::vector<float>& basis,const std::vector<float>& beta,int point_count,double* elapsed_ms=nullptr);
}
