#pragma once
#include <string>
#include <vector>
namespace staged_lm {
struct SchurSystem {int frame_count=0,frame_dim=0,beta_dim=0;std::vector<double>a,b,c,g_frame,g_beta;};
struct SchurFrameWorkspace {std::vector<double>a_damped,y_matrix,y_gradient,frame_rhs,delta;std::string factor_method;};
struct PoseShapeSchurResult {bool success=false,fallback_used=false;double damping=0;std::vector<SchurFrameWorkspace>frames;std::vector<double>c_damped,schur,schur_rhs,delta_beta,full_delta,dense_delta,linear_residual,schur_residual;std::string schur_factor_method;double damping_ms=0,a_factor_ms=0,solve_y_ms=0,solve_g_ms=0,build_schur_ms=0,beta_solve_ms=0,back_substitution_ms=0,total_ms=0,dense_solve_ms=0;};
PoseShapeSchurResult solve_pose_shape_schur(const SchurSystem&,double damping,bool dense_diagnostic=true);
}
