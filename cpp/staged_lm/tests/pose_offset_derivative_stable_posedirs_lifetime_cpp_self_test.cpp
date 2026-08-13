#include "staged_lm/pose_forward.h"

#include <iostream>
#include <type_traits>
#include <vector>

namespace {
staged_lm::PoseCameraInput make_camera(float marker) {
  staged_lm::PoseCameraInput camera;
  camera.selected_posedirs = {marker, marker + 1.0F, marker + 2.0F};
  staged_lm::promote_selected_posedirs_to_stable_owner(camera);
  return camera;
}
}

int main() {
  using staged_lm::selected_posedirs_view;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(
      selected_posedirs_view(std::declval<const staged_lm::PoseCameraInput&>()))>>);
  auto cam2 = make_camera(2.0F);
  auto cam4 = make_camera(4.0F);
  const float* p2 = selected_posedirs_view(cam2).data();
  const float* p4 = selected_posedirs_view(cam4).data();
  if (p2 == p4 || cam2.selected_posedirs.size() || cam4.selected_posedirs.size())
    return 1;
  std::vector<staged_lm::PoseCameraInput> calls;
  calls.reserve(1);
  for (int index = 0; index < 16; ++index)
    calls.push_back(index % 2 ? cam4 : cam2);
  for (int index = 0; index < 16; ++index) {
    const float* expected = index % 2 ? p4 : p2;
    if (selected_posedirs_view(calls[index]).data() != expected) return 2;
  }
  cam2 = {};
  cam4 = {};
  if (selected_posedirs_view(calls[0])[0] != 2.0F ||
      selected_posedirs_view(calls[1])[0] != 4.0F) return 3;
  std::cout << "POSE_OFFSET_DERIVATIVE_STABLE_POSEDIRS_LIFETIME_CPP_SELF_TEST=PASS\n";
  return 0;
}
