// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Host registration in a different TU from the device tag tests ID stability.
#include "collapse_cases_source.h"
#include <proteus/JitInterface.h>

void registerColorValues(std::vector<Color> &Values, int &Collections) {
  proteus::register_variant_source<ColorTag>([&](auto Emit) {
    ++Collections;
    for (auto C : Values)
      Emit(C);
  });
}
