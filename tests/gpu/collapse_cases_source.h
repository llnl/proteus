// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROTEUS_TEST_COLLAPSE_CASES_SOURCE_H
#define PROTEUS_TEST_COLLAPSE_CASES_SOURCE_H
#include <vector>

enum class Color : int { Blue, Green, Red, Yellow, Purple, Orange };
struct ColorTag {
  using value_type = Color;
};

void registerColorValues(std::vector<Color> &Values, int &Collections);
#endif
