#include "raja_style_launch.h"

#include <utility>

namespace MockMfemInterface {
template <typename DBODY> void rajaWrap(const int N, DBODY &&DBody) {
  MockRajaInterface::forall<true>(N, std::forward<DBODY>(DBody));
}

template <typename d_lambda>
inline void forallWrap(const int N, d_lambda &&DBody) {
  return rajaWrap(N, std::forward<d_lambda>(DBody));
}

template <typename lambda>
__attribute__((noinline)) void forall(int N, lambda &&Body) {
  forallWrap(N, std::forward<lambda>(Body));
}

template <typename lambda>
__attribute__((noinline)) void forall(int Nx, int Ny, lambda &&Body) {
  MockMfemInterface::forall(Nx * Ny, [=] PROTEUS_HOST_DEVICE(int idx) {
    int j = idx / Nx;
    int i = idx % Nx;
    Body(i, j);
  });
}
} // namespace MockMfemInterface
