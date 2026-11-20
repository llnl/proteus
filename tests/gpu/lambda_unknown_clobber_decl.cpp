extern "C" __device__ __attribute__((noinline)) void
overwriteSlotFromDeclaration(void **Slot, void *Replacement) {
  *Slot = Replacement;
}
