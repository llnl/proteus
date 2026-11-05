// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "proteus/VariantSource.h"
#include "proteus/impl/TransformCollapseCasesSpecialization.h"
#include <cstdlib>
#include <iostream>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>

using namespace llvm;
using namespace proteus;
static void check(bool Condition, const char *Message) {
  if (!Condition) {
    errs() << Message << "\n";
    std::abort();
  }
}
static const char *IR = R"(
declare i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32, i64, i64, i32, i32)
define i32 @color_switch(i32 %color) {
entry:
  %v = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %color, i64 42, i64 7, i32 32, i32 2)
  switch i32 %v, label %default [i32 0, label %a
                               i32 1, label %shared
                               i32 2, label %shared
                               i32 3, label %b
                               i32 4, label %c
                               i32 5, label %c]
a:
  br label %shared
shared:
  %p = phi i32 [10, %a], [20, %entry], [20, %entry]
  ret i32 %p
b:
  ret i32 30
c:
  ret i32 40
default:
  ret i32 99
}
define i32 @conditions(i32 %color, i1 %unrelated) {
entry:
  %v = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %color, i64 42, i64 7, i32 32, i32 2)
  %a = icmp eq i32 %v, 0
  br i1 %a, label %yes, label %next
next:
  %b = icmp eq i32 %v, 3
  br i1 %b, label %dynamic, label %other
dynamic:
  br i1 %unrelated, label %yes, label %other
yes:
  ret i32 10
other:
  ret i32 20
}
define i1 @signed_cast(i32 %color) {
  %v = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %color, i64 43, i64 7, i32 32, i32 2)
  %ext = sext i32 %v to i64
  %cmp = icmp slt i64 %ext, 0
  ret i1 %cmp
}
define i1 @independent(i32 %left, i32 %right) {
  %a = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %left, i64 42, i64 7, i32 32, i32 2)
  %b = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %right, i64 44, i64 7, i32 32, i32 2)
  %cmp = icmp eq i32 %a, %b
  ret i1 %cmp
}
define i1 @same_tag(i32 %left, i32 %right) {
  %a = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %left, i64 42, i64 7, i32 32, i32 2)
  %b = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %right, i64 42, i64 7, i32 32, i32 2)
  %cmp = icmp eq i32 %a, %b
  ret i1 %cmp
}
define i1 @spilled(i32 %color) {
  %local = alloca i32, addrspace(5)
  %generic = addrspacecast ptr addrspace(5) %local to ptr
  %v = call i32 @_ZN7proteus6detail21__jit_variants_markerIiEE(i32 %color, i64 42, i64 7, i32 32, i32 2)
  store i32 %v, ptr %generic
  %loaded = load i32, ptr %generic
  %cmp = icmp slt i32 %loaded, 0
  ret i1 %cmp
}
)";
static std::unique_ptr<Module> module(LLVMContext &Ctx) {
  SMDiagnostic Error;
  auto M = parseAssemblyString(IR, Error, Ctx);
  if (!M)
    Error.print("collapse_cases_ir", errs());
  check(bool(M), "parse failed");
  return M;
}
static void validate(Module &M) {
  check(!verifyModule(M, &errs()), "invalid transformed IR");
  check(variantMarkers(M).empty(), "unlowered marker");
}

struct Tag {
  using value_type = int;
};
int main() {
  LLVMContext Ctx;
  for (VariantSnapshot Snapshot : {VariantSnapshot{}, VariantSnapshot{{42, {}}},
                                   VariantSnapshot{{42, {0, 1, 3}}}}) {
    auto M = module(Ctx);
    TransformCollapseCasesSpecialization::transform(*M, Snapshot);
    validate(*M);
    auto *S = cast<SwitchInst>(
        M->getFunction("color_switch")->getEntryBlock().getTerminator());
    check(S->getNumCases() == 6, "generic switch changed");
  }
  {
    auto M = module(Ctx);
    TransformCollapseCasesSpecialization::transform(
        *M, {{42, {0, 3}}, {43, {0xfffffffe, 0xffffffff}}});
    validate(*M);
    auto *S = cast<SwitchInst>(
        M->getFunction("color_switch")->getEntryBlock().getTerminator());
    check(S->getNumCases() == 2, "six cases did not collapse to two");
    check(isa<UnreachableInst>(S->getDefaultDest()->getTerminator()),
          "exhaustive default survived");
    auto *Shared = S->getSuccessor(1)->getTerminator()->getSuccessor(0);
    auto *Return = cast<ReturnInst>(Shared->getTerminator());
    check(cast<ConstantInt>(Return->getReturnValue())->getZExtValue() == 10,
          "fallthrough PHI broken");
    for (auto &BB : *M->getFunction("conditions")) {
      if (BB.getName() == "next")
        check(cast<ConstantInt>(
                  cast<BranchInst>(BB.getTerminator())->getCondition())
                  ->isOne(),
              "else branch was not refined");
      if (BB.getName() == "dynamic")
        check(
            isa<Argument>(cast<BranchInst>(BB.getTerminator())->getCondition()),
            "unrelated condition changed");
    }
    check(cast<ConstantInt>(
              cast<ReturnInst>(
                  M->getFunction("signed_cast")->back().getTerminator())
                  ->getReturnValue())
              ->isOne(),
          "signed cast failed");
    auto *SpilledReturn =
        cast<ReturnInst>(M->getFunction("spilled")->back().getTerminator());
    check(isa<ConstantInt>(SpilledReturn->getReturnValue()) &&
              cast<ConstantInt>(SpilledReturn->getReturnValue())->isZero(),
          "local variable prevented if/else specialization");
  }
  {
    auto M = module(Ctx);
    TransformCollapseCasesSpecialization::transform(*M, {{42, {0, 99}}});
    validate(*M);
    auto *S = cast<SwitchInst>(
        M->getFunction("color_switch")->getEntryBlock().getTerminator());
    check(S->getNumCases() == 1, "unknown value should use default");
    check(isa<ReturnInst>(S->getDefaultDest()->getTerminator()),
          "reachable default removed");
  }
  {
    auto M = module(Ctx);
    TransformCollapseCasesSpecialization::transform(*M, {{42, {3}}});
    validate(*M);
    auto *S = cast<SwitchInst>(
        M->getFunction("color_switch")->getEntryBlock().getTerminator());
    check(cast<ConstantInt>(S->getCondition())->getZExtValue() == 3,
          "singleton not constant");
  }
  {
    auto M = module(Ctx);
    TransformCollapseCasesSpecialization::transform(*M,
                                                    {{42, {0, 3}}, {44, {4}}});
    validate(*M);
    auto *F = M->getFunction("independent");
    auto *Cmp = cast<ICmpInst>(
        cast<ReturnInst>(F->back().getTerminator())->getReturnValue());
    check(Cmp->getOperand(0) == F->getArg(0), "wrong expression specialized");
    check(cast<ConstantInt>(Cmp->getOperand(1))->getZExtValue() == 4,
          "independent tag not specialized");
    F = M->getFunction("same_tag");
    Cmp = cast<ICmpInst>(
        cast<ReturnInst>(F->back().getTerminator())->getReturnValue());
    check(Cmp->getOperand(0) == F->getArg(0) &&
              Cmp->getOperand(1) == F->getArg(1),
          "shared tag incorrectly implies equal values");
  }
  int Collections = 0;
  std::vector<int> Values{3, 0, 3};
  register_variant_source<Tag>([&](auto Emit) {
    ++Collections;
    for (int V : Values)
      Emit(V);
  });
  VariantSite Site{proteus::detail::variant_type_id<Tag>(),
                   proteus::detail::variant_type_id<int>(), 32, 2};
  auto Snapshot = collectVariantSources({Site, Site});
  check(Collections == 1, "duplicate collection");
  check(Snapshot.at(Site.Tag) == std::vector<uint64_t>({0, 3}),
        "set not canonical");
  Values = {1};
  check(collectVariantSources({Site}).at(Site.Tag) ==
            std::vector<uint64_t>({1}),
        "stale source");
  check(Snapshot.at(Site.Tag) == std::vector<uint64_t>({0, 3}),
        "snapshot mutated");
  Values = {0, 1, 2};
  check(collectVariantSources({Site}).empty(), "over budget was truncated");
  Values.clear();
  check(collectVariantSources({Site}).empty(), "empty source specialized");
  unregister_variant_source<Tag>();
  check(collectVariantSources({Site}).empty(), "unregistered source retained");
  register_variant_source<Tag>([](auto Emit) { Emit(-1); });
  check(collectVariantSources({Site}).at(Site.Tag) ==
            std::vector<uint64_t>({0xffffffff}),
        "negative value encoding");
  unregister_variant_source<Tag>();
  std::cout << "PASS collapse cases IR and registry\n";
}
