#include "KernelArgPtrUseVisitor.h"
#include <cstdlib>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>

using namespace llvm;
using namespace proteus;

static void require(bool OK, StringRef Message) {
  if (!OK) {
    errs() << "contextual-use-def: " << Message << '\n';
    std::exit(1);
  }
}

static SmallVector<CallBase *> calls(Function &F) {
  SmallVector<CallBase *> Result;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *CB = dyn_cast<CallBase>(&I))
        Result.push_back(CB);
  return Result;
}

// Run only the existing collection phase so a correct fallback cannot mask
// failure to discover the writer. Every popped item must remain in this frame.
static bool collect(Function &F, CallBase &Boundary,
                    std::shared_ptr<MemorySSAClobberOracle> Oracle,
                    bool ExpectWrite,
                    ConnectionContext EntryContext = nullptr) {
  LambdaInstUseVisitor Visitor(
      F.getArg(0), &Boundary, &Boundary, F.getParent()->getDataLayout(), 0,
      Oracle, 4, std::nullopt,
      MemoryLocation(F.getArg(0), LocationSize::precise(4)),
      std::move(EntryContext));
  while (!Visitor.empty() && !Visitor.failed() && !Visitor.success()) {
    auto Item = Visitor.popBack();
    require(Item.Frame == &F && Item.UseBoundary == &Boundary,
            "compressed edge changed the frame or boundary");
    if (!Visitor.seen(Item.CurVal)) {
      Visitor.markAsSeen(Item.CurVal);
      if (auto *A = dyn_cast<Argument>(Item.CurVal))
        Visitor.visitArgument(*A);
      else if (auto *I = dyn_cast<Instruction>(Item.CurVal))
        Visitor.visit(*I);
    }
    Visitor.finishCurrentWorkItem();
  }
  require(!Visitor.failed(),
          "collection failed before exposing its candidates");
  bool Found = false;
  for (const auto &Candidate : Visitor.getCollectedCandidates()) {
    if (!isa<MemTransferInst>(Candidate.WriteCandidate))
      continue;
    require(Candidate.DiscoveryOnly, "conditional write became provenance");
    require(Candidate.ClobberInfo.Frame == &F &&
                Candidate.ClobberInfo.UseBoundary == &Boundary,
            "candidate changed MemorySSA frame/boundary");
    Found = true;
  }
  require(Found == ExpectWrite, "wrong conditional memcpy candidate set");
  bool Empty = Visitor.getCollectedCandidates().empty();
  if (ExpectWrite) {
    Visitor.resolveCollectedClobbersAndContinue();
    require(Visitor.failed() && !Visitor.success(),
            "conditional write authorized replacement recovery");
  }
  return Empty;
}

static bool unchanged(Function &F, CallBase &Boundary,
                      std::shared_ptr<MemorySSAClobberOracle> Oracle,
                      ConnectionContext EntryContext,
                      CallBase *ExpectedCollected = nullptr) {
  LambdaInstUseVisitor Visitor(
      F.getArg(0), &Boundary, &Boundary, F.getParent()->getDataLayout(), 0,
      Oracle, 4, std::nullopt,
      MemoryLocation(F.getArg(0), LocationSize::precise(4)),
      std::move(EntryContext));
  while (!Visitor.failed() && !Visitor.success()) {
    while (!Visitor.empty() && !Visitor.failed() && !Visitor.success()) {
      auto Item = Visitor.popBack();
      require(Item.Frame == &F && Item.UseBoundary == &Boundary,
              "call filtering crossed a frame boundary");
      if (!Visitor.seen(Item.CurVal)) {
        Visitor.markAsSeen(Item.CurVal);
        if (auto *A = dyn_cast<Argument>(Item.CurVal))
          Visitor.visitArgument(*A);
        else if (auto *I = dyn_cast<Instruction>(Item.CurVal))
          Visitor.visit(*I);
      }
      Visitor.finishCurrentWorkItem();
    }
    if (!Visitor.failed() && !Visitor.success())
      Visitor.resolveCollectedClobbersAndContinue();
  }
  if (ExpectedCollected)
    require(llvm::any_of(Visitor.getCollectedCandidates(),
                         [&](const auto &C) {
                           return C.WriteCandidate == ExpectedCollected &&
                                  C.DiscoveryOnly;
                         }),
            "tagged writer did not enter the existing visitor's collection");
  return Visitor.success() &&
         Visitor.getAnalysisResult().Kind == LambdaPtrUseKind::LiveOnEntry;
}

// Each fixture writes through a pointer held in a separate local object.
// Disconnection must account for reachable pointers, not just whether the
// explicit call argument's own allocation overlaps the tracked bytes.
static void
checkReachableWrite(Module &M, StringRef Name,
                    std::shared_ptr<MemorySSAClobberOracle> Oracle) {
  auto *Helper = M.getFunction((Name + ".helper").str());
  auto *Root = M.getFunction((Name + ".root").str());
  require(Helper && Root, "missing reachable-write fixture");
  auto HelperCalls = calls(*Helper);
  auto RootCalls = calls(*Root);
  require(HelperCalls.size() == 2 && RootCalls.size() == 1,
          "unexpected reachable-write fixture calls");
  auto &Graph = Oracle->getConnections(M.getDataLayout());
  auto EntryContext = Graph.enter(*RootCalls.front(), nullptr);
  MemoryLocation Tracked(Helper->getArg(0), LocationSize::precise(4));
  auto Baseline = Oracle->query(Tracked, *HelperCalls.back());
  require(Baseline.Kind == MemorySSAClobberKind::Definition &&
              Baseline.ClobberingInstruction == HelperCalls.front(),
          "baseline oracle did not retain the indirect writer");

  auto Connection =
      Graph.callConnection(*HelperCalls.front(), Tracked, EntryContext);
  bool SkippedWrite = unchanged(*Helper, *HelperCalls.back(), Oracle,
                                EntryContext, HelperCalls.front());
  outs() << Name << ": call="
         << (Connection == ConnectionResult::Connected      ? "Connected"
             : Connection == ConnectionResult::Disconnected ? "Disconnected"
                                                            : "Unresolved")
         << "; visitor=" << (SkippedWrite ? "LiveOnEntry" : "retained write")
         << '\n';
  // These supported paths must be discovered, not merely protected by the
  // oracle's conservative fallback for an unresolved connection.
  require(Connection == ConnectionResult::Connected && !SkippedWrite,
          "reachable writer was incorrectly discharged");
}

int main(int Argc, char **Argv) {
  require(Argc == 2 || Argc == 3,
          "expected IR filename and optional case name");
  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyFile(Argv[1], Error, Context);
  if (!M) {
    Error.print(Argv[0], errs());
    return 1;
  }
  require(!verifyModule(*M, &errs()), "invalid test IR");
  auto Oracle = std::make_shared<MemorySSAClobberOracle>();
  if (Argc == 3) {
    checkReachableWrite(*M, Argv[2], Oracle);
    return 0;
  }
  auto &Graph = Oracle->getConnections(M->getDataLayout());
  auto *Helper = M->getFunction("helper");
  auto RootCalls = calls(*M->getFunction("root"));
  auto Edges = Graph.connections(*Helper->getArg(0));
  require(Edges.size() == 1, "expected only the CB1 alias edge");
  const auto &Edge = Edges.front();
  require(Edge.From == Helper->getArg(0) && Edge.To == Helper->getArg(1) &&
              Edge.Displacement == 0 && Edge.Calls.size() == 1 &&
              Edge.Calls.front() == RootCalls[0],
          "wrong CB1 witness");
  require(Edge.applies(Graph.enter(*RootCalls[0], nullptr)), "CB1 inactive");
  require(!Edge.applies(Graph.enter(*RootCalls[1], nullptr)),
          "CB1 leaked to CB2");
  require(Edge.applies(nullptr), "unspecified context lost potential alias");
  Edge.print(outs());
  collect(*Helper, *calls(*Helper).back(), Oracle, true);
  collect(*Helper, *calls(*Helper).back(), Oracle, true,
          Graph.enter(*RootCalls[0], nullptr));
  require(collect(*Helper, *calls(*Helper).back(), Oracle, false,
                  Graph.enter(*RootCalls[1], nullptr)),
          "conditional write leaked into CB2 collection");

  auto Nested = Graph.connections(*M->getFunction("nested")->getArg(0));
  auto Outer = calls(*M->getFunction("nested.root"));
  auto *Inner = calls(*M->getFunction("middle")).front();
  require(Nested.size() == 1 && Nested[0].Calls.size() == 2 &&
              Nested[0].Calls[0] == Inner && Nested[0].Calls[1] == Outer[0],
          "nested chain was not compressed");
  require(
      Nested[0].applies(Graph.enter(*Inner, Graph.enter(*Outer[0], nullptr))) &&
          !Nested[0].applies(
              Graph.enter(*Inner, Graph.enter(*Outer[1], nullptr))),
      "nested call contexts mixed");
  auto Displaced = Graph.connections(*M->getFunction("displaced")->getArg(0));
  require(Displaced.size() == 1 && Displaced[0].Displacement == -4,
          "incorrect displaced-pointer relation");
  auto Incoming = Graph.connections(*M->getFunction("incoming")->getArg(0));
  require(Incoming.size() == 1 && !Incoming[0].Displacement,
          "different incoming formals treated as disjoint");
  auto Reloads = Graph.connections(*M->getFunction("reloaded")->getArg(0));
  auto ReloadCalls = calls(*M->getFunction("reload.root"));
  require(Reloads.size() == 1 && Reloads[0].Displacement == 0 &&
              Reloads[0].Calls.front() == ReloadCalls[0],
          "pointer-slot overwrite was ignored");
  auto *Late = M->getFunction("late");
  require(collect(*Late, *calls(*Late).front(), Oracle, false),
          "write after boundary entered candidates");
  auto Recursive = Graph.connections(*M->getFunction("recursive")->getArg(0));
  require(!Recursive.empty() && !Recursive[0].Displacement,
          "recursion did not produce an unresolved connection");

  auto *OutputHelper = M->getFunction("output_helper");
  auto OutputCalls = calls(*OutputHelper);
  auto OutputContexts = calls(*M->getFunction("output.root"));
  MemoryLocation OutputTracked(OutputHelper->getArg(0),
                               LocationSize::precise(4));
  auto DisjointContext = Graph.enter(*OutputContexts[0], nullptr);
  auto AliasingContext = Graph.enter(*OutputContexts[1], nullptr);
  require(Oracle->query(OutputTracked, *OutputCalls.back()).Kind ==
              MemorySSAClobberKind::Definition,
          "control no longer exercises an unresolved oracle clobber");
  require(Graph.callConnection(*OutputCalls.front(), OutputTracked,
                               DisjointContext) ==
              ConnectionResult::Disconnected,
          "complete disjoint call was not classified");
  require(Graph.callConnection(*OutputCalls.front(), OutputTracked,
                               AliasingContext) == ConnectionResult::Connected,
          "aliasing writer was classified as disconnected");
  require(
      unchanged(*OutputHelper, *OutputCalls.back(), Oracle, DisjointContext),
      "collector did not discharge the disconnected call");
  require(
      !unchanged(*OutputHelper, *OutputCalls.back(), Oracle, AliasingContext),
      "collector skipped the connected writer");
  require(!unchanged(*OutputHelper, *OutputCalls.back(), Oracle, nullptr),
          "unselected caller context lost the possible writer");
  auto *Opaque = M->getFunction("opaque_helper");
  auto OpaqueCalls = calls(*Opaque);
  require(Graph.callConnection(
              *OpaqueCalls.front(),
              MemoryLocation(Opaque->getArg(0), LocationSize::precise(4)),
              nullptr) == ConnectionResult::Unresolved,
          "missing pointer operands treated as a completeness proof");
  require(!unchanged(*Opaque, *OpaqueCalls.back(), Oracle, nullptr),
          "empty candidate set skipped an unresolved effect");
  auto *IncomingDisconnected = M->getFunction("incoming_disconnected");
  auto IncomingCalls = calls(*IncomingDisconnected);
  require(Graph.callConnection(*IncomingCalls.front(),
                               MemoryLocation(IncomingDisconnected->getArg(0),
                                              LocationSize::precise(4)),
                               nullptr) == ConnectionResult::Disconnected,
          "external pointer origin vetoed a completed disconnected graph");
  require(
      unchanged(*IncomingDisconnected, *IncomingCalls.back(), Oracle, nullptr),
      "incoming pointer could not cross a disconnected utility call");
  require(Graph.callConnection(
              *IncomingCalls.front(),
              MemoryLocation(IncomingCalls.back()->getArgOperand(0),
                             LocationSize::precise(4)),
              nullptr) == ConnectionResult::Disconnected,
          "pointer integer round trip lost the incoming graph root");

  auto *GlobalArgHelper = M->getFunction("global_arg_helper");
  auto GlobalArgCalls = calls(*GlobalArgHelper);
  require(GlobalArgCalls.size() == 2,
          "unexpected global-argument fixture calls");
  require(Graph.callConnection(*GlobalArgCalls.front(),
                               MemoryLocation(GlobalArgHelper->getArg(0),
                                              LocationSize::precise(4)),
                               nullptr, GlobalArgCalls.back()) ==
              ConnectionResult::Disconnected,
          "direct global pointer argument was treated as a possible closure");

  auto *DeviceGlobalArg = M->getFunction("device_global_arg");
  auto DeviceGlobalCalls = calls(*DeviceGlobalArg);
  require(DeviceGlobalCalls.size() == 3,
          "unexpected device-style global-argument fixture calls");
  auto &FullBoundaryTags = Graph.tags(DeviceGlobalArg->getArg(0),
                                      *DeviceGlobalCalls.back(), nullptr);
  require(FullBoundaryTags.Unresolved,
          "later opaque closure use did not make the broad summary unresolved");
  auto &CandidateBoundaryTags = Graph.tags(DeviceGlobalArg->getArg(0),
                                           *DeviceGlobalCalls.front(), nullptr);
  require(!CandidateBoundaryTags.Unresolved,
          "later opaque closure use contaminated an earlier candidate query");
  require(!CandidateBoundaryTags.Calls.contains(DeviceGlobalCalls.front()),
          "global write entered the candidate-bounded connected-call set");
  require(Graph.callConnection(*DeviceGlobalCalls.front(),
                               MemoryLocation(DeviceGlobalArg->getArg(0),
                                              LocationSize::precise(4)),
                               nullptr, DeviceGlobalCalls.front()) ==
              ConnectionResult::Disconnected,
          "device-style global write was not disconnected");
  require(Graph.callConnection(*DeviceGlobalCalls.front(),
                               MemoryLocation(DeviceGlobalArg->getArg(0),
                                              LocationSize::precise(4)),
                               nullptr, DeviceGlobalCalls.back()) ==
              ConnectionResult::Unresolved,
          "broad unresolved context was silently treated as disconnected");
  outs() << "contextual use-def checks passed\n";
}
