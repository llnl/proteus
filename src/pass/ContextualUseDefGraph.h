#ifndef PROTEUS_CONTEXTUALUSEDEFGRAPH_H
#define PROTEUS_CONTEXTUALUSEDEFGRAPH_H

#include <functional>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/MathExtras.h>
#include <memory>
#include <optional>
#include <tuple>

namespace proteus {

// Contexts describe static call chains, not dynamic loop iterations. An absent
// parent leaves outer callers unspecified. The graph lives only during analysis
// of immutable IR, so instruction identity is sufficient to identify a call.
struct ConnectionCallContext {
  llvm::CallBase *Call;
  std::shared_ptr<const ConnectionCallContext> Parent;
};
using ConnectionContext = std::shared_ptr<const ConnectionCallContext>;

enum class ConnectionResult { Connected, Disconnected, Unresolved };

// These tags describe reachability, not byte provenance. In particular, a
// container can be disjoint from the closure while holding a pointer to it.
struct ConnectionTags {
  enum : unsigned { Pointer = 1, Container = 2 };
  llvm::DenseMap<llvm::Value *, unsigned> Values;
  llvm::SmallPtrSet<llvm::CallBase *, 16> Calls;
  bool Unresolved = false;
};

// Both endpoints are in the same function. Calls are evidence/conditions, NOT
// edges for the ptr-use visitor to traverse. Conditions run inner to outer.
struct ContextualConnection {
  llvm::Value *From;
  llvm::Value *To;
  llvm::SmallVector<llvm::CallBase *, 4> Calls;
  // To == From + Displacement, under Calls. Unknown is potential overlap only.
  std::optional<int64_t> Displacement;

  bool applies(ConnectionContext Context) const {
    for (auto *Call : Calls) {
      if (!Context)
        return true; // Caller not selected: retain this feasible alternative.
      if (Context->Call != Call)
        return false;
      Context = Context->Parent;
    }
    return true;
  }

  void print(llvm::raw_ostream &OS) const {
    OS << "[contextual-use] ";
    if (auto *A = llvm::dyn_cast<llvm::Argument>(From))
      OS << A->getParent()->getName() << ": ";
    From->printAsOperand(OS, false);
    OS << " -> ";
    To->printAsOperand(OS, false);
    OS << "; displacement=";
    if (Displacement)
      OS << *Displacement;
    else
      OS << "unknown";
    for (auto *Call : Calls) {
      OS << "\n  conditional on " << Call->getFunction()->getName() << ": ";
      OS << *Call;
    }
    OS << '\n';
  }
};

// A lazy index of compressed, function-local alias connections. This is not a
// reaching-write analysis. Pointer reloads and disjointness are delegated to
// the session's existing MemorySSA/AA rather than a separate memory walk.
class ContextualUseDefGraph {
  llvm::DataLayout DL;
  std::function<llvm::Value *(llvm::LoadInst &)> ResolveReload;
  std::function<bool(llvm::Value *, llvm::Value *, llvm::Function &)> NoAlias;
  llvm::DenseMap<llvm::Argument *, llvm::SmallVector<ContextualConnection, 4>>
      Edges;
  llvm::DenseMap<std::pair<llvm::Value *, llvm::Value *>,
                 llvm::SmallVector<ContextualConnection, 4>>
      Relations;
  llvm::DenseMap<std::pair<llvm::CallBase *, const ConnectionCallContext *>,
                 ConnectionContext>
      Contexts;
  llvm::DenseMap<std::tuple<llvm::Value *, llvm::Instruction *,
                            const ConnectionCallContext *>,
                 std::unique_ptr<ConnectionTags>>
      TagCache;
  static constexpr unsigned MaxWork = 4096;
  static constexpr unsigned MaxDepth = 16;

  llvm::Value *base(llvm::Value *V, int64_t &Offset, unsigned &Budget) {
    Offset = 0;
    while (Budget) {
      --Budget;
      int64_t Step = 0;
      V = llvm::GetPointerBaseWithConstantOffset(V, Step, DL);
      if (llvm::AddOverflow(Offset, Step, Offset))
        return nullptr;
      auto *Load = llvm::dyn_cast<llvm::LoadInst>(V);
      if (!Load)
        return V;
      auto *Stored = ResolveReload(*Load);
      if (!Stored)
        return V;
      V = Stored;
    }
    return nullptr;
  }

  void relate(llvm::Value *From, llvm::Value *To, ContextualConnection Edge,
              int64_t Delta, llvm::SmallVectorImpl<ContextualConnection> &Out,
              unsigned &Budget, unsigned Depth) {
    // A null pointer cannot designate any object. In particular, CUDA lowers
    // printf's unused va_list argument to a literal null pointer; leaving it
    // unresolved would prevent the connection graph from proving that the
    // call has no path to a tracked closure.
    if (llvm::isa<llvm::ConstantPointerNull>(From) ||
        llvm::isa<llvm::ConstantPointerNull>(To))
      return;

    auto Unknown = [&] {
      Edge.Displacement.reset();
      Out.push_back(std::move(Edge));
    };
    if (!Budget || Depth == MaxDepth) {
      Unknown();
      return;
    }
    --Budget;
    int64_t FromOff = 0, ToOff = 0, Difference = 0;
    From = base(From, FromOff, Budget);
    To = base(To, ToOff, Budget);
    if (!From || !To || llvm::SubOverflow(ToOff, FromOff, Difference) ||
        llvm::AddOverflow(Delta, Difference, Delta)) {
      Unknown();
      return;
    }
    if (From == To) {
      if (llvm::isa<llvm::UndefValue>(From) ||
          llvm::isa<llvm::PoisonValue>(From)) {
        Unknown();
        return;
      }
      Edge.Displacement = Delta;
      Out.push_back(std::move(Edge));
      return;
    }
    // Separate allocations are separate objects. Separate formals are not.
    if (llvm::isa<llvm::AllocaInst>(From) && llvm::isa<llvm::AllocaInst>(To))
      return;
    auto *FA = llvm::dyn_cast<llvm::Argument>(From);
    auto *TA = llvm::dyn_cast<llvm::Argument>(To);
    if (!FA || !TA || FA->getParent() != TA->getParent()) {
      Unknown();
      return;
    }
    auto &F = *FA->getParent();
    if (NoAlias(FA, TA, F))
      return;
    bool HasCaller = false;
    // Nonlocal functions may also have callers outside this module.
    bool Incomplete = !F.hasLocalLinkage();
    for (llvm::Use &U : F.uses()) {
      auto *CB = llvm::dyn_cast<llvm::CallBase>(U.getUser());
      if (!CB || !CB->isCallee(&U) || CB->getCalledFunction() != &F ||
          FA->getArgNo() >= CB->arg_size() ||
          TA->getArgNo() >= CB->arg_size()) {
        Incomplete = true;
        continue;
      }
      if (!Budget) {
        Incomplete = true;
        break;
      }
      HasCaller = true;
      auto Child = Edge;
      Child.Calls.push_back(CB);
      relate(CB->getArgOperand(FA->getArgNo()),
             CB->getArgOperand(TA->getArgNo()), std::move(Child), Delta, Out,
             Budget, Depth + 1);
    }
    if (!HasCaller || Incomplete)
      Unknown();
  }

public:
  ContextualUseDefGraph(
      const llvm::DataLayout &DL,
      std::function<llvm::Value *(llvm::LoadInst &)> ResolveReload,
      std::function<bool(llvm::Value *, llvm::Value *, llvm::Function &)>
          NoAlias)
      : DL(DL), ResolveReload(std::move(ResolveReload)),
        NoAlias(std::move(NoAlias)) {}

  ConnectionContext enter(llvm::CallBase &CB, ConnectionContext Parent) {
    auto &Context = Contexts[{&CB, Parent.get()}];
    if (!Context)
      Context = std::make_shared<ConnectionCallContext>(
          ConnectionCallContext{&CB, std::move(Parent)});
    return Context;
  }

  // Build a monotone set over SSA uses. Stored pointers tag their storage;
  // loads from that storage resume pointer propagation. Caller bindings are
  // the only interprocedural edges. Unknown contexts are merged conservatively;
  // a selected incoming call excludes other invocations of the same function.
  // No store is selected here, and an overwrite never removes a tag.
  const ConnectionTags &tags(llvm::Value *Root, llvm::Instruction &Boundary,
                             ConnectionContext Context) {
    auto [It, Inserted] =
        TagCache.try_emplace({Root, &Boundary, Context.get()});
    if (!Inserted)
      return *It->second;
    It->second = std::make_unique<ConnectionTags>();
    auto &Result = *It->second;
    llvm::DenseMap<llvm::Function *, llvm::CallBase *> SelectedCalls;
    llvm::DenseMap<llvm::Function *, llvm::Instruction *> Boundaries;
    Boundaries[Boundary.getFunction()] = &Boundary;
    for (auto C = Context; C; C = C->Parent) {
      auto *F = C->Call->getCalledFunction();
      if (!F || SelectedCalls.count(F)) {
        Result.Unresolved = true;
        return Result;
      }
      SelectedCalls[F] = C->Call;
      Boundaries[C->Call->getFunction()] = C->Call;
    }
    llvm::SmallVector<std::pair<llvm::Value *, unsigned>, 32> Pending;
    unsigned Budget = MaxWork;
    auto Add = [&](llvm::Value *V, unsigned Bits) {
      unsigned New = Bits & ~Result.Values.lookup(V);
      if (New) {
        Result.Values[V] |= New;
        Pending.emplace_back(V, New);
      }
    };
    // A tagged container may be reached through a different pointer value
    // than the store that put the tracked pointer in it. Def-use traversal
    // alone misses that case (%a and %b can alias even though they have no SSA
    // relationship). Close container tags over every pointer reload whose
    // source may alias the container. This is deliberately an overestimate;
    // AA's NoAlias result is the only reason to leave a reload out.
    auto AddAliasingReloads = [&](llvm::Value *Container, llvm::Function &F) {
      llvm::Instruction *End = Boundaries.lookup(&F);
      for (llvm::BasicBlock &BB : F)
        for (llvm::Instruction &I : BB) {
          if (!Budget) {
            Result.Unresolved = true;
            return;
          }
          --Budget;
          auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I);
          if (!LI || !LI->getType()->isPointerTy() ||
              (End && !llvm::isPotentiallyReachable(LI, End)))
            continue;
          if (!NoAlias(Container, LI->getPointerOperand(), F))
            Add(LI, ConnectionTags::Pointer | ConnectionTags::Container);
        }
    };
    Add(Root, ConnectionTags::Pointer);
    while (!Pending.empty() && Budget) {
      auto [V, Bits] = Pending.pop_back_val();
      --Budget;
      if (V->getType()->isPointerTy()) {
        auto *Base = llvm::getUnderlyingObject(V);
        if (Base != V)
          Add(Base, Bits); // Includes the enclosing object of a field address.
        if (llvm::isa<llvm::GlobalValue>(Base))
          Result.Unresolved = true; // The pointer has been published.
        // ROCm represents address-space changes as exact integer round trips.
        // A query beginning at the converted pointer must also find stores
        // and other uses of its original pointer, not just downstream users.
        llvm::Value *RoundTripSource = nullptr;
        if (auto *ITP = llvm::dyn_cast<llvm::IntToPtrInst>(Base))
          if (auto *PTI =
                  llvm::dyn_cast<llvm::PtrToIntInst>(ITP->getOperand(0))) {
            auto *Source = PTI->getPointerOperand();
            unsigned Width = PTI->getType()->getIntegerBitWidth();
            if (!DL.isNonIntegralPointerType(Source->getType()) &&
                !DL.isNonIntegralPointerType(ITP->getType()) &&
                DL.getPointerTypeSizeInBits(Source->getType()) == Width &&
                DL.getPointerTypeSizeInBits(ITP->getType()) == Width) {
              RoundTripSource = Source;
              Add(Source, Bits);
            }
          }
        if (V == Root &&
            !llvm::isa<llvm::Argument, llvm::AllocaInst, llvm::LoadInst>(
                Base) &&
            !RoundTripSource)
          Result.Unresolved = true;
        if (Bits & ConnectionTags::Container) {
          llvm::Function *F = nullptr;
          if (auto *I = llvm::dyn_cast<llvm::Instruction>(V))
            F = I->getFunction();
          else if (auto *A = llvm::dyn_cast<llvm::Argument>(V))
            F = A->getParent();
          if (F)
            AddAliasingReloads(V, *F);
        }
      }
      if (auto *A = llvm::dyn_cast<llvm::Argument>(V)) {
        auto *F = A->getParent();
        if (auto *Selected = SelectedCalls.lookup(F)) {
          Add(Selected->getArgOperand(A->getArgNo()), Bits);
        } else {
          // A device kernel argument is supplied by the launch and has no
          // ordinary device-side caller to expand. Its Function::uses() can
          // include registration references that are not pointer-flow edges.
          // Keep caller expansion for helper formals: it is required to trace
          // stored closure pointers through plans and other caller-owned
          // objects when no selected context is available.
          bool IsKernelEntry =
              F->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL ||
              F->getCallingConv() == llvm::CallingConv::PTX_Kernel;
          if (!(V == Root && IsKernelEntry)) {
            for (llvm::Use &U : F->uses()) {
              auto *CB = llvm::dyn_cast<llvm::CallBase>(U.getUser());
              if (!CB || !CB->isCallee(&U) || CB->getCalledFunction() != F ||
                  A->getArgNo() >= CB->arg_size()) {
                Result.Unresolved = true;
                continue;
              }
              Add(CB->getArgOperand(A->getArgNo()), Bits);
            }
          }
        }
      }
      // A query may start at a pointer reload. Reuse the oracle's existing
      // supplier query for that seed; ordinary container loads need no such
      // selection because their tags deliberately overapproximate all stores.
      if (V == Root)
        if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(V)) {
          if (auto *Stored = ResolveReload(*LI))
            Add(Stored, Bits);
          else
            Result.Unresolved = true;
        }
      for (llvm::Use &U : V->uses()) {
        if (!Budget) {
          Result.Unresolved = true;
          break;
        }
        --Budget;
        auto *User = U.getUser();
        if (auto *I = llvm::dyn_cast<llvm::Instruction>(User))
          if (auto *End = Boundaries.lookup(I->getFunction());
              End && !llvm::isPotentiallyReachable(I, End))
            continue;
        if (llvm::isa<llvm::GEPOperator, llvm::BitCastOperator,
                      llvm::AddrSpaceCastOperator>(User) ||
            llvm::isa<llvm::PHINode, llvm::SelectInst, llvm::PtrToIntInst,
                      llvm::IntToPtrInst>(User)) {
          Add(User, Bits);
        } else if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(User)) {
          if (SI->getValueOperand() == V)
            Add(SI->getPointerOperand(), ConnectionTags::Container);
        } else if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(User)) {
          if (Bits & ConnectionTags::Container) {
            if (LI->getType()->isPointerTy())
              Add(LI, ConnectionTags::Pointer | ConnectionTags::Container);
            else
              Result.Unresolved = true; // Encoded or aggregate pointer load.
          }
        } else if (auto *MT = llvm::dyn_cast<llvm::MemTransferInst>(User)) {
          if ((Bits & ConnectionTags::Container) && MT->getRawSource() == V)
            Add(MT->getRawDest(), ConnectionTags::Container);
        } else if (auto *CB = llvm::dyn_cast<llvm::CallBase>(User)) {
          if (!CB->isArgOperand(&U) || CB->hasOperandBundles() ||
              CB->isInlineAsm()) {
            Result.Unresolved = true;
            continue;
          }
          unsigned ArgNo = CB->getArgOperandNo(&U);
          if (!CB->onlyReadsMemory() && ((Bits & ConnectionTags::Container) ||
                                         !CB->onlyReadsMemory(ArgNo)))
            Result.Calls.insert(CB);
          if (CB->onlyReadsMemory() && CB->getType()->isVoidTy())
            continue;
          auto *F = CB->getCalledFunction();
          if (!F || F->isDeclaration() || F->isInterposable() ||
              ArgNo >= F->arg_size()) {
            // nocapture for a container's address says nothing about
            // publishing a pointer loaded from inside that container.
            if ((Bits & ConnectionTags::Container) ||
                !CB->doesNotCapture(ArgNo) || !CB->getType()->isVoidTy())
              Result.Unresolved = true;
            continue;
          }
          auto *Selected = SelectedCalls.lookup(F);
          if (!Selected || Selected == CB)
            Add(F->getArg(ArgNo), Bits);
          else if (!CB->doesNotCapture(ArgNo))
            Result.Unresolved = true;
        } else if (auto *RI = llvm::dyn_cast<llvm::ReturnInst>(User)) {
          auto *F = RI->getFunction();
          for (llvm::Use &FU : F->uses()) {
            auto *CB = llvm::dyn_cast<llvm::CallBase>(FU.getUser());
            if (!CB || !CB->isCallee(&FU) || CB->getCalledFunction() != F) {
              Result.Unresolved = true;
              continue;
            }
            auto *Selected = SelectedCalls.lookup(F);
            if (!Selected || Selected == CB)
              Add(CB, Bits);
          }
        } else if (!llvm::isa<llvm::ICmpInst, llvm::DbgInfoIntrinsic>(User)) {
          Result.Unresolved = true;
        }
      }
    }
    if (!Pending.empty())
      Result.Unresolved = true;
    return Result;
  }

  // This query considers every recovered caller alternative. An empty set
  // proves disconnection only because relate emits an explicit unresolved
  // alternative on unsupported origins, incomplete callers, or budget limits.
  ConnectionResult connection(llvm::Value *From, llvm::Value *To,
                              ConnectionContext Context) {
    auto [It, Inserted] = Relations.try_emplace({From, To});
    if (Inserted) {
      unsigned Budget = MaxWork;
      relate(From, To, {From, To, {}, std::nullopt}, 0, It->second, Budget, 0);
    }
    bool Unknown = false;
    for (const auto &Edge : It->second) {
      if (!Edge.applies(Context))
        continue;
      if (Edge.Displacement)
        return ConnectionResult::Connected;
      Unknown = true;
    }
    return Unknown ? ConnectionResult::Unresolved
                   : ConnectionResult::Disconnected;
  }

  ConnectionResult callConnection(llvm::CallBase &Call,
                                  const llvm::MemoryLocation &Tracked,
                                  ConnectionContext Context,
                                  llvm::Instruction *UseBoundary = nullptr) {
    if (Call.hasOperandBundles() || Call.isInlineAsm())
      return ConnectionResult::Unresolved;
    if (Call.onlyReadsMemory())
      return ConnectionResult::Disconnected;
    const auto &Tags = tags(const_cast<llvm::Value *>(Tracked.Ptr),
                            UseBoundary ? *UseBoundary : Call, Context);
    if (Tags.Calls.contains(&Call))
      return ConnectionResult::Connected;
    if (Tags.Unresolved)
      return ConnectionResult::Unresolved;
    // The caller graph is the available proof of whether this call can reach
    // the non-global tracked object. Inspect its pointer arguments even for an
    // opaque callee: a connected or unresolved path still rejects the call,
    // while disconnected pointer arguments prove it disjoint. With no pointer
    // arguments, an opaque call without an arg-memory-only contract may still
    // access the object through hidden state, so it remains unresolved.
    bool Unknown = false;
    bool HasPointerArgument = false;
    for (unsigned I = 0; I < Call.arg_size(); ++I) {
      auto *Arg = Call.getArgOperand(I);
      if (Arg->getType()->isAggregateType())
        return ConnectionResult::Unresolved;
      if (!Arg->getType()->isPointerTy())
        continue;
      HasPointerArgument = true;
      if (Call.onlyReadsMemory(I))
        continue;
      // LambdaASTChecker rejects registered wrapper objects with global
      // lifetime. A direct pointer to an unrelated global therefore cannot
      // denote the tracked closure bytes. Publication of the closure pointer
      // through a global slot remains unresolved in tags().
      if (llvm::isa<llvm::GlobalVariable>(llvm::getUnderlyingObject(Arg)))
        continue;
      auto Result =
          connection(const_cast<llvm::Value *>(Tracked.Ptr), Arg, Context);
      if (Result == ConnectionResult::Connected)
        return Result;
      Unknown |= Result == ConnectionResult::Unresolved;
    }
    auto *Callee = Call.getCalledFunction();
    if (!HasPointerArgument &&
        (!Callee || Callee->isDeclaration() || Callee->isInterposable()) &&
        !Call.onlyAccessesArgMemory())
      return ConnectionResult::Unresolved;
    return Unknown ? ConnectionResult::Unresolved
                   : ConnectionResult::Disconnected;
  }

  llvm::ArrayRef<ContextualConnection> connections(llvm::Argument &From) {
    auto [It, Inserted] = Edges.try_emplace(&From);
    if (!Inserted)
      return It->second;
    unsigned Budget = MaxWork;
    for (auto &To : From.getParent()->args()) {
      if (&To == &From || !To.getType()->isPointerTy() ||
          !From.getType()->isPointerTy())
        continue;
      relate(&From, &To, {&From, &To, {}, std::nullopt}, 0, It->second, Budget,
             0);
    }
    return It->second;
  }
};

} // namespace proteus
#endif
