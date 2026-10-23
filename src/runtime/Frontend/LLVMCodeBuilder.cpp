#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/Frontend/LLVMTypeMap.h"
#include "proteus/Frontend/TargetModel.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/CoreLLVMDevice.h"
#include "proteus/impl/LLVMIRFunction.h"
#include "proteus/impl/LLVMIRValue.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Host.h>

#include <deque>
#include <string>

namespace proteus {

using namespace llvm;

namespace {

bool isCudaDeviceTarget(TargetModelType TM) {
  return TM == TargetModelType::CUDA;
}

bool isHipDeviceTarget(TargetModelType TM) {
  return TM == TargetModelType::HIP;
}

void emitVoidIntrinsicCall(LLVMCodeBuilder &CB, Intrinsic::ID IID,
                           ArrayRef<Value *> Args = {}) {
  Module &M = CB.getModule();
#if LLVM_VERSION_MAJOR >= 20
  Function *Fn = Intrinsic::getOrInsertDeclaration(&M, IID);
#else
  Function *Fn = Intrinsic::getDeclaration(&M, IID);
#endif
  CB.getIRBuilder().CreateCall(Fn, Args);
}

IRValue *emitCudaBuiltin(LLVMCodeBuilder &CB, const std::string &Name,
                         IRType RetTy) {
  if (Name == "threadIdx.x")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.tid.x", RetTy);
  if (Name == "threadIdx.y")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.tid.y", RetTy);
  if (Name == "threadIdx.z")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.tid.z", RetTy);
  if (Name == "blockIdx.x")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.ctaid.x", RetTy);
  if (Name == "blockIdx.y")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.ctaid.y", RetTy);
  if (Name == "blockIdx.z")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.ctaid.z", RetTy);
  if (Name == "blockDim.x")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.ntid.x", RetTy);
  if (Name == "blockDim.y")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.ntid.y", RetTy);
  if (Name == "blockDim.z")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.ntid.z", RetTy);
  if (Name == "gridDim.x")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.nctaid.x", RetTy);
  if (Name == "gridDim.y")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.nctaid.y", RetTy);
  if (Name == "gridDim.z")
    return CB.createCall("llvm.nvvm.read.ptx.sreg.nctaid.z", RetTy);
  if (Name == "syncThreads") {
#if LLVM_VERSION_MAJOR >= 21
    // LLVM 21 replaced the legacy zero-arg barrier0 intrinsic with CTA barrier
    // variants; __syncthreads maps to barrier.cta.sync.aligned.all(0).
    Value *BarrierId = ConstantInt::get(Type::getInt32Ty(CB.getContext()), 0);
    emitVoidIntrinsicCall(CB, Intrinsic::nvvm_barrier_cta_sync_aligned_all,
                          {BarrierId});
#else
    emitVoidIntrinsicCall(CB, Intrinsic::nvvm_barrier0);
#endif
    return nullptr;
  }

  reportFatalError("Unsupported CUDA builtin " + Name);
}

IRValue *emitHipGridDimBuiltin(LLVMCodeBuilder &CB, unsigned Offset) {
  // HIP grid dimensions come from the implicit argument pointer ABI, which is
  // stable across AMDGPU generations and avoids explicit OCKL bitcode linkage.
  constexpr unsigned ConstantAddressSpace = 4;
  auto *ImplicitArgPtr =
      CB.createCall("llvm.amdgcn.implicitarg.ptr",
                    IRType{IRTypeKind::Pointer, false, 0, IRTypeKind::Void,
                           ConstantAddressSpace});
  IRValue *OffsetVal = CB.getConstantInt(IRType{IRTypeKind::Int64}, Offset);
  IRType PtrTy{IRTypeKind::Pointer, false, 0, IRTypeKind::Int32,
               ConstantAddressSpace};
  auto A = CB.getElementPtr(ImplicitArgPtr, PtrTy, OffsetVal,
                            IRType{IRTypeKind::Int32});
  auto *GEP = CB.loadAddress(A.Slot, A.AllocTy);
  return CB.createLoad(IRType{IRTypeKind::Int32}, GEP);
}

IRValue *emitHipBlockDimBuiltin(LLVMCodeBuilder &CB, unsigned Offset) {
  // HIP block dimensions are encoded as i16 in the implicit argument area;
  // zero-extend to i32 to match the frontend builtin contract.
  constexpr unsigned ConstantAddressSpace = 4;
  auto *ImplicitArgPtr =
      CB.createCall("llvm.amdgcn.implicitarg.ptr",
                    IRType{IRTypeKind::Pointer, false, 0, IRTypeKind::Void,
                           ConstantAddressSpace});
  IRValue *OffsetVal = CB.getConstantInt(IRType{IRTypeKind::Int64}, Offset);
  IRType PtrTy{IRTypeKind::Pointer, false, 0, IRTypeKind::Int16,
               ConstantAddressSpace};
  auto A = CB.getElementPtr(ImplicitArgPtr, PtrTy, OffsetVal,
                            IRType{IRTypeKind::Int16});
  auto *GEP = CB.loadAddress(A.Slot, A.AllocTy);
  auto *Load = CB.createLoad(IRType{IRTypeKind::Int16}, GEP);
  return CB.createZExt(Load, IRType{IRTypeKind::Int32});
}

IRValue *emitHipBuiltin(LLVMCodeBuilder &CB, const std::string &Name,
                        IRType RetTy) {
  if (Name == "threadIdx.x")
    return CB.createCall("llvm.amdgcn.workitem.id.x", RetTy);
  if (Name == "threadIdx.y")
    return CB.createCall("llvm.amdgcn.workitem.id.y", RetTy);
  if (Name == "threadIdx.z")
    return CB.createCall("llvm.amdgcn.workitem.id.z", RetTy);
  if (Name == "blockIdx.x")
    return CB.createCall("llvm.amdgcn.workgroup.id.x", RetTy);
  if (Name == "blockIdx.y")
    return CB.createCall("llvm.amdgcn.workgroup.id.y", RetTy);
  if (Name == "blockIdx.z")
    return CB.createCall("llvm.amdgcn.workgroup.id.z", RetTy);
  if (Name == "blockDim.x")
    return emitHipBlockDimBuiltin(CB, 6);
  if (Name == "blockDim.y")
    return emitHipBlockDimBuiltin(CB, 7);
  if (Name == "blockDim.z")
    return emitHipBlockDimBuiltin(CB, 8);
  if (Name == "gridDim.x")
    return emitHipGridDimBuiltin(CB, 0);
  if (Name == "gridDim.y")
    return emitHipGridDimBuiltin(CB, 1);
  if (Name == "gridDim.z")
    return emitHipGridDimBuiltin(CB, 2);
  if (Name == "syncThreads") {
    emitVoidIntrinsicCall(CB, Intrinsic::amdgcn_s_barrier);
    return nullptr;
  }

  reportFatalError("Unsupported HIP builtin " + Name);
}

} // namespace

struct LLVMCodeBuilder::Impl {
  // Owned context and module (only set via the owning constructor).
  std::unique_ptr<LLVMContext> OwnedCtx;
  std::unique_ptr<Module> OwnedMod;

  IRBuilder<> IRB;
  IRBuilderBase::InsertPoint IP;

  struct Scope {
    std::string File;
    int Line;
    ScopeKind Kind;
    IRBuilderBase::InsertPoint ContIP;
    Scope(const char *File, int Line, ScopeKind Kind,
          IRBuilderBase::InsertPoint ContIP)
        : File(File), Line(Line), Kind(Kind), ContIP(ContIP) {}
  };
  std::vector<Scope> Scopes;

  // Value table — deque guarantees pointer stability on growth.
  std::deque<LLVMIRValue> Values;
  // Function handle table — same stability guarantee.
  std::deque<LLVMIRFunction> Functions;

  IRValue *wrap(llvm::Value *V) {
    Values.emplace_back(V);
    return &Values.back();
  }

  IRFunction *wrapFunction(llvm::Function *Fn) {
    Functions.emplace_back(Fn);
    return &Functions.back();
  }

  static llvm::Value *unwrap(IRValue *V) {
    return static_cast<LLVMIRValue *>(V)->V;
  }

  explicit Impl(LLVMContext &Ctx) : IRB{Ctx} { init(); }

  Impl(std::unique_ptr<LLVMContext> Ctx, std::unique_ptr<Module> Mod)
      : OwnedCtx{std::move(Ctx)}, OwnedMod{std::move(Mod)}, IRB{*OwnedCtx} {
    init();
  }

  void init() {
    // Initialize IP.
    IP = IRB.saveIP();
    // Clang enables the 'contract' rewrite rule by default to enable FMA
    // instructions.
    // (Controllable via the '-ffp-contract' flag.)
    // Without this, PJ-DSL performance does not match JIT frontends
    // that use FMA instructions.
    // TODO: Make such things configurable.
    FastMathFlags FMF;
    FMF.setAllowContract(true);
    IRB.setFastMathFlags(FMF);
  }
};

LLVMCodeBuilder::LLVMCodeBuilder(std::unique_ptr<LLVMContext> Ctx,
                                 std::unique_ptr<Module> Mod,
                                 TargetModelType TM)
    : PImpl{std::make_unique<Impl>(std::move(Ctx), std::move(Mod))}, F(nullptr),
      TargetModel(TM) {
#if LLVM_VERSION_MAJOR >= 21
  getModule().setTargetTriple(Triple{getTargetTriple(TM)});
#else
  getModule().setTargetTriple(getTargetTriple(TM));
#endif
  // TODO: Create a TargetArch abstraction to encapsulate this kind of logic for
  // both host and device targets.
  auto TMExpected = detail::createTargetMachine(
      getModule(), isHostTargetModel(TM) ? sys::getHostCPUName() : "");
  if (!TMExpected)
    reportFatalError("LLVMCodeBuilder: failed to create target machine for " +
                     getTargetTriple(TM));
  getModule().setDataLayout((*TMExpected)->createDataLayout());
}

LLVMCodeBuilder::~LLVMCodeBuilder() = default;

IRBuilderBase &LLVMCodeBuilder::getIRBuilder() { return PImpl->IRB; }

Function &LLVMCodeBuilder::getFunction() {
  if (!F)
    reportFatalError("LLVMCodeBuilder: no active function");
  return *F;
}

Module &LLVMCodeBuilder::getModule() {
  if (PImpl->OwnedMod)
    return *PImpl->OwnedMod;
  if (!F)
    reportFatalError("LLVMCodeBuilder: no active function or owned module");
  return *F->getParent();
}

LLVMContext &LLVMCodeBuilder::getContext() {
  if (PImpl->OwnedCtx)
    return *PImpl->OwnedCtx;
  if (!F)
    reportFatalError("LLVMCodeBuilder: no active function or owned context");
  return F->getContext();
}

IRFunction *LLVMCodeBuilder::addFunction(const std::string &Name, IRType RetTy,
                                         const std::vector<IRType> &ArgTys,
                                         bool IsKernel) {
  if (!PImpl->OwnedMod)
    reportFatalError("addFunction requires an owning LLVMCodeBuilder");
  auto &Ctx = getContext();
  Type *LLVMRetTy = toLLVMType(RetTy, Ctx);
  std::vector<Type *> LLVMArgTys;
  LLVMArgTys.reserve(ArgTys.size());
  for (const auto &T : ArgTys)
    LLVMArgTys.push_back(toLLVMType(T, Ctx));
  auto FC = PImpl->OwnedMod->getOrInsertFunction(
      Name, FunctionType::get(LLVMRetTy, LLVMArgTys, false));
  Function *Fn = dyn_cast<Function>(FC.getCallee());
  if (!Fn)
    reportFatalError("Expected LLVM Function");
  BasicBlock::Create(Fn->getContext(), "entry", Fn);
  F = Fn;
  IRFunction *IRF = PImpl->wrapFunction(Fn);
  if (IsKernel) {
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
    setKernel(IRF);
#else
    reportFatalError("addFunction: IsKernel " + std::to_string(IsKernel) +
                     " kernel functions are not supported for host target");
#endif
  }

  return IRF;
}

void LLVMCodeBuilder::setFunctionName(IRFunction *IRF,
                                      const std::string &Name) {
  unwrapFunction(IRF)->setName(Name);
}

std::unique_ptr<LLVMContext> LLVMCodeBuilder::takeLLVMContext() {
  return std::move(PImpl->OwnedCtx);
}

std::unique_ptr<Module> LLVMCodeBuilder::takeModule() {
  return std::move(PImpl->OwnedMod);
}

// Insert point management.
void LLVMCodeBuilder::setInsertPoint(BasicBlock *BB) {
  PImpl->IRB.SetInsertPoint(BB);
}

void LLVMCodeBuilder::setInsertPointBegin(BasicBlock *BB) {
  PImpl->IP = IRBuilderBase::InsertPoint(BB, BB->begin());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::setInsertPointAtEntry() {
  auto &EntryBB = F->getEntryBlock();
  PImpl->IP = IRBuilderBase::InsertPoint(&EntryBB, EntryBB.end());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::clearInsertPoint() { PImpl->IRB.ClearInsertionPoint(); }

BasicBlock *LLVMCodeBuilder::getInsertBlock() {
  return PImpl->IRB.GetInsertBlock();
}

// Basic block operations.
std::tuple<BasicBlock *, BasicBlock *> LLVMCodeBuilder::splitCurrentBlock() {
  BasicBlock *CurBlock = PImpl->IP.getBlock();
  BasicBlock *NextBlock = CurBlock->splitBasicBlock(
      PImpl->IP.getPoint(), CurBlock->getName() + ".split");
  return {CurBlock, NextBlock};
}

BasicBlock *LLVMCodeBuilder::createBasicBlock(const std::string &Name,
                                              BasicBlock *InsertBefore) {
  BasicBlock *BB = BasicBlock::Create(getContext(), Name, F, InsertBefore);
  return BB;
}

void LLVMCodeBuilder::eraseTerminator(BasicBlock *BB) {
  if (!BB->getTerminator())
    reportFatalError("Basic block has no terminator to erase");

  BB->getTerminator()->eraseFromParent();
}

BasicBlock *LLVMCodeBuilder::getUniqueSuccessor(BasicBlock *BB) {
  auto *Succ = BB->getUniqueSuccessor();
  if (!Succ)
    reportFatalError("Expected unique successor for basic block " +
                     BB->getName().str());
  return Succ;
}

// Scope management.
void LLVMCodeBuilder::pushScope(const char *File, int Line, ScopeKind Kind,
                                BasicBlock *NextBlock) {
  IRBuilderBase::InsertPoint ContIP{NextBlock, NextBlock->begin()};
  PImpl->Scopes.emplace_back(File, Line, Kind, ContIP);
}

// High-level scope operations.
void LLVMCodeBuilder::beginFunction(IRFunction *IRF, const char *File,
                                    int Line) {
  F = unwrapFunction(IRF);
  BasicBlock *BodyBB = BasicBlock::Create(F->getContext(), "body", F);
  BasicBlock *ExitBB = BasicBlock::Create(F->getContext(), "exit", F);
  PImpl->IP =
      IRBuilderBase::InsertPoint(&F->getEntryBlock(), F->getEntryBlock().end());
  PImpl->IRB.restoreIP(PImpl->IP);
  PImpl->IRB.CreateBr(BodyBB);

  PImpl->IP = IRBuilderBase::InsertPoint(BodyBB, BodyBB->end());
  PImpl->IRB.restoreIP(PImpl->IP);
  PImpl->IRB.CreateBr(ExitBB);

  PImpl->IRB.SetInsertPoint(ExitBB);
  {
    PImpl->IRB.CreateUnreachable();
  }

  PImpl->IP = IRBuilderBase::InsertPoint(BodyBB, BodyBB->begin());
  PImpl->IRB.restoreIP(PImpl->IP);

  PImpl->Scopes.emplace_back(
      File, Line, ScopeKind::FUNCTION,
      IRBuilderBase::InsertPoint(ExitBB, ExitBB->begin()));
}

void LLVMCodeBuilder::endFunction() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected FUNCTION scope");

  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::FUNCTION)
    reportFatalError("Syntax error, expected FUNCTION end scope but found "
                     "unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));
  PImpl->Scopes.pop_back();
}

void LLVMCodeBuilder::beginIf(IRValue *Cond, const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = PImpl->IP.getBlock();
  BasicBlock *NextBlock = CurBlock->splitBasicBlock(
      PImpl->IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  PImpl->Scopes.emplace_back(File, Line, ScopeKind::IF, ContIP);

  BasicBlock *ThenBlock =
      BasicBlock::Create(F->getContext(), "if.then", F, NextBlock);
  BasicBlock *ExitBlock =
      BasicBlock::Create(F->getContext(), "if.cont", F, NextBlock);

  CurBlock->getTerminator()->eraseFromParent();
  PImpl->IRB.SetInsertPoint(CurBlock);
  {
    PImpl->IRB.CreateCondBr(PImpl->unwrap(Cond), ThenBlock, ExitBlock);
  }

  PImpl->IRB.SetInsertPoint(ThenBlock);
  {
    PImpl->IRB.CreateBr(ExitBlock);
  }

  PImpl->IRB.SetInsertPoint(ExitBlock);
  {
    PImpl->IRB.CreateBr(NextBlock);
  }

  PImpl->IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::endIf() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected IF scope");
  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::IF)
    reportFatalError("Syntax error, expected IF end scope but "
                     "found unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));

  PImpl->IP = S.ContIP;
  PImpl->Scopes.pop_back();

  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::beginFor(IRValue *IterSlot, IRType IterTy,
                               IRValue *InitVal, IRValue *UpperBoundVal,
                               IRValue *IncVal, bool IsSigned, const char *File,
                               int Line, LoopHints Hints) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = splitCurrentBlock();
  pushScope(File, Line, ScopeKind::FOR, NextBlock);

  llvm::BasicBlock *Header = createBasicBlock("loop.header", NextBlock);
  llvm::BasicBlock *LoopCond = createBasicBlock("loop.cond", NextBlock);
  llvm::BasicBlock *Body = createBasicBlock("loop.body", NextBlock);
  llvm::BasicBlock *Latch = createBasicBlock("loop.inc", NextBlock);
  llvm::BasicBlock *LoopExit = createBasicBlock("loop.end", NextBlock);

  // Erase the old terminator and branch to the header.
  eraseTerminator(CurBlock);
  setInsertPoint(CurBlock);
  {
    createBr(Header);
  }

  setInsertPoint(Header);
  {
    createStore(InitVal, IterSlot);
    createBr(LoopCond);
  }

  setInsertPoint(LoopCond);
  {
    IRValue *Iter = createLoad(IterTy, IterSlot);
    IRValue *Cond = IsSigned ? createICmpSLT(Iter, UpperBoundVal)
                             : createICmpULT(Iter, UpperBoundVal);
    createCondBr(Cond, Body, LoopExit);
  }

  setInsertPoint(Body);
  createBr(Latch);

  setInsertPoint(Latch);
  {
    IRValue *Iter = createLoad(IterTy, IterSlot);
    IRValue *Next = createAdd(Iter, IncVal);
    createStore(Next, IterSlot);
    createBr(LoopCond);
  }

  if (Hints.Unroll) {
    Instruction *BackEdgeBr = Latch->getTerminator();
    LLVMContext &Ctx = BackEdgeBr->getContext();

    SmallVector<Metadata *, 4> LoopMDOperands;
    LoopMDOperands.push_back(nullptr); // self-reference placeholder

    MDNode *UnrollEnableMD =
        MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.enable"));
    LoopMDOperands.push_back(UnrollEnableMD);

    if (Hints.UnrollCount.has_value()) {
      Metadata *CountOperands[] = {
          MDString::get(Ctx, "llvm.loop.unroll.count"),
          ConstantAsMetadata::get(
              ConstantInt::get(Type::getInt32Ty(Ctx), *Hints.UnrollCount))};
      MDNode *UnrollCountMD = MDNode::get(Ctx, CountOperands);
      LoopMDOperands.push_back(UnrollCountMD);
    }

    MDNode *LoopMD = MDNode::getDistinct(Ctx, LoopMDOperands);
    LoopMD->replaceOperandWith(0, LoopMD);
    BackEdgeBr->setMetadata(LLVMContext::MD_loop, LoopMD);
  }

  setInsertPoint(LoopExit);
  {
    createBr(NextBlock);
  }

  setInsertPointBegin(Body);
}

void LLVMCodeBuilder::endFor() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected FOR scope");

  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::FOR)
    reportFatalError("Syntax error, expected FOR end scope but "
                     "found unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));

  PImpl->IP = S.ContIP;
  PImpl->Scopes.pop_back();

  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::beginWhile(std::function<IRValue *()> CondFn,
                                 const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = splitCurrentBlock();
  pushScope(File, Line, ScopeKind::WHILE, NextBlock);

  llvm::BasicBlock *LoopCond = createBasicBlock("while.cond", NextBlock);
  llvm::BasicBlock *Body = createBasicBlock("while.body", NextBlock);
  llvm::BasicBlock *LoopExit = createBasicBlock("while.end", NextBlock);

  eraseTerminator(CurBlock);
  setInsertPoint(CurBlock);
  {
    createBr(LoopCond);
  }

  setInsertPoint(LoopCond);
  {
    IRValue *CondV = CondFn();
    createCondBr(CondV, Body, LoopExit);
  }

  setInsertPoint(Body);
  createBr(LoopCond);

  setInsertPoint(LoopExit);
  {
    createBr(NextBlock);
  }

  setInsertPointBegin(Body);
}

void LLVMCodeBuilder::endWhile() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected WHILE scope");

  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::WHILE)
    reportFatalError("Syntax error, expected WHILE end scope but "
                     "found unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));

  PImpl->IP = S.ContIP;
  PImpl->Scopes.pop_back();

  PImpl->IRB.restoreIP(PImpl->IP);
}

// Arithmetic operations.
IRValue *LLVMCodeBuilder::createAdd(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateAdd(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFAdd(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFAdd(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createSub(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateSub(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFSub(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFSub(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createMul(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateMul(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFMul(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFMul(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createUDiv(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateUDiv(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createSDiv(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateSDiv(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFDiv(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFDiv(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createURem(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateURem(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createSRem(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateSRem(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFRem(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFRem(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}

// Atomic operations.
IRValue *LLVMCodeBuilder::createAtomicAdd(IRValue *Addr, IRValue *Val) {
  Value *LLVMVal = PImpl->unwrap(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FAdd
                                                    : AtomicRMWInst::Add;

  return PImpl->wrap(
      PImpl->IRB.CreateAtomicRMW(Op, PImpl->unwrap(Addr), LLVMVal, MaybeAlign(),
                                 AtomicOrdering::SequentiallyConsistent));
}

IRValue *LLVMCodeBuilder::createAtomicSub(IRValue *Addr, IRValue *Val) {
  Value *LLVMVal = PImpl->unwrap(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FSub
                                                    : AtomicRMWInst::Sub;

  return PImpl->wrap(
      PImpl->IRB.CreateAtomicRMW(Op, PImpl->unwrap(Addr), LLVMVal, MaybeAlign(),
                                 AtomicOrdering::SequentiallyConsistent));
}

IRValue *LLVMCodeBuilder::createAtomicMax(IRValue *Addr, IRValue *Val) {
  Value *LLVMVal = PImpl->unwrap(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FMax
                                                    : AtomicRMWInst::Max;

  return PImpl->wrap(
      PImpl->IRB.CreateAtomicRMW(Op, PImpl->unwrap(Addr), LLVMVal, MaybeAlign(),
                                 AtomicOrdering::SequentiallyConsistent));
}

IRValue *LLVMCodeBuilder::createAtomicMin(IRValue *Addr, IRValue *Val) {
  Value *LLVMVal = PImpl->unwrap(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FMin
                                                    : AtomicRMWInst::Min;

  return PImpl->wrap(
      PImpl->IRB.CreateAtomicRMW(Op, PImpl->unwrap(Addr), LLVMVal, MaybeAlign(),
                                 AtomicOrdering::SequentiallyConsistent));
}

// Comparison operations.
IRValue *LLVMCodeBuilder::createICmpEQ(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpEQ(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpNE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpNE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpSLT(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpSLT(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpSGT(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpSGT(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpSGE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpSGE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpSLE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpSLE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpUGT(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpUGT(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpUGE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpUGE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpULT(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpULT(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createICmpULE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateICmpULE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpOEQ(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpOEQ(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpONE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpONE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpOLT(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpOLT(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpOLE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpOLE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpOGT(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpOGT(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpOGE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpOGE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpULT(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpULT(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createFCmpULE(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateFCmpULE(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}

// Logical operations.
IRValue *LLVMCodeBuilder::createAnd(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateAnd(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createOr(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateOr(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createXor(IRValue *LHS, IRValue *RHS) {
  return PImpl->wrap(
      PImpl->IRB.CreateXor(PImpl->unwrap(LHS), PImpl->unwrap(RHS)));
}
IRValue *LLVMCodeBuilder::createNot(IRValue *Val) {
  return PImpl->wrap(PImpl->IRB.CreateNot(PImpl->unwrap(Val)));
}

// Load/Store operations.
IRValue *LLVMCodeBuilder::createLoad(IRType Ty, IRValue *Ptr,
                                     const std::string &Name) {
  return PImpl->wrap(PImpl->IRB.CreateLoad(toLLVMType(Ty, getContext()),
                                           PImpl->unwrap(Ptr), Name));
}

void LLVMCodeBuilder::createStore(IRValue *Val, IRValue *Ptr) {
  PImpl->IRB.CreateStore(PImpl->unwrap(Val), PImpl->unwrap(Ptr));
}

// Control flow operations.
void LLVMCodeBuilder::createBr(BasicBlock *Dest) { PImpl->IRB.CreateBr(Dest); }

void LLVMCodeBuilder::createCondBr(IRValue *Cond, BasicBlock *True,
                                   BasicBlock *False) {
  PImpl->IRB.CreateCondBr(PImpl->unwrap(Cond), True, False);
}

void LLVMCodeBuilder::createRetVoid() {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRetVoid();

  TermI->eraseFromParent();
}

void LLVMCodeBuilder::createRet(IRValue *V) {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRet(PImpl->unwrap(V));

  TermI->eraseFromParent();
}

// Cast operations.
IRValue *LLVMCodeBuilder::createIntCast(IRValue *V, IRType DestTy,
                                        bool IsSigned) {
  return PImpl->wrap(PImpl->IRB.CreateIntCast(
      PImpl->unwrap(V), toLLVMType(DestTy, getContext()), IsSigned));
}

IRValue *LLVMCodeBuilder::createFPCast(IRValue *V, IRType DestTy) {
  return PImpl->wrap(PImpl->IRB.CreateFPCast(PImpl->unwrap(V),
                                             toLLVMType(DestTy, getContext())));
}

IRValue *LLVMCodeBuilder::createSIToFP(IRValue *V, IRType DestTy) {
  return PImpl->wrap(PImpl->IRB.CreateSIToFP(PImpl->unwrap(V),
                                             toLLVMType(DestTy, getContext())));
}
IRValue *LLVMCodeBuilder::createUIToFP(IRValue *V, IRType DestTy) {
  return PImpl->wrap(PImpl->IRB.CreateUIToFP(PImpl->unwrap(V),
                                             toLLVMType(DestTy, getContext())));
}
IRValue *LLVMCodeBuilder::createFPToSI(IRValue *V, IRType DestTy) {
  return PImpl->wrap(PImpl->IRB.CreateFPToSI(PImpl->unwrap(V),
                                             toLLVMType(DestTy, getContext())));
}
IRValue *LLVMCodeBuilder::createFPToUI(IRValue *V, IRType DestTy) {
  return PImpl->wrap(PImpl->IRB.CreateFPToUI(PImpl->unwrap(V),
                                             toLLVMType(DestTy, getContext())));
}
IRValue *LLVMCodeBuilder::createBitCast(IRValue *V, IRType DestTy) {
  Value *LLVMv = PImpl->unwrap(V);
  Type *DestLLVMTy = toLLVMType(DestTy, getContext());
  if (LLVMv->getType() == DestLLVMTy)
    return V;

  return PImpl->wrap(PImpl->IRB.CreateBitCast(LLVMv, DestLLVMTy));
}

IRValue *LLVMCodeBuilder::createZExt(IRValue *V, IRType DestTy) {
  return PImpl->wrap(PImpl->IRB.CreateZExt(PImpl->unwrap(V),
                                           toLLVMType(DestTy, getContext())));
}

// Constant creation.
IRValue *LLVMCodeBuilder::getConstantInt(IRType Ty, uint64_t Val) {
  return PImpl->wrap(ConstantInt::get(toLLVMType(Ty, getContext()), Val));
}
IRValue *LLVMCodeBuilder::getConstantFP(IRType Ty, double Val) {
  return PImpl->wrap(ConstantFP::get(toLLVMType(Ty, getContext()), Val));
}

// GEP operations.
IRValue *
LLVMCodeBuilder::createInBoundsGEP(IRType Ty, IRValue *Ptr,
                                   const std::vector<IRValue *> &IdxList,
                                   const std::string &Name) {
  std::vector<Value *> LLVMIdxList;
  LLVMIdxList.reserve(IdxList.size());
  for (const auto &I : IdxList)
    LLVMIdxList.push_back(PImpl->unwrap(I));
  return PImpl->wrap(PImpl->IRB.CreateInBoundsGEP(
      toLLVMType(Ty, getContext()), PImpl->unwrap(Ptr), LLVMIdxList, Name));
}
IRValue *LLVMCodeBuilder::createConstInBoundsGEP1_64(IRType Ty, IRValue *Ptr,
                                                     size_t Idx) {
  return PImpl->wrap(PImpl->IRB.CreateConstInBoundsGEP1_64(
      toLLVMType(Ty, getContext()), PImpl->unwrap(Ptr), Idx));
}

IRValue *LLVMCodeBuilder::createConstInBoundsGEP2_64(IRType Ty, IRValue *Ptr,
                                                     size_t Idx0, size_t Idx1) {
  return PImpl->wrap(PImpl->IRB.CreateConstInBoundsGEP2_64(
      toLLVMType(Ty, getContext()), PImpl->unwrap(Ptr), Idx0, Idx1));
}

// ---------------------------------------------------------------------------
// Semantic arithmetic / comparison / cast dispatch
// ---------------------------------------------------------------------------

IRValue *LLVMCodeBuilder::createArith(ArithOp Op, IRValue *LHS, IRValue *RHS,
                                      IRType Ty) {
  if (isIntegerKind(Ty)) {
    switch (Op) {
    case ArithOp::Add:
      return createAdd(LHS, RHS);
    case ArithOp::Sub:
      return createSub(LHS, RHS);
    case ArithOp::Mul:
      return createMul(LHS, RHS);
    case ArithOp::Div:
      return Ty.Signed ? createSDiv(LHS, RHS) : createUDiv(LHS, RHS);
    case ArithOp::Rem:
      return Ty.Signed ? createSRem(LHS, RHS) : createURem(LHS, RHS);
    }
  } else if (isFloatingPointKind(Ty)) {
    switch (Op) {
    case ArithOp::Add:
      return createFAdd(LHS, RHS);
    case ArithOp::Sub:
      return createFSub(LHS, RHS);
    case ArithOp::Mul:
      return createFMul(LHS, RHS);
    case ArithOp::Div:
      return createFDiv(LHS, RHS);
    case ArithOp::Rem:
      return createFRem(LHS, RHS);
    }
  }
  reportFatalError("createArith: unsupported type kind");
}

IRValue *LLVMCodeBuilder::createCmp(CmpOp Op, IRValue *LHS, IRValue *RHS,
                                    IRType Ty) {
  if (isIntegerKind(Ty)) {
    switch (Op) {
    case CmpOp::EQ:
      return createICmpEQ(LHS, RHS);
    case CmpOp::NE:
      return createICmpNE(LHS, RHS);
    case CmpOp::LT:
      return Ty.Signed ? createICmpSLT(LHS, RHS) : createICmpULT(LHS, RHS);
    case CmpOp::LE:
      return Ty.Signed ? createICmpSLE(LHS, RHS) : createICmpULE(LHS, RHS);
    case CmpOp::GT:
      return Ty.Signed ? createICmpSGT(LHS, RHS) : createICmpUGT(LHS, RHS);
    case CmpOp::GE:
      return Ty.Signed ? createICmpSGE(LHS, RHS) : createICmpUGE(LHS, RHS);
    }
  } else if (isFloatingPointKind(Ty)) {
    switch (Op) {
    case CmpOp::EQ:
      return createFCmpOEQ(LHS, RHS);
    case CmpOp::NE:
      return createFCmpONE(LHS, RHS);
    case CmpOp::LT:
      return createFCmpOLT(LHS, RHS);
    case CmpOp::LE:
      return createFCmpOLE(LHS, RHS);
    case CmpOp::GT:
      return createFCmpOGT(LHS, RHS);
    case CmpOp::GE:
      return createFCmpOGE(LHS, RHS);
    }
  }
  reportFatalError("createCmp: unsupported type kind");
}

IRValue *LLVMCodeBuilder::createCast(IRValue *V, IRType FromTy, IRType ToTy) {
  if (isIntegerKind(FromTy) && isFloatingPointKind(ToTy))
    return FromTy.Signed ? createSIToFP(V, ToTy) : createUIToFP(V, ToTy);
  if (isFloatingPointKind(FromTy) && isIntegerKind(ToTy))
    return ToTy.Signed ? createFPToSI(V, ToTy) : createFPToUI(V, ToTy);
  if (isIntegerKind(FromTy) && isIntegerKind(ToTy))
    return createIntCast(V, ToTy, FromTy.Signed);
  if (isFloatingPointKind(FromTy) && isFloatingPointKind(ToTy))
    return createFPCast(V, ToTy);
  reportFatalError("createCast: unsupported type combination");
}

VarAlloc LLVMCodeBuilder::getElementPtr(IRValue *Base, IRType BaseTy,
                                        IRValue *Index, IRType ElemTy) {
  if (BaseTy.Kind == IRTypeKind::Pointer) {
    unsigned AS = BaseTy.AddrSpace;
    IRValue *GEP = createInBoundsGEP(ElemTy, Base, {Index});
    auto A = allocPointer("elem.ptr", ElemTy, AS);
    storeAddress(A.Slot, GEP);
    return A;
  }
  if (BaseTy.Kind == IRTypeKind::Array) {
    unsigned AS = getAddressSpaceFromValue(Base);
    IRValue *Zero = getConstantInt(IRType{IRTypeKind::Int64}, 0);
    IRValue *GEP = createInBoundsGEP(BaseTy, Base, {Zero, Index});
    auto A = allocPointer("elem.ptr", ElemTy, AS);
    storeAddress(A.Slot, GEP);
    return A;
  }
  reportFatalError("getElementPtr: unsupported base type kind");
}

VarAlloc LLVMCodeBuilder::getElementPtr(IRValue *Base, IRType BaseTy,
                                        size_t Index, IRType ElemTy) {
  if (BaseTy.Kind == IRTypeKind::Pointer) {
    unsigned AS = BaseTy.AddrSpace;
    IRValue *GEP = createConstInBoundsGEP1_64(ElemTy, Base, Index);
    auto A = allocPointer("elem.ptr", ElemTy, AS);
    storeAddress(A.Slot, GEP);
    return A;
  }
  if (BaseTy.Kind == IRTypeKind::Array) {
    unsigned AS = getAddressSpaceFromValue(Base);
    IRValue *GEP = createConstInBoundsGEP2_64(BaseTy, Base, 0, Index);
    auto A = allocPointer("elem.ptr", ElemTy, AS);
    storeAddress(A.Slot, GEP);
    return A;
  }
  reportFatalError("getElementPtr: unsupported base type kind");
}

// Type accessors.
Type *LLVMCodeBuilder::getPointerType(unsigned AS) {
  return PointerType::get(getContext(), AS);
}
Type *LLVMCodeBuilder::getPointerTypeUnqual() {
  return PointerType::getUnqual(getContext());
}
Type *LLVMCodeBuilder::getInt16Ty() { return PImpl->IRB.getInt16Ty(); }
Type *LLVMCodeBuilder::getInt32Ty() { return PImpl->IRB.getInt32Ty(); }
Type *LLVMCodeBuilder::getInt64Ty() { return PImpl->IRB.getInt64Ty(); }
Type *LLVMCodeBuilder::getFloatTy() { return PImpl->IRB.getFloatTy(); }

// Type queries.
unsigned LLVMCodeBuilder::getAddressSpace(Type *Ty) {
  auto *PtrTy = dyn_cast<PointerType>(Ty);
  if (!PtrTy)
    reportFatalError("Expected LLVM PointerType for getAddressSpace");

  return PtrTy->getAddressSpace();
}

unsigned LLVMCodeBuilder::getAddressSpaceFromValue(IRValue *PtrVal) {
  return getAddressSpace(PImpl->unwrap(PtrVal)->getType());
}

bool LLVMCodeBuilder::isIntegerTy(Type *Ty) { return Ty->isIntegerTy(); }
bool LLVMCodeBuilder::isFloatingPointTy(Type *Ty) {
  return Ty->isFloatingPointTy();
}

// Call operations.
IRValue *LLVMCodeBuilder::createCall(const std::string &FName, IRType RetTy,
                                     const std::vector<IRType> &ArgTys,
                                     const std::vector<IRValue *> &Args) {
  auto &Ctx = getContext();
  Type *LLVMRetTy = toLLVMType(RetTy, Ctx);
  std::vector<Type *> LLVMArgTys;
  LLVMArgTys.reserve(ArgTys.size());
  for (const auto &T : ArgTys)
    LLVMArgTys.push_back(toLLVMType(T, Ctx));
  std::vector<Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (const auto &A : Args)
    LLVMArgs.push_back(PImpl->unwrap(A));
  Module *M = &getModule();
  FunctionType *FnTy = FunctionType::get(LLVMRetTy, LLVMArgTys, false);
  FunctionCallee Callee = M->getOrInsertFunction(FName, FnTy);
  return PImpl->wrap(PImpl->IRB.CreateCall(Callee, LLVMArgs));
}

IRValue *LLVMCodeBuilder::createCall(const std::string &FName, IRType RetTy) {
  Type *LLVMRetTy = toLLVMType(RetTy, getContext());
  Module *M = &getModule();
  FunctionType *FnTy = FunctionType::get(LLVMRetTy, {}, false);
  FunctionCallee Callee = M->getOrInsertFunction(FName, FnTy);
  return PImpl->wrap(PImpl->IRB.CreateCall(Callee));
}

IRValue *LLVMCodeBuilder::emitIntrinsic(const std::string &Name, IRType RetTy,
                                        const std::vector<IRValue *> &Args) {
  std::vector<IRType> ArgTys(Args.size(), RetTy);

  if (Name == "powf") {
    if (Args.size() != 2)
      reportFatalError("powf expects 2 arguments");
    // CUDA lowers to libdevice entry points while host/HIP use LLVM intrinsics
    // so later target-specific passes can legalize appropriately.
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_powf" : "llvm.pow.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }

  if (Name == "sqrtf") {
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_sqrtf" : "llvm.sqrt.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }
  if (Name == "expf") {
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_expf" : "llvm.exp.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }
  if (Name == "sinf") {
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_sinf" : "llvm.sin.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }
  if (Name == "cosf") {
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_cosf" : "llvm.cos.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }
  if (Name == "fabsf" || Name == "absf") {
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_fabsf" : "llvm.fabs.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }
  if (Name == "truncf") {
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_truncf" : "llvm.trunc.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }
  if (Name == "logf") {
    const std::string Callee =
        isCudaDeviceTarget(TargetModel) ? "__nv_logf" : "llvm.log.f32";
    return createCall(Callee, RetTy, ArgTys, Args);
  }

  reportFatalError("LLVMCodeBuilder::emitIntrinsic: unsupported intrinsic " +
                   Name);
}

IRValue *LLVMCodeBuilder::emitBuiltin(const std::string &Name, IRType RetTy,
                                      const std::vector<IRValue *> &Args) {
  if (!Args.empty())
    reportFatalError("LLVMCodeBuilder::emitBuiltin expects no arguments");

  if (isCudaDeviceTarget(TargetModel))
    return emitCudaBuiltin(*this, Name, RetTy);
  if (isHipDeviceTarget(TargetModel))
    return emitHipBuiltin(*this, Name, RetTy);

  reportFatalError("LLVMCodeBuilder::emitBuiltin: builtin not supported on "
                   "host target model");
}

// Alloca/array emission.
IRValue *LLVMCodeBuilder::emitAlloca(Type *Ty, const std::string &Name,
                                     AddressSpace AS) {
  const unsigned AllocaAS = getModule().getDataLayout().getAllocaAddrSpace();
  auto SaveIP = PImpl->IRB.saveIP();
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  PImpl->IRB.restoreIP(AllocaIP);
  Value *Slot = PImpl->IRB.CreateAlloca(Ty, AllocaAS, nullptr, Name);
  // Targets such as AMDGPU require allocas in a dedicated address space, so
  // reach the requested one through a cast, the same way Clang does.
  if (static_cast<unsigned>(AS) != AllocaAS)
    Slot = PImpl->IRB.CreateAddrSpaceCast(
        Slot, PointerType::get(getContext(), static_cast<unsigned>(AS)),
        Name + ".ascast");

  PImpl->IRB.restoreIP(SaveIP);
  return PImpl->wrap(Slot);
}

IRValue *LLVMCodeBuilder::emitArrayCreate(Type *Ty, AddressSpace AT,
                                          const std::string &Name) {
  if (!Ty || !Ty->isArrayTy())
    reportFatalError("Expected LLVM ArrayType for emitArrayCreate");

  auto *ArrTy = cast<ArrayType>(Ty);

  switch (AT) {
  case AddressSpace::SHARED:
  case AddressSpace::GLOBAL: {
    Module *M = &getModule();
    auto *GV = new GlobalVariable(
        *M, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
        UndefValue::get(ArrTy), Name, /*InsertBefore=*/nullptr,
        GlobalValue::NotThreadLocal, static_cast<unsigned>(AT),
        /*ExternallyInitialized=*/false);

    return PImpl->wrap(GV);
  }
  case AddressSpace::DEFAULT:
  case AddressSpace::LOCAL: {
    return emitAlloca(ArrTy, Name, AT);
  }
  case AddressSpace::CONSTANT:
    reportFatalError("Constant arrays are not supported");
  default:
    reportFatalError("Unsupported AddressSpace");
  }
}

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
void LLVMCodeBuilder::setKernel(IRFunction *IRF) {
  Function &Fn = *unwrapFunction(IRF);
  LLVMContext &Ctx = getContext();
  switch (TargetModel) {
  case TargetModelType::CUDA: {
#if LLVM_VERSION_MAJOR < 22
    NamedMDNode *MD = getModule().getOrInsertNamedMetadata("nvvm.annotations");

    Metadata *MDVals[] = {
        ConstantAsMetadata::get(&Fn), MDString::get(Ctx, "kernel"),
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1))};
    // Append metadata to nvvm.annotations.
    MD->addOperand(MDNode::get(Ctx, MDVals));
#endif

    // Add a function attribute for the kernel.
    Fn.addFnAttr(Attribute::get(Ctx, "kernel"));
#if LLVM_VERSION_MAJOR >= 20
    Fn.setCallingConv(CallingConv::PTX_Kernel);
#endif
    return;
  }
  case TargetModelType::HIP:
    Fn.setCallingConv(CallingConv::AMDGPU_KERNEL);
    return;
  case TargetModelType::HOST:
    reportFatalError("Host does not support setKernel");
  default:
    reportFatalError("Unsupported target " + getTargetTriple(TargetModel) +
                     " for setKernel");
  }
}

void LLVMCodeBuilder::setLaunchBoundsForKernel(IRFunction *IRF,
                                               int MaxThreadsPerBlock,
                                               int MinBlocksPerSM) {
  proteus::setLaunchBoundsForKernel(*unwrapFunction(IRF), MaxThreadsPerBlock,
                                    MinBlocksPerSM);
}
#endif

// ---------------------------------------------------------------------------
// Storage-aware load / store methods
// ---------------------------------------------------------------------------

IRValue *LLVMCodeBuilder::loadScalar(IRValue *Slot, IRType ValueTy) {
  return PImpl->wrap(PImpl->IRB.CreateLoad(toLLVMType(ValueTy, getContext()),
                                           PImpl->unwrap(Slot)));
}

void LLVMCodeBuilder::storeScalar(IRValue *Slot, IRValue *Val) {
  PImpl->IRB.CreateStore(PImpl->unwrap(Val), PImpl->unwrap(Slot));
}

IRValue *LLVMCodeBuilder::loadAddress(IRValue *Slot, IRType AllocTy) {
  return PImpl->wrap(PImpl->IRB.CreateLoad(toLLVMType(AllocTy, getContext()),
                                           PImpl->unwrap(Slot)));
}

void LLVMCodeBuilder::storeAddress(IRValue *Slot, IRValue *Addr) {
  PImpl->IRB.CreateStore(PImpl->unwrap(Addr), PImpl->unwrap(Slot));
}

IRValue *LLVMCodeBuilder::loadFromPointee(IRValue *Slot, IRType AllocTy,
                                          IRType ValueTy) {
  auto &Ctx = getContext();
  Value *Ptr =
      PImpl->IRB.CreateLoad(toLLVMType(AllocTy, Ctx), PImpl->unwrap(Slot));
  return PImpl->wrap(PImpl->IRB.CreateLoad(toLLVMType(ValueTy, Ctx), Ptr));
}

void LLVMCodeBuilder::storeToPointee(IRValue *Slot, IRType AllocTy,
                                     IRValue *Val) {
  Value *Ptr = PImpl->IRB.CreateLoad(toLLVMType(AllocTy, getContext()),
                                     PImpl->unwrap(Slot));
  PImpl->IRB.CreateStore(PImpl->unwrap(Val), Ptr);
}

// ---------------------------------------------------------------------------
// Alloca factory methods
// ---------------------------------------------------------------------------

VarAlloc LLVMCodeBuilder::allocScalar(const std::string &Name, IRType ValueTy) {
  Type *LLVMTy = toLLVMType(ValueTy, getContext());
  IRValue *Slot = emitAlloca(LLVMTy, Name);
  return {Slot, ValueTy, ValueTy, 0};
}

VarAlloc LLVMCodeBuilder::allocPointer(const std::string &Name, IRType ElemTy,
                                       unsigned AddrSpace) {
  Type *AllocaTy = PointerType::get(getContext(), AddrSpace);
  IRValue *Slot = emitAlloca(AllocaTy, Name);
  IRType AllocTy{IRTypeKind::Pointer, ElemTy.Signed, 0, ElemTy.Kind};
  AllocTy.AddrSpace = AddrSpace;
  return {Slot, ElemTy, AllocTy, AddrSpace};
}

VarAlloc LLVMCodeBuilder::allocArray(const std::string &Name, AddressSpace AS,
                                     IRType ElemTy, size_t NElem) {
  auto &Ctx = getContext();
  Type *ElemLLVMTy = toLLVMType(ElemTy, Ctx);
  Type *ArrTy = ArrayType::get(ElemLLVMTy, NElem);
  IRValue *Slot = emitArrayCreate(ArrTy, AS, Name);
  IRType AllocTy{IRTypeKind::Array, ElemTy.Signed, NElem, ElemTy.Kind};
  return {Slot, ElemTy, AllocTy, static_cast<unsigned>(AS)};
}

IRValue *LLVMCodeBuilder::getArg(IRFunction *IRF, size_t Idx) {
  return PImpl->wrap(unwrapFunction(IRF)->getArg(Idx));
}

llvm::Function *LLVMCodeBuilder::unwrapFunction(IRFunction *IRF) {
  return static_cast<LLVMIRFunction *>(IRF)->F;
}

} // namespace proteus
