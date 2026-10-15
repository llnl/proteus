//===-- LambdaASTChecker.cpp - Validate Proteus lambda API use -----------===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

bool isProteusFunction(const clang::CallExpr *Call, llvm::StringRef Name) {
  const clang::FunctionDecl *Callee = Call->getDirectCallee();
  return Callee && Callee->getQualifiedNameAsString() == Name;
}

const clang::Expr *ignoreTransparentExprs(const clang::Expr *Expr) {
  while (Expr) {
    const clang::Expr *Next = Expr->IgnoreParens();
    if (Next != Expr) {
      Expr = Next;
      continue;
    }
    if (const auto *E = llvm::dyn_cast<clang::CastExpr>(Expr))
      Next = E->getSubExpr();
    else if (const auto *E = llvm::dyn_cast<clang::ExprWithCleanups>(Expr))
      Next = E->getSubExpr();
    else if (const auto *E =
                 llvm::dyn_cast<clang::MaterializeTemporaryExpr>(Expr))
      Next = E->getSubExpr();
    else if (const auto *E = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(Expr))
      Next = E->getSubExpr();
    if (Next == Expr)
      break;
    Expr = Next;
  }
  return Expr;
}

const clang::Expr *ignoreCaptureInitWrappers(const clang::Expr *Expr) {
  Expr = ignoreTransparentExprs(Expr);
  if (const auto *List = llvm::dyn_cast_or_null<clang::InitListExpr>(Expr);
      List && List->getNumInits() == 1)
    return ignoreTransparentExprs(List->getInit(0));
  return Expr;
}

const clang::CXXRecordDecl *getCanonicalRecord(clang::QualType Type) {
  Type = Type.getNonReferenceType().getUnqualifiedType();
  if (const auto *Record = Type->getAsCXXRecordDecl())
    return Record->getCanonicalDecl();
  return nullptr;
}

bool isLambdaFunctorWrapper(const clang::CXXRecordDecl *Record) {
  const auto *Specialization =
      llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(Record);
  return Specialization &&
         Specialization->getSpecializedTemplate()->getQualifiedNameAsString() ==
             "proteus::detail::LambdaFunctorWrapper";
}

bool directlyReferencesTarget(
    const clang::Expr *Expr,
    const llvm::DenseSet<const clang::VarDecl *> &Targets) {
  Expr = ignoreTransparentExprs(Expr);
  const auto *Reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(Expr);
  const auto *Variable =
      Reference ? llvm::dyn_cast<clang::VarDecl>(Reference->getDecl())
                : nullptr;
  return Variable && Targets.contains(Variable->getCanonicalDecl());
}

bool takesAddressOfTarget(
    const clang::Expr *Expr,
    const llvm::DenseSet<const clang::VarDecl *> &Targets) {
  Expr = ignoreTransparentExprs(Expr);
  const auto *Operator = llvm::dyn_cast_or_null<clang::UnaryOperator>(Expr);
  return Operator && Operator->getOpcode() == clang::UO_AddrOf &&
         directlyReferencesTarget(Operator->getSubExpr(), Targets);
}

bool directlyWritesTarget(
    const clang::Expr *Expr,
    const llvm::DenseSet<const clang::VarDecl *> &Targets) {
  Expr = ignoreTransparentExprs(Expr);
  if (directlyReferencesTarget(Expr, Targets))
    return true;
  const auto *Dereference = llvm::dyn_cast_or_null<clang::UnaryOperator>(Expr);
  return Dereference && Dereference->getOpcode() == clang::UO_Deref &&
         takesAddressOfTarget(Dereference->getSubExpr(), Targets);
}

bool isMutableReference(clang::QualType Type) {
  if (!Type->isReferenceType())
    return false;
  return !Type.getNonReferenceType().isConstQualified();
}

bool isMutablePointer(clang::QualType Type) {
  return Type->isPointerType() && !Type->getPointeeType().isConstQualified();
}

const clang::VarDecl *getDirectReferencedVariable(const clang::Expr *Expr) {
  Expr = ignoreTransparentExprs(Expr);
  const auto *Reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(Expr);
  return Reference ? llvm::dyn_cast<clang::VarDecl>(Reference->getDecl())
                   : nullptr;
}

const clang::VarDecl *getAddressedVariable(const clang::Expr *Expr) {
  Expr = ignoreTransparentExprs(Expr);
  const auto *Operator = llvm::dyn_cast_or_null<clang::UnaryOperator>(Expr);
  if (!Operator || Operator->getOpcode() != clang::UO_AddrOf)
    return nullptr;
  return getDirectReferencedVariable(Operator->getSubExpr());
}

bool isPointerToLambdaFunctorWrapper(clang::QualType Type,
                                     unsigned MinimumDepth = 1) {
  unsigned Depth = 0;
  Type = Type.getNonReferenceType();
  while (Type->isPointerType()) {
    ++Depth;
    Type = Type->getPointeeType();
  }
  return Depth >= MinimumDepth &&
         isLambdaFunctorWrapper(getCanonicalRecord(Type));
}

// Whether Type is, or transitively holds by value, a registered lambda
// wrapper. This is the storage the kernel-arg analysis specializes from;
// envelopes and enclosing closures that capture a wrapper by value count.
bool typeContainsLambdaFunctorWrapper(clang::QualType Type,
                                      unsigned Depth = 0) {
  if (Depth > 8)
    return false;
  Type = Type.getNonReferenceType();
  if (const auto *Array = Type->getAsArrayTypeUnsafe())
    return typeContainsLambdaFunctorWrapper(Array->getElementType(),
                                            Depth + 1);
  const clang::CXXRecordDecl *Record = getCanonicalRecord(Type);
  if (!Record)
    return false;
  if (isLambdaFunctorWrapper(Record))
    return true;
  if (!Record->hasDefinition())
    return false;
  for (const clang::CXXBaseSpecifier &Base : Record->bases())
    if (typeContainsLambdaFunctorWrapper(Base.getType(), Depth + 1))
      return true;
  for (const clang::FieldDecl *Field : Record->fields())
    if (typeContainsLambdaFunctorWrapper(Field->getType(), Depth + 1))
      return true;
  return false;
}

// Destination of a memory intrinsic, resolved through the implicit void*
// decay: the pointee type plus the byte offset of the denoted lvalue within
// its outermost containing object (so &Envelope->Body yields Body's type at
// Body's offset inside Envelope, with the envelope as the root).
struct IntrinsicDestination {
  clang::QualType PointeeType; // type the pointer denotes
  clang::QualType RootType;    // outermost object the write lands in
  uint64_t OffsetInRoot = 0;   // byte offset of PointeeType within RootType
};

std::optional<IntrinsicDestination>
getIntrinsicDestination(const clang::Expr *Arg, clang::ASTContext &Context) {
  Arg = ignoreTransparentExprs(Arg);
  const clang::Expr *Lvalue = nullptr;
  if (const auto *AddrOf = llvm::dyn_cast_or_null<clang::UnaryOperator>(Arg);
      AddrOf && AddrOf->getOpcode() == clang::UO_AddrOf)
    Lvalue = ignoreTransparentExprs(AddrOf->getSubExpr());
  else if (Arg && Arg->getType()->isPointerType())
    return IntrinsicDestination{Arg->getType()->getPointeeType(),
                                Arg->getType()->getPointeeType(), 0};
  if (!Lvalue)
    return std::nullopt;

  // Accumulate constant member offsets through a (possibly nested) member
  // chain: &E.Prefix.Body, &E->Body, ... Bail (conservatively unknown offset
  // root) on anything else, e.g. array indexing with a dynamic index.
  uint64_t OffsetBits = 0;
  const clang::Expr *Current = Lvalue;
  while (const auto *Member =
             llvm::dyn_cast<clang::MemberExpr>(ignoreTransparentExprs(Current))) {
    const auto *Field =
        llvm::dyn_cast<clang::FieldDecl>(Member->getMemberDecl());
    if (!Field)
      return IntrinsicDestination{Lvalue->getType(), Lvalue->getType(), 0};
    OffsetBits += Context.getFieldOffset(Field);
    Current = Member->getBase();
    if (Member->isArrow()) {
      // The root is the pointee of the base pointer; stop the walk here.
      return IntrinsicDestination{Lvalue->getType(),
                                  Current->getType()->getPointeeType(),
                                  OffsetBits / 8};
    }
  }
  return IntrinsicDestination{Lvalue->getType(),
                              ignoreTransparentExprs(Current)->getType(),
                              OffsetBits / 8};
}

// Byte interval [Offset, Offset + Size) of the first wrapper found inside
// Root (by value, through bases/fields/arrays), or nullopt if none.
std::optional<std::pair<uint64_t, uint64_t>>
getWrapperIntervalInType(clang::QualType Root, clang::ASTContext &Context,
                         unsigned Depth = 0) {
  if (Depth > 8)
    return std::nullopt;
  Root = Root.getNonReferenceType();
  const clang::CXXRecordDecl *Record = getCanonicalRecord(Root);
  if (Record && isLambdaFunctorWrapper(Record))
    return std::make_pair(
        uint64_t{0}, Context.getTypeSizeInChars(Root).getQuantity());
  if (const auto *Array = Root->getAsArrayTypeUnsafe()) {
    if (auto Inner = getWrapperIntervalInType(Array->getElementType(), Context,
                                              Depth + 1))
      return Inner; // first element's instance suffices for overlap checks
    return std::nullopt;
  }
  if (!Record || !Record->hasDefinition())
    return std::nullopt;
  const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(Record);
  for (const clang::CXXBaseSpecifier &Base : Record->bases()) {
    if (auto Inner =
            getWrapperIntervalInType(Base.getType(), Context, Depth + 1)) {
      const auto *BaseRecord = getCanonicalRecord(Base.getType());
      uint64_t BaseOff =
          Layout.getBaseClassOffset(llvm::cast<clang::CXXRecordDecl>(
                                        BaseRecord))
              .getQuantity();
      return std::make_pair(BaseOff + Inner->first, Inner->second);
    }
  }
  for (const clang::FieldDecl *Field : Record->fields())
    if (auto Inner =
            getWrapperIntervalInType(Field->getType(), Context, Depth + 1))
      return std::make_pair(
          Context.getFieldOffset(Field) / 8 + Inner->first, Inner->second);
  return std::nullopt;
}

class VariableReferenceVisitor
    : public clang::RecursiveASTVisitor<VariableReferenceVisitor> {
public:
  explicit VariableReferenceVisitor(const clang::VarDecl *Target)
      : Target(Target->getCanonicalDecl()) {}

  bool VisitDeclRefExpr(clang::DeclRefExpr *Reference) {
    const auto *Variable = llvm::dyn_cast<clang::VarDecl>(Reference->getDecl());
    Found |= Variable && Variable->getCanonicalDecl() == Target;
    return !Found;
  }

  bool found() const { return Found; }

private:
  const clang::VarDecl *Target;
  bool Found = false;
};

// Whether Statement transitively contains exactly the given expression node.
class ExpressionSearchVisitor
    : public clang::RecursiveASTVisitor<ExpressionSearchVisitor> {
public:
  explicit ExpressionSearchVisitor(const clang::Expr *Target)
      : Target(Target) {}

  bool VisitStmt(clang::Stmt *Statement) {
    Found |= Statement == Target;
    return !Found;
  }

  bool found() const { return Found; }

private:
  const clang::Expr *Target;
  bool Found = false;
};

bool expressionContains(const clang::Expr *Haystack,
                        const clang::Expr *Needle) {
  ExpressionSearchVisitor Visitor(Needle);
  Visitor.TraverseStmt(const_cast<clang::Expr *>(Haystack));
  return Visitor.found();
}

bool referencesVariable(const clang::Stmt *Statement,
                        const clang::VarDecl *Variable) {
  VariableReferenceVisitor Visitor(Variable);
  Visitor.TraverseStmt(const_cast<clang::Stmt *>(Statement));
  return Visitor.found();
}

class GlobalReferenceVisitor
    : public clang::RecursiveASTVisitor<GlobalReferenceVisitor> {
public:
  bool VisitDeclRefExpr(clang::DeclRefExpr *Reference) {
    const auto *Variable = llvm::dyn_cast<clang::VarDecl>(Reference->getDecl());
    Found |= Variable && Variable->hasGlobalStorage() &&
             !llvm::isa<clang::ParmVarDecl>(Variable);
    return !Found;
  }

  bool found() const { return Found; }

private:
  bool Found = false;
};

bool referencesGlobalStorage(const clang::Stmt *Statement) {
  GlobalReferenceVisitor Visitor;
  Visitor.TraverseStmt(const_cast<clang::Stmt *>(Statement));
  return Visitor.found();
}

struct ParameterEffects {
  bool Written = false;
  bool Escaped = false;
};

class ParameterEffectVisitor
    : public clang::RecursiveASTVisitor<ParameterEffectVisitor> {
public:
  ParameterEffectVisitor(const clang::ParmVarDecl *Parameter,
                         ParameterEffects &Effects)
      : Parameter(Parameter), Effects(Effects) {}

  bool VisitBinaryOperator(clang::BinaryOperator *Operator) {
    if (!Operator->isAssignmentOp())
      return true;
    if (referencesVariable(Operator->getLHS(), Parameter))
      Effects.Written = true;
    if (referencesVariable(Operator->getRHS(), Parameter) &&
        referencesGlobalStorage(Operator->getLHS()))
      Effects.Escaped = true;
    return true;
  }

  bool VisitUnaryOperator(clang::UnaryOperator *Operator) {
    if (Operator->isIncrementDecrementOp() &&
        referencesVariable(Operator->getSubExpr(), Parameter))
      Effects.Written = true;
    return true;
  }

private:
  const clang::ParmVarDecl *Parameter;
  ParameterEffects &Effects;
};

ParameterEffects getParameterEffects(const clang::FunctionDecl *Function,
                                     unsigned ParameterIndex) {
  ParameterEffects Effects;
  if (!Function || ParameterIndex >= Function->getNumParams())
    return Effects;
  if (const clang::FunctionDecl *Definition = Function->getDefinition()) {
    ParameterEffectVisitor Visitor(Definition->getParamDecl(ParameterIndex),
                                   Effects);
    Visitor.TraverseStmt(Definition->getBody());
  }
  return Effects;
}

enum class StorageHazard : unsigned {
  MultipleCallWrites,
  TypeErasedSlot,
  EscapedSlot,
  MergedSlot,
  AmbiguousField,
  TypeErasedWrapperStorage,
  MemoryBuiltinOnWrapper,
  UnionOverWrapper,
};

class FunctionStorageHazardVisitor
    : public clang::RecursiveASTVisitor<FunctionStorageHazardVisitor> {
public:
  FunctionStorageHazardVisitor(clang::DiagnosticsEngine &Diagnostics,
                               unsigned MultipleCallWrites,
                               unsigned TypeErasedSlot, unsigned EscapedSlot,
                               unsigned TypeErasedWrapperStorage,
                               unsigned MemoryBuiltinOnWrapper,
                               llvm::DenseSet<uint64_t> &Reported)
      : Diagnostics(Diagnostics), MultipleCallWrites(MultipleCallWrites),
        TypeErasedSlot(TypeErasedSlot), EscapedSlot(EscapedSlot),
        TypeErasedWrapperStorage(TypeErasedWrapperStorage),
        MemoryBuiltinOnWrapper(MemoryBuiltinOnWrapper), Reported(Reported) {}

  bool VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *Cast) {
    if (isPointerToLambdaFunctorWrapper(Cast->getType(), 2))
      report(Cast->getExprLoc(), Cast->getSourceRange(),
             StorageHazard::TypeErasedSlot, TypeErasedSlot);
    // Record casts that erase wrapper-containing storage into a mutable
    // pointer. The cast alone is legal (container_of-style address math and
    // const reads are common); only a WRITE through the erased pointer can
    // break the whole-object discipline the IR analysis's witness model
    // assumes (docs/dev/design/lambda-closure-provenance.md). The write scan
    // happens in diagnoseErasedPointerWrites after the traversal.
    clang::QualType Source = Cast->getSubExpr()->getType();
    clang::QualType Dest = Cast->getType();
    if (Source->isPointerType() &&
        typeContainsLambdaFunctorWrapper(Source->getPointeeType()) &&
        Dest->isPointerType() &&
        !typeContainsLambdaFunctorWrapper(
            Dest->getPointeeType().getNonReferenceType()) &&
        !Dest->getPointeeType().isConstQualified())
      ErasedWrapperCasts.push_back(Cast);
    return true;
  }

  // Collect assignments; the erased-pointer write scan runs after the
  // traversal (diagnoseErasedPointerWrites), because this visitor fires in
  // pre-order -- a cast inside this assignment's LHS has not been recorded
  // yet when the assignment itself is visited.
  bool VisitBinaryOperator(clang::BinaryOperator *Operator) {
    if (Operator->isAssignmentOp())
      Assignments.push_back(Operator);
    return true;
  }

  // A write whose destination goes through an erased pointer: *p = v or
  // p[i] = v where p is a recorded erased cast or a variable holding one.
  void diagnoseErasedPointerWrites() {
    // Variables initialized from an erased cast propagate its taint:
    // auto *Words = reinterpret_cast<...>(&Copy); Words[1] = ...;
    for (const clang::VarDecl *Variable : InitializedVariables) {
      if (Variable->getType().isConstQualified() ||
          !Variable->getType()->isPointerType() ||
          Variable->getType()->getPointeeType().isConstQualified())
        continue;
      for (const clang::CXXReinterpretCastExpr *Cast : ErasedWrapperCasts)
        if (expressionContains(Variable->getInit(), Cast))
          ErasedPointerVariables.insert(Variable->getCanonicalDecl());
    }
    for (const clang::BinaryOperator *Operator : Assignments) {
      const clang::Expr *LHS = Operator->getLHS();
      bool ThroughErased = false;
      for (const clang::CXXReinterpretCastExpr *Cast : ErasedWrapperCasts)
        ThroughErased |= expressionContains(LHS, Cast);
      for (const clang::VarDecl *Variable : ErasedPointerVariables)
        ThroughErased |= referencesVariable(LHS, Variable);
      if (ThroughErased)
        report(Operator->getOperatorLoc(), Operator->getSourceRange(),
               StorageHazard::TypeErasedWrapperStorage,
               TypeErasedWrapperStorage);
    }
  }

  bool VisitDeclStmt(clang::DeclStmt *Stmt) {
    // Collected for the post-traversal scan: a variable's initializer may
    // contain a cast that has not been recorded yet in this pre-order visit.
    for (const clang::Decl *Decl : Stmt->decls())
      if (const auto *Variable = llvm::dyn_cast<clang::VarDecl>(Decl);
          Variable && Variable->hasInit())
        InitializedVariables.push_back(Variable);
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *Call) {
    const clang::FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      return true;

    // Memory builtins writing into wrapper storage can bypass the closure's
    // whole-object copy discipline: a wrong length splices two sources into
    // one functor (the classic wrong-sizeof bug). Writes that provably cover
    // the whole wrapper, or provably miss it, are the discipline itself and
    // stay legal; only a PARTIAL overlap of the wrapper's bytes -- or an
    // overlap whose extent cannot be computed -- is diagnosed. Memset is
    // stricter: any overlap destroys registered-slot provenance.
    bool IsSet = false;
    switch (Callee->getBuiltinID()) {
    case clang::Builtin::BImemset:
    case clang::Builtin::BI__builtin_memset:
    case clang::Builtin::BI__builtin_memset_inline:
      IsSet = true;
      [[fallthrough]];
    case clang::Builtin::BImemcpy:
    case clang::Builtin::BImemmove:
    case clang::Builtin::BI__builtin_memcpy:
    case clang::Builtin::BI__builtin_memmove:
    case clang::Builtin::BI__builtin_memcpy_inline: {
      if (Call->getNumArgs() < 3)
        break;
      clang::ASTContext &Context = Callee->getASTContext();
      auto Destination = getIntrinsicDestination(Call->getArg(0), Context);
      if (!Destination)
        break;
      auto Wrapper = getWrapperIntervalInType(Destination->RootType, Context);
      if (!Wrapper)
        break; // no registered wrapper in the written object

      // Wrapper interval within the root; write interval within the root.
      uint64_t WrapperBegin = Wrapper->first;
      uint64_t WrapperEnd = Wrapper->first + Wrapper->second;
      uint64_t WriteBegin = Destination->OffsetInRoot;
      clang::Expr::EvalResult Length;
      bool HasLength = Call->getArg(2)->EvaluateAsInt(Length, Context);
      if (!HasLength) {
        // Unknown length: diagnose only if the write could reach the
        // wrapper, i.e. it does not start past the wrapper's end.
        if (WriteBegin < WrapperEnd)
          report(Call->getExprLoc(), Call->getSourceRange(),
                 StorageHazard::MemoryBuiltinOnWrapper,
                 MemoryBuiltinOnWrapper);
        break;
      }
      uint64_t WriteEnd =
          WriteBegin + Length.Val.getInt().getZExtValue();
      bool Overlaps = WriteBegin < WrapperEnd && WriteEnd > WrapperBegin;
      bool Covers = WriteBegin <= WrapperBegin && WriteEnd >= WrapperEnd;
      if (Overlaps && (IsSet || !Covers))
        report(Call->getExprLoc(), Call->getSourceRange(),
               StorageHazard::MemoryBuiltinOnWrapper, MemoryBuiltinOnWrapper);
      break;
    }
    default:
      break;
    }

    const unsigned Count = std::min(Call->getNumArgs(), Callee->getNumParams());
    for (unsigned I = 0; I < Count; ++I) {
      const clang::VarDecl *Variable = getAddressedVariable(Call->getArg(I));
      if (!Variable || !isPointerToLambdaFunctorWrapper(Variable->getType()))
        continue;

      ParameterEffects Effects = getParameterEffects(Callee, I);
      if (Effects.Written && ++CallWrites[Variable->getCanonicalDecl()] == 2)
        report(Call->getExprLoc(), Call->getSourceRange(),
               StorageHazard::MultipleCallWrites, MultipleCallWrites);
      if (Effects.Escaped)
        report(Call->getExprLoc(), Call->getSourceRange(),
               StorageHazard::EscapedSlot, EscapedSlot);
    }
    return true;
  }

private:
  void report(clang::SourceLocation Location, clang::SourceRange Range,
              StorageHazard Hazard, unsigned Diagnostic) {
    const uint64_t Key =
        (static_cast<uint64_t>(Hazard) << 32) | Location.getRawEncoding();
    if (Reported.insert(Key).second)
      Diagnostics.Report(Location, Diagnostic) << Range;
  }

  clang::DiagnosticsEngine &Diagnostics;
  unsigned MultipleCallWrites;
  unsigned TypeErasedSlot;
  unsigned EscapedSlot;
  unsigned TypeErasedWrapperStorage;
  unsigned MemoryBuiltinOnWrapper;
  llvm::DenseSet<uint64_t> &Reported;
  llvm::DenseMap<const clang::VarDecl *, unsigned> CallWrites;
  llvm::SmallVector<const clang::CXXReinterpretCastExpr *, 4> ErasedWrapperCasts;
  llvm::SmallPtrSet<const clang::VarDecl *, 4> ErasedPointerVariables;
  llvm::SmallVector<const clang::BinaryOperator *, 8> Assignments;
  llvm::SmallVector<const clang::VarDecl *, 8> InitializedVariables;
};

class ReturnStorageHazardVisitor
    : public clang::RecursiveASTVisitor<ReturnStorageHazardVisitor> {
public:
  bool VisitBinaryOperator(clang::BinaryOperator *Operator) {
    if (!Operator->isAssignmentOp())
      return true;
    if (const clang::VarDecl *Variable =
            getDirectReferencedVariable(Operator->getLHS()))
      ++Assignments[Variable->getCanonicalDecl()];
    return true;
  }

  bool VisitReturnStmt(clang::ReturnStmt *Return) {
    Returns.push_back(Return);
    return true;
  }

  llvm::DenseMap<const clang::VarDecl *, unsigned> Assignments;
  llvm::SmallVector<clang::ReturnStmt *, 4> Returns;
};
class CaptureMutationVisitor
    : public clang::RecursiveASTVisitor<CaptureMutationVisitor> {
public:
  CaptureMutationVisitor(const llvm::DenseSet<const clang::VarDecl *> &Targets,
                         llvm::SmallVectorImpl<clang::SourceRange> &Mutations)
      : Targets(Targets), Mutations(Mutations) {}

  bool VisitBinaryOperator(clang::BinaryOperator *Operator) {
    if (!Operator->isAssignmentOp())
      return true;
    if (directlyWritesTarget(Operator->getLHS(), Targets) ||
        (isMutablePointer(Operator->getLHS()->getType()) &&
         takesAddressOfTarget(Operator->getRHS(), Targets)))
      Mutations.push_back(Operator->getSourceRange());
    return true;
  }

  bool VisitUnaryOperator(clang::UnaryOperator *Operator) {
    if (Operator->isIncrementDecrementOp() &&
        directlyReferencesTarget(Operator->getSubExpr(), Targets))
      Mutations.push_back(Operator->getSourceRange());
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *Variable) {
    if (!Variable->hasInit())
      return true;
    clang::QualType Type = Variable->getType();
    if (isMutableReference(Type) &&
        directlyReferencesTarget(Variable->getInit(), Targets))
      Mutations.push_back(Variable->getSourceRange());
    else if (isMutablePointer(Type) &&
             takesAddressOfTarget(Variable->getInit(), Targets))
      Mutations.push_back(Variable->getSourceRange());
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *Call) {
    const clang::FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      return true;
    // Member operator calls pass the object as argument 0, ahead of the
    // parameters.
    const auto *Method = llvm::dyn_cast<clang::CXXMethodDecl>(Callee);
    const unsigned FirstArgument =
        llvm::isa<clang::CXXOperatorCallExpr>(Call) && Method &&
                Method->isImplicitObjectMemberFunction()
            ? 1
            : 0;
    const unsigned Count =
        std::min(Call->getNumArgs() - FirstArgument, Callee->getNumParams());
    for (unsigned I = 0; I < Count; ++I) {
      const clang::Expr *Argument = Call->getArg(I + FirstArgument);
      clang::QualType ParameterType = Callee->getParamDecl(I)->getType();
      if ((isMutableReference(ParameterType) &&
           directlyReferencesTarget(Argument, Targets)) ||
          (isMutablePointer(ParameterType) &&
           takesAddressOfTarget(Argument, Targets)))
        Mutations.push_back(Argument->getSourceRange());
    }
    return true;
  }

  bool VisitCXXConstructExpr(clang::CXXConstructExpr *ConstructorCall) {
    const clang::CXXConstructorDecl *Constructor =
        ConstructorCall->getConstructor();
    const unsigned Count =
        std::min(ConstructorCall->getNumArgs(), Constructor->getNumParams());
    for (unsigned I = 0; I < Count; ++I) {
      const clang::Expr *Argument = ConstructorCall->getArg(I);
      clang::QualType ParameterType = Constructor->getParamDecl(I)->getType();
      if ((isMutableReference(ParameterType) &&
           directlyReferencesTarget(Argument, Targets)) ||
          (isMutablePointer(ParameterType) &&
           takesAddressOfTarget(Argument, Targets)))
        Mutations.push_back(Argument->getSourceRange());
    }
    return true;
  }

private:
  const llvm::DenseSet<const clang::VarDecl *> &Targets;
  llvm::SmallVectorImpl<clang::SourceRange> &Mutations;
};

class LambdaVisitor : public clang::RecursiveASTVisitor<LambdaVisitor> {
public:
  bool shouldVisitTemplateInstantiations() const { return true; }

  // Only instantiations show which closure types reach register_lambda, so
  // template patterns are skipped.
  bool TraverseDecl(clang::Decl *Decl) {
    if (const auto *Context = llvm::dyn_cast_or_null<clang::DeclContext>(Decl);
        Context && Context->isDependentContext())
      return true;
    return RecursiveASTVisitor::TraverseDecl(Decl);
  }

  bool VisitFunctionDecl(clang::FunctionDecl *Function) {
    if (Function->hasBody())
      Functions.push_back(Function);
    return true;
  }

  bool VisitLambdaExpr(clang::LambdaExpr *Lambda) {
    Lambdas.push_back(Lambda);
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *Record) {
    if (Record->isUnion() && Record->isCompleteDefinition())
      Unions.push_back(Record);
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *Call) {
    if (isProteusFunction(Call, "proteus::jit_variable"))
      JitVariableCalls.push_back(Call);
    else if (isProteusFunction(Call, "proteus::register_lambda") &&
             Call->getNumArgs() != 0 && !Call->getArg(0)->isTypeDependent()) {
      RegisterLambdaCalls.push_back(Call);
      const clang::Expr *Argument = ignoreTransparentExprs(Call->getArg(0));
      if (const auto *Lambda = llvm::dyn_cast<clang::LambdaExpr>(Argument)) {
        DirectlyRegisteredLambdas.insert(Lambda);
        DirectlyRegisteredLambdaLocations.insert(
            Lambda->getBeginLoc().getRawEncoding());
      }
      if (const auto *Record = getCanonicalRecord(Argument->getType());
          Record && Record->isLambda())
        RegisteredLambdaTypes.insert(Record);
    }
    return true;
  }

  llvm::SmallVector<clang::LambdaExpr *, 16> Lambdas;
  llvm::SmallVector<clang::CXXRecordDecl *, 8> Unions;
  llvm::SmallVector<clang::FunctionDecl *, 32> Functions;
  llvm::SmallVector<clang::CallExpr *, 16> JitVariableCalls;
  llvm::SmallVector<clang::CallExpr *, 16> RegisterLambdaCalls;
  llvm::DenseSet<const clang::LambdaExpr *> DirectlyRegisteredLambdas;
  llvm::DenseSet<unsigned> DirectlyRegisteredLambdaLocations;
  llvm::DenseSet<const clang::CXXRecordDecl *> RegisteredLambdaTypes;
};

class LambdaASTConsumer : public clang::ASTConsumer {
public:
  LambdaASTConsumer(clang::CompilerInstance &CI, bool ContractWarnOnly)
      : CI(CI), ContractWarnOnly(ContractWarnOnly) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    LambdaVisitor Visitor;
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());

    llvm::DenseMap<const clang::CallExpr *, const clang::LambdaExpr *>
        CaptureInitializers;
    llvm::DenseMap<const clang::CallExpr *, clang::QualType> CaptureTypes;
    llvm::DenseMap<const clang::LambdaExpr *,
                   llvm::SmallVector<const clang::VarDecl *, 4>>
        JitCaptureVariables;
    for (const clang::LambdaExpr *Lambda : Visitor.Lambdas) {
      auto Init = Lambda->capture_init_begin();
      for (const clang::LambdaCapture &Capture : Lambda->captures()) {
        const clang::Expr *CaptureInit = *Init++;
        if (!Capture.capturesVariable() ||
            !Capture.getCapturedVar()->isInitCapture())
          continue;
        clang::QualType CaptureType = CaptureInit->getType();
        CaptureInit = ignoreCaptureInitWrappers(CaptureInit);
        const auto *Call = llvm::dyn_cast_or_null<clang::CallExpr>(CaptureInit);
        if (Call && isProteusFunction(Call, "proteus::jit_variable")) {
          CaptureInitializers.try_emplace(Call, Lambda);
          CaptureTypes.try_emplace(Call, CaptureType);
          JitCaptureVariables[Lambda].push_back(llvm::cast<clang::VarDecl>(
              Capture.getCapturedVar()->getCanonicalDecl()));
        }
      }
    }

    clang::DiagnosticsEngine &Diagnostics = CI.getDiagnostics();
    const unsigned InvalidUse = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "proteus::jit_variable must be used directly as a lambda "
        "init-capture initializer");
    const unsigned Unregistered = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "lambda containing proteus::jit_variable must be registered with "
        "proteus::register_lambda");
    const unsigned UnsupportedType = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "proteus::jit_variable does not support capture type %0");
    const unsigned InvalidRegisterArgument = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "proteus::register_lambda requires a lambda closure object");
    const unsigned AlreadyRegistered = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "a lambda returned by proteus::register_lambda cannot be registered "
        "again");
    const unsigned MutatedCapture = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "proteus::jit_variable capture must remain read-only");
    const unsigned MultipleCallWrites = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "registered lambda pointer slot cannot be overwritten by multiple "
        "function calls");
    const unsigned TypeErasedSlot = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "registered lambda pointer slots cannot be accessed through "
        "type-erased byte offsets");
    const unsigned EscapedSlot = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "registered lambda pointer slot cannot escape to global storage");
    const unsigned MergedSlot = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "registered lambda pointer has been passed through "
        "control-flow assignments.  Analysis will try to specialize but failure to specialize lambda may occur.");
    const unsigned AmbiguousField = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "registered lambda pointer slot cannot be selected from different "
        "aggregate fields at runtime");
    const clang::DiagnosticsEngine::Level ContractLevel =
        ContractWarnOnly ? clang::DiagnosticsEngine::Warning
                         : clang::DiagnosticsEngine::Error;
    const unsigned TypeErasedWrapperStorage = Diagnostics.getCustomDiagID(
        ContractLevel,
        "storage containing a registered lambda cannot be reinterpreted "
        "through a type-erased mutable pointer; the specialization analysis "
        "requires closures to be written only as whole objects");
    const unsigned MemoryBuiltinOnWrapper = Diagnostics.getCustomDiagID(
        ContractLevel,
        "memory builtins cannot write into storage containing a registered "
        "lambda; copy the closure object instead");
    const unsigned UnionOverWrapper = Diagnostics.getCustomDiagID(
        ContractLevel,
        "a registered lambda cannot be a member of a union; union storage "
        "reuse defeats the specialization analysis");

    for (const clang::CXXRecordDecl *Union : Visitor.Unions)
      for (const clang::FieldDecl *Field : Union->fields())
        if (typeContainsLambdaFunctorWrapper(Field->getType()))
          Diagnostics.Report(Field->getLocation(), UnionOverWrapper)
              << Field->getSourceRange();

    llvm::DenseSet<uint64_t> ReportedStorageHazards;
    for (clang::FunctionDecl *Function : Visitor.Functions) {
      FunctionStorageHazardVisitor StorageVisitor(
          Diagnostics, MultipleCallWrites, TypeErasedSlot, EscapedSlot,
          TypeErasedWrapperStorage, MemoryBuiltinOnWrapper,
          ReportedStorageHazards);
      StorageVisitor.TraverseStmt(Function->getBody());
      StorageVisitor.diagnoseErasedPointerWrites();

      if (!isPointerToLambdaFunctorWrapper(Function->getReturnType(), 2))
        continue;

      ReturnStorageHazardVisitor ReturnVisitor;
      ReturnVisitor.TraverseStmt(Function->getBody());
      llvm::DenseSet<const clang::ValueDecl *> ReturnedMembers;
      for (clang::ReturnStmt *Return : ReturnVisitor.Returns) {
        const clang::Expr *Returned = Return->getRetValue();
        if (!Returned)
          continue;
        if (const clang::VarDecl *Variable =
                getDirectReferencedVariable(Returned)) {
          if (ReturnVisitor.Assignments.lookup(Variable->getCanonicalDecl()) >=
              2)
            reportStorageHazard(Diagnostics, ReportedStorageHazards,
                                Return->getReturnLoc(),
                                Return->getSourceRange(),
                                StorageHazard::MergedSlot, MergedSlot);
        }

        Returned = ignoreTransparentExprs(Returned);
        if (const auto *Address =
                llvm::dyn_cast_or_null<clang::UnaryOperator>(Returned);
            Address && Address->getOpcode() == clang::UO_AddrOf)
          Returned = ignoreTransparentExprs(Address->getSubExpr());
        const auto *Member =
            llvm::dyn_cast_or_null<clang::MemberExpr>(Returned);
        if (!Member)
          continue;
        const clang::ValueDecl *MemberDeclaration = Member->getMemberDecl();
        if (!ReturnedMembers.empty() &&
            !ReturnedMembers.contains(MemberDeclaration))
          reportStorageHazard(Diagnostics, ReportedStorageHazards,
                              Return->getReturnLoc(), Return->getSourceRange(),
                              StorageHazard::AmbiguousField, AmbiguousField);
        ReturnedMembers.insert(MemberDeclaration);
      }
    }

    for (const clang::CallExpr *Call : Visitor.RegisterLambdaCalls) {
      const clang::Expr *Argument = Call->getArg(0);
      const auto *Record = getCanonicalRecord(Argument->getType());
      if (isLambdaFunctorWrapper(Record))
        Diagnostics.Report(Argument->getExprLoc(), AlreadyRegistered)
            << Argument->getSourceRange();
      else if (!Record || !Record->isLambda())
        Diagnostics.Report(Argument->getExprLoc(), InvalidRegisterArgument)
            << Argument->getSourceRange();
    }

    for (const auto &Entry : JitCaptureVariables) {
      llvm::DenseSet<const clang::VarDecl *> Targets;
      Targets.insert(Entry.second.begin(), Entry.second.end());
      llvm::SmallVector<clang::SourceRange, 4> Mutations;
      CaptureMutationVisitor MutationVisitor(Targets, Mutations);
      MutationVisitor.TraverseStmt(Entry.first->getBody());
      for (clang::SourceRange Range : Mutations)
        Diagnostics.Report(Range.getBegin(), MutatedCapture) << Range;
    }

    for (const clang::CallExpr *Call : Visitor.JitVariableCalls) {
      // Dependent template patterns are checked when their specializations are
      // instantiated, once both the closure and captured value types are known.
      if (Call->isTypeDependent())
        continue;

      auto CaptureIt = CaptureInitializers.find(Call);
      if (CaptureIt == CaptureInitializers.end()) {
        Diagnostics.Report(Call->getExprLoc(), InvalidUse)
            << Call->getSourceRange();
        continue;
      }

      const clang::LambdaExpr *Lambda = CaptureIt->second;
      const auto *LambdaType = Lambda->getLambdaClass()->getCanonicalDecl();
      if (!Visitor.DirectlyRegisteredLambdas.contains(Lambda) &&
          !Visitor.DirectlyRegisteredLambdaLocations.contains(
              Lambda->getBeginLoc().getRawEncoding()) &&
          !Visitor.RegisteredLambdaTypes.contains(LambdaType)) {
        Diagnostics.Report(Call->getExprLoc(), Unregistered)
            << Lambda->getSourceRange();
      }

      clang::QualType Type = CaptureTypes.lookup(Call).getUnqualifiedType();
      if (!isSupportedRuntimeConstantType(Type, Context))
        Diagnostics.Report(Call->getExprLoc(), UnsupportedType)
            << Type << Call->getSourceRange();
    }
  }

private:
  static void reportStorageHazard(clang::DiagnosticsEngine &Diagnostics,
                                  llvm::DenseSet<uint64_t> &Reported,
                                  clang::SourceLocation Location,
                                  clang::SourceRange Range,
                                  StorageHazard Hazard, unsigned Diagnostic) {
    const uint64_t Key =
        (static_cast<uint64_t>(Hazard) << 32) | Location.getRawEncoding();
    if (Reported.insert(Key).second)
      Diagnostics.Report(Location, Diagnostic) << Range;
  }

  static bool isSupportedRuntimeConstantType(clang::QualType Type,
                                             clang::ASTContext &Context) {
    if (Type->isPointerType() || Type->isBooleanType() ||
        (Type->isRealFloatingType() &&
         (Type->isSpecificBuiltinType(clang::BuiltinType::Float) ||
          Type->isSpecificBuiltinType(clang::BuiltinType::Double) ||
          Type->isSpecificBuiltinType(clang::BuiltinType::LongDouble))))
      return true;

    if (!Type->isIntegralOrEnumerationType())
      return false;
    const uint64_t Width = Context.getTypeSize(Type);
    return Width == 8 || Width == 32 || Width == 64;
  }

  clang::CompilerInstance &CI;
  bool ContractWarnOnly;
};

class LambdaASTCheckerAction : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<LambdaASTConsumer>(CI, ContractWarnOnly);
  }

  bool ParseArgs(const clang::CompilerInstance &,
                 const std::vector<std::string> &Args) override {
    for (const std::string &Arg : Args)
      if (Arg == "-contract-warn")
        ContractWarnOnly = true;
    return true;
  }

private:
  // Downgrades the wrapper-storage contract diagnostics (type-erased writes,
  // partial memory builtins, unions) from error to warning. For test code
  // that deliberately violates the contract to exercise the IR analysis's
  // conservative declines -- the second line of defense for IR that never
  // passed this checker. Pass via:
  //   -Xclang -plugin-arg-proteus-lambda-checker -Xclang -contract-warn
  bool ContractWarnOnly = false;

  ActionType getActionType() override { return AddBeforeMainAction; }
};

} // namespace

static clang::FrontendPluginRegistry::Add<LambdaASTCheckerAction>
    X("proteus-lambda-checker", "validate Proteus lambda API usage");
