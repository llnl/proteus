//===-- LambdaASTChecker.cpp - Validate Proteus lambda API use -----------===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <memory>
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
};

class FunctionStorageHazardVisitor
    : public clang::RecursiveASTVisitor<FunctionStorageHazardVisitor> {
public:
  FunctionStorageHazardVisitor(clang::DiagnosticsEngine &Diagnostics,
                               unsigned MultipleCallWrites,
                               unsigned TypeErasedSlot, unsigned EscapedSlot,
                               llvm::DenseSet<uint64_t> &Reported)
      : Diagnostics(Diagnostics), MultipleCallWrites(MultipleCallWrites),
        TypeErasedSlot(TypeErasedSlot), EscapedSlot(EscapedSlot),
        Reported(Reported) {}

  bool VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *Cast) {
    if (isPointerToLambdaFunctorWrapper(Cast->getType(), 2))
      report(Cast->getExprLoc(), Cast->getSourceRange(),
             StorageHazard::TypeErasedSlot, TypeErasedSlot);
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *Call) {
    const clang::FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      return true;
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
  llvm::DenseSet<uint64_t> &Reported;
  llvm::DenseMap<const clang::VarDecl *, unsigned> CallWrites;
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
  llvm::SmallVector<clang::FunctionDecl *, 32> Functions;
  llvm::SmallVector<clang::CallExpr *, 16> JitVariableCalls;
  llvm::SmallVector<clang::CallExpr *, 16> RegisterLambdaCalls;
  llvm::DenseSet<const clang::LambdaExpr *> DirectlyRegisteredLambdas;
  llvm::DenseSet<unsigned> DirectlyRegisteredLambdaLocations;
  llvm::DenseSet<const clang::CXXRecordDecl *> RegisteredLambdaTypes;
};

class LambdaASTConsumer : public clang::ASTConsumer {
public:
  explicit LambdaASTConsumer(clang::CompilerInstance &CI) : CI(CI) {}

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
        clang::DiagnosticsEngine::Error,
        "registered lambda pointer slot cannot be selected from multiple "
        "control-flow assignments");
    const unsigned AmbiguousField = Diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "registered lambda pointer slot cannot be selected from different "
        "aggregate fields at runtime");

    llvm::DenseSet<uint64_t> ReportedStorageHazards;
    for (clang::FunctionDecl *Function : Visitor.Functions) {
      FunctionStorageHazardVisitor StorageVisitor(
          Diagnostics, MultipleCallWrites, TypeErasedSlot, EscapedSlot,
          ReportedStorageHazards);
      StorageVisitor.TraverseStmt(Function->getBody());

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
};

class LambdaASTCheckerAction : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<LambdaASTConsumer>(CI);
  }

  bool ParseArgs(const clang::CompilerInstance &,
                 const std::vector<std::string> &) override {
    return true;
  }

  ActionType getActionType() override { return AddBeforeMainAction; }
};

} // namespace

static clang::FrontendPluginRegistry::Add<LambdaASTCheckerAction>
    X("proteus-lambda-checker", "validate Proteus lambda API usage");
