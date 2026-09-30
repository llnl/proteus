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
      Reference ? llvm::dyn_cast<clang::VarDecl>(Reference->getDecl()) : nullptr;
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

bool isMutableReference(clang::QualType Type) {
  if (!Type->isReferenceType())
    return false;
  return !Type.getNonReferenceType().isConstQualified();
}

bool isMutablePointer(clang::QualType Type) {
  return Type->isPointerType() && !Type->getPointeeType().isConstQualified();
}

class CaptureMutationVisitor
    : public clang::RecursiveASTVisitor<CaptureMutationVisitor> {
public:
  CaptureMutationVisitor(
      const llvm::DenseSet<const clang::VarDecl *> &Targets,
      llvm::SmallVectorImpl<clang::SourceRange> &Mutations)
      : Targets(Targets), Mutations(Mutations) {}

  bool VisitBinaryOperator(clang::BinaryOperator *Operator) {
    if (Operator->isAssignmentOp() &&
        directlyReferencesTarget(Operator->getLHS(), Targets))
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
    const unsigned Count = std::min(Call->getNumArgs(), Callee->getNumParams());
    for (unsigned I = 0; I < Count; ++I) {
      const clang::Expr *Argument = Call->getArg(I);
      clang::QualType ParameterType = Callee->getParamDecl(I)->getType();
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
        CaptureInit = ignoreTransparentExprs(CaptureInit);
        const auto *Call = llvm::dyn_cast_or_null<clang::CallExpr>(CaptureInit);
        if (Call && isProteusFunction(Call, "proteus::jit_variable")) {
          CaptureInitializers.try_emplace(Call, Lambda);
          CaptureTypes.try_emplace(Call, CaptureType);
          JitCaptureVariables[Lambda].push_back(
              llvm::cast<clang::VarDecl>(
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
  static bool isSupportedRuntimeConstantType(clang::QualType Type,
                                             clang::ASTContext &Context) {
    if (Type->isPointerType() || Type->isBooleanType() ||
        (Type->isRealFloatingType() &&
         (Type->isSpecificBuiltinType(clang::BuiltinType::Float) ||
          Type->isSpecificBuiltinType(clang::BuiltinType::Double))))
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
