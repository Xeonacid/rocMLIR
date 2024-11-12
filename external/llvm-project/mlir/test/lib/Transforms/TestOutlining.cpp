//===- TosaPartition.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/OutlinerUtils.h"
#include "llvm/ADT/SmallVector.h"

using llvm::SmallVector;
using namespace mlir;

namespace {

struct OutlinerPass : public PassWrapper<OutlinerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlinerPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,tosa::TosaDialect>();
  }

  StringRef getArgument() const final { return "test-outliner"; }
  StringRef getDescription() const final {
    return "Test outlining certain derived regions";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto isLinalgAnchor = [&](Operation *op) { return isa<linalg::Conv2DOp,linalg::Conv2DNhwcHwcfOp>(op); };
    auto isLinalgTerminal = [&](Operation *op) { return false; };
    auto isLinalgLeading = [&](Operation *op) { return false; };
    auto isLinalgTrailing = [&](Operation *op) { return true /*op->hasTrait<OpTrait::Elementwise>()*/; };

    Outliner forLinalg(isLinalgAnchor, isLinalgLeading, isLinalgTrailing, isLinalgTerminal,
                       "kernel");
    forLinalg.outline(module);

    auto isTransposeConfigConstant = [](Operation *op) {
      return op->hasTrait<OpTrait::ConstantLike>() &&
        llvm::any_of(op->getUsers(), [&](Operation *u) {
          return isa<tosa::TransposeOp>(u) && u->getOperand(1) == op->getResult(0);
        });
    };
    auto isTosaAnchor = [&](Operation *op) { return isa<tosa::Conv2DOp>(op); };
    auto isTosaTerminal = [&](Operation *op) { return false; };
    auto isTosaLeading = [&](Operation *op) {
      return isa<tosa::TransposeOp, tosa::ReshapeOp>(op) ||
        isTransposeConfigConstant(op);
    };
    auto isTosaTrailing = [&](Operation *op) {
      return op->hasTrait<OpTrait::ResultsBroadcastableShape>();
    };

    Outliner forTosa(isTosaAnchor, isTosaLeading, isTosaTrailing, isTosaTerminal,
                     "kernel");
    forTosa.outline(module);
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestOutliner() { PassRegistration<OutlinerPass>(); }
} // namespace test
} // namespace mlir
