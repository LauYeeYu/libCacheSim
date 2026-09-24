// Per-block compute cost.
//
// The compute saving ratio is cost-weighted, so the cost model is what decides
// what "saving" means. Every model here is a function of the block's 0-based
// position within its request, which is the only compute signal a trace carries.
//
// Cost is deliberately absent from the trace: the constants below belong to one
// model on one class of hardware, so naming a model after its profile is the
// honest way to say so. Add an enumerator per profiled model rather than
// generalising the shape.

#pragma once

#include <cstdint>
#include <string>

namespace prefixsim {

enum class CostModel {
  /// Every block costs 1, so the compute saving ratio equals the block hit
  /// ratio. Use this when you want the two numbers to be directly comparable.
  kUniform,
  /// cost = position + 1. Matches the `compute` field that
  /// llm_cachesim/scripts/trace_processing/llm_qwen.py writes into .lcsllm:
  /// the number of blocks that would have to be prefilled from the root to
  /// reconstruct this one.
  kPosition,
  /// cost = 865 + 2*position: the measured per-block prefill cost of
  /// Qwen3-Coder-30B at a 16-token block size. A block costs a fixed part
  /// (projections, MLP) plus a part growing with the context it attends to,
  /// which is why the constant dominates. Matches the compute-intensity model
  /// in the vLLM prototype's logs (idx=0 -> 865, idx=100 -> 1065) and
  /// compute_intensity_transform() in the evaluate/ harness.
  ///
  /// Mind the dynamic range: position 1 vs 100 is 865 vs 1065, a factor of
  /// 1.23, against kPosition's factor of 100. Under this model a cost-aware
  /// policy ranks almost like a cost-blind one -- a property of the hardware,
  /// not a bug, and the first thing to check when compute-awareness looks like
  /// it is not paying.
  kQwen3Coder30bBlk16,
  /// cost = 3472 + 32*position: the SAME measured Qwen3-Coder-30B profile at a 64-token
  /// block size. Not an independent measurement -- it is the 16-token model re-expressed
  /// at 4x the block, and it is exact rather than fitted:
  ///     sum_{j=0..3} (865 + 2*(4p + j)) = 4*865 + 32p + 12 = 3472 + 32p
  /// i.e. a 64-token block costs exactly what its four constituent 16-token blocks cost.
  /// (Equivalently, with a linear/MLP term a per token and an attention term b per
  /// query-key pair, a block of B tokens costs [a*B + b*B^2/2] + [b*B^2]*position; the
  /// 16-token constants give a = 54, b = 1/128, and B = 64 gives 3472 + 32*position.)
  /// For the agentX trace, whose blocks are natively 64 tokens.
  kQwen3Coder30bBlk64,
};

inline double block_cost(CostModel model, int64_t position) {
  const double pos1 = static_cast<double>(position + 1);
  switch (model) {
    case CostModel::kUniform:
      return 1.0;
    case CostModel::kPosition:
      return pos1;
    case CostModel::kQwen3Coder30bBlk16:
      return 863.0 + 2.0 * pos1;  // == 865 + 2*position
    case CostModel::kQwen3Coder30bBlk64:
      return 3440.0 + 32.0 * pos1;  // == 3472 + 32*position
  }
  return 1.0;
}

inline bool parse_cost_model(const std::string &name, CostModel &out) {
  if (name == "uniform") {
    out = CostModel::kUniform;
  } else if (name == "position") {
    out = CostModel::kPosition;
  } else if (name == "qwen3coder30b_blksz_16") {
    out = CostModel::kQwen3Coder30bBlk16;
  } else if (name == "qwen3coder30b_blksz_64") {
    out = CostModel::kQwen3Coder30bBlk64;
  } else {
    return false;
  }
  return true;
}

inline const char *cost_model_name(CostModel model) {
  switch (model) {
    case CostModel::kUniform:
      return "uniform";
    case CostModel::kPosition:
      return "position";
    case CostModel::kQwen3Coder30bBlk16:
      return "qwen3coder30b_blksz_16";
    case CostModel::kQwen3Coder30bBlk64:
      return "qwen3coder30b_blksz_64";
  }
  return "?";
}

/// True when a block's cost depends on where it sits in the prefix, which means
/// two different prefixes reaching the same *raw* block id disagree about cost.
inline bool cost_is_position_dependent(CostModel model) {
  return model != CostModel::kUniform;
}

}  // namespace prefixsim
