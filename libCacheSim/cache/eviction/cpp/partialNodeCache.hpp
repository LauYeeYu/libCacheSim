// Base class for sample-based partial-node eviction algorithms.
//
// The shape mirrors RadixTreeFreeBlockManager in the vLLM prototype: the base
// owns the prefix tree, the sampling, the candidate selection and all of the
// cache_t bookkeeping; a derived algorithm supplies only a per-block score.
//
// One eviction round:
//   1. sample n_sample NODES uniformly (not blocks -- that is the difference
//      from RandomCompute and friends, which sample the flat hash table)
//   2. take each sampled node's eviction candidate: its shallowest resident
//      block by default, its deepest under evict-from=tail; see
//      `evict_from_tail`
//   3. score the candidates and evict the single worst one
//
// Evicting one block out of a multi-block node is what makes this "partial
// node": the rest of the node stays cached.
//
// To add a variant, derive from PartialNodeCache, override score(), and write
// a five-line _init that hands the instance to partial_node_cache_init().
// See PartialNodeRandomCompute.cpp.

#pragma once

#include <string>

#include "prefixRadixTree.hpp"

extern "C" {
#include "libCacheSim/cache.h"
}

namespace eviction {

class PartialNodeCache {
 public:
  virtual ~PartialNodeCache() = default;

  /**
   * Score one candidate block. LOWER scores are evicted first, matching the
   * convention in RandomCompute and the other sampling algorithms here.
   *
   * `obj` is guaranteed to be in the cache. Everything a score can depend on
   * lives on the object (cost, misc.freq, Random.last_access_vtime,
   * misc.next_access_vtime) or on the cache (n_req).
   */
  virtual double score(const cache_t *cache, const cache_obj_t *obj) const = 0;

  /**
   * Optional: called once per request with the whole block path, before those
   * blocks are accessed. The base class has already inserted the path into the
   * tree by the time this runs; override only to keep extra per-request state.
   */
  virtual void on_record_request(const obj_id_t * /*ids*/, int64_t /*n*/) {}

  /// How many blocks one eviction round takes out of the winning node.
  ///
  /// Both modes sample n_sample nodes, score each by its canonical block (see
  /// node_chunk_score, which follows vLLM's rule), and take the chunk from the
  /// best-scoring node. They differ in the cap, and in which canonical rule
  /// applies:
  ///   kDrain  -- uncapped: the winning node gives up the entire remaining
  ///              deficit if it can. One sample per round, cheapest, but it
  ///              commits to a single node for a potentially large batch.
  ///   kMicro  -- capped at micro_batch blocks, then re-sample and re-score.
  ///              More selective for the same reason drain is cheaper.
  /// Named after EVICTION_MODE in the vLLM prototype.
  ///
  /// kMicro is the default, matching the vLLM prototype. The two modes agree to
  /// ~0.002 on qwen_coder, but on the fi-new trace at 32k blocks drain gives up
  /// 0.007 of compute savings (0.8537 against micro's 0.8605): draining a whole
  /// deficit out of one node evicts blocks the score never got to look at.
  enum class Mode { kDrain, kMicro };

  PrefixRadixTree tree;

  Mode mode = Mode::kMicro;
  /// Blocks per round in kMicro. vLLM's MICRO_BATCH_SIZE.
  int64_t micro_batch = 64;

  /// Nodes drawn per eviction round. vLLM calls this ASSOCIATIVITY.
  int n_sample = 32;

  // -------------------------------------------------------------------------
  // Optional S3FIFO-style admission queue.
  //
  // Off when small_size_ratio is 0, which is the default: the cache is then
  // one tier and everything above applies to all of it.
  //
  // When on, the cache is split in two and the two tiers are evicted by
  // completely different rules. The small queue is a plain FIFO -- no
  // sampling, no scoring, no tree involvement -- because its whole job is to
  // discard one-hit wonders cheaply before they ever reach the main set.
  // Partial-node eviction applies only to the main set.
  //
  // There is no second hash table: an object in the cache that the tree has
  // NOT marked resident is by definition still in the small queue. Promotion
  // is then just tree.mark_resident() plus unlinking from the FIFO -- the
  // object never moves, and every eviction still goes through
  // cache_evict_base, so eviction-observing hooks keep working.
  // -------------------------------------------------------------------------

  /// Fraction of the cache given to the small queue. 0 disables it.
  double small_size_ratio = 0.0;
  /// Ghost capacity as a fraction of the cache. Ghost holds ids only.
  double ghost_size_ratio = 0.90;
  /// Accesses a block must collect in the small queue to earn promotion.
  int move_to_main_threshold = 2;

  /// How the small queue picks what to discard.
  ///
  /// kFifo is S3FIFO's rule and the default: drop the oldest. kCost is a
  /// diagnostic -- scan the oldest `small_window` entries and drop the
  /// lowest-scoring one, i.e. give the small queue the same cost awareness the
  /// main set has. The pair isolates how much of the small queue's damage comes
  /// from being cost-blind as opposed to from holding capacity at all.
  enum class SmallEvict { kFifo, kCost };
  SmallEvict small_evict = SmallEvict::kFifo;
  /// How deep into the FIFO kCost looks. 0 means the whole queue.
  int64_t small_window = 64;

  /// Cost-aware admission: a block whose cost exceeds `small_bypass_mult`
  /// times the running mean cost of everything admitted so far skips the queue
  /// and goes straight to main. 0 disables it, which is S3FIFO's rule.
  ///
  /// This is the knob that matters if the queue's damage turns out to be the
  /// admission gate rather than what the queue evicts: the gate makes every
  /// newly seen block miss twice before it can be cached in the cost-aware
  /// tier, and a newly seen block is a deep suffix block, i.e. an expensive
  /// one.
  double small_bypass_mult = 0.0;

  // ---- small-queue state, owned by the shared vtable ----
  cache_obj_t *small_head = nullptr;  ///< FIFO order, oldest at head
  cache_obj_t *small_tail = nullptr;
  int64_t small_bytes = 0;
  int64_t small_capacity = 0;
  /// Ghost of recently evicted small-queue ids; a hit here skips the queue.
  cache_t *ghost = nullptr;

  // ---- ARC-style adaptive probation boundary (adaptive=1) ----------------
  // The small queue's size is a hard-coded guess, and it is the wrong guess by
  // ~6x on some traces and ~0 on others. Stock S3FIFO (and this cache) keep a
  // ghost only for what the SMALL queue drops, so they only ever see evidence
  // that probation is too SMALL. Adding a ghost for MAIN-set evictions supplies
  // the missing other half, and the boundary can then be moved the way ARC
  // moves p between T1 and T2:
  //     hit in the small ghost -> probation starved -> small_target += d
  //     hit in the main  ghost -> main      starved -> small_target -= d
  // Unlike ARC, d is scaled by the block's recompute cost when
  // adaptive-cost-weighted=1, so the boundary chases compute savings rather
  // than hit count -- which is what this cache is scored on.
  bool adaptive = false;
  bool adaptive_cost_weighted = true;
  double adaptive_step = 1.0;
  double small_target = 0.0;   ///< the adaptive analogue of ARC's p, in bytes
  cache_t *main_ghost = nullptr;
  bool hit_on_main_ghost = false;
  int64_t n_small_ghost_hit = 0;
  int64_t n_main_ghost_hit = 0;
  /// Until the cache first evicts, the small queue overflowing means the cache
  /// is simply still filling, so admit straight to main.
  bool has_evicted = false;
  /// Set by find() when the request hit the ghost, consumed by insert().
  bool hit_on_ghost = false;
  bool small_enabled() const { return small_capacity > 0; }
  /// Which end of a node to evict from.
  ///
  /// Head (shallowest resident block) is the default, matching vLLM's
  /// leftmost-evictable choice. Tail looks more natural for a prefix cache --
  /// drop the private end, keep the shared root -- but it interacts badly with
  /// a position-based cost model: the deepest block of a node is also the most
  /// expensive one, so restricting candidates to tails forces the policy to
  /// evict exactly what its own score wants to keep. Measured on qwen_traceA at
  /// 8k blocks with the position cost model, tail scores 0.0129 compute saving
  /// against head's 0.0580.
  bool evict_from_tail = false;

  // -------------------------------------------------------------------------
  // Instrumentation. Off unless the `stats` param is set; printed at teardown.
  //
  // Exists to answer where a block dies and what it was worth: a cost-aware
  // main set and a cost-blind small queue should show very different mean
  // costs among the blocks they discard.
  // -------------------------------------------------------------------------
  bool print_stats = false;
  int64_t n_admit = 0;          ///< inserts, total
  int64_t n_small_admit = 0;    ///< inserts that landed in the small queue
  int64_t n_direct_admit = 0;   ///< inserts that skipped the queue (ghost or
                                ///< cost bypass)
  int64_t n_promote = 0;        ///< small -> main promotions
  double cost_small_admit = 0.0;
  double cost_direct_admit = 0.0;
  double cost_promote = 0.0;
  int64_t n_small_drop = 0;     ///< blocks discarded from the small queue
  int64_t n_main_evict = 0;     ///< blocks evicted from the main set
  int64_t n_small_hit = 0;      ///< hits served by a block still in the queue
  int64_t n_main_hit = 0;       ///< hits served by the main set
  double cost_admit = 0.0;      ///< summed cost over all inserts
  double cost_small_drop = 0.0;
  double cost_main_evict = 0.0;
};

/**
 * Build a cache_t around `impl`, which the returned cache takes ownership of.
 * Installs the shared vtable, including record_request.
 *
 * Recognised cache_specific_params: n-sample=<int>, evict-from=<tail|head>,
 * eviction-mode=<drain|micro>, micro-batch=<int>, small-size-ratio=<double>,
 * ghost-size-ratio=<double>, move-to-main-threshold=<int>, stats=<0|1>.
 */
cache_t *partial_node_cache_init(const char *cache_name,
                                 common_cache_params_t ccache_params,
                                 const char *cache_specific_params,
                                 PartialNodeCache *impl);

}  // namespace eviction
