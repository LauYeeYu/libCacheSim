#include "partialNodeCache.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "dataStructure/hashtable/hashtable.h"
#include "libCacheSim/evictionAlgo.h"
#include "libCacheSim/macro.h"
#include "utils/include/mymath.h"
}

namespace eviction {
namespace {

inline PartialNodeCache *impl_of(const cache_t *cache) {
  return static_cast<PartialNodeCache *>(cache->eviction_params);
}

/// The block this node would give up next, or nullptr if it has none left.
cache_obj_t *node_candidate(cache_t *cache, PartialNodeCache *impl,
                            const PrefixRadixTree::Node *node) {
  const int64_t pos = impl->evict_from_tail ? impl->tree.tail_resident(node)
                                            : impl->tree.head_resident(node);
  if (pos < 0) return nullptr;
  return hashtable_find_obj_id(cache->hashtable,
                               node->blocks[static_cast<size_t>(pos)]);
}

/// Evict one object from the small queue, S3FIFO style: walk the FIFO from the
/// oldest, promoting anything that earned it, until one block is actually
/// dropped. Deliberately no sampling and no scoring -- the small queue exists
/// to discard one-hit wonders cheaply.
///
/// Returns true if a block was dropped (the caller made progress).
bool evict_small(cache_t *cache, PartialNodeCache *impl) {
  while (impl->small_head != nullptr) {
    cache_obj_t *victim = impl->small_head;

    // kCost: promote everything in the window that earned it, then drop the
    // lowest-scoring survivor instead of the oldest. Diagnostic only -- see
    // PartialNodeCache::SmallEvict.
    if (impl->small_evict == PartialNodeCache::SmallEvict::kCost) {
      int64_t seen = 0;
      double best = DBL_MAX;
      cache_obj_t *pick = nullptr;
      for (cache_obj_t *o = impl->small_head; o != nullptr;
           o = o->queue.next) {
        if (impl->small_window > 0 && seen >= impl->small_window) break;
        ++seen;
        if (static_cast<int>(o->misc.freq) >= impl->move_to_main_threshold) {
          continue;  // promotable; the head walk below will take care of it
        }
        const double sc = impl->score(cache, o);
        if (sc < best) {
          best = sc;
          pick = o;
        }
      }
      if (pick != nullptr &&
          static_cast<int>(impl->small_head->misc.freq) <
              impl->move_to_main_threshold) {
        victim = pick;
      }
    }
    const obj_id_t id = victim->obj_id;
    const int64_t size = victim->obj_size;

    if (static_cast<int>(victim->misc.freq) >= impl->move_to_main_threshold) {
      // Promote: unlink from the FIFO and tell the tree it is resident. The
      // object itself does not move -- there is only one hash table.
      remove_obj_from_list(&impl->small_head, &impl->small_tail, victim);
      impl->small_bytes -= size;
      victim->Random.last_access_vtime = cache->n_req;
      impl->tree.mark_resident(id);
      ++impl->n_promote;
      impl->cost_promote += static_cast<double>(victim->cost);
      continue;
    }

    // Drop it, remembering the id so a quick re-reference skips the queue.
    remove_obj_from_list(&impl->small_head, &impl->small_tail, victim);
    impl->small_bytes -= size;
    if (impl->ghost != nullptr) {
      request_t ghost_req;
      memset(&ghost_req, 0, sizeof(ghost_req));
      ghost_req.obj_id = id;
      ghost_req.obj_size = 1;
      ghost_req.valid = true;
      impl->ghost->get(impl->ghost, &ghost_req);
    }
    ++impl->n_small_drop;
    impl->cost_small_drop += static_cast<double>(victim->cost);
    cache_evict_base(cache, victim, true);
    return true;
  }
  return false;
}

/// Bytes held by the main set, i.e. everything that is not in the small queue.
inline int64_t main_occupied(const cache_t *cache, const PartialNodeCache *impl) {
  return cache->occupied_byte - impl->small_bytes;
}

/// Defined below, next to the eviction path it shares its scoring with.
cache_obj_t *pick_victim(cache_t *cache);

// ---------------------------------------------------------------------------
// cache_t vtable
// ---------------------------------------------------------------------------

/// Where blocks die and what they were worth. See PartialNodeCache's stats
/// block for why this exists.
void pn_print_stats(const cache_t *cache, const PartialNodeCache *impl) {
  const double hits = static_cast<double>(impl->n_small_hit + impl->n_main_hit);
  const double mean_admit =
      impl->n_admit > 0 ? impl->cost_admit / impl->n_admit : 0.0;
  const double mean_small_drop =
      impl->n_small_drop > 0 ? impl->cost_small_drop / impl->n_small_drop : 0.0;
  const double mean_main_evict =
      impl->n_main_evict > 0 ? impl->cost_main_evict / impl->n_main_evict : 0.0;
  // Mean small-queue residency, in block accesses (libCacheSim's n_req counts
  // one per block, not per LLM request): a block is pushed out after about
  // small_capacity further admissions, which arrive at n_small_admit/n_req.
  const double admits_per_req =
      cache->n_req > 0 ? static_cast<double>(impl->n_small_admit) / cache->n_req
                       : 0.0;
  const double residency_req =
      admits_per_req > 0 ? impl->small_capacity / admits_per_req : 0.0;

  printf(
      "STATS algo=%s small_capacity=%lld admit=%lld small_admit=%lld "
      "direct_admit=%lld promote=%lld promote_rate=%.4f small_drop=%lld "
      "main_evict=%lld small_hit_share=%.4f small_residency_acc=%.0f "
      "mean_cost_admit=%.1f mean_cost_small_admit=%.1f "
      "mean_cost_direct_admit=%.1f mean_cost_promote=%.1f "
      "mean_cost_small_drop=%.1f mean_cost_main_evict=%.1f\n",
      cache->cache_name, static_cast<long long>(impl->small_capacity),
      static_cast<long long>(impl->n_admit),
      static_cast<long long>(impl->n_small_admit),
      static_cast<long long>(impl->n_direct_admit),
      static_cast<long long>(impl->n_promote),
      impl->n_small_admit > 0
          ? static_cast<double>(impl->n_promote) / impl->n_small_admit
          : 0.0,
      static_cast<long long>(impl->n_small_drop),
      static_cast<long long>(impl->n_main_evict),
      hits > 0 ? impl->n_small_hit / hits : 0.0, residency_req, mean_admit,
      impl->n_small_admit > 0 ? impl->cost_small_admit / impl->n_small_admit
                              : 0.0,
      impl->n_direct_admit > 0 ? impl->cost_direct_admit / impl->n_direct_admit
                              : 0.0,
      impl->n_promote > 0 ? impl->cost_promote / impl->n_promote : 0.0,
      mean_small_drop, mean_main_evict);
}

void pn_free(cache_t *cache) {
  PartialNodeCache *impl = impl_of(cache);
  if (impl->print_stats) pn_print_stats(cache, impl);
  if (impl->adaptive) {
    fprintf(stderr,
            "PN_ADAPT cache_size=%lld small_target=%.1f small_target_frac=%.4f "
            "small_ghost_hit=%lld main_ghost_hit=%lld step=%.3f cost_weighted=%d\n",
            (long long)cache->cache_size, impl->small_target,
            impl->small_target / (double)cache->cache_size,
            (long long)impl->n_small_ghost_hit,
            (long long)impl->n_main_ghost_hit, impl->adaptive_step,
            impl->adaptive_cost_weighted ? 1 : 0);
  }
  if (impl->ghost != nullptr) impl->ghost->cache_free(impl->ghost);
  if (impl->main_ghost != nullptr) impl->main_ghost->cache_free(impl->main_ghost);
  delete impl;
  cache->eviction_params = nullptr;
  cache_struct_free(cache);
}

bool pn_get(cache_t *cache, const request_t *req) {
  return cache_get_base(cache, req);
}

/**
 * How far to move the probation boundary on a ghost hit.
 *
 * ARC's rule is delta = max(|other ghost| / |this ghost|, 1) -- a confidence
 * scaling, one unit of cache per unit of evidence. Because this cache is scored
 * on COMPUTE SAVINGS rather than hits, the default also scales by how expensive
 * the returning block is relative to the running mean admitted cost: a deep
 * block that was thrown away is worth more than a shallow one, and the boundary
 * should move further for it.
 */
static double pn_adapt_delta(cache_t *cache, PartialNodeCache *impl,
                             const request_t *req, bool from_small_ghost) {
  const double b_small =
      impl->ghost != nullptr
          ? static_cast<double>(impl->ghost->get_occupied_byte(impl->ghost))
          : 0.0;
  const double b_main =
      impl->main_ghost != nullptr
          ? static_cast<double>(
                impl->main_ghost->get_occupied_byte(impl->main_ghost))
          : 0.0;
  const double here = from_small_ghost ? b_small : b_main;
  const double there = from_small_ghost ? b_main : b_small;
  double delta = (here > 0.0 && there / here > 1.0) ? there / here : 1.0;
  if (impl->adaptive_cost_weighted && impl->n_admit > 0) {
    const double mean_cost = impl->cost_admit / static_cast<double>(impl->n_admit);
    if (mean_cost > 0.0) {
      double w = static_cast<double>(req->cost) / mean_cost;
      if (w < 0.1) w = 0.1;
      if (w > 10.0) w = 10.0;
      delta *= w;
    }
  }
  return delta * impl->adaptive_step;
}

cache_obj_t *pn_find(cache_t *cache, const request_t *req, bool update_cache) {
  PartialNodeCache *impl = impl_of(cache);
  if (update_cache && impl->small_enabled()) {
    impl->hit_on_ghost = false;
    impl->hit_on_main_ghost = false;
  }

  cache_obj_t *obj = cache_find_base(cache, req, update_cache);
  if (obj != nullptr) {
    if (update_cache) obj->Random.last_access_vtime = cache->n_req;
    if (update_cache && impl->print_stats) {
      if (impl->small_enabled() && !impl->tree.is_resident(obj->obj_id)) {
        ++impl->n_small_hit;
      } else {
        ++impl->n_main_hit;
      }
    }
    return obj;
  }

  // A miss that the ghost recognises means this block was in the small queue
  // recently and came back: admit it straight to main next time.
  if (update_cache && impl->small_enabled() && impl->ghost != nullptr &&
      impl->ghost->remove(impl->ghost, req->obj_id)) {
    impl->hit_on_ghost = true;
    if (impl->adaptive) {
      ++impl->n_small_ghost_hit;
      impl->small_target =
          std::min(impl->small_target + pn_adapt_delta(cache, impl, req, true),
                   static_cast<double>(cache->cache_size));
    }
  } else if (update_cache && impl->adaptive && impl->main_ghost != nullptr &&
             impl->main_ghost->remove(impl->main_ghost, req->obj_id)) {
    impl->hit_on_main_ghost = true;
    ++impl->n_main_ghost_hit;
    impl->small_target =
        std::max(impl->small_target - pn_adapt_delta(cache, impl, req, false),
                 0.0);
  }
  return nullptr;
}

cache_obj_t *pn_insert(cache_t *cache, const request_t *req) {
  PartialNodeCache *impl = impl_of(cache);
  cache_obj_t *obj = cache_insert_base(cache, req);
  obj->Random.last_access_vtime = cache->n_req;

  ++impl->n_admit;
  impl->cost_admit += static_cast<double>(obj->cost);

  if (!impl->small_enabled()) {
    impl->tree.mark_resident(obj->obj_id);
    return obj;
  }

  // Straight to main if the ghost vouched for it, or if the cache is still
  // filling and the small queue is already full.
  const bool too_expensive_to_gate =
      impl->small_bypass_mult > 0.0 && impl->n_admit > 1 &&
      static_cast<double>(obj->cost) >
          impl->small_bypass_mult * (impl->cost_admit / impl->n_admit);
  const int64_t small_cap = impl->adaptive
                                ? static_cast<int64_t>(impl->small_target)
                                : impl->small_capacity;
  const bool to_main =
      impl->hit_on_ghost || impl->hit_on_main_ghost || too_expensive_to_gate ||
      (!impl->has_evicted && impl->small_bytes >= small_cap);
  impl->hit_on_ghost = false;
  impl->hit_on_main_ghost = false;

  if (to_main) {
    impl->tree.mark_resident(obj->obj_id);
    ++impl->n_direct_admit;
    impl->cost_direct_admit += static_cast<double>(obj->cost);
  } else {
    append_obj_to_tail(&impl->small_head, &impl->small_tail, obj);
    impl->small_bytes += obj->obj_size;
    ++impl->n_small_admit;
    impl->cost_small_admit += static_cast<double>(obj->cost);
  }
  return obj;
}

cache_obj_t *pn_to_evict(cache_t *cache, const request_t * /*req*/) {
  cache_obj_t *victim = pick_victim(cache);
  cache->to_evict_candidate = victim;
  cache->to_evict_candidate_gen_vtime = cache->n_req;
  return victim;
}

/// Drain orphans, which sampling can never reach. Returns how many went.
int64_t evict_orphans(cache_t *cache, PartialNodeCache *impl, int64_t n) {
  int64_t evicted = 0;
  while (evicted < n && impl->tree.has_orphans()) {
    const obj_id_t id = impl->tree.any_orphan();
    cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, id);
    impl->tree.mark_evicted(id);
    if (obj == nullptr) continue;  // tree drifted; the id is gone either way
    ++impl->n_main_evict;
    impl->cost_main_evict += static_cast<double>(obj->cost);
    cache_evict_base(cache, obj, true);
    ++evicted;
  }
  return evicted;
}

/**
 * Score a node by one representative block -- vLLM's "canonical block".
 *
 * Scoring a whole node by one block is what makes node sampling affordable, and
 * the representative has to sit inside the chunk about to be evicted, or the
 * score describes blocks that are not going anywhere. The rule is taken from the
 * vLLM prototype verbatim so the two implementations rank nodes identically:
 *
 *   drain: start at rank min(chunk, node_size) / 2 and scan outwards over the
 *          whole node   (_get_canonical_block_for_radix_tree_node)
 *   micro: start at rank first_resident + chunk/2, clamped to the last rank,
 *          and scan outwards but never before first_resident
 *          (_get_chunk_canonical_block)
 *
 * Note both walk *ranks*, resident or not, so a sparsely-resident node can be
 * represented by a block some way from the chunk's centre. That is vLLM's
 * behaviour, not an accident of this port.
 *
 * Returns false when the node has nothing to give.
 */
bool node_chunk_score(cache_t *cache, PartialNodeCache *impl,
                      const PrefixRadixTree::Node *node, int64_t chunk,
                      double &score_out) {
  const int64_t total = static_cast<int64_t>(node->blocks.size());
  if (total == 0 || node->n_resident == 0) return false;

  obj_id_t rep = 0;
  bool found = false;
  if (impl->mode == PartialNodeCache::Mode::kMicro) {
    const int64_t first =
        impl->tree.first_resident_rank(node, impl->evict_from_tail);
    if (first < 0) return false;
    const int64_t target = std::min(first + chunk / 2, total - 1);
    found = impl->tree.canonical_resident(node, impl->evict_from_tail, target,
                                          first, rep);
  } else {
    const int64_t start = std::min(chunk, total) / 2;
    found = impl->tree.canonical_resident(node, impl->evict_from_tail, start,
                                          0, rep);
  }
  if (!found) return false;

  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, rep);
  if (obj == nullptr) return false;
  score_out = impl->score(cache, obj);
  return true;
}

/// Take up to `max_take` blocks from `node`, in eviction order.
int64_t take_from_node(cache_t *cache, PartialNodeCache *impl,
                       const PrefixRadixTree::Node *node, int64_t max_take,
                       std::vector<obj_id_t> &scratch) {
  // Snapshot the ids first: evicting the node's last resident block prunes the
  // node, and anything read from it afterwards is a use-after-free.
  scratch.clear();
  impl->tree.collect_resident(node, impl->evict_from_tail, max_take, scratch);

  int64_t took = 0;
  for (const obj_id_t id : scratch) {
    cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, id);
    if (obj == nullptr) continue;
    impl->tree.mark_evicted(id);
    ++impl->n_main_evict;
    impl->cost_main_evict += static_cast<double>(obj->cost);
    if (impl->main_ghost != nullptr) {
      request_t gr;
      memset(&gr, 0, sizeof(gr));
      gr.obj_id = id;
      gr.obj_size = 1;
      gr.valid = true;
      impl->main_ghost->get(impl->main_ghost, &gr);
    }
    cache_evict_base(cache, obj, true);
    ++took;
  }
  return took;
}

/**
 * Evict up to n objects. `n` is a hard cap.
 *
 * Per round: sample n_sample nodes, score each by its chunk midpoint, and take
 * the chunk from the best-scoring one. kDrain lets that chunk be the whole
 * remaining deficit; kMicro caps it at micro_batch and re-samples. Either way
 * the sampling cost is paid once per round rather than once per evicted block.
 *
 * Counterpart of _sample_and_evict_drain / _sample_and_evict_micro in the vLLM
 * prototype.
 */
int64_t pn_evict_n(cache_t *cache, const request_t * /*req*/, int64_t n) {
  if (n <= 0) return 0;
  PartialNodeCache *impl = impl_of(cache);
  impl->has_evicted = true;
  int64_t evicted = evict_orphans(cache, impl, n);

  // With a small queue, drain it first whenever the main set is within its
  // share. Only what survives promotion out of the queue is ever subject to
  // partial-node eviction below.
  while (impl->small_enabled() && evicted < n && impl->small_bytes > 0 &&
         (impl->adaptive
              ? static_cast<double>(impl->small_bytes) > impl->small_target
              : main_occupied(cache, impl) <=
                    cache->cache_size - impl->small_capacity)) {
    if (!evict_small(cache, impl)) break;
    ++evicted;
  }

  std::vector<obj_id_t> scratch;
  while (evicted < n) {
    const int64_t want = n - evicted;
    const int64_t chunk = (impl->mode == PartialNodeCache::Mode::kMicro)
                              ? std::min(want, impl->micro_batch)
                              : want;

    // The candidate pool is often smaller than n_sample -- a path-compressed
    // prefix tree has few, long nodes, and on the qwen traces the pool averages
    // well under a hundred. Random draws would then be strictly worse than
    // looking at everything: sampling with replacement from a pool of 78 with
    // 128 draws still misses about a fifth of it, while costing more score
    // evaluations than a full scan. So scan exhaustively once the pool fits,
    // and sample only when it genuinely does not.
    const int64_t pool = impl->tree.n_sampleable();
    if (pool == 0) {
      // Nothing in the main set to sample. If the small queue still holds
      // something, fall back to it rather than reporting no progress.
      if (impl->small_enabled() && impl->small_bytes > 0 &&
          evict_small(cache, impl)) {
        ++evicted;
        continue;
      }
      break;
    }
    const bool exhaustive = pool <= impl->n_sample;
    const int64_t draws = exhaustive ? pool : impl->n_sample;

    const PrefixRadixTree::Node *victim = nullptr;
    double best = DBL_MAX;
    for (int64_t i = 0; i < draws; ++i) {
      const PrefixRadixTree::Node *node =
          exhaustive ? impl->tree.node_at(i) : impl->tree.sample_node(next_rand());
      if (node == nullptr) break;
      double score = 0.0;
      if (!node_chunk_score(cache, impl, node, chunk, score)) continue;
      if (score < best) {
        best = score;
        victim = node;
      }
    }
    if (victim == nullptr) break;

    const int64_t took = take_from_node(cache, impl, victim, chunk, scratch);
    if (took == 0) break;  // a whole round achieved nothing; stop rather than spin
    evicted += took;
  }

  cache->to_evict_candidate_gen_vtime = -1;
  return evicted;
}

/**
 * The single block eviction would take next, without taking it.
 *
 * Shares node_chunk_score with the eviction path at chunk = 1, so to_evict and
 * evict cannot disagree about the victim. It is only a faithful preview at
 * chunk 1: with a larger chunk eviction takes a run of blocks and scores the
 * node by a different canonical block.
 */
cache_obj_t *pick_victim(cache_t *cache) {
  PartialNodeCache *impl = impl_of(cache);

  if (impl->tree.has_orphans()) {
    cache_obj_t *obj =
        hashtable_find_obj_id(cache->hashtable, impl->tree.any_orphan());
    if (obj != nullptr) return obj;
  }

  const int64_t pool = impl->tree.n_sampleable();
  if (pool == 0) return nullptr;
  const bool exhaustive = pool <= impl->n_sample;
  const int64_t draws = exhaustive ? pool : impl->n_sample;

  const PrefixRadixTree::Node *victim = nullptr;
  double best = DBL_MAX;
  for (int64_t i = 0; i < draws; ++i) {
    const PrefixRadixTree::Node *node =
        exhaustive ? impl->tree.node_at(i) : impl->tree.sample_node(next_rand());
    if (node == nullptr) break;
    double score = 0.0;
    if (!node_chunk_score(cache, impl, node, 1, score)) continue;
    if (score < best) {
      best = score;
      victim = node;
    }
  }
  return victim == nullptr ? nullptr : node_candidate(cache, impl, victim);
}

void pn_evict(cache_t *cache, const request_t *req) {
  // Single-object eviction is the n = 1 case, so there is only one place where
  // victims get chosen.
  if (pn_evict_n(cache, req, 1) == 0) {
    DEBUG_ASSERT(cache->n_obj == 0);
    WARN("no object can be evicted\n");
  }
}

bool pn_remove(cache_t *cache, obj_id_t obj_id) {
  PartialNodeCache *impl = impl_of(cache);
  cache_obj_t *obj = hashtable_find_obj_id(cache->hashtable, obj_id);
  if (obj == nullptr) return false;
  if (impl->small_enabled() && !impl->tree.is_resident(obj_id)) {
    remove_obj_from_list(&impl->small_head, &impl->small_tail, obj);
    impl->small_bytes -= obj->obj_size;
  }
  impl->tree.mark_evicted(obj_id);
  cache_remove_obj_base(cache, obj, true);
  return true;
}

void pn_record_request(cache_t *cache, const obj_id_t *ids, int64_t n) {
  if (ids == nullptr || n <= 0) return;
  PartialNodeCache *impl = impl_of(cache);
  impl->tree.add_sequence(ids, n);
  impl->on_record_request(ids, n);
  if (impl->tree.n_ambiguous_blocks() > 0) {
    WARN_ONCE(
        "%s: block ids are not prefix-unique, so the same block appears under "
        "more than one path and the prefix tree cannot represent it. Use "
        "prefix-hashed block ids.\n",
        cache->cache_name);
  }
}

void pn_parse_params(cache_t *cache, const char *cache_specific_params) {
  PartialNodeCache *impl = impl_of(cache);
  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != nullptr && params_str[0] != '\0') {
    char *key = strsep(&params_str, "=");
    char *value = strsep(&params_str, ",");
    while (params_str != nullptr && *params_str == ' ') ++params_str;

    if (strcasecmp(key, "n-sample") == 0) {
      impl->n_sample = static_cast<int>(strtol(value, nullptr, 0));
    } else if (strcasecmp(key, "evict-from") == 0) {
      if (strcasecmp(value, "tail") == 0) {
        impl->evict_from_tail = true;
      } else if (strcasecmp(value, "head") == 0) {
        impl->evict_from_tail = false;
      } else {
        ERROR("%s: evict-from must be tail or head, got %s\n",
              cache->cache_name, value);
      }
    } else if (strcasecmp(key, "eviction-mode") == 0) {
      if (strcasecmp(value, "drain") == 0) {
        impl->mode = PartialNodeCache::Mode::kDrain;
      } else if (strcasecmp(value, "micro") == 0) {
        impl->mode = PartialNodeCache::Mode::kMicro;
      } else {
        ERROR("%s: eviction-mode must be drain or micro, got %s\n",
              cache->cache_name, value);
      }
    } else if (strcasecmp(key, "small-size-ratio") == 0) {
      impl->small_size_ratio = strtod(value, nullptr);
    } else if (strcasecmp(key, "ghost-size-ratio") == 0) {
      impl->ghost_size_ratio = strtod(value, nullptr);
    } else if (strcasecmp(key, "adaptive") == 0) {
      impl->adaptive = (atoi(value) != 0);
    } else if (strcasecmp(key, "adaptive-step") == 0) {
      impl->adaptive_step = strtod(value, nullptr);
    } else if (strcasecmp(key, "adaptive-cost-weighted") == 0) {
      impl->adaptive_cost_weighted = (atoi(value) != 0);
    } else if (strcasecmp(key, "move-to-main-threshold") == 0) {
      impl->move_to_main_threshold = static_cast<int>(strtol(value, nullptr, 0));
    } else if (strcasecmp(key, "small-evict") == 0) {
      if (strcasecmp(value, "fifo") == 0) {
        impl->small_evict = PartialNodeCache::SmallEvict::kFifo;
      } else if (strcasecmp(value, "cost") == 0) {
        impl->small_evict = PartialNodeCache::SmallEvict::kCost;
      } else {
        ERROR("%s: small-evict must be fifo or cost, got %s\n",
              cache->cache_name, value);
      }
    } else if (strcasecmp(key, "small-bypass-mult") == 0) {
      impl->small_bypass_mult = strtod(value, nullptr);
    } else if (strcasecmp(key, "small-window") == 0) {
      impl->small_window = strtoll(value, nullptr, 0);
    } else if (strcasecmp(key, "stats") == 0) {
      impl->print_stats = strtol(value, nullptr, 0) != 0;
    } else if (strcasecmp(key, "micro-batch") == 0) {
      impl->micro_batch = strtoll(value, nullptr, 0);
    } else if (strcasecmp(key, "print") == 0) {
      printf("n-sample=%d,evict-from=%s,eviction-mode=%s,micro-batch=%lld\n",
             impl->n_sample, impl->evict_from_tail ? "tail" : "head",
             impl->mode == PartialNodeCache::Mode::kMicro ? "micro" : "drain",
             static_cast<long long>(impl->micro_batch));
      exit(0);
    } else {
      ERROR("%s does not have parameter %s, support n-sample, evict-from, "
            "eviction-mode, micro-batch, small-size-ratio, "
            "ghost-size-ratio, move-to-main-threshold, small-evict, "
            "small-window, small-bypass-mult, stats\n",
            cache->cache_name, key);
    }
  }
  free(old_params_str);
}

}  // namespace

cache_t *partial_node_cache_init(const char *cache_name,
                                 common_cache_params_t ccache_params,
                                 const char *cache_specific_params,
                                 PartialNodeCache *impl) {
  cache_t *cache =
      cache_struct_init(cache_name, ccache_params, cache_specific_params);
  cache->cache_free = pn_free;
  cache->get = pn_get;
  cache->find = pn_find;
  cache->insert = pn_insert;
  cache->evict = pn_evict;
  cache->remove = pn_remove;
  cache->to_evict = pn_to_evict;
  cache->record_request = pn_record_request;
  cache->evict_n = pn_evict_n;
  cache->obj_md_size = 0;
  cache->eviction_params = impl;

  if (cache_specific_params != nullptr) {
    pn_parse_params(cache, cache_specific_params);
  }

  // adaptive=1 runs the queue whatever small-size-ratio says: the ratio only
  // sets where the boundary STARTS, and 0 would mean "no queue at all".
  if (impl->adaptive && impl->small_size_ratio <= 0.0) {
    impl->small_size_ratio = 0.10;
  }

  if (impl->small_size_ratio > 0.0) {
    impl->small_capacity =
        static_cast<int64_t>(ccache_params.cache_size * impl->small_size_ratio);
    if (impl->small_capacity < 1) impl->small_capacity = 1;
    impl->small_target = static_cast<double>(impl->small_capacity);
    // With the boundary moving, small_capacity must not also cap the queue --
    // it only gates small_enabled() from here on.
    if (impl->adaptive) impl->small_capacity = ccache_params.cache_size;

    const int64_t ghost_capacity =
        static_cast<int64_t>(ccache_params.cache_size * impl->ghost_size_ratio);
    if (ghost_capacity > 0) {
      common_cache_params_t ghost_params = ccache_params;
      ghost_params.cache_size = ghost_capacity;
      impl->ghost = FIFO_init(ghost_params, nullptr);
      snprintf(impl->ghost->cache_name, CACHE_NAME_ARRAY_LEN, "FIFO-ghost");
      if (impl->adaptive) {
        common_cache_params_t mg_params = ccache_params;
        mg_params.cache_size = ghost_capacity;
        impl->main_ghost = FIFO_init(mg_params, nullptr);
        snprintf(impl->main_ghost->cache_name, CACHE_NAME_ARRAY_LEN,
                 "FIFO-main-ghost");
      }
    }
  }
  return cache;
}

}  // namespace eviction
