//
//  This version (S3FIFO.c) differs from the original S3-FIFO (S3FIFOv0.c) in
//  that when the small queue is full, but the cache is not full, the original
//  S3-FIFO will insert into the small queue, but this version will insert into
//  the main queue. This version is in general better than the original S3-FIFO
//  because
//    1. the objects inserted after the cache is full are evicted more quickly
//    2. the objects inserted between the small queue is full and the cache is
//    full are kept slightly longer
//
//  10% small FIFO + 90% main FIFO (2-bit Clock) + ghost
//  insert to small FIFO if not in the ghost, else insert to the main FIFO
//  evict from small FIFO:
//      if object in the small is accessed,
//          reinsert to main FIFO,
//      else
//          evict and insert to the ghost
//  evict from main FIFO:
//      if object in the main is accessed,
//          reinsert to main FIFO,
//      else
//          evict
//
//
//  S3FIFO.c
//  libCacheSim
//
//  Created by Juncheng on 12/4/24.
//  Copyright © 2018 Juncheng. All rights reserved.
//

#include "dataStructure/hashtable/hashtable.h"
#include "libCacheSim/evictionAlgo.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  cache_t *small_fifo;
  cache_t *ghost_fifo;
  cache_t *main_fifo;
  bool hit_on_ghost;

  int move_to_main_threshold;
  double small_size_ratio;
  double ghost_size_ratio;

  bool has_evicted;
  request_t *req_local;

  /* ---- ARC-style adaptive probation boundary (off unless adaptive=1) ----
   * Stock S3FIFO hard-codes the small queue at 10% of the cache and keeps a
   * ghost only for what the SMALL queue drops, so it never sees evidence that
   * the MAIN set is the one starved.  With adaptive=1 we add the missing
   * second ghost and move a soft boundary `small_target` between the two,
   * exactly the way ARC moves p between T1 and T2:
   *   hit in the small ghost  -> probation was too small -> small_target += d
   *   hit in the main  ghost  -> main was too small      -> small_target -= d
   * The queues themselves stay S3FIFO's (FIFO small with lazy freq>=threshold
   * promotion, CLOCK main) -- only the sizing rule is borrowed. */
  bool adaptive;
  /* Which rule moves small_target.
   *   0 arc      -- ARC's two-ghost rule (needs main_ghost_fifo)
   *   1 witness  -- NO main ghost. Grow on a small-ghost hit; shrink when MAIN
   *                 evicts a block that was hit during its main residency.
   *                 A prospective witness ("we knew it was useful") replaces
   *                 ARC's retrospective one ("it came back"), at 1 bit/block.
   *   2 density  -- NO ghost needed. Move toward whichever tier has the higher
   *                 hits-per-byte over the last cache-size worth of accesses.
   *   4 split    -- NO main ghost. Same signal as depth, but the threshold is
   *                 the ghost's MIDPOINT, not small_target: a hit in the ghost's
   *                 front half means a modestly bigger probation catches it
   *                 (grow), a hit in the back half means the block needs
   *                 main-style retention (shrink). Two-sided and balanced by
   *                 construction -- the closest single-ghost analogue of ARC's
   *                 two-ghost balance -- and with no absorbing state, which is
   *                 exactly what kills mode 3.
   *   5 witbal   -- witness, with ARC's rate normalisation: each event moves the
   *                 boundary by max(count of the OTHER event / count of this
   *                 one, 1), so a rare shrink signal still balances a frequent
   *                 grow signal. Mode 1 runs away on traceB without this.
   *   3 depth    -- NO main ghost. Use WHERE in the small ghost a hit lands:
   *                 shallow (came back soon after leaving probation) means a
   *                 slightly bigger probation would have caught it -> grow;
   *                 deep means probation was never going to hold it and the
   *                 space belongs to main -> shrink. One threshold, no constants. */
  int adapt_mode;
  bool main_ghost_on;   /* second ghost without the moving boundary */
  int64_t ghost_ins_seq;        /* depth mode: ghost insertions so far */
  double h_small, h_main;       /* density mode: decaying per-tier hit counts */
  int64_t density_countdown;
  int64_t n_witness_shrink;
  cache_t *main_ghost_fifo;
  double small_target;        /* bytes; the adaptive analogue of ARC's p */
  double adaptive_step;       /* damping on the ARC delta, 1.0 = ARC's own */
  bool hit_on_main_ghost;
  int64_t n_small_ghost_hit;
  int64_t n_main_ghost_hit;
} S3FIFO_params_t;

static const char *DEFAULT_CACHE_PARAMS =
    "small-size-ratio=0.10,ghost-size-ratio=0.90,move-to-main-threshold=2,"
    "adaptive=0,adaptive-step=1.0,main-ghost=0,adapt-mode=arc";

// ***********************************************************************
// ****                                                               ****
// ****                   function declarations                       ****
// ****                                                               ****
// ***********************************************************************
static void S3FIFO_free(cache_t *cache);
static bool S3FIFO_get(cache_t *cache, const request_t *req);

static cache_obj_t *S3FIFO_find(cache_t *cache, const request_t *req,
                                bool update_cache);
static cache_obj_t *S3FIFO_insert(cache_t *cache, const request_t *req);
static cache_obj_t *S3FIFO_to_evict(cache_t *cache, const request_t *req);
static void S3FIFO_evict(cache_t *cache, const request_t *req);
static void S3FIFO_evict_once(cache_t *cache, const request_t *req);
static bool S3FIFO_remove(cache_t *cache, obj_id_t obj_id);
static inline int64_t S3FIFO_get_occupied_byte(const cache_t *cache);
static inline int64_t S3FIFO_get_n_obj(const cache_t *cache);
static inline bool S3FIFO_can_insert(cache_t *cache, const request_t *req);
static void S3FIFO_parse_params(cache_t *cache,
                                const char *cache_specific_params);

static void S3FIFO_evict_small(cache_t *cache, const request_t *req);
static void S3FIFO_evict_main(cache_t *cache, const request_t *req);

// ***********************************************************************
// ****                                                               ****
// ****                   end user facing functions                   ****
// ****                                                               ****
// ***********************************************************************

cache_t *S3FIFO_init(const common_cache_params_t ccache_params,
                     const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("S3FIFO", ccache_params, cache_specific_params);
  cache->cache_init = S3FIFO_init;
  cache->cache_free = S3FIFO_free;
  cache->get = S3FIFO_get;
  cache->find = S3FIFO_find;
  cache->insert = S3FIFO_insert;
  cache->evict = S3FIFO_evict;
  cache->remove = S3FIFO_remove;
  cache->to_evict = S3FIFO_to_evict;
  cache->get_n_obj = S3FIFO_get_n_obj;
  cache->get_occupied_byte = S3FIFO_get_occupied_byte;
  cache->can_insert = S3FIFO_can_insert;

  cache->obj_md_size = 0;

  cache->eviction_params = malloc(sizeof(S3FIFO_params_t));
  memset(cache->eviction_params, 0, sizeof(S3FIFO_params_t));
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  params->req_local = new_request();
  params->hit_on_ghost = false;

  S3FIFO_parse_params(cache, DEFAULT_CACHE_PARAMS);
  if (cache_specific_params != NULL) {
    S3FIFO_parse_params(cache, cache_specific_params);
  }

  int64_t small_fifo_size =
      (int64_t)(ccache_params.cache_size * params->small_size_ratio);
  int64_t main_fifo_size = ccache_params.cache_size - small_fifo_size;
  int64_t ghost_fifo_size =
      (int64_t)(ccache_params.cache_size * params->ghost_size_ratio);

  if (small_fifo_size <= 0 || main_fifo_size <= 0) {
    ERROR(
        "Invalid cache size configuration: small_fifo=%lld bytes, "
        "main_fifo=%lld "
        "bytes\n",
        (long long)small_fifo_size, (long long)main_fifo_size);
  }

  common_cache_params_t ccache_params_local = ccache_params;
  ccache_params_local.cache_size = small_fifo_size;
  params->small_fifo = FIFO_init(ccache_params_local, NULL);
  params->has_evicted = false;

  if (ghost_fifo_size > 0) {
    ccache_params_local.cache_size = ghost_fifo_size;
    params->ghost_fifo = FIFO_init(ccache_params_local, NULL);
    snprintf(params->ghost_fifo->cache_name, CACHE_NAME_ARRAY_LEN,
             "FIFO-ghost");
  } else {
    params->ghost_fifo = NULL;
  }

  ccache_params_local.cache_size = main_fifo_size;
  params->main_fifo = FIFO_init(ccache_params_local, NULL);

  params->small_target = (double)small_fifo_size;
  params->hit_on_main_ghost = false;
  params->density_countdown = ccache_params.cache_size;
  params->main_ghost_fifo = NULL;
  if (params->adaptive) {
    /* The split is enforced by small_target from here on, so neither sub-cache
     * may impose its own cap -- give both the full size. */
    params->small_fifo->cache_size = ccache_params.cache_size;
    params->main_fifo->cache_size = ccache_params.cache_size;
  }
  if (((params->adaptive && params->adapt_mode == 0) || params->main_ghost_on) &&
      ghost_fifo_size > 0) {
    ccache_params_local.cache_size = ghost_fifo_size;
    params->main_ghost_fifo = FIFO_init(ccache_params_local, NULL);
    snprintf(params->main_ghost_fifo->cache_name, CACHE_NAME_ARRAY_LEN,
             "FIFO-main-ghost");
  }

  if (params->adaptive) {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN, "S3FIFOAdaptive-%.4lf-%d",
             params->small_size_ratio, params->move_to_main_threshold);
  } else {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN, "S3FIFO-%.4lf-%d",
             params->small_size_ratio, params->move_to_main_threshold);
  }

  return cache;
}

/**
 * free resources used by this cache
 *
 * @param cache
 */
static void S3FIFO_free(cache_t *cache) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  if (params->adaptive || params->main_ghost_on) {
    fprintf(stderr,
            "S3FIFO_ADAPT cache_size=%lld small_target=%.1f small_target_frac=%.4f "
            "small_ghost_hit=%lld main_ghost_hit=%lld witness_shrink=%lld "
            "mode=%d step=%.3f\n",
            (long long)cache->cache_size, params->small_target,
            params->small_target / (double)cache->cache_size,
            (long long)params->n_small_ghost_hit,
            (long long)params->n_main_ghost_hit,
            (long long)params->n_witness_shrink, params->adapt_mode,
            params->adaptive_step);
  }
  free_request(params->req_local);
  params->small_fifo->cache_free(params->small_fifo);
  if (params->ghost_fifo != NULL) {
    params->ghost_fifo->cache_free(params->ghost_fifo);
  }
  if (params->main_ghost_fifo != NULL) {
    params->main_ghost_fifo->cache_free(params->main_ghost_fifo);
  }
  params->main_fifo->cache_free(params->main_fifo);
  free(cache->eviction_params);
  cache_struct_free(cache);
}

/**
 * @brief this function is the user facing API
 * it performs the following logic
 *
 * ```
 * if obj in cache:
 *    update_metadata
 *    return true
 * else:
 *    if cache does not have enough space:
 *        evict until it has space to insert
 *    insert the object
 *    return false
 * ```
 *
 * @param cache
 * @param req
 * @return true if cache hit, false if cache miss
 */
static bool S3FIFO_get(cache_t *cache, const request_t *req) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  DEBUG_ASSERT(params->small_fifo->get_occupied_byte(params->small_fifo) +
                   params->main_fifo->get_occupied_byte(params->main_fifo) <=
               cache->cache_size);

  bool cache_hit = cache_get_base(cache, req);

  return cache_hit;
}

// ***********************************************************************
// ****                                                               ****
// ****       developer facing APIs (used by cache developer)         ****
// ****                                                               ****
// ***********************************************************************
/**
 * @brief find an object in the cache
 *
 * @param cache
 * @param req
 * @param update_cache whether to update the cache,
 *  if true, the object is promoted
 *  and if the object is expired, it is removed from the cache
 * @return the object or NULL if not found
 */
static cache_obj_t *S3FIFO_find(cache_t *cache, const request_t *req,
                                bool update_cache) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;

  // if update cache is false, we only check the fifo and main caches
  if (!update_cache) {
    cache_obj_t *obj = params->small_fifo->find(params->small_fifo, req, false);
    if (obj != NULL) {
      return obj;
    }
    obj = params->main_fifo->find(params->main_fifo, req, false);
    if (obj != NULL) {
      return obj;
    }
    return NULL;
  }

  /* update cache is true from now */
  params->hit_on_ghost = false;
  params->hit_on_main_ghost = false;
  cache_obj_t *obj = params->small_fifo->find(params->small_fifo, req, true);
  if (obj != NULL) {
    obj->S3FIFO.freq += 1;
    if (params->adaptive && params->adapt_mode == 2) params->h_small += 1.0;
    return obj;
  }

  int64_t ghost_depth = -1;
  if (params->ghost_fifo != NULL && params->adaptive &&
      (params->adapt_mode == 3 || params->adapt_mode == 4)) {
    /* depth mode needs the ghost position, so read before removing */
    cache_obj_t *g = params->ghost_fifo->find(params->ghost_fifo, req, false);
    if (g != NULL) ghost_depth = params->ghost_ins_seq - g->S3FIFO.insertion_time;
  }
  if (params->ghost_fifo != NULL &&
      params->ghost_fifo->remove(params->ghost_fifo, req->obj_id)) {
    // if object in ghost_fifo, remove will return true
    params->hit_on_ghost = true;
    params->n_small_ghost_hit++;
    if (params->adaptive && params->adapt_mode == 1) {
      /* witness: probation lost a block that came back -> grow it */
      params->small_target = MIN(params->small_target + params->adaptive_step,
                                 (double)cache->cache_size);
    } else if (params->adaptive && params->adapt_mode == 5) {
      /* witbal: rate-normalised grow step */
      const double ng = (double)(params->n_small_ghost_hit);
      const double ns = (double)(params->n_witness_shrink);
      double d = (ng > 0 && ns / ng > 1.0) ? ns / ng : 1.0;
      params->small_target = MIN(params->small_target + d * params->adaptive_step,
                                 (double)cache->cache_size);
    } else if (params->adaptive && params->adapt_mode == 4) {
      /* split: front half of the ghost -> grow, back half -> shrink */
      const double half =
          0.5 * (double)params->ghost_fifo->get_occupied_byte(params->ghost_fifo);
      const double d = params->adaptive_step;
      if (ghost_depth >= 0 && (double)ghost_depth < half) {
        params->small_target = MIN(params->small_target + d,
                                   (double)cache->cache_size);
      } else {
        params->small_target = MAX(params->small_target - d, 0.0);
      }
    } else if (params->adaptive && params->adapt_mode == 3) {
      /* depth: shallow -> a slightly bigger probation catches it; deep -> the
       * block needs main-style retention, so hand the space to main. */
      const double d = params->adaptive_step;
      if (ghost_depth >= 0 && (double)ghost_depth < params->small_target) {
        params->small_target = MIN(params->small_target + d,
                                   (double)cache->cache_size);
      } else {
        params->small_target = MAX(params->small_target - d, 0.0);
      }
    }
    if (params->adaptive && params->adapt_mode == 0) {
      /* ARC case II: the block died in probation and came back -> grow it. */
      const double b1 = (double)params->ghost_fifo->get_occupied_byte(
          params->ghost_fifo);
      const double b2 =
          params->main_ghost_fifo != NULL
              ? (double)params->main_ghost_fifo->get_occupied_byte(
                    params->main_ghost_fifo)
              : 0.0;
      double delta = (b1 > 0 && b2 / b1 > 1.0) ? b2 / b1 : 1.0;
      delta *= params->adaptive_step;
      params->small_target = MIN(params->small_target + delta,
                                 (double)cache->cache_size);
    }
  } else if (params->main_ghost_fifo != NULL &&
             params->main_ghost_fifo->remove(params->main_ghost_fifo,
                                             req->obj_id)) {
    /* ARC case III: the block was pushed out of MAIN and came back -> the
     * protected set is the starved one, so shrink probation. */
    params->hit_on_main_ghost = true;
    params->n_main_ghost_hit++;
    if (params->adaptive) {
    const double b1 =
        params->ghost_fifo != NULL
            ? (double)params->ghost_fifo->get_occupied_byte(params->ghost_fifo)
            : 0.0;
    const double b2 = (double)params->main_ghost_fifo->get_occupied_byte(
        params->main_ghost_fifo);
    double delta = (b2 > 0 && b1 / b2 > 1.0) ? b1 / b2 : 1.0;
    delta *= params->adaptive_step;
    params->small_target = MAX(params->small_target - delta, 0.0);
    }
  }

  obj = params->main_fifo->find(params->main_fifo, req, true);
  if (obj != NULL) {
    obj->S3FIFO.freq += 1;
    obj->S3FIFO.main_insert_freq = 1;  /* sticky "was useful in main" witness */
    if (params->adaptive && params->adapt_mode == 2) params->h_main += 1.0;
  }

  /* density mode: every cache-size worth of accesses, move the boundary toward
   * whichever tier is currently earning more hits per byte, then decay. */
  if (params->adaptive && params->adapt_mode == 2 &&
      --params->density_countdown <= 0) {
    params->density_countdown = cache->cache_size;
    const double s_bytes = MAX(params->small_target, 1.0);
    const double m_bytes = MAX((double)cache->cache_size - params->small_target, 1.0);
    const double ds = params->h_small / s_bytes;
    const double dm = params->h_main / m_bytes;
    const double nudge = params->adaptive_step * (double)cache->cache_size * 0.02;
    if (ds > dm) {
      params->small_target = MIN(params->small_target + nudge,
                                 (double)cache->cache_size);
    } else if (dm > ds) {
      params->small_target = MAX(params->small_target - nudge, 0.0);
    }
    params->h_small *= 0.5;
    params->h_main *= 0.5;
  }

  return obj;
}

/**
 * @brief insert an object into the cache,
 * update the hash table and cache metadata
 * this function assumes the cache has enough space
 * eviction should be
 * performed before calling this function
 *
 * @param cache
 * @param req
 * @return the inserted object
 */
static cache_obj_t *S3FIFO_insert(cache_t *cache, const request_t *req) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  cache_obj_t *obj = NULL;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (params->hit_on_ghost || params->hit_on_main_ghost) {
    /* insert into main FIFO */
    params->hit_on_ghost = false;
    params->hit_on_main_ghost = false;
    obj = main_fifo->insert(main_fifo, req);
    obj->S3FIFO.main_insert_freq = 0;
  } else {
    /* insert into small fifo */
    // NOTE: Inserting an object whose size equals the size of small fifo is
    // NOT allowed. Doing so would completely fill the small fifo, causing all
    // objects in small fifo to be evicted. This scenario may occur
    // when using a tiny cache size.
    const int64_t small_guard = params->adaptive
                                    ? (int64_t)params->small_target
                                    : small_fifo->cache_size;
    if (small_guard > 0 && req->obj_size >= small_guard &&
        !params->adaptive) {
      return NULL;
    }

    const int64_t small_cap =
        params->adaptive ? (int64_t)params->small_target
                         : small_fifo->cache_size;
    if (!params->has_evicted &&
        small_fifo->get_occupied_byte(small_fifo) >= small_cap) {
      obj = main_fifo->insert(main_fifo, req);
    } else {
      obj = small_fifo->insert(small_fifo, req);
    }
  }

  obj->S3FIFO.freq = 0;

  return obj;
}

/**
 * @brief find the object to be evicted
 * this function does not actually evict the object or update metadata
 * not all eviction algorithms support this function
 * because the eviction logic cannot be decoupled from finding eviction
 * candidate, so use assert(false) if you cannot support this function
 *
 * @param cache the cache
 * @return the object to be evicted
 */
static cache_obj_t *S3FIFO_to_evict(cache_t *cache, const request_t *req) {
  assert(false);
  return NULL;
}

static void S3FIFO_evict_small(cache_t *cache, const request_t *req) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  cache_t *small_fifo = params->small_fifo;
  cache_t *ghost_fifo = params->ghost_fifo;
  cache_t *main_fifo = params->main_fifo;

  bool has_evicted = false;
  while (!has_evicted && small_fifo->get_occupied_byte(small_fifo) > 0) {
    cache_obj_t *obj_to_evict = small_fifo->to_evict(small_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);
    // need to copy the object before it is evicted
    copy_cache_obj_to_request(params->req_local, obj_to_evict);

    if (obj_to_evict->S3FIFO.freq >= params->move_to_main_threshold) {
      main_fifo->insert(main_fifo, params->req_local);
    } else {
      // insert to ghost
      if (ghost_fifo != NULL) {
        ghost_fifo->get(ghost_fifo, params->req_local);
        if (params->adaptive &&
            (params->adapt_mode == 3 || params->adapt_mode == 4)) {
          cache_obj_t *g =
              ghost_fifo->find(ghost_fifo, params->req_local, false);
          if (g != NULL) g->S3FIFO.insertion_time = ++params->ghost_ins_seq;
        }
      }
      has_evicted = true;
    }

    // remove from small fifo, but do not update stat
    bool removed = small_fifo->remove(small_fifo, params->req_local->obj_id);
    DEBUG_ASSERT(removed);
  }
}

static void S3FIFO_evict_main(cache_t *cache, const request_t *req) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  cache_t *main_fifo = params->main_fifo;

  bool has_evicted = false;
  while (!has_evicted && main_fifo->get_occupied_byte(main_fifo) > 0) {
    cache_obj_t *obj_to_evict = main_fifo->to_evict(main_fifo, req);
    DEBUG_ASSERT(obj_to_evict != NULL);
    int freq = obj_to_evict->S3FIFO.freq;
    copy_cache_obj_to_request(params->req_local, obj_to_evict);
    const int32_t witness = obj_to_evict->S3FIFO.main_insert_freq;
    if (freq >= 1) {
      // we need to evict first because the object to insert has the same obj_id
      main_fifo->remove(main_fifo, obj_to_evict->obj_id);
      obj_to_evict = NULL;

      cache_obj_t *new_obj = main_fifo->insert(main_fifo, params->req_local);
      // clock with 2-bit counter
      new_obj->S3FIFO.freq = MIN(freq, 3) - 1;
      new_obj->S3FIFO.main_insert_freq = witness;  /* carry the witness across */

    } else {
      bool removed = main_fifo->remove(main_fifo, obj_to_evict->obj_id);
      DEBUG_ASSERT(removed);
      if (params->main_ghost_fifo != NULL) {
        params->main_ghost_fifo->get(params->main_ghost_fifo,
                                     params->req_local);
      }
      if (params->adaptive && params->adapt_mode == 5 && witness) {
        const double ng = (double)(params->n_small_ghost_hit);
        const double ns = (double)(params->n_witness_shrink);
        double d = (ns > 0 && ng / ns > 1.0) ? ng / ns : 1.0;
        params->small_target = MAX(params->small_target - d * params->adaptive_step, 0.0);
        params->n_witness_shrink++;
      } else if (params->adaptive && params->adapt_mode == 1 && witness) {
        /* main was forced to give up a block it had served -> main is starved */
        params->small_target = MAX(params->small_target - params->adaptive_step,
                                   0.0);
        params->n_witness_shrink++;
      }

      has_evicted = true;
    }
  }
}

/**
 * @brief evict an object from the cache
 * it needs to call cache_evict_base before returning
 * which updates some metadata such as n_obj, occupied size, and hash table
 *
 * @param cache
 * @param req not used
 * @param evicted_obj if not NULL, return the evicted object to caller
 */
static void S3FIFO_evict(cache_t *cache, const request_t *req) {
  /* An eviction must free something.
   *
   * S3FIFO_evict_once below can instead PROMOTE a block out of the small queue
   * into the main queue, which leaves total occupancy unchanged. cache_get_base()
   * hides that by looping until it has room, so it never mattered there -- but a
   * caller that asks for one eviction and gets none (e.g. prefixsim's allocate
   * loop) cannot distinguish "made no progress" from "nothing left to evict", and
   * has to treat it as failure.
   *
   * So retry until the cache actually shrinks. This terminates: every iteration
   * either frees an object or moves one out of the small queue, and once the
   * small queue is empty the main branch always frees. The guard bounds it
   * anyway rather than risking a hang if that ever stops holding. Mirrors the
   * fix already in S3FIFOd_evict / QDLP / S3FIFOCompute / LIRS / CAR.
   */
  const int64_t occupied_before = cache->get_occupied_byte(cache);
  int64_t attempts = 0;
  const int64_t limit = cache->get_n_obj(cache) + 16;
  do {
    S3FIFO_evict_once(cache, req);
    if (++attempts > limit) {
      ERROR("S3FIFO_evict: %lld attempts freed nothing (occupied %lld)\n",
            (long long)attempts, (long long)cache->get_occupied_byte(cache));
    }
  } while (cache->get_occupied_byte(cache) >= occupied_before &&
           cache->get_occupied_byte(cache) > 0);
}

static void S3FIFO_evict_once(cache_t *cache, const request_t *req) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  params->has_evicted = true;

  cache_t *small_fifo = params->small_fifo;
  cache_t *main_fifo = params->main_fifo;

  if (params->adaptive) {
    /* ARC's REPLACE, with small_target playing the role of p. */
    const int64_t small_occ = small_fifo->get_occupied_byte(small_fifo);
    const int64_t main_occ = main_fifo->get_occupied_byte(main_fifo);
    const bool take_small =
        (small_occ > 0 && (double)small_occ > params->small_target) ||
        main_occ == 0;
    if (take_small) {
      S3FIFO_evict_small(cache, req);
    } else {
      S3FIFO_evict_main(cache, req);
    }
  } else if (main_fifo->get_occupied_byte(main_fifo) > main_fifo->cache_size ||
             small_fifo->get_occupied_byte(small_fifo) == 0) {
    S3FIFO_evict_main(cache, req);
  } else {
    S3FIFO_evict_small(cache, req);
  }
}

/**
 * @brief remove an object from the cache
 * this is different from cache_evict because it is used to for user trigger
 * remove, and eviction is used by the cache to make space for new objects
 *
 * it needs to call cache_remove_obj_base before returning
 * which updates some metadata such as n_obj, occupied size, and hash table
 *
 * @param cache
 * @param obj_id
 * @return true if the object is removed, false if the object is not in the
 * cache
 */
static bool S3FIFO_remove(cache_t *cache, obj_id_t obj_id) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  bool removed = false;
  removed = removed || params->small_fifo->remove(params->small_fifo, obj_id);
  removed = removed || (params->ghost_fifo &&
                        params->ghost_fifo->remove(params->ghost_fifo, obj_id));
  removed = removed || params->main_fifo->remove(params->main_fifo, obj_id);

  return removed;
}

static inline int64_t S3FIFO_get_occupied_byte(const cache_t *cache) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  return params->small_fifo->get_occupied_byte(params->small_fifo) +
         params->main_fifo->get_occupied_byte(params->main_fifo);
}

static inline int64_t S3FIFO_get_n_obj(const cache_t *cache) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;
  return params->small_fifo->get_n_obj(params->small_fifo) +
         params->main_fifo->get_n_obj(params->main_fifo);
}

static inline bool S3FIFO_can_insert(cache_t *cache, const request_t *req) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)cache->eviction_params;

  return req->obj_size <= params->small_fifo->cache_size &&
         cache_can_insert_default(cache, req);
}

// ***********************************************************************
// ****                                                               ****
// ****                parameter set up functions                     ****
// ****                                                               ****
// ***********************************************************************
static const char *S3FIFO_current_params(S3FIFO_params_t *params) {
  static __thread char params_str[128];
  snprintf(params_str, 128,
           "small-size-ratio=%.4lf,ghost-size-ratio=%.4lf,move-to-main-"
           "threshold=%d\n",
           params->small_size_ratio, params->ghost_size_ratio,
           params->move_to_main_threshold);
  return params_str;
}

static void S3FIFO_parse_params(cache_t *cache,
                                const char *cache_specific_params) {
  S3FIFO_params_t *params = (S3FIFO_params_t *)(cache->eviction_params);

  char *params_str = strdup(cache_specific_params);
  char *old_params_str = params_str;

  while (params_str != NULL && params_str[0] != '\0') {
    /* different parameters are separated by comma,
     * key and value are separated by = */
    char *key = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    // skip the white space
    while (params_str != NULL && *params_str == ' ') {
      params_str++;
    }

    if (strcasecmp(key, "fifo-size-ratio") == 0 ||
        strcasecmp(key, "small-size-ratio") == 0) {
      params->small_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "ghost-size-ratio") == 0) {
      params->ghost_size_ratio = strtod(value, NULL);
    } else if (strcasecmp(key, "move-to-main-threshold") == 0) {
      params->move_to_main_threshold = atoi(value);
    } else if (strcasecmp(key, "adaptive") == 0) {
      params->adaptive = (atoi(value) != 0);
    } else if (strcasecmp(key, "main-ghost") == 0) {
      params->main_ghost_on = (atoi(value) != 0);
    } else if (strcasecmp(key, "adapt-mode") == 0) {
      if (strcasecmp(value, "arc") == 0) params->adapt_mode = 0;
      else if (strcasecmp(value, "witness") == 0) params->adapt_mode = 1;
      else if (strcasecmp(value, "density") == 0) params->adapt_mode = 2;
      else if (strcasecmp(value, "depth") == 0) params->adapt_mode = 3;
      else if (strcasecmp(value, "split") == 0) params->adapt_mode = 4;
      else if (strcasecmp(value, "witbal") == 0) params->adapt_mode = 5;
      else { ERROR("unknown adapt-mode %s\n", value); exit(1); }
    } else if (strcasecmp(key, "adaptive-step") == 0) {
      params->adaptive_step = strtod(value, NULL);
    } else if (strcasecmp(key, "print") == 0) {
      printf("parameters: %s\n", S3FIFO_current_params(params));
      exit(0);
    } else {
      ERROR("%s does not have parameter %s\n", cache->cache_name, key);
      exit(1);
    }
  }

  free(old_params_str);
}

#ifdef __cplusplus
}
#endif
