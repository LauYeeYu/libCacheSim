#include <assert.h>

#include "dataStructure/hashtable/hashtable.h"
#include "lhd.hpp"
#include "lhd_request_compute.hpp"
#include "repl.hpp"
#include "candidate.hpp"

using namespace repl;

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: Since LHDRequest uses internal data struct to represent cache_obj_t, to suit
// the interface, we use a dummy pointer to represent the cache_obj_t.
// Specifically, for find and insert, we return a dummy pointer when the object
// is found or inserted and NULL when the object is not found.
static cache_obj_t *const DUMMY_CACHE_OBJ_PTR =
    reinterpret_cast<cache_obj_t *>(1);

typedef struct {
  void *LHDRequest_cache;

  int associativity;
  int admission;

  repl::candidate_t to_evict_candidate;
} LHDRequest_params_t;

// ***********************************************************************
// ****                                                               ****
// ****                   function declarations                       ****
// ****                                                               ****
// ***********************************************************************

static void LHDRequest_free(cache_t *cache);
static bool LHDRequest_get(cache_t *cache, const request_t *req);

static cache_obj_t *LHDRequest_find(cache_t *cache, const request_t *req,
                                   const bool update_cache);
static cache_obj_t *LHDRequest_insert(cache_t *cache, const request_t *req);
static cache_obj_t *LHDRequest_to_evict(cache_t *cache, const request_t *req);
static void LHDRequest_evict(cache_t *cache, const request_t *req);
static bool LHDRequest_remove(cache_t *cache, const obj_id_t obj_id);
static int64_t LHDRequest_get_occupied_byte(const cache_t *cache);
static int64_t LHDRequest_get_n_obj(const cache_t *cache);

// ***********************************************************************
// ****                                                               ****
// ****                   end user facing functions                   ****
// ****                                                               ****
// ****                       init, free, get                         ****
// ***********************************************************************

/**
 * @brief initialize the cache
 *
 * @param ccache_params some common cache parameters
 * @param cache_specific_params cache specific parameters, see parse_params
 * function or use -e "print" with the cachesim binary
 */
cache_t *LHDRequest_init(const common_cache_params_t ccache_params,
                   const char *cache_specific_params) {
#ifdef SUPPORT_TTL
  if (ccache_params.default_ttl < 30 * 86400) {
    ERROR("LHDRequest does not support expiration\n");
    abort();
  }
#endif

  cache_t *cache =
      cache_struct_init("LHDRequest", ccache_params, cache_specific_params);
  cache->cache_init = LHDRequest_init;
  cache->cache_free = LHDRequest_free;
  cache->get = LHDRequest_get;
  cache->find = LHDRequest_find;
  cache->insert = LHDRequest_insert;
  cache->evict = LHDRequest_evict;
  cache->to_evict = LHDRequest_to_evict;
  cache->remove = LHDRequest_remove;
  cache->can_insert = cache_can_insert_default;
  cache->get_occupied_byte = LHDRequest_get_occupied_byte;
  cache->get_n_obj = LHDRequest_get_n_obj;
  cache->to_evict_candidate =
      static_cast<cache_obj_t *>(malloc(sizeof(cache_obj_t)));

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 8 * 3 + 1;  // two age, one time stamp
  } else {
    cache->obj_md_size = 0;
  }

  auto *params = my_malloc(LHDRequest_params_t);
  memset(params, 0, sizeof(LHDRequest_params_t));
  cache->eviction_params = params;

  if (params->admission == 0) {
    params->admission = 8;
  }
  if (params->associativity == 0) {
    params->associativity = 32;
  }

  params->LHDRequest_cache = static_cast<void *>(
      new LHDRequestCompute(params->associativity, params->admission, cache));

  return cache;
}

/**
 * free resources used by this cache
 *
 * @param cache
 */
static void LHDRequest_free(cache_t *cache) {
  auto *params = static_cast<LHDRequest_params_t *>(cache->eviction_params);
  auto *lhd_request = static_cast<LHDRequestCompute *>(params->LHDRequest_cache);
  delete lhd_request;
  free(cache->to_evict_candidate);
  my_free(sizeof(LHDRequest_params_t), params);
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
static bool LHDRequest_get(cache_t *cache, const request_t *req) {
  return cache_get_base(cache, req);
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
static cache_obj_t *LHDRequest_find(cache_t *cache, const request_t *req,
                                   const bool update_cache) {
  auto *params = static_cast<LHDRequest_params_t *>(cache->eviction_params);
  auto *lhd_request = static_cast<LHDRequestCompute *>(params->LHDRequest_cache);

  auto id = repl::candidate_t::make(req);
  auto itr = lhd_request->getSizeMap().find(id);
  bool hit = (itr != lhd_request->getSizeMap().end());

  if (!hit) {
    return NULL;
  }

  if (update_cache) {
    if (itr->second != req->obj_size) {
      cache->occupied_byte -= itr->second;
      cache->occupied_byte += req->obj_size;
      itr->second = req->obj_size;
    }
    lhd_request->update(id, req);
  }

  return DUMMY_CACHE_OBJ_PTR;
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
static cache_obj_t *LHDRequest_insert(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LHDRequest_params_t *>(cache->eviction_params);
  auto *lhd_request = static_cast<LHDRequestCompute *>(params->LHDRequest_cache);
  auto id = repl::candidate_t::make(req);

#if defined(TRACK_EVICTION_V_AGE)
  id.create_time = CURR_TIME(cache, req);
#endif

  lhd_request->getSizeMap()[id] = req->obj_size;
  lhd_request->update(id, req);

  cache->occupied_byte += req->obj_size + cache->obj_md_size;
  cache->n_obj += 1;

  return DUMMY_CACHE_OBJ_PTR;
}

/**
 * @brief find an eviction candidate, but do not evict from the cache,
 * and do not update the cache metadata
 * note that eviction must evicts this object, so if we implement this function
 * and it uses random number, we must make sure that the same object is evicted
 * when we call evict
 *
 * @param cache
 * @param req
 * @return cache_obj_t*
 */
static cache_obj_t *LHDRequest_to_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LHDRequest_params_t *>(cache->eviction_params);
  auto *lhd_request = static_cast<LHDRequestCompute *>(params->LHDRequest_cache);

  cache->to_evict_candidate_gen_vtime = cache->n_req;

  repl::candidate_t victim = lhd_request->rank(req);
  auto victimItr = lhd_request->getSizeMap().find(victim);
  assert(victimItr != lhd_request->getSizeMap().end());

  params->to_evict_candidate = victim;

  cache->to_evict_candidate->obj_id = victim.id;
  cache->to_evict_candidate->obj_size = victimItr->second;

  return cache->to_evict_candidate;
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
static void LHDRequest_evict(cache_t *cache, const request_t *req) {
  auto *params = static_cast<LHDRequest_params_t *>(cache->eviction_params);
  auto *lhd_request = static_cast<LHDRequestCompute *>(params->LHDRequest_cache);

  repl::candidate_t victim;
  if (cache->to_evict_candidate_gen_vtime == cache->n_req) {
    victim = params->to_evict_candidate;
    cache->to_evict_candidate_gen_vtime = -1;
  } else {
    victim = lhd_request->rank(req);
  }

  auto victimItr = lhd_request->getSizeMap().find(victim);
  if (victimItr == lhd_request->getSizeMap().end()) {
    std::cerr << "Couldn't find victim: " << victim << std::endl;
  }
  assert(victimItr != lhd_request->getSizeMap().end());

  DEBUG_ASSERT(cache->occupied_byte >= victimItr->second);
  cache->occupied_byte -= (victimItr->second + cache->obj_md_size);
  cache->n_obj -= 1;

#if defined(TRACK_EVICTION_V_AGE)
  {
    cache_obj_t obj;
    obj.obj_id = victim.id;
    obj.create_time = victim.create_time;
    record_eviction_age(cache, &obj,
                        CURR_TIME(cache, req) - victim.create_time);
  }
#endif

  lhd_request->replaced(victim);
  lhd_request->getSizeMap().erase(victimItr);
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
static bool LHDRequest_remove(cache_t *cache, const obj_id_t obj_id) {
  auto *params = static_cast<LHDRequest_params_t *>(cache->eviction_params);
  auto *lhd_request = static_cast<LHDRequestCompute *>(params->LHDRequest_cache);
  repl::candidate_t id{DEFAULT_APP_ID, (int64_t)obj_id};

  auto itr = lhd_request->getSizeMap().find(id);
  if (itr == lhd_request->getSizeMap().end()) {
    return false;
  }

  cache->occupied_byte -= (itr->second + cache->obj_md_size);
  cache->n_obj -= 1;
  lhd_request->getSizeMap().erase(itr);
  auto idx = lhd_request->getIndices()[id];
  lhd_request->getIndices().erase(id);
  lhd_request->getTags()[idx] = lhd_request->getTags().back();
  lhd_request->getTags().pop_back();

  if (idx < lhd_request->getTags().size()) {
    lhd_request->getIndices()[lhd_request->getTags()[idx].id] = idx;
  }

  return true;
}

static int64_t LHDRequest_get_occupied_byte(const cache_t *cache) {
  return cache->occupied_byte;
}

static int64_t LHDRequest_get_n_obj(const cache_t *cache) { return cache->n_obj; }

#ifdef __cplusplus
}
#endif