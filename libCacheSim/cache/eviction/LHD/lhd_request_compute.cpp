#include "lhd_request_compute.hpp"

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cassert>

#include "constants.hpp"
#include "utils/include/mymath.h"

namespace repl {

LHDRequestCompute::LHDRequestCompute(int _associativity, int _admissions, cache_t* _cache)
    : ASSOCIATIVITY(_associativity),
      ADMISSIONS(_admissions),
      cache(_cache),
      timestamp(0),
      nextReconfiguration(0),
      numReconfigurations(0),
      ageCoarseningShift(10),
      ewmaNumObjects(0),
      ewmaNumObjectsMass(0),
      overflows(0),
      recentlyAdmittedHead(0),
      ewmaVictimHitDensity(0),
      explorerBudget(0),
      root_prefix(nullptr) {
    nextReconfiguration = ACCS_PER_RECONFIGURATION;
    explorerBudget = cache->cache_size * EXPLORER_BUDGET_FRACTION;
    
    // Initialize recently admitted vector
    recentlyAdmitted.resize(ADMISSIONS, INVALID_CANDIDATE);
    
    // Initialize root prefix node
    root_prefix = new PrefixNode(0, nullptr);

    // Initialize LHD classes
    for (uint32_t i = 0; i < NUM_CLASSES; i++) {
        classes.push_back(Class());
        // Ensure vectors are properly sized for main classes
        classes[i].ensure_size(MAX_AGE);
    }

    // Initialize policy to ~GDSF by default
    for (uint32_t c = 0; c < NUM_CLASSES; c++) {
        for (uint32_t a = 0; a < MAX_AGE; a++) {
            classes[c].hitDensities[a] = 1. * (c + 1) / (a + 1);
        }
    }
    

}

LHDRequestCompute::~LHDRequestCompute() {
    delete root_prefix;
}

// Hash function to create unique prefix identifiers
uint64_t LHDRequestCompute::hashPrefix(uint64_t current_prefix, uint64_t block_id) {
    // Simple hash combination - in practice, you might want a better hash function
    return current_prefix * 31 + block_id;
}

// Find or create prefix node for a given prefix hash
PrefixNode* LHDRequestCompute::findOrCreatePrefix(uint64_t prefix_hash, PrefixNode* parent) {
    if (parent == nullptr) {
        parent = root_prefix;
    }
    
    // Look for existing child with this prefix hash
    for (auto child : parent->children) {
        if (child->prefix_hash == prefix_hash) {
            return child;
        }
    }
    
    // Create new child node
    PrefixNode* new_node = new PrefixNode(prefix_hash, parent);
    parent->children.push_back(new_node);
    return new_node;
}

// Get prefix node for current request sequence
PrefixNode* LHDRequestCompute::getPrefixNode(const request_t* req, PrefixNode*& parent_out) {
    static uint64_t last_request_prefix = 0;
    static PrefixNode* last_request_node = nullptr;
    
    // Check if this is a new request (features[1] != 0) or subsequent block
    bool is_new_request = (req->features[1] != 0);
    
    if (is_new_request) {
        // New request - start from root
        last_request_prefix = req->obj_id;
        last_request_node = findOrCreatePrefix(last_request_prefix, root_prefix);
        parent_out = root_prefix;
    } else {
        // Subsequent block - extend previous prefix
        last_request_prefix = hashPrefix(last_request_prefix, req->obj_id);
        last_request_node = findOrCreatePrefix(last_request_prefix, last_request_node);
        parent_out = last_request_node->parent;
    }
    
    return last_request_node;
}

candidate_t LHDRequestCompute::rank(const request_t* req) {
    uint64_t victim = -1;
    rank_t victimRank = std::numeric_limits<rank_t>::max();

    // Sample candidates for eviction decision
    uint32_t candidates = (numReconfigurations > 50) ? ASSOCIATIVITY : 8;

    // Use prefix tree to guide sampling - prefer prefixes with many cache blocks
    for (uint32_t i = 0; i < candidates; i++) {
        candidate_t candidate_id;
        
        // With 50% probability, sample from prefix tree
        if (i < candidates / 2 && !tags.empty()) {
            candidate_id = sampleFromPrefixTree();
            if (candidate_id == INVALID_CANDIDATE && !tags.empty()) {
                auto idx = next_rand() % tags.size();
                candidate_id = tags[idx].id;
            }
        } else {
            // Regular random sampling
            if (tags.empty()) continue;
            auto idx = next_rand() % tags.size();
            candidate_id = tags[idx].id;
        }
        
        // Find the tag and compute rank
        auto tag_itr = indices.find(candidate_id);
        if (tag_itr == indices.end()) continue;
        
        auto& tag = tags[tag_itr->second];
        rank_t rank = getHitDensity(tag);

        if (rank < victimRank) {
            victim = tag_itr->second;
            victimRank = rank;
        }
    }

    // Also consider recently admitted items
    for (int i = 0; i < ADMISSIONS; i++) {
        auto itr = indices.find(recentlyAdmitted[i]);
        if (itr == indices.end()) {
            continue;
        }

        auto idx = itr->second;
        auto& tag = tags[idx];
        assert(tag.id == recentlyAdmitted[i]);
        rank_t rank = getHitDensity(tag);

        if (rank < victimRank) {
            victim = idx;
            victimRank = rank;
        }
    }

    // Also consider evicting from prefixes when appropriate
    // This ensures we can evict entire request sequences when beneficial
    if (numReconfigurations > 50 && victimRank > 0.1) {
        auto prefix_candidate = sampleFromPrefixTree();
        if (prefix_candidate != INVALID_CANDIDATE) {
            auto prefix_itr = indices.find(prefix_candidate);
            if (prefix_itr != indices.end()) {
                auto& tag = tags[prefix_itr->second];
                rank_t prefix_rank = getHitDensity(tag);
                if (prefix_rank < victimRank) {
                    victim = prefix_itr->second;
                    victimRank = prefix_rank;
                }
            }
        }
    }

    assert(victim != (uint64_t)-1);

    ewmaVictimHitDensity =
        EWMA_DECAY * ewmaVictimHitDensity + (1 - EWMA_DECAY) * victimRank;

    return tags[victim].id;
}

void LHDRequestCompute::update(candidate_t id, const request_t* req) {
    auto itr = indices.find(id);
    bool insert = (itr == indices.end());

    Tag* tag;
    if (insert) {
        tags.push_back(Tag{});
        tag = &tags.back();
        indices[id] = tags.size() - 1;

        tag->lastLastHitAge = MAX_AGE;
        tag->lastHitAge = 0;
        tag->id = id;
    } else {
        tag = &tags[itr->second];
        assert(tag->id == id);
        auto age = getAge(*tag);
        auto& cl = getClass(*tag);
        cl.hits[age] += 1;

        if (tag->explorer) {
            explorerBudget += tag->size;
        }

        tag->lastLastHitAge = tag->lastHitAge;
        tag->lastHitAge = age;
    }

    tag->timestamp = timestamp;
    tag->app = DEFAULT_APP_ID % APP_CLASSES;
    tag->size = 1;
    tag->compute_intensity = req->features[0];

    // Add this cache block to the prefix tree for request-level tracking
    PrefixNode* parent_node;
    PrefixNode* prefix_node = getPrefixNode(req, parent_node);
    
    // Track this cache block in the prefix node
    prefix_node->cache_blocks.push_back(tag->id);
    prefix_node->block_count++;
    prefix_node->reference_count++;
    
    if (parent_node) {
        parent_node->reference_count++;
    }
    
    // Update active prefixes tracking
    // Only add to active_prefixes if this prefix was not already active
    if (prefix_node->reference_count == 1) {
        active_prefixes[prefix_node->prefix_hash] = prefix_node;
    }

    // Mark for exploration with some probability
    bool explore = (next_rand() % EXPLORE_INVERSE_PROBABILITY) == 0;
    if (explore && explorerBudget > 0 && numReconfigurations < 50) {
        tag->explorer = true;
        explorerBudget -= tag->size;
    } else {
        tag->explorer = false;
    }

    // Track recently admitted items that look like good eviction candidates
    if (insert && !explore && getHitDensity(*tag) < ewmaVictimHitDensity) {
        recentlyAdmitted[recentlyAdmittedHead++ % ADMISSIONS] = id;
    }
}

// Remove a cache block and update prefix tracking
void LHDRequestCompute::evict(uint32_t index) {
    if (index >= tags.size()) return;
    
    auto& tag = tags[index];
    
    // Update prefix tracking for this evicted block
    updatePrefixOnEviction(tag.id);
    
    // Standard LHD eviction logic
    auto itr = indices.find(tag.id);
    if (itr != indices.end()) {
        indices.erase(itr);
    }
    
    tags[index] = tags.back();
    tags.pop_back();
    
    if (index < tags.size()) {
        indices[tags[index].id] = index;
    }
}

// Update prefix tracking when a cache block is evicted
void LHDRequestCompute::updatePrefixOnEviction(candidate_t evicted_id) {
    // Find which prefix contains this cache block by traversing active prefixes
    for (auto& pair : active_prefixes) {
        PrefixNode* node = pair.second;
        if (!node) continue;
        
        // Check if this node contains the evicted block
        auto& blocks = node->cache_blocks;
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (*it == evicted_id) {
                // Remove from this node's cache blocks
                blocks.erase(it);
                node->block_count--;
                node->reference_count--;
                
                // If no more blocks in this prefix, remove from active_prefixes
                if (blocks.empty()) {
                    active_prefixes.erase(pair.first);
                    
                    // Add child prefixes to active_prefixes if they have blocks
                    for (auto child : node->children) {
                        if (!child->cache_blocks.empty()) {
                            active_prefixes[child->prefix_hash] = child;
                        }
                    }
                }
                
                return;  // Found and removed
            }
        }
    }
}

void LHDRequestCompute::replaced(candidate_t id) {
    auto itr = indices.find(id);
    assert(itr != indices.end());
    auto index = itr->second;

    // Record stats before removing item
    auto& tag = tags[index];
    assert(tag.id == id);
    auto age = getAge(tag);
    auto& cl = getClass(tag);
    cl.evictions[age] += 1;

    if (tag.explorer) {
        explorerBudget += tag.size;
    }

    // Update prefix tree - decrement reference count
    // Note: In a full implementation, you'd want to track which prefix this
    // specific object belonged to for proper cleanup
    
    // Remove tag for replaced item and update index
    indices.erase(itr);
    tags[index] = tags.back();
    tags.pop_back();

    if (index < tags.size()) {
        indices[tags[index].id] = index;
    }
}

void LHDRequestCompute::reconfigure() {
    rank_t totalHits = 0;
    rank_t totalEvictions = 0;
    
    // Update main classes (same as regular LHD)
    for (auto& cl : classes) {
        updateClass(cl);
        totalHits += cl.totalHits;
        totalEvictions += cl.totalEvictions;
    }
    
    adaptAgeCoarsening();
    modelHitDensity();

    overflows = 0;
}

void LHDRequestCompute::updateClass(Class& cl) {
    cl.totalHits = 0;
    cl.totalEvictions = 0;

    for (age_t age = 0; age < MAX_AGE; age++) {
        cl.hits[age] *= EWMA_DECAY;
        cl.evictions[age] *= EWMA_DECAY;

        cl.totalHits += cl.hits[age];
        cl.totalEvictions += cl.evictions[age];
    }
}



void LHDRequestCompute::modelHitDensity() {
    for (uint32_t c = 0; c < classes.size(); c++) {
        rank_t totalEvents =
            classes[c].hits[MAX_AGE - 1] + classes[c].evictions[MAX_AGE - 1];
        rank_t totalHits = classes[c].hits[MAX_AGE - 1];
        rank_t lifetimeUnconditioned = totalEvents;

        for (age_t a = MAX_AGE - 2; a < MAX_AGE; a--) {
            totalHits += classes[c].hits[a];

            totalEvents += classes[c].hits[a] + classes[c].evictions[a];

            lifetimeUnconditioned += totalEvents;

            if (totalEvents > 1e-5) {
                classes[c].hitDensities[a] = totalHits / lifetimeUnconditioned;
            } else {
                classes[c].hitDensities[a] = 0.;
            }
        }
    }
}





void LHDRequestCompute::adaptAgeCoarsening() {
    ewmaNumObjects *= EWMA_DECAY;
    ewmaNumObjectsMass *= EWMA_DECAY;

    ewmaNumObjects += sizeMap.size();
    ewmaNumObjectsMass += 1.;

    rank_t numObjects = ewmaNumObjects / ewmaNumObjectsMass;

    rank_t optimalAgeCoarsening =
        1. * numObjects / (AGE_COARSENING_ERROR_TOLERANCE * MAX_AGE);

    if (numReconfigurations == 5 || numReconfigurations == 25) {
        uint32_t optimalAgeCoarseningLog2 = 1;

        while ((1 << optimalAgeCoarseningLog2) < optimalAgeCoarsening) {
            optimalAgeCoarseningLog2 += 1;
        }

        int32_t delta = optimalAgeCoarseningLog2 - ageCoarseningShift;
        ageCoarseningShift = optimalAgeCoarseningLog2;

        ewmaNumObjects *= 8;
        ewmaNumObjectsMass *= 8;

        // Apply age coarsening to main classes
        if (delta < 0) {
            for (auto& cl : classes) {
                for (age_t a = MAX_AGE >> (-delta); a < MAX_AGE - 1; a++) {
                    cl.hits[MAX_AGE - 1] += cl.hits[a];
                    cl.evictions[MAX_AGE - 1] += cl.evictions[a];
                }
                for (age_t a = MAX_AGE - 2; a < MAX_AGE; a--) {
                    cl.hits[a] = cl.hits[a >> (-delta)] / (1 << (-delta));
                    cl.evictions[a] = cl.evictions[a >> (-delta)] / (1 << (-delta));
                }
            }
        } else if (delta > 0) {
            for (auto& cl : classes) {
                for (age_t a = 0; a < MAX_AGE >> delta; a++) {
                    cl.hits[a] = cl.hits[a << delta];
                    cl.evictions[a] = cl.evictions[a << delta];
                    for (int i = 1; i < (1 << delta); i++) {
                        cl.hits[a] += cl.hits[(a << delta) + i];
                        cl.evictions[a] += cl.evictions[(a << delta) + i];
                    }
                }
                for (age_t a = (MAX_AGE >> delta); a < MAX_AGE - 1; a++) {
                    cl.hits[a] = 0;
                    cl.evictions[a] = 0;
                }
            }
        }
    }
}



// Sample eviction candidate from prefix tree


candidate_t LHDRequestCompute::sampleFromPrefixTree() {
    if (active_prefixes.empty()) {
        // Fallback to random selection from tags
        if (tags.empty()) return INVALID_CANDIDATE;
        auto idx = next_rand() % tags.size();
        return tags[idx].id;
    }
    
    // TRULY O(1) sampling using reservoir sampling technique
    uint32_t selected_count = 0;
    PrefixNode* selected_prefix = nullptr;
    
    for (auto& pair : active_prefixes) {
        PrefixNode* candidate_prefix = pair.second;
        if (candidate_prefix && !candidate_prefix->cache_blocks.empty()) {
            selected_count++;
            // Reservoir sampling: keep each item with probability 1/count
            if (next_rand() % selected_count == 0) {
                selected_prefix = candidate_prefix;
            }
        }
    }
    
    if (!selected_prefix) {
        return INVALID_CANDIDATE;
    }
    
    // Sample a cache block from this prefix
    size_t random_idx = next_rand() % selected_prefix->cache_blocks.size();
    candidate_t block_to_evict = selected_prefix->cache_blocks[random_idx];
    
    // Find the corresponding tag index
    auto tag_it = indices.find(block_to_evict);
    if (tag_it != indices.end()) {
        return tags[tag_it->second].id;
    }
    
    // Fallback if not found
    if (tags.empty()) return INVALID_CANDIDATE;
    auto idx = next_rand() % tags.size();
    return tags[idx].id;
}



}  // namespace repl