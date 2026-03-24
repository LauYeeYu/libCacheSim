#ifndef LHD_REQUEST_COMPUTE_HPP
#define LHD_REQUEST_COMPUTE_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>

#include "libCacheSim/cache.h"
#include "candidate.hpp"
#include "constants.hpp"

namespace repl {

// Forward declaration
struct PrefixNode;

// Type definitions (matching original LHDCompute)
typedef uint64_t timestamp_t;
typedef uint64_t age_t;
typedef float rank_t;

// Tag structure for LHD tracking
struct Tag {
    double compute_intensity;
    timestamp_t timestamp;
    age_t lastHitAge;
    age_t lastLastHitAge;
    uint32_t app;
    
    candidate_t id;
    rank_t size;  // stored redundantly with cache
    bool explorer;
    
    Tag() : compute_intensity(1.0), timestamp(0), lastHitAge(0), lastLastHitAge(0),
            app(0), id(), size(1), explorer(false) {}
};

// Class structure for hit density modeling
struct Class {
    std::vector<rank_t> hits;
    std::vector<rank_t> evictions;
    std::vector<rank_t> hitDensities;
    rank_t totalHits = 0;
    rank_t totalEvictions = 0;
    
    Class() {
        // Vectors will be resized lazily when needed
    }
    
    void ensure_size(size_t size) {
        if (hits.size() < size) {
            hits.resize(size, 0);
            evictions.resize(size, 0);
            hitDensities.resize(size, 0);
        }
    }
};

// Prefix Tree Node for tracking LLM request sequences
struct PrefixNode {
    uint64_t prefix_hash;  // Hash of the prefix sequence
    std::vector<PrefixNode*> children;  // Child nodes for next blocks
    PrefixNode* parent;  // Parent node
    std::vector<candidate_t> cache_blocks;  // Cache block IDs belonging to this prefix
    uint64_t block_count;  // Number of blocks in this prefix sequence
    uint64_t reference_count;  // Number of cache objects referencing this prefix
    
    PrefixNode(uint64_t hash = 0, PrefixNode* p = nullptr) 
        : prefix_hash(hash), parent(p), block_count(0), reference_count(0) {}
    
    ~PrefixNode() {
        for (auto child : children) {
            delete child;
        }
    }
};

class LHDRequestCompute {
public:
    LHDRequestCompute(int _associativity, int _admissions, cache_t* _cache);
    ~LHDRequestCompute();

    // Main LHD interface functions
    candidate_t rank(const request_t* req);
    void update(candidate_t id, const request_t* req);
    void replaced(candidate_t id);

    // Public accessors for interface functions
    auto& getTags() { return tags; }
    auto& getIndices() { return indices; }
    auto& getSizeMap() { return sizeMap; }

private:
    // Core LHD components
    const int ASSOCIATIVITY;
    const int ADMISSIONS;
    cache_t* cache;
    
    // LHD state
    std::vector<Tag> tags;
    std::unordered_map<candidate_t, uint32_t> indices;
    std::unordered_map<candidate_t, uint32_t> sizeMap;
    
    // LHD configuration and statistics
    static constexpr timestamp_t ACCS_PER_RECONFIGURATION = (1 << 20);
    static constexpr rank_t EWMA_DECAY = 0.9;
    static constexpr age_t MAX_AGE = 20000;
    static constexpr uint32_t HIT_AGE_CLASSES = 16;
    static constexpr uint32_t APP_CLASSES = 16;
    static constexpr uint32_t NUM_CLASSES = HIT_AGE_CLASSES * APP_CLASSES;
    static constexpr rank_t AGE_COARSENING_ERROR_TOLERANCE = 0.01;
    static constexpr uint32_t DEFAULT_APP_ID = ::DEFAULT_APP_ID;
    static constexpr rank_t EXPLORER_BUDGET_FRACTION = 0.01;
    static constexpr uint32_t EXPLORE_INVERSE_PROBABILITY = 32;
    static constexpr bool DUMP_RANKS = false;
    
    // LHD state variables
    timestamp_t timestamp = 0;
    timestamp_t nextReconfiguration = 0;
    int numReconfigurations = 0;
    timestamp_t ageCoarseningShift = 10;
    rank_t ewmaNumObjects = 0;
    rank_t ewmaNumObjectsMass = 0;
    uint64_t overflows = 0;
    int recentlyAdmittedHead = 0;
    rank_t ewmaVictimHitDensity = 0;
    int64_t explorerBudget = 0;
    std::vector<candidate_t> recentlyAdmitted;
    
    // LHD classes for hit density modeling
    std::vector<Class> classes;
    
    // Request-level prefix tree structure
    PrefixNode* root_prefix;
    std::unordered_map<uint64_t, PrefixNode*> active_prefixes;  // Direct hash to prefix node mapping
    
    // Helper functions for LHD computation
    inline rank_t getHitDensity(const Tag& tag) {
        // Apply compute intensity transform
        double transformed_intensity = cache->compute_intensity_transform != NULL
            ? cache->compute_intensity_transform(tag.compute_intensity)
            : tag.compute_intensity;
        return getClass(tag).hitDensities[getAge(tag)] * (rank_t)transformed_intensity;
    }
    
    inline Class& getClass(const Tag& tag) {
        return classes[getClassId(tag)];
    }
    
    inline uint32_t getClassId(const Tag& tag) const {
        uint32_t hitAgeId = hitAgeClass(tag.lastHitAge + tag.lastLastHitAge);
        return tag.app * HIT_AGE_CLASSES + hitAgeId;
    }
    
    inline uint32_t hitAgeClass(age_t age) const {
        if (age == 0) {
            return HIT_AGE_CLASSES - 1;
        }
        uint32_t log = 0;
        while (age < MAX_AGE && log < HIT_AGE_CLASSES - 1) {
            age <<= 1;
            log += 1;
        }
        return log;
    }
    
    inline age_t getAge(Tag tag) {
        timestamp_t age = (timestamp - (timestamp_t)tag.timestamp) >> ageCoarseningShift;
        
        if (age >= MAX_AGE) {
            ++overflows;
            return MAX_AGE - 1;
        } else {
            return (age_t)age;
        }
    }
    
    // Random number generation for sampling
    inline uint64_t next_rand() {
        static uint64_t state = 123456789;
        state = state * 1103515245 + 12345;
        return state >> 16;
    }
    
    // Prefix tree management functions
    uint64_t hashPrefix(uint64_t current_prefix, uint64_t block_id);
    PrefixNode* findOrCreatePrefix(uint64_t prefix_hash, PrefixNode* parent = nullptr);
    PrefixNode* getPrefixNode(const request_t* req, PrefixNode*& parent_out);
    
    // LHD reconfiguration and modeling
    void reconfigure();
    void updateClass(Class& cl);
    void modelHitDensity();
    void adaptAgeCoarsening();
    void dumpClassRanks(Class& cl);
    
    // Prefix tree specific functions
    candidate_t sampleFromPrefixTree();
    void evict(uint32_t index);
    void updatePrefixOnEviction(candidate_t evicted_id);
};

} // namespace repl

#endif // LHD_REQUEST_COMPUTE_HPP