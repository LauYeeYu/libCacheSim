# LHDRequest: LLM Request-Level Compute with Prefix Tree

## Overview

LHDRequest is a new version of LHD Compute that operates on LLM request level instead of block level. It maintains a prefix tree structure to track request sequences and uses this information for intelligent eviction decisions.

## Key Features

### Request-Level Tracking
- Distinguishes between new requests (`req->features[1] != 0`) and subsequent blocks (`req->features[1] == 0`)
- For new requests: starts a new sequence from the root
- For subsequent blocks: extends the previous prefix sequence

### Prefix Tree Structure
- **PrefixNode**: Each node represents a prefix of a request sequence
- **Hash-based Identification**: Uses hash function to create unique prefix identifiers
- **Hierarchical Organization**: Tree structure where each level represents a block in the sequence
- **Reference Counting**: Tracks how many cache objects reference each prefix
- **Automatic Cleanup**: Removes prefix nodes when no references remain and no children exist

### Intelligent Eviction
- **LHD-based Sampling**: Uses the same sampling methodology as LHD/LHD Compute
- **Prefix-Aware**: Considers the request-level context when making eviction decisions
- **Adaptive**: Maintains hit density models for both individual objects and prefix sequences

## Architecture

```
Root Prefix
├── Request A Block 1
│   ├── Request A Block 2
│   └── Request A Block 3
├── Request B Block 1
│   └── Request B Block 2
└── Request C Block 1
```

## Implementation Details

### Core Components

1. **PrefixNode Structure**
   ```cpp
   struct PrefixNode {
       uint64_t prefix_hash;           // Hash of the prefix sequence
       std::vector<PrefixNode*> children;  // Child nodes
       PrefixNode* parent;             // Parent node
       uint64_t block_count;          // Number of blocks in sequence
       uint64_t reference_count;      // Cache object references
       Class lhd_class;               // LHD statistics for this prefix
   };
   ```

2. **Request Processing Logic**
   - `getPrefixNode()`: Determines the appropriate prefix node for each request
   - `hashPrefix()`: Creates unique identifiers for prefix sequences
   - Automatic prefix creation and maintenance

3. **LHD Integration**
   - Maintains all LHD statistics and hit density modeling
   - Uses LHD's sampling methodology for eviction candidates
   - Supports exploration and admission mechanisms

### Memory Management

- **Automatic Cleanup**: Prefix nodes are removed when:
  - Reference count reaches zero (no cache objects reference this prefix)
  - No child nodes exist (no suffixes for this prefix)
- **Space Efficiency**: Prevents excessive memory usage by removing unused prefixes

## Usage

### Initialization
```cpp
cache_t* cache = LHDRequest_init(cc_params, NULL);
```

### Integration with Evaluation Framework
The algorithm is available as "LHDRequest" in the evaluation framework:
```cpp
// In evaluate.cpp algorithms array
"LHDRequest"
```

### Command Line Usage
```bash
./evaluate --trace-file <trace> --cache-size <size>
```

## Algorithm Parameters

LHDRequest inherits all LHD parameters:
- **Associativity**: 32 (default)
- **Admissions**: 8 (default)
- **Explorer Budget Fraction**: 1%
- **Age Coarsening**: Adaptive based on cache size
- **Reconfiguration Interval**: 2^20 operations

## Differences from Block-Level LHD

| Aspect | Block-Level LHD | LHDRequest |
|--------|------------------|------------|
| **Granularity** | Individual blocks | LLM request sequences |
| **Context Awareness** | No | Yes (prefix tracking) |
| **Sequence Modeling** | No | Yes (prefix tree) |
| **Memory Overhead** | Lower | Higher (prefix tree) |
| **Eviction Intelligence** | Basic LHD | LHD + request context |

## Performance Considerations

### Advantages
- **Better LLM Cache Performance**: Understands request-level patterns
- **Reduced Redundancy**: Avoids evicting prefixes with active suffixes
- **Adaptive Learning**: Learns request sequence patterns over time

### Trade-offs
- **Memory Overhead**: Additional memory for prefix tree structure
- **Computational Complexity**: More complex prefix management
- **Initialization Time**: Tree structure setup for new sequences

## Testing and Validation

The implementation includes:
1. **Unit Tests**: Core prefix tree operations
2. **Integration Tests**: Cache interface compliance
3. **Performance Tests**: Comparison with other LHD variants
4. **Memory Tests**: Prefix cleanup validation

## Future Enhancements

Potential improvements:
1. **Dynamic Prefix Pruning**: More sophisticated cleanup strategies
2. **Sequence Prediction**: Use prefix information for future access prediction
3. **Adaptive Tree Structure**: Different tree layouts based on access patterns
4. **Hybrid Approaches**: Combine with other eviction algorithms

## Files

- `lhd_request_compute.hpp`: Header file with class definitions
- `lhd_request_compute.cpp`: Implementation
- `LHDRequest_Interface.cpp`: C interface wrapper
- Integration in CMakeLists.txt and evaluation framework

This implementation provides a foundation for request-aware caching in LLM systems while maintaining compatibility with the existing libCacheSim framework.