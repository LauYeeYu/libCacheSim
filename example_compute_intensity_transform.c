/**
 * Example: Custom Compute Intensity Transform Function
 *
 * This example demonstrates how to use a custom compute intensity
 * transformation function with compute-aware caching algorithms.
 *
 * Supported algorithms:
 * - GDSF_compute: Uses transform in priority calculation (sqrt by default)
 * - BeladyCompute: Uses transform in eviction scoring - reuse_distance / transform(compute_intensity) (identity by default)
 * - S3FIFOCompute: Uses transform in main queue eviction decisions (identity by default)
 * - LHD_compute: Uses transform in hit density calculation (identity by default)
 * - LHDRequest: Uses transform in hit density calculation (identity by default)
 *
 * The transform function allows you to experiment with different
 * formulas without modifying or recompiling libCacheSim source code.
 */

#include "libCacheSim/include/libCacheSim/cache.h"

// Example 1: Square root transform (this is the default)
double sqrt_transform(double raw_intensity) {
  return sqrt(raw_intensity);
}

// Example 2: Logarithmic scaling
double log_transform(double raw_intensity) {
  return log(1.0 + raw_intensity);
}

// Example 3: Normalized transform (prevents very large values)
double normalized_transform(double raw_intensity) {
  return raw_intensity / (1000.0 + raw_intensity);
}

// Example 4: Linear with threshold
double threshold_transform(double raw_intensity) {
  if (raw_intensity < 100.0) {
    return raw_intensity * 0.1;
  } else {
    return 10.0 + (raw_intensity - 100.0) * 0.01;
  }
}

// Example 5: Power transform
double power_transform(double raw_intensity) {
  return pow(raw_intensity, 0.3);  // Less aggressive than sqrt (0.5)
}

// Example 6: Identity (no transform)
double identity_transform(double raw_intensity) {
  return raw_intensity;
}

/**
 * Usage example:
 */
void usage_example() {
  // Initialize cache with default parameters
  common_cache_params_t params = default_common_cache_params();
  params.cache_size = 1024 * 1024 * 1024;  // 1 GB

  // Works with any compute-aware algorithm
  cache_t *cache = GDSF_compute_init(params, NULL);
  // OR: cache_t *cache = BeladyCompute_init(params, NULL);
  // OR: cache_t *cache = S3FIFOCompute_init(params, NULL);
  // OR: cache_t *cache = LHD_compute_init(params, NULL);
  // OR: cache_t *cache = LHDRequest_init(params, NULL);

  // Option 1: Use the default transform (already set based on algorithm)
  // GDSF_compute defaults to sqrt, others default to identity
  // No action needed

  // Option 2: Override with a custom transform
  cache->compute_intensity_transform = normalized_transform;

  // Option 3: Use logarithmic scaling
  cache->compute_intensity_transform = log_transform;

  // Option 4: No transformation (use raw values)
  cache->compute_intensity_transform = identity_transform;

  // Now use the cache normally with cache->get(cache, req)
  // The transform will be applied automatically

  // Clean up
  cache->cache_free(cache);
}

/**
 * Advanced: Dynamic transform selection based on workload
 */
double adaptive_transform(double raw_intensity) {
  static int call_count = 0;
  static double avg_intensity = 0.0;

  // Update running average (simple moving average)
  avg_intensity = (avg_intensity * call_count + raw_intensity) / (call_count + 1);
  call_count++;

  // Normalize based on observed average
  if (avg_intensity > 0) {
    return raw_intensity / avg_intensity;
  }
  return raw_intensity;
}

/**
 * Notes:
 *
 * 1. The transform function is called during cache operations, so keep it
 *    lightweight for performance.
 *
 * 2. The function receives the raw compute_intensity value from req->features[0]
 *    and returns the transformed value used in eviction/priority decisions.
 *
 * 3. Algorithm-specific formulas:
 *    - GDSF_compute: priority = pri_last_evict + freq * transform(compute_intensity)
 *    - BeladyCompute: score = reuse_distance / transform(compute_intensity)
 *    - S3FIFOCompute: ratio = transform(mean_intensity) / transform(obj_intensity)
 *    - LHD_compute/LHDRequest: hit_density = class_hit_density * transform(compute_intensity)
 *
 * 4. You can change the transform function at runtime without restarting
 *    the program or recompiling libCacheSim.
 *
 * 5. Default transforms vary by algorithm:
 *    - GDSF_compute: sqrt() - balances between linear and logarithmic
 *    - Others: identity() - no transformation
 */
