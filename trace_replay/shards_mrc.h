#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// Self-contained Fixed-Rate SHARDS implementation for online MRC generation.
//
// ProcessAccess() is called on every block cache access. It uses hash-based
// spatial sampling to maintain a stack distance histogram incrementally.
// The histogram can be dumped to a binary file at any time, compatible with
// the offline online_mrc plotting scripts (plot_mrc.py,
// plot_shards_vs_groundtruth.py).
//
// Not thread-safe. Callers must hold an external lock.
class ShardsMRC {
 public:
  // sampling_ratio: fraction of unique blocks to track (e.g. 0.01 = 1%)
  // num_bins:       number of histogram bins
  // bin_size:       stack distance units per bin (in blocks)
  ShardsMRC(double sampling_ratio = 0.01, uint64_t num_bins = 10000,
            uint64_t bin_size = 10);
  ~ShardsMRC() = default;

  ShardsMRC(const ShardsMRC&) = delete;
  ShardsMRC& operator=(const ShardsMRC&) = delete;

  // Process one block cache access. block_key is the unique block identifier.
  void ProcessAccess(const Slice& block_key);

  // Write MRC binary file in online_mrc format:
  //   header:  (num_bins: uint64, bin_size: uint64)
  //   entries: (index: uint64, miss_rate: float64) x num_bins
  // Compatible with plot_shards_vs_groundtruth.py. Returns false on error.
  bool DumpMRC(const std::string& path) const;

  // Returns a JSON string of the raw histogram for logging.
  std::string GetHistogramJSON() const;

  // Apply exponential decay to the histogram: multiply every bin by
  // decay_factor. Call after each periodic snapshot to down-weight old
  // observations so the MRC adapts to distribution shifts.
  void Decay(double decay_factor);

  uint64_t num_entries_seen() const { return num_entries_seen_; }
  uint64_t num_entries_processed() const { return num_entries_processed_; }

 private:
  // Order-statistics tree keyed by timestamp.
  // order_of_key(t) returns the count of elements strictly less than t in
  // O(log n), which lets us compute stack distance (rank from the right)
  // without a full scan.
  using OrderedSet =
      __gnu_pbds::tree<uint64_t, __gnu_pbds::null_type, std::less<uint64_t>,
                       __gnu_pbds::rb_tree_tag,
                       __gnu_pbds::tree_order_statistics_node_update>;

  double sampling_ratio_;
  uint64_t threshold_;  // sample if GetSliceNPHash64(key) <= threshold
  double scale_;        // 1.0 / sampling_ratio — each sampled hit counts as scale_ accesses

  uint64_t num_bins_;
  uint64_t bin_size_;

  uint64_t current_timestamp_{0};
  uint64_t num_entries_seen_{0};
  uint64_t num_entries_processed_{0};

  // Maps block_key -> last-seen timestamp (sampled blocks only).
  std::unordered_map<std::string, uint64_t> last_seen_;

  // Ordered set of last-seen timestamps for O(log n) rank queries.
  OrderedSet timestamp_tree_;

  // histogram_[i] accumulates scaled counts for stack distances in
  // [i*bin_size, (i+1)*bin_size).
  // histogram_[num_bins_] is the overflow/infinite bin (first-time accesses).
  std::vector<double> histogram_;
};

}  // namespace ROCKSDB_NAMESPACE
