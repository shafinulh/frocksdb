#include "trace_replay/shards_mrc.h"

#include <cassert>
#include <cstdio>
#include <limits>
#include <sstream>

#include "util/hash.h"

namespace ROCKSDB_NAMESPACE {

ShardsMRC::ShardsMRC(double sampling_ratio, uint64_t num_bins,
                     uint64_t bin_size)
    : sampling_ratio_(sampling_ratio),
      threshold_(sampling_ratio >= 1.0
                     ? std::numeric_limits<uint64_t>::max()
                     : static_cast<uint64_t>(
                           sampling_ratio *
                           static_cast<double>(
                               std::numeric_limits<uint64_t>::max()))),
      scale_(sampling_ratio >= 1.0 ? 1.0 : 1.0 / sampling_ratio),
      num_bins_(num_bins),
      bin_size_(bin_size),
      histogram_(num_bins + 1, 0.0) {}

void ShardsMRC::ProcessAccess(const Slice& block_key) {
  ++num_entries_seen_;
  ++current_timestamp_;

  // Spatial sampling: include this block if its hash falls below the threshold.
  // All accesses to a sampled block are tracked (hash is deterministic per key).
  uint64_t h = GetSliceNPHash64(block_key);
  if (h > threshold_) {
    // Not sampled. Timestamp still advances so stack distances through the
    // unsampled access stream remain correct for sampled blocks.
    return;
  }
  ++num_entries_processed_;

  std::string key(block_key.data(), block_key.size());
  auto it = last_seen_.find(key);

  if (it != last_seen_.end()) {
    uint64_t old_ts = it->second;

    // Stack distance = number of distinct sampled blocks accessed between
    // old_ts and now = number of timestamps strictly greater than old_ts.
    // order_of_key(t) = count of elements strictly less than t, so:
    //   elements > old_ts = tree.size() - order_of_key(old_ts) - 1
    uint64_t rank_left = timestamp_tree_.order_of_key(old_ts);
    uint64_t distance = timestamp_tree_.size() - rank_left - 1;

    // Update tree and map to reflect current access.
    timestamp_tree_.erase(old_ts);
    timestamp_tree_.insert(current_timestamp_);
    it->second = current_timestamp_;

    // Record in histogram, scaled by 1/sampling_ratio.
    uint64_t bin = (bin_size_ > 0) ? (distance / bin_size_) : distance;
    if (bin >= num_bins_) {
      histogram_[num_bins_] += scale_;
    } else {
      histogram_[bin] += scale_;
    }
  } else {
    // First time seeing this block — infinite stack distance (compulsory miss).
    last_seen_[key] = current_timestamp_;
    timestamp_tree_.insert(current_timestamp_);
    histogram_[num_bins_] += scale_;
  }
}

bool ShardsMRC::DumpMRC(const std::string& path) const {
  // Compute total scaled accesses across all bins.
  double total = 0.0;
  for (double v : histogram_) {
    total += v;
  }
  if (total <= 0.0) {
    return false;
  }

  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    return false;
  }

  // Write header: (num_bins: uint64, bin_size: uint64)
  uint64_t header[2] = {num_bins_, bin_size_};
  if (fwrite(header, sizeof(uint64_t), 2, f) != 2) {
    fclose(f);
    return false;
  }

  // Write MRC entries: (index: uint64, miss_rate: float64) x num_bins.
  // miss_rate[i] = fraction of accesses with stack distance >= i * bin_size
  //              = suffix_sum[i] / total
  double suffix = total;
  for (uint64_t i = 0; i < num_bins_; ++i) {
    double miss_rate = suffix / total;
    if (fwrite(&i, sizeof(uint64_t), 1, f) != 1 ||
        fwrite(&miss_rate, sizeof(double), 1, f) != 1) {
      fclose(f);
      return false;
    }
    suffix -= histogram_[i];
    if (suffix < 0.0) {
      suffix = 0.0;
    }
  }

  fclose(f);

  // Write human-readable CSV alongside the binary.
  std::string txt_path = path + ".txt";
  FILE* tf = fopen(txt_path.c_str(), "w");
  if (tf) {
    fprintf(tf, "cache_size_bytes,miss_ratio\n");
    double sfx = total;
    for (uint64_t i = 0; i < num_bins_; ++i) {
      fprintf(tf, "%llu,%.6f\n", (unsigned long long)(i * bin_size_),
              sfx / total);
      sfx -= histogram_[i];
      if (sfx < 0.0) sfx = 0.0;
    }
    fclose(tf);
  }

  return true;
}

void ShardsMRC::Decay(double decay_factor) {
  for (double& v : histogram_) {
    v *= decay_factor;
  }
}

std::string ShardsMRC::GetHistogramJSON() const {
  std::ostringstream oss;
  oss << "{\"sampling_ratio\":" << sampling_ratio_
      << ",\"num_bins\":" << num_bins_
      << ",\"bin_size\":" << bin_size_
      << ",\"entries_seen\":" << num_entries_seen_
      << ",\"entries_processed\":" << num_entries_processed_
      << ",\"histogram\":[";
  for (uint64_t i = 0; i <= num_bins_; ++i) {
    if (i > 0) oss << ",";
    oss << histogram_[i];
  }
  oss << "]}";
  return oss.str();
}

}  // namespace ROCKSDB_NAMESPACE
