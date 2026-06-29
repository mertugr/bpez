#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bpez {

/// Byte-level BPE tokenizer.
///
/// Vocabulary always starts with the 256 raw byte tokens (ids 0..255).
/// Training learns merge rules that produce higher token ids.
/// Operates on raw bytes (not Unicode code points / chars).
/// Guarantees: decode(encode(x)) == x for any byte sequence x.
class BPE {
public:
  using TokenId = int32_t;
  using Byte = uint8_t;
  using Pair = std::pair<TokenId, TokenId>;

  struct PairHash {
    std::size_t operator()(const Pair& p) const noexcept {
      return (static_cast<std::size_t>(static_cast<uint32_t>(p.first)) << 32) ^
             static_cast<std::size_t>(static_cast<uint32_t>(p.second));
    }
  };

  BPE();

  /// Train on raw bytes in `text`. Target vocabulary size includes the base 256 bytes.
  /// If vocab_size <= 256, no merges are learned.
  void train(std::string_view text, int vocab_size);

  /// Encode arbitrary bytes to token ids using learned merges.
  [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;

  /// Decode token ids back to the original byte sequence.
  [[nodiscard]] std::string decode(const std::vector<TokenId>& ids) const;

  /// Save model (vocab merges) to a text file.
  void save(const std::string& path) const;

  /// Load model from a file produced by save().
  void load(const std::string& path);

  [[nodiscard]] int vocab_size() const { return static_cast<int>(vocab_.size()); }
  [[nodiscard]] const std::vector<Pair>& merges() const { return merges_; }

private:
  /// vocab_[id] = byte sequence represented by that token.
  std::vector<std::string> vocab_;
  /// merges_ in priority order (first merge has highest priority).
  std::vector<Pair> merges_;
  /// pair -> merge rank (lower = applied first).
  std::unordered_map<Pair, int, PairHash> merge_ranks_;

  void rebuild_ranks();
  [[nodiscard]] std::string token_bytes(TokenId id) const;
};

}  // namespace bpez
