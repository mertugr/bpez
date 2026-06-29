#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bpez {

/// Byte-level BPE tokenizer with GPT-2-style pre-tokenization and special tokens.
///
/// Vocabulary layout:
///   - ids 0..255: raw byte tokens
///   - ids 256..256+M-1: learned merge tokens (M = merges_.size())
///   - remaining ids: special tokens (atomic strings, never BPE-merged)
///
/// Before BPE, text is split with a GPT-2-style regex over Unicode:
///   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
/// Letter/number classes use Unicode categories (not ASCII a-zA-Z0-9), so Turkish
/// and other non-ASCII letters (ğ, ş, ı, ç, ö, ü, …) stay in the same word piece.
/// BPE runs independently inside each pre-token chunk.
/// Guarantees: decode(encode(x)) == x for any input byte sequence x.
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

  /// Train on `text`. `vocab_size` is the total vocabulary size including the base
  /// 256 byte tokens and any `special_tokens`. Special tokens are appended after merges
  /// and are treated as atomic during encode (never split / merged across).
  /// Default special token matches GPT-2: "<|endoftext|>".
  void train(std::string_view text, int vocab_size,
             const std::vector<std::string>& special_tokens = {"<|endoftext|>"});

  /// Encode bytes → token ids (GPT-2 pre-tokenize, then BPE per chunk; specials atomic).
  [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;

  /// Decode token ids → original byte sequence (including special-token strings).
  [[nodiscard]] std::string decode(const std::vector<TokenId>& ids) const;

  /// GPT-2-style pre-tokenization only (no BPE). Useful for tests / inspection.
  /// Special tokens present in the model are emitted as whole chunks when they occur.
  [[nodiscard]] std::vector<std::string> pretokenize(std::string_view text) const;

  void save(const std::string& path) const;
  void load(const std::string& path);

  [[nodiscard]] int vocab_size() const { return static_cast<int>(vocab_.size()); }
  [[nodiscard]] const std::vector<Pair>& merges() const { return merges_; }
  [[nodiscard]] const std::vector<std::string>& special_tokens() const { return special_tokens_; }
  /// Returns special-token id, or -1 if unknown.
  [[nodiscard]] TokenId special_token_id(std::string_view token) const;

private:
  std::vector<std::string> vocab_;
  std::vector<Pair> merges_;
  std::unordered_map<Pair, int, PairHash> merge_ranks_;
  std::vector<std::string> special_tokens_;
  /// special string -> token id
  std::unordered_map<std::string, TokenId> special_to_id_;

  void reset_base_vocab();
  void rebuild_ranks();
  void rebuild_special_index();
  [[nodiscard]] std::string token_bytes(TokenId id) const;
  [[nodiscard]] std::vector<TokenId> encode_chunk(std::string_view chunk) const;
  /// Partition text into chunks: specials (atomic) and GPT-2 pre-tokens.
  [[nodiscard]] std::vector<std::string> split_for_encode(std::string_view text) const;
};

/// Public for testing: GPT-2 pre-tokenizer over UTF-8 (no special tokens).
[[nodiscard]] std::vector<std::string> gpt2_pretokenize(std::string_view text);

}  // namespace bpez
