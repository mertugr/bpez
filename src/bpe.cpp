#include "bpez/bpe.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace bpez {

namespace {

struct PairCountHash {
  std::size_t operator()(const BPE::Pair& p) const noexcept {
    return (static_cast<std::size_t>(static_cast<uint32_t>(p.first)) << 32) ^
           static_cast<std::size_t>(static_cast<uint32_t>(p.second));
  }
};

using PairCount = std::unordered_map<BPE::Pair, int64_t, PairCountHash>;

PairCount count_pairs(const std::vector<std::vector<BPE::TokenId>>& corpus) {
  PairCount counts;
  for (const auto& word : corpus) {
    if (word.size() < 2) {
      continue;
    }
    for (std::size_t i = 0; i + 1 < word.size(); ++i) {
      ++counts[{word[i], word[i + 1]}];
    }
  }
  return counts;
}

void merge_pair_in_word(std::vector<BPE::TokenId>& word, BPE::Pair pair, BPE::TokenId new_id) {
  if (word.size() < 2) {
    return;
  }
  std::vector<BPE::TokenId> out;
  out.reserve(word.size());
  for (std::size_t i = 0; i < word.size();) {
    if (i + 1 < word.size() && word[i] == pair.first && word[i + 1] == pair.second) {
      out.push_back(new_id);
      i += 2;
    } else {
      out.push_back(word[i]);
      ++i;
    }
  }
  word.swap(out);
}

}  // namespace

BPE::BPE() {
  vocab_.resize(256);
  for (int i = 0; i < 256; ++i) {
    vocab_[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(static_cast<unsigned char>(i)));
  }
}

void BPE::rebuild_ranks() {
  merge_ranks_.clear();
  merge_ranks_.reserve(merges_.size());
  for (int i = 0; i < static_cast<int>(merges_.size()); ++i) {
    merge_ranks_[merges_[static_cast<std::size_t>(i)]] = i;
  }
}

std::string BPE::token_bytes(TokenId id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= vocab_.size()) {
    throw std::out_of_range("bpez: invalid token id in decode");
  }
  return vocab_[static_cast<std::size_t>(id)];
}

void BPE::train(std::string_view text, int vocab_size) {
  if (vocab_size < 256) {
    throw std::invalid_argument("bpez: vocab_size must be >= 256");
  }

  // Reset to base 256-byte vocabulary.
  vocab_.assign(256, std::string());
  for (int i = 0; i < 256; ++i) {
    vocab_[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(static_cast<unsigned char>(i)));
  }
  merges_.clear();
  merge_ranks_.clear();

  if (vocab_size == 256 || text.empty()) {
    return;
  }

  // Treat entire training blob as one continuous byte sequence (no whitespace pretokenization).
  // This is true byte-level BPE on the raw byte stream.
  std::vector<std::vector<TokenId>> corpus(1);
  corpus[0].reserve(text.size());
  for (unsigned char c : text) {
    corpus[0].push_back(static_cast<TokenId>(c));
  }

  const int num_merges = vocab_size - 256;
  for (int m = 0; m < num_merges; ++m) {
    const PairCount counts = count_pairs(corpus);
    if (counts.empty()) {
      break;
    }

    // Pick most frequent pair; tie-break by smaller (a,b) for determinism.
    Pair best{-1, -1};
    int64_t best_count = 0;
    for (const auto& kv : counts) {
      if (kv.second > best_count ||
          (kv.second == best_count &&
           (kv.first.first < best.first ||
            (kv.first.first == best.first && kv.first.second < best.second)))) {
        best = kv.first;
        best_count = kv.second;
      }
    }
    if (best_count <= 0 || best.first < 0) {
      break;
    }

    const TokenId new_id = static_cast<TokenId>(vocab_.size());
    vocab_.push_back(token_bytes(best.first) + token_bytes(best.second));
    merges_.push_back(best);

    for (auto& word : corpus) {
      merge_pair_in_word(word, best, new_id);
    }
  }

  rebuild_ranks();
}

std::vector<BPE::TokenId> BPE::encode(std::string_view text) const {
  if (text.empty()) {
    return {};
  }

  std::vector<TokenId> tokens;
  tokens.reserve(text.size());
  for (unsigned char c : text) {
    tokens.push_back(static_cast<TokenId>(c));
  }

  if (merges_.empty()) {
    return tokens;
  }

  // Greedy BPE: repeatedly merge the pair with the lowest merge rank until no more merges apply.
  while (tokens.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    std::size_t best_i = 0;
    bool found = false;
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      const auto it = merge_ranks_.find({tokens[i], tokens[i + 1]});
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
        found = true;
      }
    }
    if (!found) {
      break;
    }
    // new token id = 256 + rank
    const TokenId new_id = static_cast<TokenId>(256 + best_rank);
    tokens[best_i] = new_id;
    tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
  }

  return tokens;
}

std::string BPE::decode(const std::vector<TokenId>& ids) const {
  std::string out;
  // Pre-size conservatively.
  out.reserve(ids.size() * 4);
  for (TokenId id : ids) {
    out += token_bytes(id);
  }
  return out;
}

void BPE::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("bpez: failed to open file for writing: " + path);
  }
  out << "# bpez model v1\n";
  out << "vocab_size " << vocab_.size() << "\n";
  out << "merges " << merges_.size() << "\n";
  for (const auto& p : merges_) {
    out << p.first << " " << p.second << "\n";
  }
  if (!out) {
    throw std::runtime_error("bpez: failed while writing model: " + path);
  }
}

void BPE::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("bpez: failed to open model: " + path);
  }

  std::string line;
  // Skip header comment lines.
  while (std::getline(in, line)) {
    if (!line.empty() && line[0] != '#') {
      break;
    }
  }

  int vs = 0;
  if (sscanf(line.c_str(), "vocab_size %d", &vs) != 1 || vs < 256) {
    throw std::runtime_error("bpez: invalid vocab_size in model");
  }

  if (!std::getline(in, line)) {
    throw std::runtime_error("bpez: missing merges line");
  }
  int nm = 0;
  if (sscanf(line.c_str(), "merges %d", &nm) != 1 || nm < 0) {
    throw std::runtime_error("bpez: invalid merges count in model");
  }
  if (vs != 256 + nm) {
    throw std::runtime_error("bpez: vocab_size and merges count mismatch");
  }

  vocab_.assign(256, std::string());
  for (int i = 0; i < 256; ++i) {
    vocab_[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(static_cast<unsigned char>(i)));
  }
  merges_.clear();
  merges_.reserve(static_cast<std::size_t>(nm));

  for (int i = 0; i < nm; ++i) {
    if (!std::getline(in, line)) {
      throw std::runtime_error("bpez: truncated merges list");
    }
    int a = 0, b = 0;
    if (sscanf(line.c_str(), "%d %d", &a, &b) != 2) {
      throw std::runtime_error("bpez: bad merge line");
    }
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= vocab_.size() ||
        static_cast<std::size_t>(b) >= vocab_.size()) {
      throw std::runtime_error("bpez: merge references unknown token");
    }
    merges_.emplace_back(static_cast<TokenId>(a), static_cast<TokenId>(b));
    vocab_.push_back(vocab_[static_cast<std::size_t>(a)] + vocab_[static_cast<std::size_t>(b)]);
  }

  rebuild_ranks();
}

}  // namespace bpez
