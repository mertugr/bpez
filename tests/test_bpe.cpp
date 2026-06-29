#include "bpez/bpe.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;
int g_passed = 0;

#define CHECK(cond)                                                                              \
  do {                                                                                           \
    if (!(cond)) {                                                                               \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n";              \
      ++g_failed;                                                                                \
    } else {                                                                                     \
      ++g_passed;                                                                                \
    }                                                                                            \
  } while (0)

void expect_roundtrip(const bpez::BPE& tok, const std::string& x, const char* label) {
  const auto ids = tok.encode(x);
  const std::string y = tok.decode(ids);
  if (y != x) {
    std::cerr << "FAIL roundtrip [" << label << "] size_in=" << x.size() << " size_out=" << y.size()
              << "\n";
    ++g_failed;
  } else {
    ++g_passed;
  }
}

void test_empty() {
  bpez::BPE tok;
  tok.train("", 300);
  expect_roundtrip(tok, "", "empty");
  CHECK(tok.encode("").empty());
  CHECK(tok.decode({}).empty());
}

void test_ascii() {
  const std::string corpus =
      "hello world\n"
      "the quick brown fox jumps over the lazy dog\n"
      "Hello, ASCII! 0123456789\n";
  bpez::BPE tok;
  tok.train(corpus, 400);
  expect_roundtrip(tok, "", "ascii/empty");
  expect_roundtrip(tok, "hello", "ascii/hello");
  expect_roundtrip(tok, corpus, "ascii/corpus");
  expect_roundtrip(tok, "the quick brown fox", "ascii/phrase");
  expect_roundtrip(tok, std::string("\0\1\2\xff", 4), "ascii/binary-nuls");
}

void test_emoji() {
  // UTF-8 emoji and multi-byte sequences as raw bytes.
  const std::string corpus =
      "hello 👋 world 🌍\n"
      "emoji: 😀😁😂🤣😃😄😅\n"
      "flags: 🇹🇷 🇺🇸\n"
      "zwj family: 👨‍👩‍👧\n";
  bpez::BPE tok;
  tok.train(corpus, 512);
  expect_roundtrip(tok, "👋", "emoji/wave");
  expect_roundtrip(tok, "🌍", "emoji/globe");
  expect_roundtrip(tok, "🇹🇷", "emoji/flag-tr");
  expect_roundtrip(tok, "👨‍👩‍👧", "emoji/zwj-family");
  expect_roundtrip(tok, corpus, "emoji/corpus");
  // Mixed ASCII + emoji
  expect_roundtrip(tok, "hi 👋", "emoji/mixed");
}

void test_all_bytes() {
  std::string all(256, '\0');
  for (int i = 0; i < 256; ++i) {
    all[static_cast<std::size_t>(i)] = static_cast<char>(static_cast<unsigned char>(i));
  }
  bpez::BPE tok;
  tok.train(all + all, 300);
  expect_roundtrip(tok, all, "all_bytes");
  expect_roundtrip(tok, std::string("\0", 1), "nul_byte");
  expect_roundtrip(tok, std::string("\xff", 1), "0xff");
}

void test_base_vocab_only() {
  bpez::BPE tok;
  tok.train("abcabcabc", 256);  // no merges
  CHECK(tok.vocab_size() == 256);
  CHECK(tok.merges().empty());
  const auto ids = tok.encode("abc");
  CHECK(ids.size() == 3);
  CHECK(ids[0] == 'a');
  CHECK(ids[1] == 'b');
  CHECK(ids[2] == 'c');
  expect_roundtrip(tok, "abcabc", "base_only");
}

void test_save_load_roundtrip() {
  const std::string corpus = "banana bandana band bandana banana\n";
  bpez::BPE a;
  a.train(corpus, 280);

  const char* path = "test_model_tmp.bpez";
  a.save(path);
  bpez::BPE b;
  b.load(path);
  std::remove(path);

  CHECK(a.vocab_size() == b.vocab_size());
  CHECK(a.merges().size() == b.merges().size());

  const std::string sample = "banana band";
  CHECK(a.encode(sample) == b.encode(sample));
  expect_roundtrip(b, sample, "save_load");
}

void test_decode_encode_property_randomish() {
  // Odd / adversarial byte patterns
  const std::string corpus = std::string(100, 'a') + std::string(50, 'b') + "abababab" +
                             std::string("\0\0\xff\xff\0\xff", 6);
  bpez::BPE tok;
  tok.train(corpus, 320);

  const std::vector<std::string> cases = {
      "",
      "a",
      "aa",
      "ab",
      "ba",
      std::string(20, 'a'),
      std::string("\0\xff\0\xff", 4),
      "abababababab",
  };
  for (const auto& c : cases) {
    expect_roundtrip(tok, c, "property");
  }
}

}  // namespace

int main() {
  test_empty();
  test_ascii();
  test_emoji();
  test_all_bytes();
  test_base_vocab_only();
  test_save_load_roundtrip();
  test_decode_encode_property_randomish();

  std::cout << "passed=" << g_passed << " failed=" << g_failed << "\n";
  return g_failed == 0 ? 0 : 1;
}
