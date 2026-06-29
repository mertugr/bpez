#include "bpez/bpe.hpp"

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
  tok.train("", 300, {});
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
  expect_roundtrip(tok, "hi 👋", "emoji/mixed");
}

void test_all_bytes() {
  std::string all(256, '\0');
  for (int i = 0; i < 256; ++i) {
    all[static_cast<std::size_t>(i)] = static_cast<char>(static_cast<unsigned char>(i));
  }
  bpez::BPE tok;
  tok.train(all + all, 300, {});
  expect_roundtrip(tok, all, "all_bytes");
  expect_roundtrip(tok, std::string("\0", 1), "nul_byte");
  expect_roundtrip(tok, std::string("\xff", 1), "0xff");
}

void test_base_vocab_only() {
  bpez::BPE tok;
  tok.train("abcabcabc", 256, {});  // no merges, no specials
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
  CHECK(a.special_tokens() == b.special_tokens());

  const std::string sample = "banana band";
  CHECK(a.encode(sample) == b.encode(sample));
  expect_roundtrip(b, sample, "save_load");
}

void test_decode_encode_property_randomish() {
  const std::string corpus = std::string(100, 'a') + std::string(50, 'b') + "abababab" +
                             std::string("\0\0\xff\xff\0\xff", 6);
  bpez::BPE tok;
  tok.train(corpus, 320, {});

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

void test_gpt2_pretokenize_basic() {
  // Leading space sticks to following word (GPT-2 style).
  const auto parts = bpez::gpt2_pretokenize("Hello world");
  CHECK(parts.size() == 2);
  CHECK(parts[0] == "Hello");
  CHECK(parts[1] == " world");

  const auto punct = bpez::gpt2_pretokenize("hi!");
  CHECK(punct.size() == 2);
  CHECK(punct[0] == "hi");
  CHECK(punct[1] == "!");

  const auto contraction = bpez::gpt2_pretokenize("don't");
  // d o n 't  -> "don" + "'t" with GPT-2 pattern (letters then contraction)
  CHECK(contraction.size() == 2);
  CHECK(contraction[0] == "don");
  CHECK(contraction[1] == "'t");
}

void test_gpt2_whitespace_matches_reference() {
  // GPT-2 / tiktoken: multi-space between words uses `\s+(?!\S)` backtracking so
  // one space (or n-1 of a run) is its own chunk and the last space joins the
  // following word via ` ?\p{L}+`. Reference: "a  b" -> ['a', ' ', ' b']
  // (we previously wrongly emitted ['a', '  ', 'b']).
  {
    const auto p = bpez::gpt2_pretokenize("a  b");
    CHECK(p.size() == 3);
    CHECK(p[0] == "a");
    CHECK(p[1] == " ");
    CHECK(p[2] == " b");
  }
  {
    const auto p = bpez::gpt2_pretokenize("a   b");
    CHECK(p.size() == 3);
    CHECK(p[0] == "a");
    CHECK(p[1] == "  ");
    CHECK(p[2] == " b");
  }
  {
    const auto p = bpez::gpt2_pretokenize("a b");
    CHECK(p.size() == 2);
    CHECK(p[0] == "a");
    CHECK(p[1] == " b");
  }
  {
    const auto p = bpez::gpt2_pretokenize("a ");
    CHECK(p.size() == 2);
    CHECK(p[0] == "a");
    CHECK(p[1] == " ");
  }
  {
    const auto p = bpez::gpt2_pretokenize("a  ");
    CHECK(p.size() == 2);
    CHECK(p[0] == "a");
    CHECK(p[1] == "  ");
  }
  {
    const auto p = bpez::gpt2_pretokenize("  b");
    CHECK(p.size() == 2);
    CHECK(p[0] == " ");
    CHECK(p[1] == " b");
  }
  {
    const auto p = bpez::gpt2_pretokenize("hello  world");
    CHECK(p.size() == 3);
    CHECK(p[0] == "hello");
    CHECK(p[1] == " ");
    CHECK(p[2] == " world");
  }
  // Tabs: optional space in ` ?\p{L}+` is only ASCII ' ', so each tab splits.
  {
    const auto p = bpez::gpt2_pretokenize("a\t\tb");
    CHECK(p.size() == 4);
    CHECK(p[0] == "a");
    CHECK(p[1] == "\t");
    CHECK(p[2] == "\t");
    CHECK(p[3] == "b");
  }
}

void test_turkish_and_accented_letters_same_chunk() {
  // Non-ASCII letters must stay in the same \p{L}+ run (not split as "other").
  const auto tr = bpez::gpt2_pretokenize("çağdaş");
  CHECK(tr.size() == 1);
  CHECK(tr[0] == "çağdaş");

  const auto tr2 = bpez::gpt2_pretokenize("İstanbul");
  CHECK(tr2.size() == 1);
  CHECK(tr2[0] == "İstanbul");

  const auto tr3 = bpez::gpt2_pretokenize("ışık");
  CHECK(tr3.size() == 1);
  CHECK(tr3[0] == "ışık");

  const auto mixed = bpez::gpt2_pretokenize("naïve café");
  CHECK(mixed.size() == 2);
  CHECK(mixed[0] == "naïve");
  CHECK(mixed[1] == " café");

  // Leading space + Turkish word is one chunk.
  const auto lead = bpez::gpt2_pretokenize(" şişe");
  CHECK(lead.size() == 1);
  CHECK(lead[0] == " şişe");

  // Train/encode still round-trips Turkish text.
  const std::string corpus = "Türkiye'nin çağdaş ışık kaynağı şişe ve öğrenci\n";
  bpez::BPE tok;
  tok.train(corpus, 400);
  expect_roundtrip(tok, "çağdaş", "tr/cagdas");
  expect_roundtrip(tok, "İstanbul ışık", "tr/istanbul");
  expect_roundtrip(tok, corpus, "tr/corpus");

  // Whole Turkish word is a single pre-token (maybe with leading space).
  const auto chunks = tok.pretokenize("öğrenci");
  CHECK(chunks.size() == 1);
  CHECK(chunks[0] == "öğrenci");
}

void test_special_tokens_atomic() {
  const std::string corpus = "hello world hello world\n";
  bpez::BPE tok;
  tok.train(corpus, 300, {"<|endoftext|>", "<|user|>"});

  CHECK(tok.special_token_id("<|endoftext|>") >= 256);
  CHECK(tok.special_token_id("<|user|>") >= 256);
  CHECK(tok.special_token_id("<|nope|>") == -1);

  const std::string text = "hello<|endoftext|>world<|user|>!";
  expect_roundtrip(tok, text, "special/mixed");

  const auto ids = tok.encode(text);
  // Specials must appear as single ids equal to their registered ids.
  bool saw_eot = false;
  bool saw_user = false;
  for (auto id : ids) {
    if (id == tok.special_token_id("<|endoftext|>")) {
      saw_eot = true;
    }
    if (id == tok.special_token_id("<|user|>")) {
      saw_user = true;
    }
  }
  CHECK(saw_eot);
  CHECK(saw_user);

  // Pretokenize keeps specials intact as their own chunks.
  const auto chunks = tok.pretokenize("ab<|user|>cd");
  CHECK(chunks.size() == 3);
  CHECK(chunks[0] == "ab");
  CHECK(chunks[1] == "<|user|>");
  CHECK(chunks[2] == "cd");
}

void test_bpe_independent_per_chunk() {
  // Merges must not cross GPT-2 chunk boundaries.
  // Train on repeated "ab ab" so "ab" may merge, but space-separated chunks are independent.
  const std::string corpus = "ab ab ab ab ab ab\n";
  bpez::BPE tok;
  tok.train(corpus, 260, {});
  const auto chunks = tok.pretokenize("ab ab");
  CHECK(chunks.size() == 2);
  CHECK(chunks[0] == "ab");
  CHECK(chunks[1] == " ab");
  expect_roundtrip(tok, "ab ab", "indep/chunks");
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
  test_gpt2_pretokenize_basic();
  test_gpt2_whitespace_matches_reference();
  test_turkish_and_accented_letters_same_chunk();
  test_special_tokens_atomic();
  test_bpe_independent_per_chunk();

  std::cout << "passed=" << g_passed << " failed=" << g_failed << "\n";
  return g_failed == 0 ? 0 : 1;
}
