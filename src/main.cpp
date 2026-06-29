#include "bpez/bpe.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "bpez — byte-level BPE tokenizer (GPT-2 pre-tokenization + special tokens)\n\n"
      << "Usage:\n"
      << "  " << argv0
      << " train  --input <file> --model <path> --vocab-size <N> [--special <tok>]...\n"
      << "  " << argv0 << " encode --model <path> [--input <file>] [--output <file>]\n"
      << "  " << argv0 << " decode --model <path> [--input <file>] [--output <file>]\n\n"
      << "Notes:\n"
      << "  • Pre-tokenizes with GPT-2 rules (Unicode \\p{L}/\\p{N}, incl. Turkish/accented).\n"
      << "  • BPE runs independently inside each pre-token chunk.\n"
      << "  • Default special token: <|endoftext|>. Pass --special '' to disable, or\n"
      << "    repeat --special for multiple. Use --no-special for none.\n"
      << "  • encode writes space-separated token ids; decode writes raw bytes.\n";
}

std::string read_all(std::istream& in) {
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open input: " + path);
  }
  return read_all(in);
}

void write_file(const std::string& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output: " + path);
  }
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!out) {
    throw std::runtime_error("failed while writing: " + path);
  }
}

std::string get_opt(int argc, char** argv, const std::string& name) {
  for (int i = 0; i < argc - 1; ++i) {
    if (name == argv[i]) {
      return argv[i + 1];
    }
  }
  return {};
}

std::vector<std::string> get_opt_all(int argc, char** argv, const std::string& name) {
  std::vector<std::string> out;
  for (int i = 0; i < argc - 1; ++i) {
    if (name == argv[i]) {
      out.emplace_back(argv[i + 1]);
    }
  }
  return out;
}

bool has_flag(int argc, char** argv, const std::string& name) {
  for (int i = 0; i < argc; ++i) {
    if (name == argv[i]) {
      return true;
    }
  }
  return false;
}

int cmd_train(int argc, char** argv) {
  const std::string input = get_opt(argc, argv, "--input");
  const std::string model = get_opt(argc, argv, "--model");
  const std::string vs_s = get_opt(argc, argv, "--vocab-size");
  if (input.empty() || model.empty() || vs_s.empty()) {
    usage(argv[0]);
    return 2;
  }
  const int vocab_size = std::stoi(vs_s);
  const std::string text = read_file(input);

  std::vector<std::string> specials;
  if (has_flag(argc, argv, "--no-special")) {
    specials.clear();
  } else {
    specials = get_opt_all(argc, argv, "--special");
    if (specials.empty()) {
      specials.push_back("<|endoftext|>");
    }
  }

  bpez::BPE tok;
  tok.train(text, vocab_size, specials);
  tok.save(model);
  std::cerr << "trained vocab_size=" << tok.vocab_size() << " merges=" << tok.merges().size()
            << " specials=" << tok.special_tokens().size() << " -> " << model << "\n";
  return 0;
}

int cmd_encode(int argc, char** argv) {
  const std::string model = get_opt(argc, argv, "--model");
  if (model.empty()) {
    usage(argv[0]);
    return 2;
  }
  const std::string input_path = get_opt(argc, argv, "--input");
  const std::string output_path = get_opt(argc, argv, "--output");

  bpez::BPE tok;
  tok.load(model);
  const std::string text = input_path.empty() ? read_all(std::cin) : read_file(input_path);
  const auto ids = tok.encode(text);

  std::ostringstream oss;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) {
      oss << ' ';
    }
    oss << ids[i];
  }
  oss << '\n';
  const std::string out = oss.str();
  if (output_path.empty()) {
    std::cout << out;
  } else {
    write_file(output_path, out);
  }
  return 0;
}

int cmd_decode(int argc, char** argv) {
  const std::string model = get_opt(argc, argv, "--model");
  if (model.empty()) {
    usage(argv[0]);
    return 2;
  }
  const std::string input_path = get_opt(argc, argv, "--input");
  const std::string output_path = get_opt(argc, argv, "--output");

  bpez::BPE tok;
  tok.load(model);

  const std::string ids_text = input_path.empty() ? read_all(std::cin) : read_file(input_path);
  std::vector<bpez::BPE::TokenId> ids;
  std::istringstream iss(ids_text);
  long v = 0;
  while (iss >> v) {
    ids.push_back(static_cast<bpez::BPE::TokenId>(v));
  }

  const std::string bytes = tok.decode(ids);
  if (output_path.empty()) {
    std::cout.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  } else {
    write_file(output_path, bytes);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || has_flag(argc, argv, "-h") || has_flag(argc, argv, "--help")) {
    usage(argv[0]);
    return argc < 2 ? 2 : 0;
  }

  try {
    const std::string cmd = argv[1];
    if (cmd == "train") {
      return cmd_train(argc, argv);
    }
    if (cmd == "encode") {
      return cmd_encode(argc, argv);
    }
    if (cmd == "decode") {
      return cmd_decode(argc, argv);
    }
    std::cerr << "unknown command: " << cmd << "\n\n";
    usage(argv[0]);
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
