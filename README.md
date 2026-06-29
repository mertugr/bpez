# bpez

Byte-level BPE (Byte Pair Encoding) tokenizer in C++17, with **GPT-2-style pre-tokenization** and **special tokens**.

## Features

- **Library** (`bpez::BPE`): `train`, `encode`, `decode`, `pretokenize`
- **CLI** (`bpez`): `train`, `encode`, `decode`
- **GPT-2 pre-tokenization** before BPE (Unicode `\p{L}` / `\p{N}`, not ASCII-only)
  - Turkish / accented letters stay in the same word piece: `çağdaş`, `İstanbul`, `ışık`, `öğrenci`, `naïve`, …
- **BPE runs independently inside each pre-token chunk** (no merges across boundaries)
- **Special tokens** (default `<|endoftext|>`) are atomic — never split or merge-trained
- Operates on **raw bytes**; invariant: **`decode(encode(x)) == x`**

Vocabulary layout: `0..255` bytes → learned merges → special tokens.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## CLI

```bash
# Train (default special: <|endoftext|>)
./build/bpez train --input corpus.txt --model model.bpez --vocab-size 512

# Custom specials
./build/bpez train --input corpus.txt --model model.bpez --vocab-size 512 \
  --special '<|endoftext|>' --special '<|user|>'

# No special tokens
./build/bpez train --input corpus.txt --model model.bpez --vocab-size 512 --no-special

./build/bpez encode --model model.bpez --input sample.txt
./build/bpez decode --model model.bpez --input ids.txt
```

## Library

```cpp
#include "bpez/bpe.hpp"

bpez::BPE tok;
tok.train(training_bytes, 512, {"<|endoftext|>", "<|user|>"});
auto chunks = tok.pretokenize("çağdaş ışık");  // GPT-2 chunks
auto ids = tok.encode("hi<|endoftext|>");
std::string out = tok.decode(ids);  // out == input bytes
```

Pre-tokenizer pattern (GPT-2 / tiktoken style):

```
's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
```

## License

MIT
