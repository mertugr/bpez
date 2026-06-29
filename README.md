# bpez

Byte-level BPE (Byte Pair Encoding) tokenizer in C++17.

## Features

- **Library** (`bpez::BPE`): `train(text, vocab_size)`, `encode(text)`, `decode(ids)`
- **CLI** (`bpez`): `train`, `encode`, `decode`
- Operates on **raw bytes** (not Unicode code points / `char` semantics)
- Invariant: **`decode(encode(x)) == x`** for any input byte sequence `x`
- Base vocabulary is always the 256 byte values (`0..255`); merges grow the vocab

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Binaries: `build/bpez`, `build/bpez_tests`

## CLI

```bash
# Train a model (vocab_size includes the 256 base byte tokens)
./build/bpez train --input corpus.txt --model model.bpez --vocab-size 512

# Encode bytes → space-separated token ids
./build/bpez encode --model model.bpez --input sample.txt --output ids.txt
# or pipe:
printf 'hello' | ./build/bpez encode --model model.bpez

# Decode token ids → raw bytes
./build/bpez decode --model model.bpez --input ids.txt --output out.bin
```

## Library usage

```cpp
#include "bpez/bpe.hpp"

bpez::BPE tok;
tok.train(training_bytes, /*vocab_size=*/512);
auto ids = tok.encode(input_bytes);
std::string out = tok.decode(ids);  // out == input_bytes
tok.save("model.bpez");
tok.load("model.bpez");
```

## License

MIT
