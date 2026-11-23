# Sprint 09 post outline: quantization

Working title: **"Half the size, same words, five times slower (and why)"**

## 1. The artifact

- The quality table: int8 gives the same text and the same bits/byte at
  half the size.

## 2. A weight in 8 bits

- Scale, round, the half-step bound. One worked example with four numbers.
- The outlier test: per-tensor vs per-channel, with the error numbers.

## 3. Down to 4 bits

- Groups of 32, nibble packing (show the byte layout).
- The quality cost, group 32 vs 128.

## 4. Why it got slower

- The profile tables, float vs int8.
- The plain explanation: NumPy can't multiply bytes, so it expands them.
- What a real kernel does instead, and why that's a memory-bandwidth win.

## 5. The series so far

- From a tokenizer to a quantized model behind your own server: the one
  paragraph that ties the serving track together.
