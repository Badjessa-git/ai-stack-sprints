# Sprint 06 post outline: an inference server

Working title: **"curl my own LLM"**

## 1. The artifact

- The curl command and its streamed output, from your server, running
  GPT-2 on your code.

## 2. What an HTTP response is

- Status line, headers, body: show the raw bytes once.
- Why streaming breaks Content-Length, and how SSE gets around it.

## 3. The two bugs streaming creates

- Half a character: show the garbage, then the fix.
- The stop string split across tokens: the example, and why you have to
  hold text back.

## 4. One at a time

- The load-test table: flat throughput, growing TTFT.
- Plain explanation: every request reads all the weights by itself.
- The bridge to Sprint 7: make them share.

## 5. What production servers add

- Only what the README's checklist covers.
