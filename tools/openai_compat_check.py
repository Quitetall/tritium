"""OpenAI-client compat check against tritium-serve (the P3 verification gate).

Run against a live server (default port below):
    tritium-serve --model <model.gguf> --backend cuda --port 8179 &
    uv run --with openai python tools/openai_compat_check.py

Exercises the exact call shapes OpenWebUI/continue.dev/LM-Studio-class
clients use: model listing, non-stream chat, streamed chat with
include_usage, stop sequences, temperature."""
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8179/v1", api_key="unused")

models = [m.id for m in c.models.list()]
assert models == ["tritium"], models
print("models.list ok:", models)

r = c.chat.completions.create(
    model="tritium", max_tokens=32,
    messages=[{"role": "user", "content": "Name the largest planet in our solar system."}])
print("nonstream ok:", repr(r.choices[0].message.content), r.choices[0].finish_reason, r.usage)
assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens

chunks, usage = [], None
stream = c.chat.completions.create(
    model="tritium", max_tokens=32, stream=True,
    stream_options={"include_usage": True},
    messages=[{"role": "user", "content": "Say hello in French."}])
for ev in stream:
    if ev.usage is not None:
        usage = ev.usage
    if ev.choices and ev.choices[0].delta.content:
        chunks.append(ev.choices[0].delta.content)
text = "".join(chunks)
print("stream ok:", repr(text))
print("stream usage:", usage)
assert usage is not None and usage.completion_tokens > 0

r = c.chat.completions.create(
    model="tritium", max_tokens=48, stop=["."],
    temperature=0.0,
    messages=[{"role": "user", "content": "What is the capital of Japan?"}])
print("stop+temp0 ok:", repr(r.choices[0].message.content), r.choices[0].finish_reason)
assert "." not in r.choices[0].message.content

print("ALL COMPAT CHECKS PASSED")
