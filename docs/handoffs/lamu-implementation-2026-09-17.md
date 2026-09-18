# LAMU/Tritium implementation handoff

Date: 2026-09-17
Owner: `/root/lamu_implementation`

## Scope

Diagnose and fix LAMU REPL integration with Tritium's OpenAI-compatible CUDA
server. Keep transport/UI behavior separate from ternary model quality.

## Tritium state

- Repository: `/home/brianklam/Desktop/Tritium`
- HEAD: `ecde6c2f69f8b21d00670bef8fde7d22c92ec148`
- Branch: `main`, clean and synchronized with `origin/main`
- Existing fix: Tritium accepts LAMU's `model: "default"` as alias for
  configured model ID.
- Unit gate: `cargo test -p tritium-serve --features serve --lib` — 39 passed.

## Runtime

- Tritium binary: `/home/brianklam/.local/bin/tritium-serve-cuda`
- Bundle: `/mnt/4tb/tmp/qwen36-ptq-b3-r2-r3-565abdee`
- Endpoint: `http://127.0.0.1:8101/v1/chat/completions`
- Model ID: `qwen36-ternary-salt`
- Profile: `compact-v1`
- GPU memory: approximately 7.7 GiB

## Verified behavior

- Direct non-stream request with `model: "default"`: HTTP 200.
- Direct stream request: valid `data: {...}` SSE chunks followed by
  `data: [DONE]`.
- Ternary output is incoherent (`Testing` produced fragments such as
  `this, of in`). This is a model-quality blocker, not proof of transport
  failure.

## LAMU implementation result

Repository: `/home/brianklam/local-llm/lamu-rs`

- Commit: `8795c09 fix(chat): fit local completion limits and show HTTP errors`
- Pushed to `Quitetall/lamu` `main`.
- Root cause: LAMU TUI sent `max_tokens: 65536`; Tritium rejects values above
  4096 with HTTP 400. LAMU ignored non-SSE JSON errors, leaving spinner state.
- Fix: cap interactive OpenAI-compatible payload at 4096 and surface HTTP
  errors visibly.
- Tests:
  - `cargo fmt --all -- --check`
  - `cargo test -p lamu-providers` — 36 passed
  - `cargo test -p lamu chat_tui` — 15 passed
- Manual TUI test: streamed tokens render against Tritium endpoint.

## Review status

Required external review attempted for LAMU commit. Exact result:
`Transport closed`.

## Next action

Use:

```bash
lamu repl http://127.0.0.1:8101/v1/chat/completions
```

Expect several seconds startup/decode latency and incoherent output until
ternary artifact quality is repaired.
