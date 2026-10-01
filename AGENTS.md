# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

- Fork of `tmc/langchaingo`, module `github.com/vxcontrol/langchaingo`, Go version in `go.mod`. Maintained for [PentAGI](https://github.com/vxcontrol/pentagi), which pins the fork in its `go.mod` by a release tag or by the pseudo-version of a pushed commit: a change reaches it only when that pin is bumped.
- Branches: `main` mirrors upstream, `main-pull-requests` adds unmerged upstream PRs, `main-vxcontrol` is the default branch and the base of every PR, `release/v*` carry the tags `vX.Y.Z-update.N`.
- Most of the fork's own work is in `llms/`, `llms/reasoning/` and `vectorstores/pgvector`.

## Contract of a provider adapter

Each package under `llms/<vendor>` is a door: it translates the caller's `llms.CallOption`s into the request the vendor's API accepts.

- The vendor's official documentation is the source of truth. If it does not mention a parameter for a model, the parameter is unsupported. A measurement through a gateway (LiteLLM, OpenRouter) is not a vendor rule.
- What the vendor accepts reaches the wire unchanged. What it rejects is never sent.
- An option that cannot go out as asked is dropped, clamped or substituted and reported in `ContentResponse.Warnings` (`llms/warning.go`). An intent that cannot be honoured at all fails with a typed error before any request is sent.
- A model name no table knows passes through as is: the vendor API decides, so a new model never regresses.
- The same model name means different APIs on different hosts (the vendor's own API, OpenRouter, vLLM, Azure, DashScope, a LiteLLM route). A rule about one vendor's API must check the host, as `reasoning.ServedByDeepSeek(model, host)` and `reasoning.DashScopeRoute(model, host)` do.

## Layout

| Path | What |
|---|---|
| `llms/` | `Model` interface, call options (`options.go`), warnings, reasoning hints (`reasoning_support.go`), stop reasons, tool-choice classification (`turn.go`) |
| `llms/reasoning/` | Model capability tables and wire rules shared by all doors: name normalization (`name.go`), per-family tables (`*_capability.go`), effort and sampling rules (`effort_wire.go`), how to disable thinking (`off.go`). Must not import `llms` |
| `llms/openai` | OpenAI and every OpenAI-compatible host (Azure, DeepSeek, DashScope, OpenRouter, xAI, vLLM, LiteLLM) |
| `llms/anthropic`, `llms/bedrock`, `llms/googleai` (+ `vertex`), `llms/ollama`, `llms/mistral`, `llms/huggingface` | Other doors. Bedrock has the Converse API and legacy per-family providers in `internal/bedrockclient`; googleai and vertex use `google.golang.org/genai` |
| `embeddings/`, `vectorstores/`, `tools/` | Embedders, vector stores (`pgvector` carries fork-specific metadata indexes and filters), agent tools |
| `internal/httprr` | HTTP record and replay for tests |
| `internal/toolcall` | Decoding of tool-call arguments |
| `internal/devtools` | `lint` (repo-specific checks), `rrtool` (lists packages with cassettes, reports uncompressed ones), `examples-updater` |
| `testing/llmtest` | Conformance checks for an `llms.Model` |
| `examples/` | One module per directory, with `replace` to `../..` |
| `docs/` | Docusaurus site |

## Commands

```sh
go build ./...
go vet ./...
go test ./...                                  # replays cassettes; stores with testcontainers need Docker
go test ./llms/openai -run TestName -v
go test ./llms/openai -run TestName -httprecord=.   # re-record with the vendor key set
make lint                                      # golangci-lint v2; CI pins v2.12.2
make lint-testing                              # httprr test patterns
make lint-architecture
make build-examples                            # compiles every example to /dev/null
```

Never run `go build` inside `examples/*`: it leaves binaries that `.gitignore` does not cover.

CI (`.github/workflows/ci.yaml`) runs on pushes and PRs to `main-vxcontrol`: golangci-lint, `go build ./...`, `go test -race ./...`.

## Tests

- Vendor HTTP is recorded by `internal/httprr` into `testdata/<TestName>.httprr` and replayed by default. Open a recorder with `httprr.OpenForTest(t, transport)`; skip a test that has neither a key nor a cassette with `httprr.SkipIfNoCredentialsAndRecordingMissing(t, "VENDOR_API_KEY")`.
- A replay miss (the request differs from the recorded one) returns an error, not a skip. A test that only asserts `require.Error` then passes without testing anything: assert the error's type or code, or the request body.
- Pin a door's behaviour by the request body it sends, not by calling the capability helper the door itself calls.
- Re-record a cassette instead of editing it by hand.
- Environment:
  - pgvector, cloudsql and alloydb tests start a `pgvector/pgvector:pg16` container unless `PGVECTOR_CONNECTION_STRING` names a database, and skip without Docker.
  - ollama cloud tests replay with a throwaway signing key when `~/.ollama/id_ed25519` is missing; recording needs the account's key.
  - Bedrock replay needs no AWS credentials but fails when `AWS_CA_BUNDLE` is set or `AWS_PROFILE` names a profile missing from the shared config: unset both.
  - The textsplitter token tests download the `cl100k_base` tiktoken encoding and skip when it cannot be loaded.

## Changing model capabilities

1. Find the vendor documentation line that states the rule. No line, no table change.
2. Check which forms `modelSpellings` in `llms/reasoning/name.go` gives the tables: it drops the path prefix (`openrouter/…`, `anthropic/…`), `ft:` wrappers and Bedrock region prefixes (`us.`), returns both the bare and the prefixed form for platform prefixes (`anthropic.`) and the dash-written `zai-`, and maps the vendor-backed aliases listed in `earlierNames`. Other aliases (`-latest`), dated snapshots and `5-3` versus `5.3` are left to each table, so a table keyed on one spelling misses the others.
3. Change the table, then add door-level tests on the wire body: one case per spelling and per host that behaves differently.

## Style

- English in code, comments, docs and commit messages.
- Comments carry only what the code cannot show: an external contract, a non-obvious invariant, the reason for a workaround. No restating the code, no history ("used to", "no longer"), no references to trackers, review reports or anything outside the repository. A vendor fact goes into a test, not a comment.
- Markdown: one line per paragraph, list item and table row; no hard wrapping.
- Commit subject `type(scope): summary`, e.g. `fix(reasoning): …`, `test(pgvector): …`; the body says why.
