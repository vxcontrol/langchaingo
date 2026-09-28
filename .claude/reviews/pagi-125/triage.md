# Сортировка замечаний ревью PAGI-125 (PR #1) — черновик

Состояние на 2026-09-28, голова PR `355dc71`. Это черновик: приоритеты предложены, их утверждает автор PR.

## Правило

Критерий взят из описания самого PR: «изменение, которое отнимает у вызывающего работавшее поведение, — дефект, а не компромисс совместимости».

- **P0 — блокирует релиз.** Регрессия по этому критерию: на базе `da7016e` вызов работал, на PR падает или теряет поведение. Сюда же поломки компиляции, которых нет в списке breaking changes PR, и утечка ключа в тексте ошибки.
- **P1 — исправить в этом PR.** Дефекты новой функциональности, которой не было в прошлом релизе, и тесты, которые её не проверяют.
- **P2 — следующий патч.** Проблемы, которые были и в прошлом релизе (pre-existing), nit-комментарии и примеры. Это согласуется с разделом PR «Not in this PR».

Итого: **P0 — 39, P1 — 35, P2 — 25** (87 замечаний второго ревью и 12 подтверждённых из первого, помечены `m<N>` — номер комментария первого ревью). Три ложных замечания первого ревью (#7, #13, #15) исправлять не нужно — их треды стоит закрыть.

Флаг **вендор** — вред зависит от поведения вендора, которое проверено по документации, а не живым вызовом. Таких в P0 — 15. Для регрессий безопасный путь не требует живой проверки: вернуть поведение базы там, где правило вендора не измерено.

Строки — на `355dc71`; комментарии к ним есть в PR: [первое ревью](https://github.com/vxcontrol/langchaingo/pull/1#pullrequestreview-5334862871), [второе ревью](https://github.com/vxcontrol/langchaingo/pull/1#pullrequestreview-5336758859).

## A. Правила reasoning по имени модели без учёта хоста и алиасов — P0 12 · P1 1 · P2 0

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | 31 | high | `llms/reasoning/off.go:195` | WithReasoningDisabled now fails locally for Qwen/QwQ names outside DashScope (vLLM, OpenRouter, Together) |  |
| P0 | 1 | high | `llms/reasoning/effort_wire.go:211` | TakesNoThinkingDepth refuses budgets for GLM/Kimi/MiniMax on every host, breaking OpenRouter calls that worked on base |  |
| P0 | 2 | high | `llms/reasoning/effort_wire.go:76` | RejectsSamplingWhileThinking keys on a classified OpenAI name, so deployments, aliases and unclassified GPT-5 variants now send temperature with reasoning_effort | вендор |
| P0 | 3 | medium | `llms/reasoning/effort_wire.go:180` | AcceptsEffortWire drops reasoning effort on OpenRouter/Groq and turns WithReasoningDisabled into a local error |  |
| P0 | 6 | medium | `llms/reasoning/off.go:171` | Thinking-object off wire is chosen by model name, so OpenRouter gets Z.ai/DeepSeek-native `thinking` instead of its reasoning field | вендор |
| P0 | 8 | medium | `llms/reasoning/effort_wire.go:301` | EffortWithTools checks only the model name, so OpenRouter's openai/gpt-5.4, 5.5 and 5.6 lose effort-with-tools | вендор |
| P0 | 9 | medium | `llms/reasoning/effort_wire.go:49` | onMiniMaxAPI reads OpenRouter's minimax/ slug as MiniMax's own API: structured output refused, top_k dropped | вендор |
| P0 | 10 | medium | `llms/reasoning/claude_capability.go:49` | Undated OpenRouter slugs anthropic/claude-opus-4 and claude-sonnet-4 are unclassified, so temperature now goes out with thinking | вендор |
| P0 | 12 | medium | `llms/reasoning/effort_wire.go:29` | ServedByDeepSeek treats OpenRouter's deepseek/ slug as DeepSeek's own API on every host | вендор |
| P0 | m2 | medium | `llms/openai/structured_output.go:235` | TakesNoJSONSchema ignores the host |  |
| P0 | m5 | medium | `llms/openai/openaillm.go:311` | Qwen/DashScope rules applied on any host |  |
| P0 | m6 | medium | `llms/reasoning/effort_wire.go:12` | RejectsPenalties ignores the host |  |
| P1 | 37 | low | `llms/reasoning/effort_wire.go:212` | The dashscope/ exemption in budgetHasNoField also lets MiniMax guests through, which DashScope gives no budget |  |

## B. Неполные таблицы моделей — P0 2 · P1 4 · P2 3

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | 66 | medium | `llms/googleai/googleai.go:1143` | Thinking config dropped for gemini-pro-latest and other thinking names the table misses |  |
| P0 | 72 | low | `llms/reasoning/gemini_capability.go:49` | Gemini 3 image models lose their thinking config, and the door warns that they do not think | вендор |
| P1 | 4 | medium | `llms/reasoning/claude_capability.go:89` | Claude Opus 5.5 is classified as disablable (Opus 5 rules), but the vendor documents its thinking as always on | pre-existing, вендор |
| P1 | 7 | medium | `llms/reasoning/claude_capability.go:129` | Bedrock effort gate removes xhigh and max from Fable 5.1 / Mythos 5.1 although their AWS model cards list them | вендор |
| P1 | m9 | low | `llms/reasoning/claude_capability.go:285` | noPrefillClaude misses the -latest aliases |  |
| P1 | m10 | low | `llms/reasoning_support.go:115` | OpenAI effort caps not limited to ProviderOpenAI |  |
| P2 | 5 | medium | `llms/reasoning/qwen_capability.go:46` | qwenFlagThinkers matches exact names only, so DashScope -latest and dated snapshots never get enable_thinking or thinking_budget | pre-existing, вендор |
| P2 | 11 | low | `llms/reasoning/openai_capability.go:116` | glm-latest and glm-flash-latest skip the GLM 5.3 closed effort enum, although other tables treat them as 5.3 | pre-existing, вендор |
| P2 | 13 | low | `llms/reasoning/off.go:97` | ResolveOff on Google sends a thinking config to TTS and image surfaces that GeminiSupportsThinking excludes | pre-existing |

## C. googleai/vertex после перехода на GenAI SDK — P0 6 · P1 3 · P2 2

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | m8 | high | `llms/googleai/googleai.go:1096` | Penalties forwarded to every Gemini model | вендор |
| P0 | 99 | medium | `llms/googleai/new.go:71` | googleai.New fails when an API key and a credentials file/JSON are both present, and the error text contains the API key |  |
| P0 | 65 | medium | `llms/googleai/new.go:73` | WithEndpoint in gRPC host:port form now breaks every request |  |
| P0 | 71 | medium | `llms/googleai/new.go:64` | Vertex silently ignores caller ClientOptions (token source, API key) and authenticates with ADC instead |  |
| P0 | 103 | medium | `llms/googleai/vertex/vertex.go:48` | vertex.New refuses a missing location that the base and both SDKs resolve (default region, GOOGLE_CLOUD_REGION) |  |
| P0 | 68 | medium | `llms/googleai/vertex/vertex.go:16` | vertex package drops exported error vars and constants, so callers no longer compile |  |
| P1 | 102 | medium | `llms/googleai/vertex/vertex.go:28` | vertex.New with WithHTTPClient silently drops the caller's credentials and ADC |  |
| P1 | 73 | low | `llms/googleai/models.go:20` | ListModels on the Vertex backend returns publisher-prefixed ids, not the model names callers pass |  |
| P1 | 105 | low | `docs/docs/getting-started/guide-gemini.mdx:67` | Gemini guide still says Vertex cannot turn thinking off on default-thinking models |  |
| P2 | 75 | low | `llms/googleai/option.go:144` | WithGRPCClient and WithGRPCConn docs still advertise options that now make New fail | pre-existing |
| P2 | 25 | low | `llms/googleai/usage_stream_test.go:50` | "Counters add up" test uses a fixture without thoughts; on a thinking response the totals do not add up | pre-existing |

## D. tool_choice и аргументы инструментов — P0 7 · P1 1 · P2 0

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | 32 | medium | `llms/turn.go:65` | A named tool_choice whose function value is not map[string]any is rewritten to "required" |  |
| P0 | 14 | medium | `llms/anthropic/anthropicllm.go:1060` | Tool-choice translation drops disable_parallel_tool_use from a map choice |  |
| P0 | 55 | medium | `llms/bedrock/internal/bedrockclient/bedrockclient_converse.go:774` | Converse now forwards a forced tool choice (any/named) to model families Bedrock rejects it for | вендор |
| P0 | 58 | medium | `llms/bedrock/internal/bedrockclient/provider_anthropic.go:260` | Legacy Claude door sends tool_choice even when the request has no tools | вендор |
| P0 | 70 | medium | `llms/googleai/googleai.go:154` | toolConfig sent even when the call has no tools | вендор |
| P0 | m4 | medium | `internal/toolcall/arguments.go:30` | Tool-call arguments `null` are rejected | вендор |
| P0 | m11 | medium | `llms/anthropic/anthropicllm.go:922` | Tool message with extra parts fails |  |
| P1 | 80 | low | `internal/toolcall/arguments.go:50` | DecodeFields silently accepts and discards data after the arguments object |  |

## E. Стриминг, усечение, частичные ответы — P0 0 · P1 5 · P2 2

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P1 | 78 | medium | `llms/stopreason.go:11` | IsTruncated misses model_context_window_exceeded, so truncated Claude answers pass as complete | вендор |
| P1 | 21 | medium | `llms/anthropic/anthropicllm.go:335` | Partial stream response contains a made-up "{}" tool call when the stream breaks mid tool_use |  |
| P1 | 17 | medium | `llms/anthropic/anthropicllm.go:448` | InferenceSpeed (and ServiceTier) are empty on every streamed response |  |
| P1 | 24 | medium | `llms/googleai/googleai.go:1230` | Empty-stream test only covers a tiny max_tokens; with the default limit every empty stream is reported as token_limit |  |
| P1 | 22 | low | `llms/anthropic/internal/anthropicclient/completions.go:121` | Legacy completions stream leaks one goroutine each time a consumer aborts |  |
| P2 | 40 | medium | `llms/openai/internal/openaiclient/chat.go:873` | Cancelled or timed-out stream returns a truncated answer with a nil error about half the time | pre-existing |
| P2 | 23 | low | `llms/anthropic/internal/anthropicclient/messages.go:451` | Stream that ends at EOF without message_stop still drops delivered text and reports "no response" | pre-existing |

## F. Опции и предупреждения; клиент anthropic — P0 3 · P1 7 · P2 4

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | 76 | medium | `llms/options.go:184` | ValidateReasoning refuses effort "none", which the base forwarded and OpenAI documents |  |
| P0 | 38 | medium | `llms/openai/openaillm.go:471` | Delegated adaptive reasoning on Claude is dropped with no warning when no adaptive object is sent |  |
| P0 | 41 | medium | `llms/openai/internal/openaiclient/chat.go:808` | WithExtraBody deep-merges two json_schema response formats into one invalid schema |  |
| P1 | 35 | medium | `llms/openai/openaillm.go:299` | WithLogProbs/WithTopLogProbs are sent but the logprobs are never returned; top_logprobs goes out without logprobs | вендор |
| P1 | 18 | medium | `llms/anthropic/internal/anthropicclient/federation.go:101` | Federated token with a 60s lifetime is never cached: every request re-exchanges | вендор |
| P1 | 16 | low | `llms/anthropic/anthropicllm.go:967` | Fast-mode beta header silently replaces client-level beta headers |  |
| P1 | 81 | low | `llms/warning.go:93` | WithInferenceSpeed is missing from the unread-options catalogue, so every door except anthropic drops it silently |  |
| P1 | 51 | low | `llms/ollama/warnings.go:74` | Token-budget warning says nothing was sent while a level derived from the budget is on the wire |  |
| P1 | 53 | low | `llms/huggingface/warnings.go:74` | HuggingFace extra-body warning blames a vendor SDK the door does not use |  |
| P1 | 82 | low | `llms/structuredoutput/validator.go:27` | Compile's doc says the schema is used verbatim, but admitNullable now rewrites type |  |
| P2 | 36 | medium | `llms/openai/openaillm.go:408` | The effort-with-tools rule reads only opts.Tools, not opts.Functions that also become tools | pre-existing, вендор |
| P2 | 20 | nit | `llms/anthropic/sampling_warnings.go:152` | ExtraBody drop warning names a vendor SDK the Anthropic door does not use |  |
| P2 | 83 | nit | `llms/options.go:294` | InferenceSpeed doc points to Speed "above", but Speed is declared below |  |
| P2 | m14 | nit | `llms/options.go:130` | GetEffort doc comment attached to the wrong method |  |

## G. Bedrock — P0 2 · P1 1 · P2 2

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | 54 | medium | `llms/bedrock/internal/bedrockclient/bedrockclient_util.go:115` | Structured output refused on Converse for application inference profile and provisioned-model ARNs |  |
| P0 | 56 | medium | `llms/bedrock/internal/bedrockclient/provider_nova.go:180` | Legacy Nova door turns an adaptive request with no effort into top effort and drops the caller's maxTokens/temperature/topP |  |
| P1 | 59 | low | `llms/bedrock/internal/bedrockclient/provider_ai21.go:360` | Raw input_tokens/output_tokens stay int32 on four legacy families while the Claude doors now report int |  |
| P2 | 57 | low | `llms/bedrock/internal/bedrockclient/bedrockclient_converse.go:217` | Converse drops top_p=0.95 under budget thinking because the floor is compared after a float32 round trip | pre-existing |
| P2 | 60 | nit | `llms/bedrock/internal/bedrockclient/bedrockclient_converse.go:760` | Two doc comments sit on the wrong functions after the refactor |  |

## H. Ollama, Hugging Face, Mistral — P0 3 · P1 0 · P2 2

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | m1 | high | `llms/ollama/ollamallm.go:350` | `think` is sent to models that cannot think (Ollama 400) |  |
| P0 | 46 | medium | `llms/ollama/ollamallm.go:350` | String think levels are refused by Ollama servers up to v0.17 for every model except gpt-oss | вендор |
| P0 | 49 | medium | `llms/huggingface/internal/huggingfaceclient/embeddings.go:31` | Embeddings put an extra /hf-inference segment on a caller URL that already names the provider | вендор |
| P2 | 48 | low | `llms/huggingface/internal/huggingfaceclient/inference.go:48` | Provider-prefixed chat URL is wrong for groq, novita, fireworks-ai and hf-inference | pre-existing, вендор |
| P2 | 50 | low | `llms/mistral/warnings.go:34` | Mistral and HuggingFace accept WithStructuredOutput, only warn, and return unvalidated output with a nil error | pre-existing |

## I. pgvector, pinecone, эмбеддинги, tools — P0 4 · P1 4 · P2 4

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P0 | 87 | medium | `embeddings/jina/options.go:87` | jina WithBatchSize(0) or a negative value now panics |  |
| P0 | 86 | medium | `vectorstores/pinecone/pinecone.go:217` | Pinecone results re-sorted by descending score, which reverses euclidean indexes | вендор |
| P0 | m3 | medium | `vectorstores/pgvector/pgvector.go:490` | Filter keys must be bare identifiers |  |
| P0 | 90 | low | `vectorstores/pgvector/pgvector.go:343` | SimilaritySearch now depends on an embedding-table `uuid` column, so it fails on tables created by Python langchain_postgres |  |
| P1 | 84 | medium | `vectorstores/pgvector/metadata_index.go:136` | Exclude partial indexes are never used by the store's own queries |  |
| P1 | 85 | medium | `vectorstores/pgvector/metadata_index.go:191` | ANALYZE in init fails New on mixed-dimension tables with pgvector < 0.7 |  |
| P1 | 92 | medium | `vectorstores/pgvector/metadata_index.go:160` | Every store start holds a ShareLock on the embeddings table across ANALYZE, blocking all writers; the doc calls later starts 'a catalog lookup' |  |
| P1 | m12 | low | `vectorstores/pgvector/metadata_index.go:191` | ANALYZE on every store start |  |
| P2 | 95 | medium | `embeddings/bedrock/bedrock.go:53` | Cohere batches on Bedrock use the default batch size 512, above the vendor's 96-texts-per-call limit (pre-existing) | pre-existing, вендор |
| P2 | 89 | medium | `tools/perplexity/perplexity.go:25` | Perplexity tool: two constants removed (compile break); the four kept were retired 2026-09-27 | pre-existing, вендор |
| P2 | 91 | low | `vectorstores/pgvector/pgvector.go:350` | The dimension guard is written before the metadata filters, so every candidate vector is detoasted before a cheap filter can reject its row; the WithMetadataIndexes doc describes the wrong mechanism | pre-existing |
| P2 | 88 | low | `vectorstores/pgvector/pgvector.go:502` | Float filter values in exponent form never match stored JSON numbers | pre-existing |

## J. Тесты, CI, примеры, кассеты — P0 0 · P1 9 · P2 6

| Приоритет | № | Серьёзность | Где | Суть | Флаги |
|---|---|---|---|---|---|
| P1 | 26 | medium | `vectorstores/pgvector/narrowing_test.go:67` | The seven new pgvector integration tests always skip in CI |  |
| P1 | 52 | medium | `llms/ollama/ollama_cloud_test.go:50` | Cloud cassette tests skip whenever there is no ~/.ollama/id_ed25519, although any throwaway key replays them |  |
| P1 | 42 | medium | `llms/openai/reasoning_wire_test.go:379` | Grok is pinned to the deprecated max_tokens field, and the xAI cassettes that showed otherwise were rewritten by hand | вендор |
| P1 | 43 | low | `llms/openai/reasoning_wire_test.go:193` | The positive check in TestABudgetLargerThanTheAnswerLimitRaisesTheLimit matches the budget, not the answer limit |  |
| P1 | 44 | low | `llms/openai/sampling_warnings_test.go:558` | The 'reasons anyway' silence test uses gpt-4o, a model that never reasons, and pins a silent loss |  |
| P1 | 28 | low | `llms/reasoning_sampling_hint_test.go:30` | Sampling-hint tests compare the hint with the same function it is computed from |  |
| P1 | 61 | low | `llms/bedrock/internal/bedrockclient/nova_reasoning_only_test.go:27` | TestAnEmptyAnswerIsStillAnError never checks for an error |  |
| P1 | 62 | low | `llms/bedrock/converse_stream_usage_test.go:22` | Converse streamed-usage fixture sends a totalTokens the vendor never sends, and the test pins it |  |
| P1 | 97 | low | `internal/httprr/gateway_headers_test.go:56` | Recording sweeps skip .httprr.gz; two compressed cassettes still carry a real OpenAI project id | pre-existing |
| P2 | 63 | low | `llms/bedrock/internal/bedrockclient/bedrockclient_integration_test.go:432` | Integration tests the PR updated still test local copies, not the production parsers | pre-existing |
| P2 | 30 | low | `llms/count_tokens_test.go:18` | TestCountTokens still downloads a tiktoken encoding and fails offline | pre-existing |
| P2 | 98 | low | `llms/bedrock/bedrockllm_test.go:54` | Bedrock replay is not hermetic against AWS_CA_BUNDLE or AWS_PROFILE; tests fail spuriously | pre-existing |
| P2 | 93 | low | `examples/json-mode-example/json_mode_example.go:47` | The json-mode example's googleai backend still requests the shut-down gemini-1.5-flash, two lines below the fix for the retired Claude model | pre-existing, вендор |
| P2 | 94 | low | `examples/anthropic-extended-capabilities/README.md:29` | The extended-capabilities example was migrated to WithReasoning but still runs the retired claude-3-7-sonnet-20250219 | pre-existing, вендор |
| P2 | 45 | nit | `llms/openai/reasoning_effort_passthrough_test.go:74` | Stale comment states the opposite of the renamed test's assertion |  |
