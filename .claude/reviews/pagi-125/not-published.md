# Что не попало в PR

Заметки агентов второго ревью: что проверено и сознательно не опубликовано (намеренные изменения, проблемы вне строк diff'а, мелочи), 11 отклонённых проверкой замечаний и причины, по которым три замечания первого ревью признаны ложными. Эти пункты никто отдельно не перепроверял — это кандидаты в задачи, а не подтверждённые дефекты.

## Отклонённые проверкой замечания (11)

### `llms/reasoning_support.go:95` — Hint reports default-on Claude as disablable on ProviderOpenAI while the openai door refuses the off on OpenRouter

Суть: For a default-on Claude on ProviderOpenAI, CannotDisable comes from ResolveOff, which answers OffDisableThinkingObject for an anthropic/ route. The openai door refuses that off on public hosts (sendsClaudeThinkingObject needs !publicProviderHost), so the hint and the door disagree on OpenRouter.

Почему отклонено: The behaviour is real, but it is neither a defect nor a regression, so it does not need an inline comment. At HEAD, `ReasoningSupportFor("anthropic/claude-sonnet-5", ProviderOpenAI)` calls `ResolveOff`, which returns `OffDisableThinkingObject` (off.go:69-75), so `CannotDisable` is false. On an OpenRouter base URL, the door's `setReasoningOff` (openaillm.go:605-607) refuses the off because `sendsClaudeThinkingObject` requires `!publicProviderHost`. The disagreement comes from the API's design. `ReasoningSupportFor(model, provider)` has no host parameter. The same `anthropic/…` name on `ProviderOpenAI` is served both by LiteLLM-style private gateways, where the door does send `thinking:{type:disabled}`, and by OpenRouter, where it refuses. So the hint is correct for one host class and cannot tell the two apart. The type is documented as a best-effort hint that is "never authoritative". The PR made the part it can see from the name route-aware on purpose: `ClaudeThinkingObjectRoute` in `ResolveOff` gives `CannotDisable` for deepinfra/, openrouter/ and similar prefixes. When the hint is wrong, the caller gets a typed `ErrReasoningOffUnsupported` before any request is sent, not silent misbehaviour. BASE was worse on the same path. I checked BASE: its hint also said `CannotDisable=false` (`ResolveOff` returned `OffDisableClaude` for default-on Claude on `ProviderOpenAI`), but its `setReasoningOff` handled only `OffUnsupported` and `OffEffortNone`. The off was therefore a silent no-op on every host, and the model kept thinking. The PR makes this path strictly better. Fixing the remaining gap would need a host-aware hint API, which is a feature request, not a release blocker.

### `llms/reasoning/claude_capability.go:274` — Exported ClaudeSupportsEffortWithBudget changed signature (added Provider)

Суть: The public function reasoning.ClaudeSupportsEffortWithBudget(model string) became ClaudeSupportsEffortWithBudget(model string, p Provider). It is the only exported signature change in the package, and no README or doc in the diff mentions it.

Почему отклонено: Not a defect worth an inline comment. The signature change is real: BASE claude_capability.go:107 has ClaudeSupportsEffortWithBudget(model string), and HEAD :274 has ClaudeSupportsEffortWithBudget(model string, p Provider). Commit d07f16d made the change on purpose. Its measurements show that Opus 4.5 accepts output_config.effort with a thinking budget on the first-party API but gets a 400 on Bedrock, so a predicate that looks only at the model name cannot be right for both providers. All three in-repo callers (anthropicllm.go:215, bedrockclient_converse.go:211, provider_anthropic.go:865) pass the provider explicitly. The break shows up at compile time, and the fix for a caller is obvious. The release rule is about not breaking callers silently, and this break is not silent. The module is github.com/vxcontrol/langchaingo with no /vN suffix, so it is v0 semver with no stability promise. The function first appeared in the previous release (1fc1299, 2026-07-17). The package doc describes it as a low-level capability helper shared by the provider adapters, and the high-level public entry point for applications is llms.ReasoningSupportFor. A repo-wide comparison of exported signatures shows it is the only change here that breaks existing call sites; the other changes are moves, type aliases or added variadic parameters. At most it deserves a line in the release notes. It does not need a code change.

### `llms/reasoning_support.go:28` — Efforts doc says empty means unknown or no effort, but Gemini 3 is Known, takes levels and has empty Efforts

Суть: The new doc on ReasoningSupport.Efforts says it is empty only when the model is unknown or takes no effort. The Google branch returns Known=true, Mechanism=Adaptive and Efforts=nil for Gemini 3.x, whose door maps minimal, low, medium and high to thinking_level.

Почему отклонено: This is at most a wording gap in the docs, and the failure scenario does not follow from them. The facts check out: at HEAD the Google branch returns Known=true, Mechanism=Adaptive and Efforts=nil for Gemini 3.x (reasoning_support.go:139-153), and the googleai door maps minimal, low, medium and high to thinking_level (googleai.go:1155-1157, 1243-1258).

The documented contract does not tell a UI to hide the effort selector:
- The field doc (lines 28-29) says to read Mechanism before offering a control. The Mechanism doc (lines 31-32) defines Adaptive as taking an effort level, so a UI that follows the docs offers an effort control for Gemini 3.
- The type-level doc (lines 14-15) already covers this case: 'unknown effort tiers and default state are left nil rather than guessed'. The Google tiers are left unclassified on purpose, as the branch comment says ('Efforts stays unset: this package does not classify the Google level names').

Commit 53c3215 wrote the field doc to warn that budget-only models also report empty Efforts. The field doc's list of empty cases leaves out 'Known level model whose tiers are not listed', but read together with the type doc and the Mechanism doc the contract holds. The behaviour is also preexisting: BASE returned Known=true with nil Efforts for Gemini, and the PR changes neither the wire path nor what Gemini 3 reports in Efforts. A comment here would only suggest a wording tweak and is not worth a release-review inline comment.

### `llms/anthropic/anthropicllm.go:235` — Anthropic door refuses forced tool choice plus reasoning for non-Claude models, unlike the shared guard

Суть: The door's inline check refuses any model whose thinking resolved to "enabled". Every model the door does not classify, including non-Claude models behind Anthropic-compatible endpoints, resolves to budget thinking, so all of them are refused locally. The shared llms.CheckClaudeTurnLimits, and the Bedrock door after commit cfd6ad0, only apply the rule to Claude thinking generations. The two copies of the rule already disagree.

Почему отклонено: The mechanism is real, but it does not break anything. For any model name it does not classify, the Anthropic door sends thinking {type:"enabled", budget_tokens} on the wire (anthropicllm.go ~207-211). The check at line 235 keys off that exact payload, so it follows the rule for what is actually sent.

Commit cfd6ad0 gated only the Bedrock door, and its message gives the reason: "no thinking payload is ever sent for them" on Bedrock. The shared CheckClaudeTurnLimitsOnWire is used by doors (Bedrock, OpenAI) that send no Anthropic thinking payload for non-Claude families. So the two copies of the rule differ because the doors send different requests, not because one of them is wrong.

Vendor evidence goes against the failure scenario for every example model:
- DeepSeek returns HTTP 400 "Thinking mode does not support this tool_choice" for forced or named tool_choice in thinking mode (deepseek-ai/DeepSeek-V3#1376, langchain#40769, any-llm#1384).
- Kimi docs: with thinking enabled, tool_choice can only be "auto" or "none".
- MiniMax's Anthropic-compatible API supports only auto and none for tool_choice.

The BASE request would have come back 400 from the vendor anyway. HEAD only turns that round trip into a typed local error. Any endpoint where the refusal is actually wrong would be a speculative proxy case with no evidence behind it.

### `llms/anthropic/anthropicllm.go:235` — Forced-tool guard misses Opus 5.5, Fable 5.1 and Mythos 5.1, which reject forced tool use on every request

Суть: The new guard refuses a forced tool choice only under manual (budget) thinking. Anthropic documents that Claude Opus 5.5, Fable 5.1 and Mythos 5.1 reject tool_choice any/tool on every request, so these calls still go out and come back as a vendor 400 instead of the typed error.

Почему отклонено: The vendor claim is accurate. Anthropic's docs, and the bundled claude-api reference, say Claude Opus 5.5, Fable 5.1 and Mythos 5.1 return 400 for tool_choice any/tool on every request. The guard at anthropicllm.go:235 does not cover them, so those calls reach the vendor. It is still not a release defect worth an inline comment, for four reasons.
(1) It is not a regression. BASE had no guard, and it sent tool_choice as the bare string "any", a shape bce683d fixed as invalid ("send the tool choice in the shape Messages accepts"). So forced tool choice did not work at BASE for any model.
(2) The failure is loud, not silent. The caller gets the vendor's 400 ("tool_choice: type \"tool\" and \"any\" are not supported for this model"), only not the local typed error.
(3) The guard does what it says. 272ae0b scopes ErrForcedToolUseWithThinking to the manual-thinking rule, a different, mode-dependent restriction; a model-wide ban would need its own error. The Bedrock twin in llms/turn.go is scoped the same way.
(4) The maintainers weighed this exact vendor statement and declined it on purpose. Commit 4d72434 quotes it and says Fable 5.1 and Mythos 5.1 "have no route here, so the restriction cannot be checked and must not be guessed onto the family". It also pins TestFableTakesAForcedToolChoiceWithoutThinking, because claude-fable-5 accepts forced tool choice and a family-wide gate would break it.
Opus 5.5 appears only as a name in a reasoning-detection test and is otherwise unclassified. Adding it would be a follow-up enhancement, not a blocker for this release.

### `vectorstores/inmemory/inmemory_test.go:108` — In-memory result-ordering fix has no test; the existing test tolerates unsorted results

Суть: inmemory.SimilaritySearch now sorts results by score instead of reversing the hnsw heap. The change is in a commit titled refactor(tests) and has no test pinning it. TestMockSimilarityScoreCalculation still searches for the highest score itself, so it passes with the old unsorted output and would not notice the bug coming back.

Почему отклонено: The central claim is false: removing the sort would not pass every test in the package. I ran a mutation through an overlay, replacing slices.SortStableFunc at inmemory.go:181 with the old slices.Reverse. Two tests fail with an httprr cassette mismatch: TestInMemoryAsRetrieverWithScoreThreshold and TestInMemoryAsRetrieverWithMetadataFilterNotSelected. Their recorded chat/completions request bodies pin the order in which retrieved documents are put into the RetrievalQA prompt. An unsorted result changes the request body, and replay fails. The unmodified package passes. The ordering is therefore pinned, if indirectly. That TestMockSimilarityScoreCalculation itself tolerates unsorted output is at most a nice-to-have for an explicit unit test, not an unguarded regression.

### `llms/mistral/mistralembed.go:72` — WithMaxRetries(0) means one attempt for embeddings but five for chat on the same Model

Суть: The new embeddings retry loop treats maxRetries as a total attempt count and turns 0 into 1. The chat SDK turns 0 into its default of 5, and the option's doc says 'maximum number of retries'. So one setting gives different retry behaviour on the two surfaces of one Model, and WithMaxRetries(1) allows no retry at all on embeddings.

Почему отклонено: Only half the finding holds up, and that half is not a defect worth a comment.

The WithMaxRetries(1) claim is wrong. mistral-go v1.1.0 client.go loops `for i := 0; i < c.maxRetries; i++`, so the SDK also treats the value as a total attempt count. WithMaxRetries(1) gives one attempt on chat, on embeddings, and at BASE. There is no divergence. The "maximum number of retries" doc wording was already wrong before this PR, for both surfaces.

The only real difference is an explicit WithMaxRetries(0). Embeddings now make 1 attempt (mistralembed.go:72-75). Chat makes 5, because NewMistralClient maps 0 to DefaultMaxRetries. The default is unaffected: New() sets maxRetries to sdk.DefaultMaxRetries (mistralmodel.go:35), so callers who never set the option get 5 on both surfaces.

The 0 -> 1 mapping is deliberate. Commit 55c4161 added TestZeroRetriesStillSendsTheRequestOnce for it, and it matches the literal meaning of "zero retries". The chat side's 0 -> 5 comes from the vendor SDK and predates this PR. Copying it would send 5 requests to a caller who asked for none.

The wire impact vs BASE is also small. The SDK sends the same *http.Request again after its body has been read, so a 503 that carries a JSON body (the normal Mistral case) got 1 real request plus local "ContentLength=N with Body length 0" failures. That is what the reproduce verifier observed. At BASE, SDK-routed embeddings in practice did not retry either.

At most this is a cosmetic inconsistency caused by a pre-existing SDK quirk, not a release regression.

### `llms/bedrock/internal/bedrockclient/bedrockclient_converse.go:670` — Converse door now fails any BinaryPart that is not one of four image types, including text/plain

Суть: convertUserOrAssistantMessage treats every BinaryContent (message Type "image") as an image and returns ErrUnsupportedImageFormat when the MIME type is not jpeg/png/webp/gif. A text/plain part that the base Converse door sent as a text block now fails before the request. The sentinel lives in an internal package, so callers cannot match it with errors.Is.

Почему отклонено: The probes are right that the behaviour changed. At BASE, WithConverseAPI sent a text/plain BinaryPart as a text block; HEAD refuses it before the network with ErrUnsupportedImageFormat. It is still not a defect worth an inline comment:
(1) BASE only handled text/plain because of a bug. The BASE Converse converter never read msg.Type, so it sent every BinaryContent's raw bytes as a text block. PNG and JPEG arrived as replacement-rune garbage and PDFs as mojibake. Commit 132377f (PAGI-256) fixes that and says why: "A mime type Converse has no format for now fails before the network instead of travelling as text."
(2) Across the library, a BinaryPart means an image. The bedrock README says so (BinaryPart -> Message{Type:"image"}), and bedrockllm.go:249-255 maps every BinaryContent to Type "image". The legacy Bedrock Anthropic and Nova paths and the direct anthropic door (anthropicllm.go:793) send it as an image block, the openai door sends it as image_url, and ollama puts it in images. On the default Bedrock door, a text/plain BinaryPart got a vendor 400 at BASE. HEAD's ab2eae3 refuses it locally with the same allow-list and sentinel, so Converse and legacy now agree. Text belongs in llms.TextPart.
(3) The new failure is loud and happens before the request, not a silent break. It replaces silently sending corrupted bytes for every non-text binary input.
(4) The rest is cosmetic or was already there. The error text says "image" for a text MIME. The sentinel sits in an internal package, but package bedrock re-exports no sentinels at all, so this is not a break from an existing pattern. The WithConverseAPI "documents" claim was not changed by this PR, and it was not true at BASE either.

### `llms/bedrock/internal/bedrockclient/bedrockclient_converse_test.go:1150` — Saturation table breaks the bedrockclient test build on 32-bit targets

Суть: TestConverseMaxTokensSaturatesInsteadOfWrapping puts the untyped constants math.MaxInt32+1 and 3_000_000_000 into an `int` struct field. That field overflows where int is 32 bits, so the whole bedrockclient test package stops compiling on 386 and arm.

Почему отклонено: The overflow is real. On HEAD, `GOARCH=386 go vet ./llms/bedrock/internal/bedrockclient/` fails at bedrockclient_converse_test.go:1150:25 (`math.MaxInt32 + 1` into an `int` field), and the same package is clean on BASE. It does not harm anyone who uses the library, though. It sits only in a _test.go file, which importers never compile, and `GOARCH=386 go build ./llms/bedrock/...` on HEAD succeeds, so the production code still builds on 32-bit. 32-bit has never been a target of this repo either. CI (.github/workflows/ci.yaml) runs only ubuntu-latest amd64. BASE already fails to compile on 386 in production code: `GOARCH=386 go vet ./vectorstores/pgvector/` on BASE reports overflows at pgvector.go:144/161/184, where the int64 advisory-lock constants are passed as int. HEAD also adds the same 64-bit-only test pattern in three other new tests: internal/numutil/num_test.go:18, llms/googleai/googleai_unit_test.go:1148 and llms/openai/extra_body_merge_test.go:59. This spot is one case of a repo-wide 64-bit assumption, not a release regression. An inline comment here would be cosmetic, so I am dropping it.

### `internal/httprr/rr.go:1189` — Response scrubber strips only X-Litellm-* and OpenAI org/project; LiteLLM llm_provider-* and other vendors' org/project ids are recorded

Суть: The default response scrubber, extended in this PR to mask Openai-Project and drop x-litellm-* headers, still records the same ids under other names: LiteLLM's llm_provider-* copies of upstream headers, Anthropic-Organization-Id and Msh-Project-Id.

Почему отклонено: The recorder behaviour is real: the default response scrubber leaves Llm_provider-* and Anthropic-Organization-Id headers in place, as both probes show. The failure scenario still does not come from this PR.

1. The only gateway workflow the PR documents is Gemini through a LiteLLM pass-through path (GOOGLE_BASE_URL=.../gemini). Pass-through routes forward upstream headers unprefixed. The pre-scrub Bedrock gateway cassette (93680bd^:llms/bedrock/testdata/TestLLM.httprr) confirms this: it has X-Amzn-* and X-Litellm-* headers and no llm_provider-* ones. No cassette in HEAD has an llm_provider-* header either. The llm_provider- prefix only appears on LiteLLM's OpenAI-compatible routes, and no documented recording path uses them. Gemini does not return org or project ids at all. If OpenAI's openai-organization or openai-project arrived unprefixed, the scrubber would mask them, and the PR extended exactly that masking.
2. The concrete leaks are pre-existing and untouched by the PR. The 32 cassettes with Anthropic-Organization-Id are unchanged between da7016e and 355dc71, and so is TestMoonshot_MultiTurnToolCallWithReasoning.httprr (same blob bcdd34b). The BASE scrubber did not handle these headers either.
3. These values are account/org identifiers, not credentials.

A general hardening follow-up (mask *-organization-id / *-project-id and llm_provider-* headers) is optional. It is not a regression or a defect in this release worth an inline comment.

### `testing/llmtest/llmtest.go:644` — Exported MockLLM.GenerateContentStream removed from the public llmtest package

Суть: testing/llmtest is an exported package. The PR deletes the exported method (*MockLLM).GenerateContentStream, so any downstream test that streamed through the mock no longer compiles.

Почему отклонено: The removal is real. The PR diff deletes (*MockLLM).GenerateContentStream, and in HEAD MockLLM has only Call and GenerateContent (llmtest.go:598-640). But it is a deliberate API cleanup, not a defect. Commit f9ca9d5 removed it because no interface in the library declares GenerateContentStream, so no library code could ever consume it. The suite's streaming detection built on that method was dead code for every real door. The mock now streams the way real doors do, through llms.WithStreamingFunc in GenerateContent, and doc.go was rewritten to match. The only affected caller is downstream test code that calls this mock-only method directly, which is very unlikely. That break is loud, a compile error, while the release criterion is not to break callers silently. The same release also makes other deliberate compile-breaking changes to exported APIs, which I checked: reasoning.ClaudeSupportsEffortWithBudget went from (model string) to (model string, p Provider). So exported source compatibility is not held as a release constraint here, and removing a method from a test-helper mock deserves at most a release-notes line, not an inline review comment.

## Ложные замечания первого ревью (3)

### #7 `llms/googleai/option.go:52` — default max output tokens raised to 16384

This is a false positive, so there is nothing to post. The change is deliberate and fixes a real bug. Commit 476f3a6 ("stop truncating answers at 2048 tokens") moved googleai onto the library-wide llms.DefaultMaxTokens (16384), which already existed at BASE and is the default for the other doors. The old 2048 default cut Gemini answers off with finishReason MAX_TOKENS. For thinking models the cap also has to cover thought tokens, so 2048 was actively harmful there. Sixteen cassettes were re-recorded against gemini-2.5-* and 3.x models and all returned 200. BASE also always sent a default maxOutputTokens (2048) on both the googleai and the vertex doors, so the only new thing is the value. The finding's failure scenario needs gemini-2.0-flash or flash-lite, the 8k-output models. Google shut those down on 2026-06-01, per litellm PR #42850 and the Vertex and Gemini API retirement notices. Before today (2026-09-28), that request already fails no matter what maxOutputTokens says. Every Gemini model still served (2.5-flash, 2.5-pro, 2.5-flash-lite, 3.x) accepts up to 65536 output tokens, well above 16384. The vendor error "supported range is from 1 (inclusive) to 8193 (exclusive)" is real, but no live model has been shown to trigger it at 16384. The only leftover case is Gemma served through the Gemini API. Nothing shows that it rejects 16384; third-party listings even give gemma-3-27b-it a 16384 output limit. That case is also not what the finding claims. A raised cap does not raise cost unless an answer would otherwise have been truncated, and truncation is exactly the bug the change fixes. Callers who want a lower cap still have WithMaxTokens and WithDefaultMaxTokens.

### #13 `llms/ollama/ollamallm.go:358` — duplicated gpt-oss mapping

The stated failure cannot happen. resolveThink (llms/ollama/ollamallm.go:338) and reportOllamaThinking (llms/ollama/warnings.go:47) reach gptOSSLevel only when ResolveMode()==ReasoningOn. On that path GetEffort (llms/options.go:141-172) never returns "": ResolveMode gives ReasoningOn either for Mode==ReasoningOn, and GetEffort then falls back to "high", or because Adaptive, Effort or Tokens is set, and each of those yields a non-empty effort. So gptOSSLevel("") is never evaluated. For every effort that can arrive (minimal/low/medium/high/xhigh/max), gptOSSLevel and reasoning.GptOssEffort (gptOssCaps.ClampEffort) return the same value: minimal->low, low->low, medium->medium, and high/xhigh/max->high. The two name checks also agree on Ollama tags. takesOnlyGPTOSSLevels lowercases the name, strips everything up to the last '/' and matches the gpt-oss prefix, which is what modelSpellings does. modelSpellings adds only Bedrock region, platform-dot and fine-tune-wrapper handling, and those forms do not occur in Ollama tags. The helpers also feed different wires: GptOssEffort is used only by Bedrock's reasoning_effort field, while gptOSSLevel builds the Ollama think level. What is left is code duplication with no behaviour difference a caller can hit, so it does not warrant an inline defect comment in a release review.

### #15 `embeddings/jina/options.go:86` — jina comment vs BatchSize

Not a defect. The comment at embeddings/jina/options.go:86 ("_models holds vector dimensions, not batch sizes") is true: 512/768/1024 are the jina-embeddings-v2 small/base/large dimensions. Using those numbers as the default BatchSize is unchanged from BASE, which also did `o.BatchSize = _models[o.Model]` after the option loop. Commit fe380f3 ("fix(jina): keep the batch size the caller asked for", PAGI-237) changed one thing: the map no longer overwrites an explicit WithBatchSize. The unit tests pin 512/768/1024 as defaults on purpose, and TestTheCallersBatchSizeReachesTheWire covers the fix. So the comment and the code do not contradict each other, as the finding says ("either the comment or the code is wrong"). The comment warns about an old quirk the PR kept for compatibility. At most, the comment could also say why the dimension-derived defaults are kept, which is a wording nit and not worth an inline comment. The earlier thread can be closed as a false positive. Separate observation, not part of this finding and not verified with a probe: at HEAD, WithBatchSize(0) sets batchSizeFromCaller, so BatchSize stays 0. embeddings.BatchTexts then computes len(texts)/batchSize (embedding.go:110), which panics with integer divide by zero. BASE overwrote the 0 with the model's value for known models. If callers pass 0 to mean "default", this is a regression worth checking.

## Заметки агентов по частям diff'а

### reasoning

**Первый проход.**

Scope covered: every non-test file in llms/reasoning (name.go, reasoning.go, claude_capability.go, bedrock_capability.go, effort_wire.go, off.go, openai_capability.go, gemini_capability.go, qwen_capability.go, detect_hint.go, blocks.go, errors_claude.go, mechanism.go, doc.go) and llms/reasoning_support.go, all read at HEAD and diffed against BASE.

- **Callers:** traced into the anthropic, openai (setReasoning, setReasoningOff, sampling policy, dropFieldsTheModelTakesNot) and bedrock (converse and InvokeModel) doors.
- **Intent:** read the commit messages for the rules I questioned.
- **Name normalization:** checked with probes for vendor prefixes, region and profile prefixes, @ and date suffixes, dotted versions, -latest aliases, ':free'/':tag' suffixes, ft: wrappers and the vendor-dash form.
- **Probes** (overlay test files under scratchpad/probes/rsn1, rsn2, rsn3), each run in HEAD and, where relevant, in BASE:
  - reasoning package predicate dump
  - openai door request-body capture for OpenRouter, Groq, DashScope, Azure, LiteLLM-alias and api.openai.com
  - anthropic door capture for Opus 5.5

**Vendor documentation:**
- Anthropic docs fetched: models overview, deprecations, effort, thinking-troubleshooting, structured-outputs, Opus 5.5 migration guide. These confirmed that the existing tables are consistent except for Opus 5.5, and that the Opus 4.1 structured-output exclusion is correct.
- AWS, OpenRouter and Groq docs are egress-blocked. Those facts come from web-search summaries, so the related findings are marked vendor_dependent.

**Not reported:**
- Duplicates of earlier findings #2, #5, #6, #9 and #10.
- Deliberate and documented changes: gpt-5.6 without max, kimi-k3 mandatory thinking, Bedrock Claude structured-output allowlist, Bedrock Opus 5 / Sonnet 5 disable, Haiku/Opus 4.6 interleaving.
- Minor items with no concrete harm: bare OpenRouter "claude-sonnet-4" unclassified (model retired first-party), ServedByDeepSeek treating the "deepseek/" OpenRouter slug as DeepSeek's API, the vendor-dash form making mistral.magistral* read as non-reasoning, the RejectsTopK/FixesSampling host-agnostic drops (same family as #6), and the missing mutual-exclusion entry for Opus 4.1 (retired).
- Bedrock openai.gpt-5.x getting MechanismNone: also none on base, and adjacent to #10.

**Not covered in depth:** the internal/cmd snapshot generators (dev-only; skimmed); Gemini and xAI vendor facts against Google/xAI docs, which I did not fetch; test-file review beyond the tests that pin the behaviours above. The Bedrock door tests were not run.

**Второй проход.**

What I covered:
- Every non-test file in the slice, read at HEAD in full (llms/reasoning/*.go and llms/reasoning_support.go) and diffed against BASE.
- Commit messages for the key rules (about 25 commits read in full).
- The exported API surface compared between BASE and HEAD with go doc. The only exported signature change is ClaudeSupportsEffortWithBudget, which the first reviewer already reported.
- A behaviour-diff probe (probes/r2diff) ran about 900 model spellings through both checkouts: every name in models_dev.json with and without provider prefixes, plus OpenRouter, Bedrock, Vertex, Ollama and HF spellings. It compared IsReasoningModel, ClaudeReasoningKindFor and the Claude predicates, OpenAIReasoningCapsFor, the Gemini predicates and ResolveOff for five providers, and every difference was reviewed.
- Door-level probes: openai door (probes/r2door, BASE vs HEAD, host set via base URL with a dialing transport), googleai resolveThinkingConfig (probes/r2gem) and a table probe (probes/r2tables).
- Callers outside the slice: the openai, anthropic, bedrock converse and InvokeModel, googleai and ollama doors.
- Vendor facts checked against current Anthropic docs (thinking, troubleshooting table, effort, extended thinking, structured outputs, models overview). Claude tables for sampling, prefill, always-on/default-on, effort ladders, interleaving and structured output matched the docs, apart from the issues already reported.

Findings I checked and left out:
- Duplicates of earlier or first-review items: Opus 5.5 always-on, Qwen bare names on any host, AcceptsEffortWire and TakesNoThinkingDepth on OpenRouter (including the Qwen/Qwen3-8B off regression), the hint effort branch, and Bedrock top efforts.
- Deliberate and measured changes: gpt-5.6 max clamped to xhigh (models.dev lists max, the author measured a 400), the Grok 4.3 xhigh ladder, deepseek.r1 and gpt-oss off returning unsupported on Bedrock, and the Bedrock structured-output positive list (matches Anthropic docs).
- Not regressions or not real: FixesSampling for Kimi on OpenRouter (same root as the earlier penalties finding), ContentReasoning JSON round-trip, Blocks and Collector, ClaudeMaxTokensForBudget (less aggressive than BASE), and reasoning_effort "none" for opt-in DeepSeek V3.1 on vLLM (vLLM accepts "none").

What I could not do:
- No live vendor calls (no keys).
- openrouter.ai is blocked by the egress proxy, so OpenRouter behaviour rests on web-search snippets of its model pages. That is why the OpenRouter findings are marked vendor_dependent.
- Bash was intermittently refused by the auto-mode classifier, which slowed but did not block the work.
- I did not review every one of the roughly 5,600 lines of slice tests. I read the ones that pin the behaviours above: openai_capability_test, models_dev_drift_test and reasoning_support_test, plus the commit-linked tests.

### anthropic

**Первый проход.**

What I reviewed: the full net diff (da7016e..355dc71) of every non-test file under llms/anthropic/, and the HEAD version of each: anthropicllm.go, anthropicllm_option.go, errors.go, models.go, options.go, sampling_warnings.go, and in internal/anthropicclient anthropicclient.go, auth.go, completions.go, federation.go, messages.go and models.go. I also read the callees outside the slice that the door depends on: llms/turn.go, llms/stopreason.go, llms/warning.go, the llms/options.go reasoning helpers, llms/reasoning/claude_capability.go, blocks.go, off.go and reasoning.go. I read the commit messages for tool choice, forced tool use, prefill and empty-turn handling, and compared BASE behaviour.

Vendor facts were checked against the Anthropic Messages and beta Messages API docs (top_k/temperature/top_p deprecation, tool_choice variants, stop reasons, usage.speed, output_tokens_details), the WIF reference and authentication pages (exchange endpoint /v1/oauth/token and Bearer use were confirmed correct), and the claude-api skill (thinking and effort per model, fast mode, Opus 5.5).

Probes: everything ran through a go overlay with a fake HTTP Doer, because local ports were exhausted by other reviewers and httptest could not bind. Probe files are in /tmp/claude-0/-home-user-langchaingo/190be914-a4ca-51a0-89e2-3fbdbb6fc0c9/scratchpad/probes/anth-a1/. `go test ./llms/anthropic/...` passes on HEAD. No t.Skip was added in the anthropic tests.

Checked and judged not defects (deliberate or correct): these were checked and cleared:
- empty end_turn answers are no longer ErrEmptyResponse (commit 20acb40);
- a refusal now returns the response alongside the error;
- all text blocks are concatenated;
- max_tokens is no longer raised to 2x the budget (b89ada1/3c93a54);
- ErrEffortHasNoBudget, ErrForcedToolUseWithThinking and the prefill refusal;
- top_k is now carried (top_k minimum is 0 per the docs);
- redacted_thinking parsing and replay, and thinking-block placement via GroupByToolCalls;
- UseNumber decoding of tool arguments;
- partial responses on stream errors;
- the ErrModelRefusal alias (fields and message format compatible);
- the federation exchange shape;
- ListModels capabilities mapping.

Pre-existing issues not reported because the PR did not introduce them:
- the stream goroutine can leak after an SSE error event, or after the consumer stops, when the body closes first;
- a message_stop pointer race if events follow it;
- tool arguments "null" become input null;
- client beta headers were already dropped by the auto interleaved-thinking header;
- a c.baseURL write race when baseURL is empty.

Also not reported, as not regressions:
- adaptive thinking with forced tool_choice is not refused locally, including Opus 5.5 and Fable 5.1, whose forced-choice 400 is documented;
- the forced-tool check on this door is not model-gated, unlike CheckClaudeTurnLimits for Bedrock, and only matters for Anthropic-compatible third-party endpoints (vendor-dependent);
- the legacy completions path does not warn when it drops InferenceSpeed, because InferenceSpeed is missing from the core unread catalogue (core slice).

Nothing was verified against the live API; no keys were available.

**Второй проход.**

What I covered:
- The full net diff of the non-test files in llms/anthropic/: anthropicllm.go, anthropicllm_option.go, errors.go, models.go, options.go, sampling_warnings.go, and internal/anthropicclient/{anthropicclient,auth,completions,federation,messages,models}.go. I read the HEAD versions and compared them with BASE.
- The llms helpers these files call: turn.go (ClassifyToolChoice, ForcesToolUse, HasAssistantPrefill, CheckClaudeTurnLimits), stopreason.go, options.go (GetTokens, GetEffort, ValidateReasoning), warning.go, reasoning/blocks.go and reasoning/claude_capability.go. For consistency I also read the Bedrock converse door's Claude path.
- Commit messages for the budget/max_tokens changes (891554d, 3c93a54, b89ada1), the empty-content-with-stop-reason change (20acb40), top_k (52f5f37), the forced-tool guard (272ae0b, cfd6ad0, 22a87a8) and the partial-stream changes.
- Vendor docs: workload identity federation and the WIF reference (exchange endpoint, error shape, lifetime, jti single-use), steering thinking (thinking_tokens field), and the thinking overview (tool_choice limits, disabled-thinking models, sampling rules, redacted blocks, prefill).

Probes (overlay, under scratchpad/probes/anth2-*): anth2-partialtool, anth2-leak and anth2-forced ran on HEAD. anth2-partialtool and anth2-leak also ran on BASE; anth2-nonclaude ran on HEAD only. The probes cover the partial tool call, the clean-EOF stream, the legacy-stream goroutine leak, forced tool choice on Opus 5.5 / Fable 5.1 / Mythos 5.1, and forced tool choice on non-Claude models.

Checked and not reported (deliberate or fine): max_tokens no longer doubled when the caller's limit already exceeds the budget (891554d); an end_turn stop with no content block is now a successful empty answer (20acb40); text blocks are concatenated (9997d13); top_k is sent (52f5f37); thinking_tokens exists in real cassettes and the docs; the prefill guard for 4.6+ models is correct per the vendor; the refusal alias keeps the same fields and message.

Not covered:
- Live vendor calls (no keys). The acceptance of forced tool choice with thinking on DeepSeek, Kimi, GLM and MiniMax Anthropic-compatible endpoints is unverified: the egress proxy blocked the docs host.
- The slice's cassette-backed integration tests: I did not rerun them.
- A line-by-line review of every one of the ~30 new test files. I read the ones that pin door behaviour: truncation, tool_choice_wire, thinking_blocks, refusal, partial_stream, budget_interleaving, unread_parity and parts of anthropic_thinking_test.

### tests-anthropic-googleai

**Первый проход.**

What I covered: every changed *_test.go in llms/anthropic/** and llms/googleai/**. That includes llms/anthropic/internal/anthropicclient, llms/googleai/vertex, llms/googleai/palm and llms/googleai/shared_test. I also read the deleted vertex test files and the net diff of the related production code.

Runs on HEAD:
- go test -p 2 on all 7 packages: pass. They also pass with -race, and with -count=8 -shuffle=on (no flakes, no races).
- Anthropic cassette tests: they skip whenever ANTHROPIC_API_KEY is unset (the newHTTPRRClient gate predates this PR; this PR adds no such skips). With ANTHROPIC_API_KEY=test-api-key they all replay and pass. TestLLM is excluded because it is live-only and fails with 401 by design.
- googleai cassette tests with GOOGLE_API_KEY=test-api-key: pass.

Skip audit:
- Vertex shared suite: all 14 subtests skip; no Vertex cassettes exist or can be recorded, which the new message explains. This is pre-existing: the base had no cassettes either.
- PaLM tests: skip, gRPC is not recordable.
- The PR adds no skips that hide failures. In googleai_test.go it removes the skips on "cached HTTP response not found", which makes the tests stricter.

Cassettes:
- Replay matching compares the full request, and every test in the slice matches its cassette.
- 13 modified googleai cassettes were edited by hand. Their request bodies were rewritten (temperature/topK/topP dropped, 2048→16384 max tokens only in ErrorHandling), and the recorded responses are from Jan–Aug 2026 runs of the old requests.
- The affected cassettes: BatchEmbedding, CreateEmbedding, ErrorHandling, WithOptions, ExplicitCaching, the 4 ImplicitCaching cassettes, and the 5 *WithSignature* cassettes.
- The edits only remove fields, so I found no vendor rejection they could hide. I did not report this; note it if cassette fidelity matters.

Vendor checks: Anthropic behaviours were checked against the bundled claude-api reference:
- Sampling is rejected on Sonnet 5.
- xhigh effort starts at Opus 4.7, so it is lowered on 4.6.
- Prefill is rejected on 4.6 and later.
- Forced tool_choice is allowed on Sonnet 5 and Fable 5 through the Claude API.
- Fast-mode beta header name.
- Interleaved beta not needed with adaptive thinking.
- output_tokens_details.thinking_tokens appears in the real recorded cassettes.

The tests match all of these.

Not reported as findings:
- Tests encoding already-reported issues: #8 penalties (vertex_backend_test.go, googleai_core_unit_test.go) and #11 tool message with a TextContent part (tool_results_test.go).
- Weak but correct tests: legacy unknown effort only asserts Error (a probe confirmed no request is sent); credentials tests only assert New succeeds; TestWithEndpointFeedsBothDoors still asserts ClientOptions, which the vertex door no longer reads.
- A stale comment in googleai/llmtest_test.go names gemini-3.8-flash while the model is gemini-3.5-flash.

Not covered:
- Live vendor calls; there are no keys.
- The Vertex backend end to end; no ADC and no cassettes are possible.

### tests-reasoning-core-rest

**Первый проход.**

What I covered: I read the net diff (da7016e..355dc71) of every changed *_test.go in the slice: llms/reasoning (all 30 test files plus internal/cmd anthropicmodels and mistralmodels), top-level llms tests, llms/structuredoutput, llms/cache, llms/fake, internal/toolcall, internal/numutil, internal/imageutil, chains (including chains/constitution), embeddings (bedrock, jina, voyageai, core), agents, memory, and vectorstores (pgvector, inmemory, dolt, redisvector, opensearch, pinecone, mongovector, weaviate, chroma).

Test runs on HEAD with -p 2 (Bedrock with env -u AWS_CA_BUNDLE):
- All non-Docker slice packages pass except llms/TestCountTokens, which needs the network and also fails on BASE.
- With -race -count=3, embeddings/*, llms/cache, llms/fake, internal/toolcall, memory, llms and llms/reasoning show no data races.
- Packages that need Docker/testcontainers panic with 'rootless Docker not found' and were not run: pgvector container tests, alloydb, cloudsql, chroma, milvus, mongovector, redisvector, weaviate, memory/mongo. Dolt tests skip because there is no dolt binary. opensearch, pinecone and jina integration tests skip for lack of env/cassettes.
- The pgvector unit tests (filter, metadata_index, scripted init) pass.
- For vectorstore tests that switched to WithModel("gpt-4.1-nano"), I checked by script that their cassettes record gpt-4.1-nano. Tests run offline replay their cassettes successfully.

Mutation and probe checks were done through Go overlays, with nothing written into the checkouts.

Not reported again because they are already in the earlier review, although tests in this slice pin them:
- #3: filter_test.go TestFilterPredicatesRejectAKeyThatIsNotAnIdentifier.
- #9: claude_capability_test.go TestClaudeRejectsAssistantPrefill has no -latest alias.
- #10: reasoning_support_test.go TestOpenAIHintSurvivesTheTransportLabel asserts that OpenAI efforts are advertised behind the Bedrock, GoogleAI and Anthropic labels.
- #5: qwen_capability_test.go and qwen_off_test.go assert host-free qwen3-<N>b rules.
- #2 and #6: effort_wire_test.go TestTakesNoJSONSchema.
- #12: init_test.go expects the ANALYZE step.

Limits:
- Most table rows in llms/reasoning encode per-model vendor measurements (effort sets, off wires, opt-in). I checked them for internal consistency across tests and against production branches, not each against vendor docs.
- ai.google.dev is blocked by the egress proxy, so Gemini 3 'minimal as off' semantics could not be confirmed. Search results say minimal does not guarantee thinking is off. I left this out as a deliberate mapping.
- The live Anthropic and Bedrock behaviour for model_context_window_exceeded was checked only against documentation and the vendored SDK enum, with no live call.

### openai

**Первый проход.**

I read the full net diff of every non-test Go file in llms/openai/: apikey.go, doc.go, llm.go, openaillm.go, openaillm_option.go, options.go, sampling_warnings.go, structured_output.go, and internal/openaiclient chat.go, completions.go, embeddings.go, openaiclient.go and openaitts.go. For each changed path I also read the reasoning-package helpers it calls: effort_wire.go, off.go, qwen_capability.go, name.go, openai_capability.go, and parts of claude_capability.go and reasoning.go. From core llms I read turn.go (ClassifyToolChoice), warning.go, stopreason.go and the options.go reasoning methods. I read the commit messages for extra-body merge, cost, refusal, the MiniMax/DeepSeek host rules, the Claude JSON mode rule and the effort-wire rules. Every probe used httptest captures run through the Go overlay in both HEAD and BASE, except the logprobs, Functions and DashScope probes, which ran in HEAD only; two probes used a redirecting transport so the real OpenRouter and DashScope base URLs were exercised.

Checked and found no new defect:
- Extra-body deep merge and type-variant replacement (deliberate, documented).
- statusError and the sanitizeHTTPError unwrap.
- Optional API key and header omission (deliberate).
- Streaming: 8 MB line buffer, producer cancellation, partial response returned with the error, accumulators.
- decodeContent for Mistral content arrays; Thinking and KeepsEmptyReasoning marshalling; withThinkTags does not alias the caller's slice.
- Structured-output emulation: schema injection, fence and think unwrap.
- Refusal as an error and the cost values changing to float64 (both deliberate per 28ff148 and cc3ef5e).
- Embedding length mismatch returning nil (deliberate).
- Claude thinking-object host gating and the sampling-policy branches.

Findings already reported in the earlier review (#2, #5, #6 and #10) were not repeated; the new ones are distinct rules or code paths with different failures.

Not covered or not verifiable:
- No live vendor calls, since there are no keys. OpenRouter documentation could not be fetched (egress blocked), so vendor behaviour for the OpenRouter findings rests on search-result summaries.
- Pre-existing issues not introduced by the PR were left out. Example: a cancelled stream can return a truncated response with a nil error because the producer's select may pick ctx.Done over the error send.
- The reasoning package's own rule tables were reviewed only as the openai door uses them.

**Второй проход.**

What I covered. I read the full net diff of every changed non-test file in llms/openai: openaillm.go, structured_output.go, sampling_warnings.go, apikey.go, llm.go, options.go, openaillm_option.go and doc.go. In internal/openaiclient I read chat.go, completions.go, embeddings.go, openaiclient.go and openaitts.go. I also read the HEAD versions of these files.

Outside the slice, I read the helpers the door calls: reasoning/effort_wire.go, off.go, qwen_capability.go, openai_capability.go, name.go and parts of reasoning.go and claude_capability.go, plus llms turn.go, warning.go and the options.go ReasoningConfig methods. I read the commit messages for the extra-body merge, the adaptive delegation, JSON mode, refusal, Kimi sampling and the structured-output fallback.

Probe tests (Go overlay, HEAD vs BASE) covered:
- sampling on OpenAI reasoning models and on DeepSeek/grok
- delegated adaptive reasoning on Claude across hosts and formats
- extra-body merge of response_format
- effort "none"
- context cancellation mid-stream

Checked and not reported as new:
- Tool-message text parts turning into an assistant message placed before the tool result: pre-existing.
- UpstreamInference*Cost changing type from *float64 to float64 in GenerationInfo: deliberate.
- refusal returned as an error on non-structured calls: deliberate.
- CreateEmbedding returning nil on a length mismatch: deliberate.
- sanitizeHTTPError unwrapping url.Error: benign.
- statusError now including the raw body: benign.
- API-key host list: reviewed.
- Kimi FixesSampling and the DashScope thinking_budget for bare qwen names on any host: same host-agnostic root cause as the reported #2/#5/#6 and the first reviewer's #3.
- Mid-stream provider error events (finish_reason "error") parsed as success: pre-existing in base.
- validateStructuredResponse checks "stop" choices that carry tool calls, although the fallback doc says tool-call turns are not validated: not probed, and the native path has the same code in base.
- Misplaced comments (the processResponse doc now above partialWithTruncation, and the addNonZeroChange comment above a const): unexported, nit, not reported.

Not covered:
- No live vendor calls (no keys), so vendor-side acceptance of reasoning:{enabled:true} and OpenAI strict-mode behaviour comes from docs and memory.
- I did not re-run the slice's full test suite.
- I did not review every one of the 51 test files in depth; I sampled the adaptive, tool-choice, API-key, validation, partial-stream and fallback tests.

Note: the Bash permission classifier failed several times mid-review. All probes whose output is quoted above were run successfully.

### tests-openai

**Первый проход.**

What I covered: all 51 changed test files in llms/openai (including internal/openaiclient), read against the net diff da7016e..355dc71. I also read the main production changes they test: openaillm.go, sampling_warnings.go, structured_output.go, apikey.go, llm.go and openaiclient/chat.go.

Test runs on HEAD:
- `go test -p 2 ./llms/openai/...` passes (287 top-level PASS). The only skip is TestLLM (OPENAI_API_KEY), which was already there.
- `-race` passes.
- `-count=2 -shuffle=on` passes.
- `-count=4 -shuffle=on` failed once in TestZeroConfigTemperaturePinUsesDefaultModel because httptest could not listen. /proc/net/sockstat showed ~20k TIME_WAIT sockets, from ephemeral ports used up by concurrent reviewers and my repeated runs. This is the sandbox environment, not a test defect.

Leak test: TestAGivenUpStreamDoesNotLeaveItsProducerBehind has real power. With the 1eecb8f producer fix reverted through an overlay it fails, both alone and in the full package. Run 30x on its own it is stable.

Cassettes, checked by script: the 192 llms/openai testdata cassettes map one-to-one to tests that ran. There are no orphans, and no httprr test is skipped for a missing cassette. The recording-mode matcher compares request bodies. The xAI and DeepSeek cassettes were rewritten by hand, as their commits state. The DeepSeek rewrite is justified by vendor probes in 4c237ca. The xAI rewrite is reported above.

Not reported:
- Name-based host-agnostic assertions that repeat earlier findings #2, #5 and #6: deepseek/glm json_schema refusal, qwen3-<N>b, penalties on deepseek/grok.
- Deliberate, documented behaviour changes: DeepSeek penalties and temperature, stop still sent to grok, cost now a float64 with no key when absent, the TestXAI_ReasoningModel_RejectsPenalties cassette removed.
- Vendor claims about future models (gpt-5.5/5.6/6 effort-with-tools refusal, Kimi/GLM effort enums, Anthropic compat endpoint) that I could not check against docs. docs.x.ai is blocked by egress.

I found no tests that never run (misnamed tests, build tags, missing t.Run), no loopvar capture problems (go 1.26), no new t.Skip, no parallel tests that change env vars, and no data races.

### mistral-huggingface-ollama

**Первый проход.**

What I covered:
- The full net diff of every non-test file in llms/ollama (ollamallm.go, options.go, warnings.go, README), llms/mistral (mistralmodel.go, mistralembed.go, client_options.go, warnings.go) and llms/huggingface (huggingfacellm.go, the option file, warnings.go, and the huggingfaceclient files inference.go, huggingfaceclient.go and embeddings.go).
- The HEAD version of each of those files, plus the commit messages for every commit that touches them.
- The helpers they call, compared with BASE: llms/options.go (GetEffort, DelegatesDepth, the GetTemperature change), warning.go (AddUnreadOptions catalogue), stopreason.go, turn.go (ClassifyToolChoice), reasoning/off.go (ResolveOff, takesOnlyOllamaLevels), reasoning_support.go, structuredoutput/prompt.go and internal/toolcall.
- The vendored SDKs: the ollama v0.32.5 api client, ThinkValue and the server ChatHandler, and mistral-go v1.1.0 chat, embeddings and request.
- Ollama server source at tags v0.12–v0.32 and huggingface.js provider routes, fetched from raw.githubusercontent.com. huggingface.co itself is egress-blocked.
- `go test ./llms/{mistral,huggingface,ollama}/...` passes on HEAD. Probes: an ollama think probe against a simulated <=0.17 server, and HF provider-route probes, both run on HEAD and BASE.

Tests I read: ollama think_request, partial_stream, structured_output_fallback (the placement part), warnings, truncation, ollama_thinking, sampling_request, toolcall_numbers, adaptive_delegation and the ollama_test / ollama_cloud_test diffs; mistral sampling_wire, chat_wire, stream_callbacks, default_model, embed_endpoint and embed_transport; HF reasoning_wire, wire, limits_wire, truncation, call_model and embeddings_wire, plus the huggingfacellm_test and huggingfaceclient_test diffs. I skimmed rather than read line by line: mistral call_errors, call_parity, nil_callbacks, embed_short_response, embed_model_wire, warnings_test and mistral truncation_test; HF warnings_test, adaptive_delegation, embeddings_short, chat_wire and example_provider.

Test-infrastructure observations:
- All TestCloud* tests skip when ~/.ollama/id_ed25519 is missing, so the new or changed cloud cassettes (TestCloudStructuredOutputFallback-*.httprr, TestCloudJSONMode.httprr) are never replayed here or in CI.
- TestMistralEmbed still skips with the reason "Mistral SDK doesn't support HTTP client injection", although WithEmbeddingHTTPClient now exists. That file is unchanged in the PR.
- HF TestHuggingFaceLLMGenerateContent and TestClient_CreateEmbedding are still t.Skip("temporarily skip").

Observed but not reported:
- Deliberate or pre-existing: Mistral picks a random seed per request (6444b4e). Mistral now sends response_format json_object and tool_choice. Mistral embeddings keep the endpoint path while chat still goes through the SDK, which drops it (5c41dcd/f4dcd0f), so a gateway prefix now differs between chat and embeddings. WithMaxRetries(0) means 5 attempts for chat but 1 for embeddings.
- The Mistral streaming loop returns early while the SDK goroutine blocks on an unbuffered channel, which leaks the goroutine and the response body. This existed in BASE.
- HF sends reasoning_effort unmapped, and \"none\" for a disable (e83061c, deliberate). HF WithInferenceProvider no longer overrides WithURL (f40c2a9, deliberate).
- Ollama refuses WithReasoningDisabled on gpt-oss (f76f0fa, deliberate). Ollama Cloud JSON mode is dropped and schemas are refused (ebb99f3/010688e, deliberate and backed by the vendor doc quoted in the commit). A stream cut before its final frame is returned as a success with StopReason \"\" (deliberate, and a test pins it).
- Not reported again because the earlier review already has them: #1 (think sent to models that cannot think), #4 (null tool-call arguments on the ollama path) and #13 (gpt-oss detection duplicated in the door).
- Could not verify live: HF router behaviour (egress block, no keys) and real Ollama servers of the affected versions. The finding about old Ollama servers rests on vendor source at the release tags plus a simulated server.

**Второй проход.**

What I read: the full net diff and the HEAD version of every changed non-test file in the slice. Mistral: client_options.go, mistralembed.go, mistralmodel.go, warnings.go. HuggingFace: huggingfacellm.go, huggingfacellm_option.go, warnings.go, and internal/huggingfaceclient/{embeddings,huggingfaceclient,inference}.go. Ollama: ollamallm.go, options.go, warnings.go, README.md. I compared each against BASE. I also read the callers and callees these files rely on outside the slice: llms/options.go (reasoning config, WithStructuredOutput contract), llms/warning.go, stopreason.go, turn.go, structured_output.go, internal/toolcall, llms/structuredoutput, reasoning/off.go and name.go, embeddings/huggingface, chains/options.go, the mistral-go v1.1.0 SDK and ollama v0.32.5 (api types, client signing, server think handling). I read the commit messages behind each change and every added or changed test file in the three packages.

What I ran: the slice tests pass on HEAD. Probes (overlay, under probes/mho2-*) covered:
- the ollama and HF budget warnings;
- an ollama cut stream with WithFailOnTruncation (the cut stream is accepted as a normal completion; not reported, because BASE also returned no error and the "Ollama <=0.34.0 repetition stop" claim cannot be checked);
- HF structured output and a system-first conversation (only the system text reaches the wire; pre-existing and already warned);
- Mistral structured output, endpoint-prefix paths (chat drops a gateway prefix through the SDK while embeddings now keep it; not reported, since chat never worked behind a prefix);
- Mistral zero retries;
- HF embeddings with WithURL(.../hf-inference) on HEAD and BASE;
- the ollama cloud cassette suite with a generated key.

Not reported, as deliberate or already covered:
- Cloud schema refusal and JSON-mode drop (commit ebb99f3 cites the Ollama docs; BASE validated locally, but the refusal is intended).
- HF reasoning_effort sent as written, including "none" (commit e83061c accepts provider rejections). Web search shows Groq returns 400 'reasoning_effort is not supported' for llama-3.1-8b.
- Mistral per-call random seed (6444b4e).
- Mistral tool_choice now sent even without tools (vendor behaviour unknown).
- The mistral-go streaming goroutine leak on early return and the tool messages without tool_call_id (both pre-existing).
- The first reviewer's three items and the 15 earlier ones.

Limits: huggingface.co docs are blocked by egress, so the 'BASE worked live' part of the HF finding rests on web-search evidence of the documented migration URL, and there is no vendor confirmation that hf-inference accepted BASE's top-level use_gpu/wait_for_model keys. No live vendor calls were possible (no keys). I did not review cassette contents beyond the ollama cloud model and the empty TestMistralEmbed cassette; mistralembed_test.go is unchanged in the PR, so it cannot take an inline comment.

### bedrock

**Первый проход.**

Covered: the full net diff of every non-test Go file in llms/bedrock (bedrockllm.go, models_list.go, warnings.go) and internal/bedrockclient (bedrockclient.go, bedrockclient_converse.go, bedrockclient_util.go, converse_warnings.go, warnings.go, reasoning_stream.go, and the ai21, amazon, anthropic, cohere, deepseek, meta and nova providers). I read the HEAD version of each file and compared it with BASE. I also read the reasoning helpers Bedrock calls: mechanism.go, bedrock_capability.go, claude_capability.go, off.go, name.go, blocks.go, the GetEffort/GetTokens helpers in llms/options.go and turn.go. I read the README diff and the commit messages behind the main behaviour changes: structured-output allowlists, tool choice, effort clamps, Opus 5/Sonnet 5 off, the Nova legacy reasoningConfig, delegation, off refusals, counter types and the image refusal. The HEAD bedrock test suite passes with AWS_CA_BUNDLE unset. I ran probes through overlays on HEAD and BASE for ARN structured output, tool choice on the wire, Nova delegation on both doors, BinaryPart text MIME types on Converse, and the top_p floor.

Checked but not reported:
- Deliberate and documented behaviour changes: ErrReasoningOffUnsupported for deepseek.r1, gpt-oss, minimax and kimi-thinking; the forced-tool plus budget-thinking guard; ErrEffortHasNoBudget for effort "minimal"; the Claude structured-output allowlist (Opus 4.7+ refused); xhigh/max clamps on Bedrock; int counter types.
- Converse now refuses a non-image BinaryPart (text/plain, application/json) with ErrUnsupportedImageFormat, where BASE sent it as text. The commit declares this intended. In BASE such a part usually produced consecutive user messages anyway.
- The legacy Anthropic door now sends tool_choice even when no tools are given. I found no vendor evidence that this is rejected, and the first-party anthropic door does the same.
- Converse sends top_k to Claude and would send top_k:0 for WithTopK(0). Whether Sonnet 4.6 accepts top_k is unclear.
- The legacy Nova reasoningConfig placement inside inferenceConfig matches the schema the commit cites. I could not verify it because docs.aws.amazon.com and aws.amazon.com are blocked by the egress proxy.
- Converse sending Nova reasoningConfig without maxReasoningEffort under delegation depends on the vendor, and documentation was inconclusive.
- Already-reported items (#4 null tool args, #10 hint for Bedrock) were not repeated.

Not covered: no live AWS calls (no keys); vendor behaviour was taken from AWS SDK doc comments and GitHub issues because AWS docs were unreachable. Test files were read only where they bear on a source finding.

**Второй проход.**

What I covered: the full net diff of the non-test .go files in llms/bedrock/ (bedrockllm.go, warnings.go, models_list.go) and llms/bedrock/internal/bedrockclient/ (bedrockclient.go, bedrockclient_converse.go, bedrockclient_util.go, converse_warnings.go, warnings.go, reasoning_stream.go, provider_anthropic/nova/ai21/amazon/cohere/deepseek/meta.go). I read the HEAD version of each file and the README diff, and compared against BASE. I also followed the code these files call outside the slice: reasoning.ResolveMechanism / ResolveOff / claude_capability / bedrock_capability / name.go / blocks.go, llms turn.go (CheckClaudeTurnLimits, ClassifyToolChoice), llms options.go (GetEffort/GetTokens/DelegatesDepth/ValidateReasoning), llms stopreason.go, llms warning.go, toolcall.Decode, reasoning_support.go and the first-party anthropic door for parity. For intent I read the commit messages behind the main changes: 32e4f8c, 6dcf0e0, 1f5a224, 28ff148, d9ff7dc, 7ee9de2, 936a2cc, 8240420, 82ad622, 200f1a6, 2a02688, deedac0.

Probes, all run with env -u AWS_CA_BUNDLE:
- legacy tool_choice with no tools: HEAD and BASE compared;
- Converse BinaryPart text/plain, both with and without a text part: HEAD and BASE compared;
- GenerationInfo counter types per family: HEAD only.
`go test ./llms/bedrock/...` passes on HEAD.

Vendor checks: docs.aws.amazon.com is blocked by the egress proxy, so AWS model-card claims were checked only through search snippets. That covers the Nova reasoningConfig placement inside inferenceConfig, which the Nova schema confirms. The legacy tool_choice finding rests on Anthropic GitHub issues, not on a live call.

Considered and not reported, because the commit history shows the change is deliberate and backed by cited docs or live cassettes:
- Opus 5 / Sonnet 5 now take thinking:disabled on Bedrock;
- structured output is refused for Claude models outside the Bedrock list;
- ErrEffortHasNoBudget for "minimal";
- ErrReasoningOffUnsupported for gpt-oss, kimi, minimax and DeepSeek R1;
- forced tool choice plus budget thinking is refused locally where base silently sent auto;
- ErrModelRefusal on the legacy door;
- PromptTokens now includes cache tokens, and counters changed from int32 to int on the Claude doors;
- Kimi/MiniMax no longer receive a thinking budget;
- effort is derived from tokens alongside a budget, matching the first-party door.

Checked and found no problem: the Converse human-turn merging, interleaved thinking placement (GroupByToolCalls) on both doors, index-keyed streaming tool calls and the salvage path, legacy streamed usage merging, Nova stream event parsing, the Converse stopReason and truncation flags, budget/max_tokens ceilings on both doors, TopK carriage and its warnings, and the nil-config paths.

Not covered, and why:
- Tests in the slice, beyond reading the ones that pin behaviour relevant to the code paths above. They belong to the test-review slice.
- Speculative Grok-on-Bedrock behaviour: GrokEffort's "none" branch is unreachable from the door, but I cannot verify vendor semantics.
- top_k=0 on Converse for Claude: the door sends it, but I found no evidence that non-deprecated models reject it.
- Pre-existing issues outside the diff hunks, left out as not regressions: legacy temperature 0 dropped by omitempty; the Converse tool-schema json.Number re-encoding in convertToolCallInput; a nil tool.Function panicking on the legacy door.

I did not repeat the first reviewer's four findings or the 15 already-reported ones.

### tests-bedrock

**Первый проход.**

What I covered:
- I read the net diff of all 60 changed test files under llms/bedrock/** (53 added, 7 modified, one test renamed, none deleted) and the helpers they share: bedrockLLMAgainst, legacyLLMCapturing, writeLegacyChunk/writeConverseEvent, the recorders, bedrockWarningsFor, conformanceClient, and the replay credential helpers.
- For context I read the production code the tests exercise: provider_anthropic streaming and usage, applyConverseUsage, provider_nova, reasoning_stream.go, warnings.go, and the SDK event-stream reader.

Test runs on HEAD 355dc71 (all with `env -u AWS_CA_BUNDLE`):
- `go test -p 2 ./llms/bedrock/...` passes: 673 PASS lines, 0 SKIP.
- `-race -count=3` passes.
- In replay, no test in the slice skips; TestLLM, TestLLMConverse and the TestAmazon* cassette tests all run.
- httprr matches the full request, including the body, so each cassette must match what its test sends. TestLLM.httprr and TestLLMConverse.httprr each carry one streamed request (invoke-with-response-stream / converse-stream); their request lines point at the vendor host and they have no Authorization header.
- The re-recorded TestAmazonReasoningLegacyAPI cassette really contains `output_tokens_details.thinking_tokens`, which backs the new ReasoningTokens assertion.

One unexplained failure:
- In my first verbose full-slice run, TestLegacyStreamKeepsWhatArrivedWhenTheConsumerGivesUp failed once (0.00s). The failure message was not captured.
- It did not reproduce in 3000 isolated iterations (-cpu 1,4), 25 full-package runs, 300 combined stream-test iterations, or 3x -race.
- A later 60x stress run died with `httptest: failed to listen on a port`, which points to ephemeral-port exhaustion on this shared machine.
- I also checked whether the handler's `require.NoError(enc.Encode)` could fail after the client disconnects. A probe with 50ms server delays showed the write errors never surface for these fixtures: net/http buffers the writes, and three frames after the RST would be needed.
- So I attribute the failure to the environment and did not report it.

Vendor claims I did not report:
- Several assertions encode Bedrock capability claims: xhigh only on Opus 5 on Bedrock, no output_config.effort on Opus 4.5, effort next to budget thinking on Opus 4.6, Nova Invoke carrying reasoningConfig inside inferenceConfig, and the structured-output model lists.
- Commit messages (8240420, 82ad622) show these are deliberate.
- docs.aws.amazon.com and aws.amazon.com are blocked by the egress proxy, so I could not check the AWS pages directly. Search snippets agree with the Nova placement.
- The claude-api skill's per-model table lists xhigh for Sonnet 5 on the first-party API, so the Bedrock-only restriction is unverified. I did not report it.

Minor issues I judged not worth reporting:
- Several provider_anthropic_reasoning tests ignore the error returned by applyAnthropicReasoning. The error returns before any field is written, so the assertions would still catch it.
- The turn_limits negative subtests accept any error except the named type.

Probe files are under scratchpad/probes/tb-bedrock/ (probe_test.go, mut1/, mut2/, p2/).

### googleai

**Первый проход.**

Covered: every non-test file in the net diff of llms/googleai against BASE da7016e, read in full on HEAD 355dc71. That is googleai.go, new.go, option.go, credentials.go, embeddings.go, models.go and warnings.go; vertex/vertex.go, with the deleted vertex new.go, embeddings.go and structured_output.go and the old vertex.go read on BASE; palm/palm_llm.go; internal/palmclient/palmclient.go; the deleted internal/cmd/generate-vertex.go; and README.md.
Also read, outside the slice: callee code in llms/reasoning/gemini_capability.go and off.go, llms/options.go (ReasoningConfig), llms/warning.go, llms/turn.go (ClassifyToolChoice), llms/stopreason.go, internal/toolcall/arguments.go, internal/imageutil, and structuredoutput.ValidateFinalChoices.
Also read in the module cache: google.golang.org/genai v1.42.0 (NewClient backend and credentials rules, createAPIURL, EmbedContent routing, per-backend parameter rejections, Models.All) and cloud.google.com/go/vertexai v0.12.0 (NewClient endpoint, inferLocation), to compare BASE vertex behaviour.
Read the commit messages for every googleai/vertex change: sampling defaults, thinking/levels/MINIMAL/gemma-4/adaptive, empty-stream, blocked prompt, stream IDs, credentials, endpoint, tool choice.
Probes, run through overlays in both checkouts under probes/gai-rev1/: credentials plus API key, WithEndpoint in host:port form, gemini-pro-latest thinking, toolConfig without tools, vertex location inference, vertex credentials with WithHTTPClient.
`go test -p 2 ./llms/googleai/...` passes on HEAD.
Reviewed and judged not defects, or already reported: #4 (null args), #7 (max tokens 16384), #8 (penalties). Also reviewed without a finding:
- signCurrentTurn placeholder logic, streaming goto and partial-response handling, per-call stream IDs.
- The ErrEffortHasNoBudget refusal for minimal on 2.5 (deliberate, a3bd748/7d0e773).
- Removed sampling defaults (deliberate 35c8901/b51b009), the blocked-prompt error typing, parameterless tool declarations (BASE already failed at the vendor), and embeddings count checks.
- The Vertex :predict embedding path (same as BASE), responseJsonSchema on Vertex, and the palm project/location swap fix.
Not covered: live vendor calls (no keys, and ai.google.dev is blocked by the egress proxy), so the toolConfig-without-tools vendor response and Gemini 3 Pro MEDIUM level support stay unverified. Of the slice's tests I read only the ones that bore on source behaviour: thinking_wire_test, tool_choice_wire_test, credentials_test. The rest of the slice's tests and the cassettes were not reviewed.

**Второй проход.**

I read the full net diff (da7016e..355dc71) of every non-test file in llms/googleai: googleai.go, new.go, option.go, credentials.go, embeddings.go, models.go, warnings.go, vertex/vertex.go (and the deleted vertex/new.go, embeddings.go and structured_output.go on BASE), internal/palmclient, palm/palm_llm.go, README and the deleted generator. I also read the HEAD versions of these files and the commit messages for the main changes: dc2be39, 6aa6de9, 35c8901, c61cb2d, a3bd748, 7d0e773, eb58c4f, fe7ff17, 618c86f, a459111, a389dd9, e93a00b and 96a8bd1.

Code outside the slice that I traced:
- llms/reasoning: gemini_capability.go and ResolveOff in off.go.
- llms/options.go: ReasoningConfig, GetEffort, GetTokens, DelegatesDepth.
- llms/warning.go, llms/stopreason.go, llms/turn.go (ClassifyToolChoice), structuredoutput.ValidateFinalChoices, internal/toolcall and internal/imageutil.
- genai v1.42.0: NewClient credential and env precedence, the Vertex URL builder, Models.All and tModelsURL, the tModel name mapping.

Probes, run through Go overlays in probes/g2rev-a on HEAD and BASE:
- Vertex ListModels.
- Empty stream with the default max tokens.
- Gemma 4 adaptive wire and warnings.
- Custom ClientOptions (a token source) and WithAPIKey on the Vertex backend.
- Gemini 3 image models with reasoning on and with reasoning off.

Investigated and not reported:
- Gemma 4 adaptive sends only includeThoughts with no level. The warnings path assumes HIGH, but I could not confirm Gemma 4's API default: a vendor forum thread about disabling Gemma 4 thinking suggests it thinks by default, so I dropped it.
- WithCachedContent plus WithToolChoice now sends toolConfig. Gemini rejects tool_config together with cached content, whereas BASE ignored ToolChoice. This is a further failure mode of the first reviewer's #7 (tool config sent without tools) and is fixed by the same gate.
- toolcall.Decode accepts trailing data after the first object (outside the slice, lenient rather than harmful).
- Vertex embeddings now go through genai EmbedContent in batches of 100. BASE sent all texts in one palmclient request, so there is no regression.
- ImageURLContent now sends "image/"+subtype. This is an improvement, although a non-image or empty Content-Type is still wrong, as on BASE.
- The palm argument swap and the palmclient location change fix a BASE bug.
- Streamed call IDs, signCurrentTurn turn detection, the budget clamp, the minimal-effort mapping, the response-format and JSON-schema path on Vertex (vendor docs say JSON Schema is supported on all active models) and responseJsonSchema all look consistent.

Not covered:
- No live vendor calls; there are no keys.
- Vertex express-mode API-key behaviour against the real endpoint.
- Tests in the slice were read selectively: warnings, thinking wire, adaptive delegation, thought signature, tool choice, empty stream, credentials, vertex, models, shared_test diff. I did not audit every test or cassette.

### llms-core

**Первый проход.**

What I covered: the full net diff and the HEAD version of every non-test file in the slice.
- llms/options.go: ValidateReasoning, DelegatesDepth and HasExplicitTokens, ReasoningEffortBudget, the switch of GetEffort/GetTokens to DefaultMaxTokens, removal of the GetTemperature override, the new options, ExtraBody.
- llms/errors.go (ErrTruncated, ErrModelRefusal and the anthropic alias's compatibility), llms/stopreason.go, llms/turn.go (ClassifyToolChoice, CheckClaudeTurnLimitsOnWire), llms/warning.go, llms/reasoning_support.go, llms/count_tokens.go, llms/generatecontent.go, llms/structured_output.go.
- llms/structuredoutput (schema closure walk, admitNullable, the jsonschema loader change, prompt/fence unwrapping, ValidateFinalChoices), llms/cache, llms/fake, internal/toolcall, internal/numutil, internal/testutil/testctr (whitespace only).

For each helper I traced its callers in the doors: the ValidateReasoning callers (openai, anthropic, bedrock); every ClassifyToolChoice translator (openai, anthropic, bedrock converse and invoke, googleai, mistral); the IsTruncated/CheckTruncation sites; the toolcall.Decode callers (googleai, ollama, bedrock); the RequireClosedObjects/Validate callers; the AddUnreadOptions carried lists; and the openai ExtraBody merge, checked against its doc comment. I read the intent commits for each file.

Probes run in HEAD and BASE:
- none-effort on the openai and bedrock doors;
- anthropic tool_choice wire body;
- anthropic model_context_window_exceeded handling;
- structuredoutput Compile with $schema draft-07 and 2020-12 (the empty SchemeURLLoader does not break the built-in metaschemas), nullable handling, and RequireClosedObjects strictness;
- ReasoningSupportFor for unknown models.

Checked and not reported, because the change is deliberate or harmless:
- The GetTemperature override removal (f2d9082): no door relies on it; Anthropic and Bedrock set temperature themselves.
- DefaultMaxTokens replacing 8192 (5c7f33b).
- The ExtraBody move out of Metadata (33b8381); a test pins that the old key is no longer read.
- DelegatesDepth turning an adaptive request with no effort into the default mode (a459111, warned).
- Cache returning the partial response and stripping warnings.
- Fake LLM now streaming.
- CountTokens behaviour: it is identical to BASE, since tiktoken-go has no gpt2 encoding.
- toolcall leniency toward trailing data. Its null rejection is already #4.

Minor and not reported: ValidateFinalChoices panics on a nil choice (no door produces one), and nullable+enum without null still rejects null (this matches OpenAPI 3.0.3).

The known items #10 and #14 overlap this slice and were not reported again.

Test run: the slice's unit tests pass, except TestCountTokens. It fails here only because the sandbox blocks the tiktoken encoding download (403); this is environmental, not a code defect.

Not covered: live vendor calls (no keys). The model_context_window_exceeded finding rests on Anthropic docs and the AWS SDK enum, not on a real API response.

**Второй проход.**

I read the full net diff (da7016e..355dc71) of every non-test .go file in the slice:
- llms/{options,reasoning_support,stopreason,turn,warning,errors,count_tokens,generatecontent,structured_output}.go
- llms/structuredoutput/{prompt,schema,validator}.go
- llms/cache/cache.go and llms/fake/fakellm.go
- internal/toolcall/arguments.go and internal/numutil/num.go
- internal/testutil/testctr/testctr.go (whitespace-only change)

I also read the HEAD versions and the relevant commit messages (f2d9082, 5c7f33b, 1f5a224, 33b8381, e1be9fd, 28ff148, ff1a98a, 7fb2a4c, bb80316, 80840e1, 7f2f91c and others). I followed callers outside the slice: the tool-choice translators in the openai, anthropic, googleai, bedrock and mistral doors; every GetTokens/GetEffort caller; every CheckTruncation/IsTruncated site; the ValidateFinalChoices and Validate callers; the ExtraBody merge in openaiclient; the toolcall.Decode callers; and the ErrModelRefusal construction sites.

Probes, run through overlays:
- ReasoningSupportFor over about 55 model/provider pairs, BASE vs HEAD. The differences I found are either deliberate per the commits (Efforts cleared for budget-only Claude, Known dropped for unclassified OpenAI models) or already reported (#10).
- Schema compile with draft-04/07/2019-09/2020-12 $schema and $id refs, BASE vs HEAD. No regression from SchemeURLLoader{}.
- toolcall trailing-data probe.
- ClassifyToolChoice/openaiToolChoice probe, plus an end-to-end openai tool_choice wire probe, BASE vs HEAD.
- InferenceSpeed warning probe.

Slice tests pass except TestCountTokens, which fails only because the sandbox blocks the tiktoken encoding download (403 Forbidden). That is environmental.

Checked and not reported:
- The GetTemperature pin removal is deliberate (f2d9082).
- The maxTokens default change from 8192 to 16384 in GetEffort/GetTokens is deliberate (5c7f33b). All callers pass GetMaxTokens().
- GetTokens returning -1 for ReasoningMinimal is deliberate and becomes ErrEffortHasNoBudget (1f5a224).
- The cache and fake changes are consistent with their tests.
- ErrModelRefusal's InputTokens semantics match the doors (Anthropic input_tokens is uncached-only; openai subtracts cached_tokens).
- There are no removed exported symbols (go doc diff).
- IsTruncatedError uses errors.As, so on an errors.Join whose first member is an *llms.Error with another code it can return false. This is the same pattern as the existing IsXxxError helpers, and in practice the doors' causes are not *llms.Error, so I did not report it.

Related to first-reviewer #3 and not re-reported: legacy Bedrock Cohere's "ERROR_LIMIT" (context limit reached) is also missing from the truncation table.

Not verified: whether Anthropic adaptive thinking (not only budget thinking) also rejects forced tool_choice. CheckClaudeTurnLimitsOnWire refuses only manual thinking, and I made no live vendor calls.

### vectorstores-chains-embeddings

**Первый проход.**

What I covered: every non-test file changed in my slice, read in full at HEAD and compared with BASE.
- pgvector: pgvector.go, metadata_index.go, options.go.
- Other vectorstores: inmemory.go, pinecone.go.
- chains/map_rerank_documents.go.
- Embeddings: embedding.go, bedrock/bedrock.go (plus the unchanged amazon/cohere provider code), jina/jina.go and options.go, voyageai/options.go and voyageai.go.
- Also memory/token_buffer.go (and llms.CountTokens), prompts/internal/sanitization/sanitize.go, tools/perplexity/perplexity.go, doc.go and the example diffs.

Commit messages I read for intent: ec0ee19, 504595c, da097ac, abcf47d, a35a0a3, 40b6fb1, 7ab52b8, 2ba6eae, fe380f3, 884772f.

Live checks:
- I started a private PostgreSQL 16.13 in my probe dir (user-namespace initdb) and loaded pgvector 0.6.0 from the Ubuntu .deb into a relocated copy of the install, so nothing outside the probe dir was changed. The server is stopped.
- Against it I ran the PR's pgvector integration tests (init_test, narrowing_test, filter/metadata tests). All passed except TestStoreCreatesTheDeclaredMetadataIndexes: it fails because of the ANALYZE and mixed-dimension issue reported above.
- I built all 75 example modules (they use replace ../..) against HEAD with -p 2; all build. Both checkouts are still clean per git status.

Checked and not reported:
- The init rollback, cancellation and advisory-lock flow.
- Placeholder numbering and parameter types in SimilaritySearch and Search.
- The collection subquery.
- Score-threshold binding (only float32 rounding differences).
- The MATERIALIZED CTE ordering of vector_dims and <=>, verified on a mixed-width table.
- Index name derivation and truncation, and the case-fold guard.
- quoteLiteral escaping.
- The new map_rerank parser and sort: more lenient than the regex it replaces, no regression found.
- CheckEmbeddings in BatchedEmbed against every in-repo EmbedderClient: all return one vector per input.
- voyageai WithBaseURL.
- The inmemory re-sort by score.
- The doc.go streaming snippet against the streaming.Callback API.

I did not repeat earlier findings #3 (non-identifier filter keys, including the 63-character length limit) or #12 (ANALYZE cost); the ANALYZE finding above is a separate hard failure. Non-identifier explicit index Names that are SQL reserved words fail with a syntax error at New; I judged this too minor to report.

Not covered:
- Tests in other vectorstores beyond skimming.
- Live Pinecone or Perplexity calls (no keys; vendor docs are blocked by the proxy, so those two findings rely on secondary sources).
- pgvector >= 0.7, where the ANALYZE failure does not occur.

**Второй проход.**

Covered: the full net diff and the HEAD version of every changed non-test file in the slice. That is vectorstores/pgvector (pgvector.go, metadata_index.go, options.go), vectorstores/inmemory, vectorstores/pinecone, chains/map_rerank_documents.go, embeddings (embedding.go, jina, voyageai, bedrock), memory/token_buffer.go, prompts sanitize.go, tools/perplexity, doc.go, and the example changes. I also read the commit messages for each file, callers outside the slice (every EmbedderClient.CreateEmbedding implementation against the new CheckEmbeddings, llms.CountTokens for the token-buffer comment, streaming.Callback for the doc.go snippet, the RegexParser the map-rerank parser replaced), and the slice's pgvector, chains and jina tests.

Probes and runs:
- Ran a private PostgreSQL 16.13 with pgvector 0.6.0 in a user namespace, using binaries extracted in another probe directory. Against it, the HEAD pgvector package tests all pass except TestStoreCreatesTheDeclaredMetadataIndexes, which fails with "different vector dimensions 64 and 32" from ANALYZE. That is first-reviewer finding #2, reproduced.
- Probes compared HEAD and BASE on Python-layout tables, measured writer blocking during init, and checked predicate-order cost with EXPLAIN ANALYZE.
- Built all 75 example modules against HEAD with go build -p 2: all compile. Both checkouts were left clean.

Checked and found fine:
- Map-rerank parser edge cases.
- inmemory sorting.
- jina, voyage and bedrock short-response checks, and voyage WithBaseURL.
- init rollback and cancellation paths.
- Literal and identifier quoting.
- Placeholder numbering.
- Index-name hashing, truncation and case folding.
- Advisory-lock re-entry.

Not covered or skipped:
- Behaviour on pgvector 0.7 or later.
- Live vendor calls; no keys.
- Examples outside the diff that still name retired models: reasoning-tokens (claude-3-7-sonnet, claude-3-sonnet), prompt-caching (claude-3-5-sonnet-20241022), anthropic-tool-call-example (claude-3-haiku-20240307), googleai-streaming-example (gemini-1.5-pro). They are not reportable inline.
- New() leaks the pgx.Conn it opens itself when Ping or init fails. This predates the PR and falls outside the hunks.
- Nit: ErrNoEmbedding's text says "for the query" but is also returned for document batches.

Possible overlap with #12: my ANALYZE/ShareLock finding shares a root cause with earlier finding #12. It is reported separately because its impact is different: ordinary writers block, the doc claim is false, and it is measured.

### test-infra

**Первый проход.**

What I reviewed: the full net diff (da7016e..355dc71) of internal/httprr (rr.go, README, 4 test files), testing/llmtest (llmtest.go, doc.go, llmtest_test.go), internal/devtools/normalize-recordings (gofmt only) and internal/testutil/testctr (gofmt only), plus the commit intent for each.

Things I checked and found sound:
- The record-mode body split (realReq with its own Body/GetBody).
- The x-goog-api-client normalizer change. All 70 cassettes carrying the header use the new "google-genai-sdk/X.XX.X gl-go/goX.XX.X" shape, and no .gz cassette carries it.
- The case-insensitive openai-project request regex. No cassette request carries that header, so replay keys are unaffected.
- The gateway-header deletion loop.
- The llmtest option plumbing: WithCallOptions and WithoutStreaming/WithoutToolCalls, the tool-drop warning probe (door warnings use Option "WithTools" consistently), assertStreams, and the verdictRecorder tests.
- `go test -race ./internal/httprr ./testing/llmtest` passes.

Cassette integrity. Scripts only; no cassette bodies were read into context.
- The PR touches 222 cassettes: 59 added, 157 modified, 6 deleted.
  - All 216 present ones parse byte-accurately and re-parse via http.ReadRequest/ReadResponse with full bodies (453 unique request keys, 0 errors).
  - None of the 6 deleted cassettes still has a test.
- I overlaid rr.go with open/hit/miss logging and ran all 41 httprr-using packages offline in HEAD and BASE (env -u AWS_CA_BUNDLE):
  - No test opened a nonexistent cassette.
  - The 23 tests that skip for lack of a cassette are the same in BASE and HEAD, and none of them was touched by the PR.
  - 192 changed cassettes were opened. Every record of every non-vectorstore changed cassette was replayed; no stale records.
- Two groups could not be replayed here, so their use was checked statically: each maps to an existing test function and subtest name.
  - 11 dolt cassettes: need the Dolt binary.
  - 13 weaviate cassettes: need Docker.
- The chroma, milvus, opensearch, pgvector, pinecone and redisvector cassettes are opened but get zero hits here. Docker is missing, and the pinecone tests require PINECONE_API_KEY.
- Across the whole repo, no cassette is orphaned: every one of the 460 maps to a test function.

Credential and personal-data scan of the 216 changed cassettes:
- Every Authorization value is "Bearer test-api-key" or "AWS4-HMAC-SHA256 test-api-key", and every X-Goog-Api-Key is test-api-key.
- No key/token/sig query parameters other than the placeholder.
- No matches for sk-, AKIA/ASIA, AIza, gsk_, hf_, xai-, nvapi-, pcsk_, gh*_, JWT, non-placeholder bearer tokens, private keys, emails, IPs, home paths or ARNs.
- No cookie, x-amz-security-token, x-litellm or llm_provider headers.
- Request hosts are vendor hosts only.
- Real org/project ids exist only in cassettes this PR did not change: the Anthropic-Organization-Id, the Msh-Project-Id and the two .gz project ids reported above.

Pre-existing issues, not reported as PR defects:
- milvus/v2 tests fail in replay with "missing the OpenAI API key" in both BASE and HEAD, so their 2 changed cassettes never replay.
- The anthropic tests that have cassettes skip without ANTHROPIC_API_KEY.
- The Vertex shared tests never run because no cassettes exist.
- llms/googleai/llmtest_test.go's comment names gemini-3.8-flash while the code uses gemini-3.5-flash (live-only test, nit).
- The doc comment of getDefaultResponseScrubbers now sits on the gatewayHeaderPrefix const (godoc nit, same class as #14).

Vendor docs: I tried WebFetch on docs.litellm.ai, but egress blocked it; the llm_provider- prefix claim comes from web search results.

### cross-cutting

**Первый проход.**

What was done:
- Public API compatibility over the whole module: apidiff -m on export data of da7016e and 355dc71. The full list of incompatible changes is in /tmp/claude-0/-home-user-langchaingo/190be914-a4ca-51a0-89e2-3fbdbb6fc0c9/scratchpad/probes/cross/incompat.txt, and the full diff is in all.txt in the same folder. Package lists: only internal packages were added or removed.
- Removals and changes left unreported because the commit history shows them as deliberate or they are harmless:
  - perplexity.ModelR11776 and ModelSonarReasoning removed (models retired, 884772f).
  - llmtest.TestLLM gained variadic options, and MockLLM.GenerateContentStream was removed (test helper, f9ca9d5).
  - reasoning.OffEffortNone and OffUnsupported were renumbered because iota values were inserted; this matters only if values are persisted.
  - anthropic.ErrModelRefusal is now an alias of llms.ErrModelRefusal, with a superset of fields.
  - Vertex.CallbacksHandler is still reachable as a promoted field.
  - Also treated as deliberate: CallOptions.GetTemperature no longer forcing 1.0 (f2d9082), googleai no longer sending default temperature, topK and topP (35c8901), GetTokens/GetEffort default 8192 → DefaultMaxTokens (5c7f33b), openai.WithExtraBody changing from a shallow replace to a deep merge, and googleai WithGRPCConn/WithGRPCClient now being refused.
- go vet ./...: clean on both HEAD and BASE.
- golangci-lint:
  - The CI version, v2.12.2, built with go1.26.8 because the preinstalled v2.5.0 refuses go 1.26.5: 0 issues on HEAD and on BASE.
  - v2.5.0 found one new issue on HEAD, a prealloc in the test file llms/ollama/toolcall_numbers_test.go:46. It is not a defect, so it is not reported.
- go test -race -p 2 on all changed packages, with Bedrock under env -u AWS_CA_BUNDLE:
  - No data races. Bedrock and bedrockclient pass.
  - Failures are environmental only: testcontainers needs Docker (pgvector, chroma, mongovector, redisvector, weaviate panic with "rootless Docker not found"), and llms TestCountTokens needs to download the tiktoken encoding (403 in the sandbox). BASE fails TestCountTokens identically.
  - pgvector ran without Docker only as far as the test harness allowed.
- go.mod/go.sum: removing vertexai matches the code, `go mod tidy -diff` is clean on both sides, and `go mod verify` passes.
- examples/: all 75 examples build with `go build -mod=readonly` against HEAD, which is the new CI mode.
- docs/, doc.go and README changes were checked against the code. The identifiers used in the snippets exist. The only contradiction found is the Gemini/Vertex note.

Not covered in depth:
- The behaviour of each door's wire mapping; other slices own it. I dug into googleai/vertex and pinecone only because they surfaced from API and docs checks.
- Pinecone euclidean score semantics could not be fetched from docs.pinecone.io (egress blocked); it is confirmed only via search results.
- No live vendor calls (no keys).
