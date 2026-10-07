package reasoning

import (
	"slices"
	"strings"
)

// ClaudeReasoningKind classifies how a Claude model accepts extended thinking.
// It is the single source of truth for adaptive-vs-budget thinking and whether
// sampling params are permitted, so every provider path resolves the wire shape
// the same way instead of re-deriving it from scattered model-string checks.
type ClaudeReasoningKind int

const (
	// ClaudeReasoningUnknown is a non-Claude model or a Claude model without
	// extended thinking; an unlisted version of a listed tier answers as the
	// release it follows. It is handled as literal pass-through (the caller's
	// requested mechanism is sent unchanged).
	ClaudeReasoningUnknown ClaudeReasoningKind = iota
	// ClaudeReasoningAdaptiveOnly is the newest generation (Opus 4.7/4.8/5,
	// Sonnet 5, Haiku 5.5, Fable 5, Mythos 5): it is sent thinking.type=adaptive
	// and never budget_tokens.
	ClaudeReasoningAdaptiveOnly
	// ClaudeReasoningAdaptiveAndBudget accepts both adaptive and budget thinking
	// (Opus 4.6, Sonnet 4.6, Mythos Preview).
	ClaudeReasoningAdaptiveAndBudget
	// ClaudeReasoningBudgetOnly is the extended-thinking generation before
	// adaptive existed (Opus 4.5, Sonnet 4.5, Haiku 4.5): it accepts
	// thinking.type=enabled + budget_tokens and rejects adaptive.
	ClaudeReasoningBudgetOnly
)

// adaptiveOnlyClaude, dualClaude, and budgetOnlyClaude are the explicit model
// sets. Substrings match both the first-party IDs (claude-opus-4-7) and the
// Bedrock IDs (us.anthropic.claude-opus-4-7). Add a new release to exactly one
// set once claudeReleases in lines.go lists it; until then it answers as the
// release it follows.
var (
	adaptiveOnlyClaude = []string{
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
		"claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
		"claude-haiku-5-5",
	}
	dualClaude = []string{
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-mythos-preview",
	}
	budgetOnlyClaude = []string{
		"claude-opus-4-5", "claude-opus-4-1", "claude-opus-4-0",
		"claude-sonnet-4-5", "claude-sonnet-4-0",
		"claude-haiku-4-5",
		"claude-3-7",
	}
)

// ClaudeReasoningKindFor classifies a Claude model string. Matching is
// case-insensitive and substring-based so provider/region prefixes and
// -vN:0 suffixes do not affect the result.
func ClaudeReasoningKindFor(model string) ClaudeReasoningKind {
	m := canonicalClaude(model)
	if containsAny(m, adaptiveOnlyClaude) {
		return ClaudeReasoningAdaptiveOnly
	}
	if containsAny(m, dualClaude) {
		return ClaudeReasoningAdaptiveAndBudget
	}
	if containsAny(m, budgetOnlyClaude) {
		return ClaudeReasoningBudgetOnly
	}
	return ClaudeReasoningUnknown
}

// ClaudeSupportsThinking reports whether the model is a known extended-thinking
// Claude generation (any tier except Unknown).
func ClaudeSupportsThinking(model string) bool {
	return ClaudeReasoningKindFor(model) != ClaudeReasoningUnknown
}

// alwaysOnClaude models think unconditionally and reject an explicit disable.
// defaultOnClaude models think when thinking is omitted (so forcing them off
// needs an explicit disable, not just omission). Opus 5 and Sonnet 5 default to
// thinking on — a breaking change from Opus 4.8, which defaults off — but,
// unlike Fable 5 / Mythos 5, they still accept an explicit disable (subject to
// the API's own effort ceiling: Anthropic rejects thinking.disabled combined
// with effort xhigh/max on Opus 5, a constraint this package does not model
// because this SDK never sends effort alongside a disable request).
var (
	alwaysOnClaude = []string{
		"claude-fable-5", "claude-mythos-5", "claude-mythos-preview", "claude-opus-5-5",
	}
	betweenToolsOffClaude = []string{"claude-sonnet-5-5"}
	defaultOnClaude       = []string{
		"claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5",
		"claude-fable-5", "claude-mythos-5", "claude-mythos-preview",
	}
)

// ClaudeThinkingAlwaysOn reports whether the model's thinking cannot be disabled,
// so an explicit disable has to be refused rather than sent.
func ClaudeThinkingAlwaysOn(model string) bool {
	return containsAny(canonicalClaude(model), alwaysOnClaude)
}

func ClaudeTurnsOffBetweenTools(model string) bool {
	return containsAny(canonicalClaude(model), betweenToolsOffClaude)
}

// ClaudeThinkingDefaultsOn reports whether the model thinks when thinking is
// omitted, so disabling it requires an explicit disable rather than omission.
func ClaudeThinkingDefaultsOn(model string) bool {
	return containsAny(canonicalClaude(model), defaultOnClaude)
}

var claudeThinkingObjectRoutes = []string{"anthropic", "bedrock", "vertex_ai"}

// ClaudeThinkingObjectRoute reports whether a Claude name on an OpenAI-shaped
// door is bare or carries a LiteLLM route that passes Anthropic's thinking
// object on to a vendor that documents it.
func ClaudeThinkingObjectRoute(model string) bool {
	route, _, prefixed := strings.Cut(model, "/")
	return !prefixed || slices.Contains(claudeThinkingObjectRoutes, route)
}

// claudeEffortsByKind lists the effort levels each generation accepts.
var claudeEffortsByKind = map[ClaudeReasoningKind][]string{
	ClaudeReasoningAdaptiveOnly:      {"low", "medium", "high", "xhigh", "max"},
	ClaudeReasoningAdaptiveAndBudget: {"low", "medium", "high", "max"},
	ClaudeReasoningBudgetOnly:        {"low", "medium", "high"},
}

var bedrockTopEfforts = map[string][]string{
	"xhigh": {"claude-opus-5", "claude-haiku-5-5"},
	"max":   {"claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-5", "claude-haiku-5-5"},
}

func claudeEffortsOn(model string, p Provider) []string {
	accepted := claudeEffortsByKind[ClaudeReasoningKindFor(model)]
	if p != ProviderBedrock {
		return accepted
	}
	served := make([]string, 0, len(accepted))
	for _, level := range accepted {
		if families, gated := bedrockTopEfforts[level]; gated && !claudeNamedIn(model, families) {
			continue
		}
		served = append(served, level)
	}
	return served
}

func claudeNamedIn(model string, families []string) bool {
	for _, form := range modelSpellings(model) {
		name := canonicalClaude(form)
		for _, family := range families {
			if hasGeneration(name, family) {
				return true
			}
		}
	}
	return false
}

// ClaudeEffortsFor returns the effort levels the model's generation accepts on
// the given provider, or nil for a model this package does not classify.
func ClaudeEffortsFor(model string, p Provider) []string {
	kind := ClaudeReasoningKindFor(model)
	if kind == ClaudeReasoningBudgetOnly && !ClaudeSupportsEffortWithBudget(model, p) {
		return nil
	}
	return slices.Clone(claudeEffortsOn(model, p))
}

var claudeEffortRank = map[string]int{"minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5, "max": 6}

// ClaudeClampEffort moves an effort the model does not accept on the provider to
// the nearest one it does: down to the highest accepted level below it, or up
// to the lowest accepted level when every one of them is higher. An
// unclassified model and an unknown level pass through unchanged.
func ClaudeClampEffort(model, effort string, p Provider) string {
	want, ok := claudeEffortRank[effort]
	if !ok {
		return effort
	}
	accepted := claudeEffortsOn(model, p)
	if len(accepted) == 0 || slices.Contains(accepted, effort) {
		return effort
	}
	var below, lowest string
	var belowRank, lowestRank int
	for _, level := range accepted {
		rank := claudeEffortRank[level]
		if rank <= want && rank > belowRank {
			below, belowRank = level, rank
		}
		if lowestRank == 0 || rank < lowestRank {
			lowest, lowestRank = level, rank
		}
	}
	if below == "" {
		return lowest
	}
	return below
}

// ClaudeMinThinkingBudget is the smallest budget_tokens Anthropic accepts.
const ClaudeMinThinkingBudget = 1024

// ClaudeSpendsThinkingBudget reports whether the model pays for thinking out of a
// budget that the answer limit has to exceed.
func ClaudeSpendsThinkingBudget(model string) bool {
	switch ClaudeReasoningKindFor(model) {
	case ClaudeReasoningBudgetOnly, ClaudeReasoningAdaptiveAndBudget:
		return true
	case ClaudeReasoningUnknown, ClaudeReasoningAdaptiveOnly:
		return false
	}
	return false
}

// ClaudeClampBudget raises a budget below the vendor floor up to it, and leaves
// every other model and a zero budget untouched.
func ClaudeClampBudget(model string, budget int) int {
	if budget <= 0 || budget >= ClaudeMinThinkingBudget {
		return budget
	}
	switch ClaudeReasoningKindFor(model) {
	case ClaudeReasoningBudgetOnly, ClaudeReasoningAdaptiveAndBudget:
		return ClaudeMinThinkingBudget
	case ClaudeReasoningUnknown, ClaudeReasoningAdaptiveOnly:
		return budget
	}
	return budget
}

// ClaudeMaxTokensForBudget returns a ceiling that still leaves the answer room
// once the budget is spent.
func ClaudeMaxTokensForBudget(budget, maxTokens int) int {
	if budget <= 0 || maxTokens > budget {
		return maxTokens
	}
	return budget * 2
}

var budgetInterleavingClaude = []string{
	"claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-5", "claude-sonnet-4-0", "claude-sonnet-4-5", "claude-sonnet-4-6",
}

// ClaudeInterleavesOnBudget reports whether budget thinking on the model
// interleaves with tool calls once the interleaved-thinking beta is on.
func ClaudeInterleavesOnBudget(model string) bool {
	return claudeNamedIn(model, budgetInterleavingClaude)
}

// budgetEffortClaude are budget-thinking models that also accept an effort
// output_config alongside manual thinking (introduced with Opus 4.5). Newer
// generations use adaptive thinking, where effort is always available.
var budgetEffortClaude = []string{
	"claude-opus-4-5", "claude-mythos-preview",
	"claude-opus-4-6", "claude-sonnet-4-6",
}

var bedrockRejectsBudgetEffortClaude = []string{"claude-opus-4-5"}

// ClaudeSupportsEffortWithBudget reports whether the model accepts
// output_config.effort together with manual (budget) thinking on the given
// provider.
func ClaudeSupportsEffortWithBudget(model string, p Provider) bool {
	m := canonicalClaude(model)
	if !containsAny(m, budgetEffortClaude) {
		return false
	}
	return p != ProviderBedrock || !containsAny(m, bedrockRejectsBudgetEffortClaude)
}

var noPrefillClaude = []string{"claude-mythos-preview"}

var noForcedToolClaude = []string{
	"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1",
}

// ClaudeRejectsForcedToolUse reports whether the model answers a forced tool
// choice (any or a named tool) with a 400, whatever the thinking settings.
func ClaudeRejectsForcedToolUse(model string) bool {
	return containsAny(canonicalClaude(model), noForcedToolClaude)
}

// ClaudeRejectsAssistantPrefill reports whether the model rejects a prefilled
// assistant response outright, so the request must not be sent: every Claude the
// name asks for at 4.6 or later, whatever release the tables answer it with.
func ClaudeRejectsAssistantPrefill(model string) bool {
	m := claudeName(model)
	if _, id, ok := claudeID(m); ok {
		tier, major, minor, ok := claudeVersion(id)
		if ok && claudeReleases[tier] != nil && (major > 4 || major == 4 && minor >= 6) {
			return true
		}
	}
	return containsAny(m, noPrefillClaude)
}

// mutuallyExclusiveSamplingClaude models reject temperature and top_p set
// together (only one may be provided).
var mutuallyExclusiveSamplingClaude = []string{
	"claude-opus-4-1", "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
	"claude-sonnet-4-6", "claude-opus-4-6",
}

// ClaudeMutuallyExclusiveSampling reports whether the model returns a 400 when
// temperature and top_p are set together, so the caller must send at most one.
func ClaudeMutuallyExclusiveSampling(model string) bool {
	return containsAny(canonicalClaude(model), mutuallyExclusiveSamplingClaude)
}

// legacyNoStructuredClaude are Claude generations known to predate structured
// outputs (the output_config.format JSON Schema mode). A structured-output request
// on them is rejected locally rather than sent for a guaranteed 4xx.
var legacyNoStructuredClaude = []string{
	"claude-2", "claude-v2", "claude-instant",
	"claude-3-",
	"claude-opus-4-1",
	"claude-opus-4-0",
	"claude-sonnet-4-0",
}

// ClaudeSupportsStructuredOutput reports whether the model can be asked for schema
// constrained output. Known-legacy families are rejected; every current model and
// any unrecognized (newer) name passes through so the provider API stays the final
// arbiter and the local table never blocks a future model.
func ClaudeSupportsStructuredOutput(model string) bool {
	return !containsAny(canonicalClaude(model), legacyNoStructuredClaude)
}

var bedrockStructuredClaude = []string{
	"claude-opus-4-5", "claude-opus-4-6", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-haiku-4-5",
}

// ClaudeSupportsStructuredOutputOnBedrock reports whether Amazon Bedrock serves
// schema constrained output for the Claude model.
func ClaudeSupportsStructuredOutputOnBedrock(model string) bool {
	return !ClaudeStructuredOutputBlockedByProfile(model) && claudeNamedIn(model, bedrockStructuredClaude)
}

func ClaudeStructuredOutputBlockedByProfile(model string) bool {
	return strings.HasPrefix(baseModelName(model), "in.") && claudeNamedIn(model, []string{"claude-haiku-4-5"})
}

// ResolveClaudeAdaptive returns whether to send adaptive thinking (true) or
// budget thinking (false) for a Claude model, given the caller's preference
// (adaptivePreferred is true when the caller used WithAdaptiveReasoning).
//
// The rule is deterministic and honors the caller's preference whenever the
// model supports it, falling back only where the preferred mechanism would be
// rejected:
//   - AdaptiveOnly  → always adaptive (budget would 400).
//   - BudgetOnly    → always budget   (adaptive would 400).
//   - AdaptiveAndBudget / Unknown → the caller's preference, unchanged.
//
// So a currently-accepted call keeps its mechanism; only a currently-rejected
// (400) combination is redirected to the mechanism the model accepts.
func ResolveClaudeAdaptive(model string, adaptivePreferred bool) bool {
	if ClaudePredatesAdaptive(model) {
		return false
	}
	switch ClaudeReasoningKindFor(model) {
	case ClaudeReasoningAdaptiveOnly:
		return true
	case ClaudeReasoningBudgetOnly:
		return false
	default: // AdaptiveAndBudget, Unknown
		return adaptivePreferred
	}
}

// preAdaptiveClaude are known Claude generations released before adaptive thinking
// existed (adaptive arrived with Opus 4.6 / Sonnet 4.6). They are "unknown" to the
// reasoning-kind table only because legacy models are not enumerated there; an
// adaptive request on them must not be forwarded as thinking.type=adaptive, which
// these models reject with a 400.
var preAdaptiveClaude = []string{
	"claude-2", "claude-v2",
	"claude-instant",
	"claude-3", // claude-3, claude-3-5, claude-3-7 all predate adaptive
	"claude-opus-4-0", "claude-opus-4-1", "claude-sonnet-4-0",
}

// ClaudePredatesAdaptive reports whether the model is a known pre-adaptive Claude
// generation, so an adaptive request must be gated (not sent verbatim) rather than
// optimistically forwarded the way a genuinely newer, unclassified model is.
func ClaudePredatesAdaptive(model string) bool {
	return containsAny(canonicalClaude(model), preAdaptiveClaude)
}

// rejectsSamplingClaude models are sent no temperature, top_p or top_k on any
// request, whether or not thinking is requested.
var rejectsSamplingClaude = []string{
	"claude-fable-5", "claude-mythos-5", "claude-mythos-preview",
	"claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5",
	"claude-haiku-5-5",
}

// ClaudeThinkingTopPFloor is the lowest top_p Anthropic accepts while the model
// is thinking.
const ClaudeThinkingTopPFloor = 0.95

// ClaudeKeepsTopPWhileThinking reports whether the caller's top_p survives on
// the wire once the model is thinking.
func ClaudeKeepsTopPWhileThinking(model string, topP float64) bool {
	return ClaudeSupportsThinking(model) &&
		!ClaudeRejectsSampling(model) &&
		topP >= ClaudeThinkingTopPFloor
}

func ClaudeClampTemperature(model string, temperature float64) float64 {
	if !isClaudeModel(model) {
		return temperature
	}
	return min(max(temperature, 0), 1)
}

// ClaudeRejectsSampling reports whether the model rejects temperature/top_p
// outright, so sampling params must be dropped even when no thinking is
// requested.
func ClaudeRejectsSampling(model string) bool {
	return containsAny(canonicalClaude(model), rejectsSamplingClaude)
}

func canonicalClaude(model string) string {
	m := claudeName(model)
	if head, id, ok := claudeID(m); ok {
		if documented, ok := documentedClaude(id); ok {
			return head + documented
		}
	}
	return m
}

func claudeName(model string) string {
	m := strings.ToLower(model)
	m = strings.ReplaceAll(m, "@", "-")
	var b strings.Builder
	b.Grow(len(m))
	for i := range len(m) {
		if m[i] == '.' && i > 0 && i+1 < len(m) && isDigit(m[i-1]) && isDigit(m[i+1]) {
			b.WriteByte('-')
			continue
		}
		b.WriteByte(m[i])
	}
	return b.String()
}

func claudeID(m string) (head, id string, ok bool) {
	segment := m[strings.LastIndex(m, "/")+1:]
	idx := strings.Index(segment, "claude-")
	if idx == -1 {
		return "", "", false
	}
	cut := len(m) - len(segment) + idx
	return m[:cut], m[cut:], true
}

func isClaudeTier(s string) bool {
	_, listed := claudeReleases[s]
	return listed
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func containsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
