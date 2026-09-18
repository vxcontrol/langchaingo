package reasoning

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// RejectsPenalties reports whether frequency_penalty and presence_penalty
// must stay off the wire.
func RejectsPenalties(model string) bool {
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "grok") || strings.HasPrefix(form, "deepseek") {
			return true
		}
	}
	return false
}

// deepSeekAPIModels mirrors the model names DeepSeek's own API serves, as its
// Models & Pricing page lists them.
var deepSeekAPIModels = []string{"deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}

const deepSeekAPIHost = "api.deepseek.com"

func ServedByDeepSeek(model, host string) bool {
	m := strings.ToLower(model)
	if rest, ok := strings.CutPrefix(m, "deepseek/"); ok {
		return slices.Contains(deepSeekAPIModels, rest)
	}
	return host == deepSeekAPIHost && slices.Contains(deepSeekAPIModels, m)
}

// RejectsTopK reports whether top_k must stay off the wire.
func RejectsTopK(model string) bool {
	return openAIProperName(model)
}

func TakesNoTopK(model string) bool {
	return onMiniMaxAPI(model, "minimax-")
}

func TakesNoResponseFormat(model string) bool {
	return onMiniMaxAPI(model, "minimax-m")
}

func onMiniMaxAPI(model, family string) bool {
	return strings.HasPrefix(strings.TrimPrefix(strings.ToLower(model), "minimax/"), family)
}

// RejectsRepetitionPenalty reports whether repetition_penalty must stay off the wire.
func RejectsRepetitionPenalty(model string) bool {
	return openAIProperName(model)
}

// openAIProperName reports whether the name belongs to OpenAI's own catalogue
// rather than to another vendor this door serves.
func openAIProperName(model string) bool {
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "gpt-oss") {
			return false
		}
		if strings.HasPrefix(form, "gpt-") || strings.HasPrefix(form, "chatgpt-") ||
			strings.HasPrefix(form, "o1") || strings.HasPrefix(form, "o3") ||
			strings.HasPrefix(form, "o4-") {
			return true
		}
	}
	return false
}

// RejectsSamplingWhileThinking reports whether a thinking request must travel
// without the caller's temperature and top_p.
func RejectsSamplingWhileThinking(model string) bool {
	return openAIProperName(model) && OpenAIReasoningCapsFor(model).Known
}

// FixesSampling reports whether the model runs on fixed temperature, top_p and
// penalties, so any value the caller sets for them stays off the wire.
func FixesSampling(model string) bool {
	for _, form := range modelSpellings(model) {
		if hasGeneration(form, "kimi-k3") || hasGeneration(form, "kimi-k2.6") ||
			hasGeneration(form, "kimi-k2.7-code") {
			return true
		}
	}
	return false
}

// TakesNoJSONSchema reports whether the vendor's chat completions response_format
// takes only text and json_object, so a JSON schema cannot be asked for.
func TakesNoJSONSchema(model string) bool {
	if ServedByMistral(model) {
		return false
	}
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "deepseek") || strings.HasPrefix(form, "glm-") {
			return true
		}
	}
	return false
}

func TakesNoJSONObject(model string) bool {
	return isClaudeModel(model)
}

// RejectsMinP reports whether min_p must stay off the wire.
func RejectsMinP(model string) bool {
	return isClaudeModel(model) || openAIProperName(model)
}

// ReplaysReasoningOnEveryTurn reports whether reasoning_content goes back on every
// earlier assistant turn, not only on the turns that called a tool.
func ReplaysReasoningOnEveryTurn(model string) bool {
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "deepseek") || strings.HasPrefix(form, "kimi-") || strings.HasPrefix(form, "glm-") ||
			strings.HasPrefix(form, "qwen") {
			return true
		}
	}
	return false
}

// ReplaysThinkingInContent reports, for a model Mistral serves, whether its
// earlier assistant turns take their reasoning back, which they take only as a
// thinking chunk at the head of content, on every turn.
func ReplaysThinkingInContent(model string) bool {
	if !ServedByMistral(model) {
		return false
	}
	for _, form := range modelSpellings(model) {
		if mistralWithoutReasoning(form) {
			return false
		}
	}
	return true
}

func ReplaysReasoningInThinkTags(model string) bool {
	return onMiniMaxAPI(model, "minimax-m")
}

// UsesLegacyMaxTokens reports whether the output limit must travel as
// max_tokens rather than max_completion_tokens.
func UsesLegacyMaxTokens(model string) bool {
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "grok") ||
			strings.HasPrefix(form, "qwen") ||
			strings.HasPrefix(form, "qwq") ||
			strings.HasPrefix(form, "qvq") ||
			strings.HasPrefix(form, "deepseek") {
			return true
		}
	}
	return false
}

// AcceptsEffortWire reports whether the model's OpenAI-compatible door takes the
// reasoning_effort field. A door that refuses it fails the whole request, so such
// a model also has no way to spell "off": the disable token rides on this field.
func AcceptsEffortWire(model string) bool {
	for _, form := range modelSpellings(model) {
		if hasGeneration(form, "qwen3.8") {
			return true
		}
		if strings.HasPrefix(form, "qwen") || strings.HasPrefix(form, "qwq") {
			return false
		}
		if strings.HasPrefix(form, "gpt-3.5") || strings.HasPrefix(form, "gpt-4") {
			return false
		}
		if mistralWithoutReasoning(form) {
			return false
		}
		if thinksBySwitchOnly(model, form) {
			return false
		}
		if form == "grok-build-latest" {
			return true
		}
		if strings.HasPrefix(form, "grok-code-fast") ||
			strings.HasPrefix(form, "grok-build") ||
			strings.HasPrefix(form, "grok-4.20") {
			return false
		}
	}
	return true
}

// TakesNoThinkingDepth reports whether the model's own door takes neither a
// thinking budget nor an effort level.
func TakesNoThinkingDepth(model string) bool {
	if dashScopeGuestSpelling(model) != "" {
		return false
	}
	for _, form := range modelSpellings(model) {
		if thinksBySwitchOnly(model, form) {
			return true
		}
	}
	return false
}

// ErrThinkingBudgetUnsupported reports a thinking budget asked of a model that
// TakesNoThinkingDepth.
type ErrThinkingBudgetUnsupported struct{ Model string }

func (e *ErrThinkingBudgetUnsupported) Error() string {
	return fmt.Sprintf("model %q takes no thinking budget or effort level; leave the budget unset", e.Model)
}

func thinksBySwitchOnly(model, form string) bool {
	return hasGeneration(form, "kimi-k2") || glmBeforeEffortField(model, form) ||
		strings.HasPrefix(form, "minimax-m")
}

func glmBeforeEffortField(model, form string) bool {
	if ServedByMistral(model) {
		return false
	}
	major, minor, ok := glmGeneration(form)
	if !ok {
		return false
	}
	firstMinor := 2
	if dashScopeGuestSpelling(model) != "" {
		firstMinor = 1
	}
	return major < 5 || major == 5 && minor < firstMinor
}

func glmGeneration(form string) (major, minor int, ok bool) {
	rest, found := strings.CutPrefix(form, "glm-")
	if !found {
		return 0, 0, false
	}
	major, rest, ok = leadingNumber(rest)
	if !ok {
		return 0, 0, false
	}
	if after, dotted := strings.CutPrefix(rest, "."); dotted {
		minor, _, _ = leadingNumber(after)
	}
	return major, minor, true
}

func leadingNumber(s string) (n int, rest string, ok bool) {
	end := 0
	for end < len(s) && isDigit(s[end]) {
		end++
	}
	if end == 0 {
		return 0, s, false
	}
	n, err := strconv.Atoi(s[:end])
	return n, s[end:], err == nil
}

// EffortToolsRule reports what to do with reasoning_effort when function tools
// ride on the same request.
type EffortToolsRule int

const (
	EffortToolsFree EffortToolsRule = iota
	EffortToolsOmit
	EffortToolsDisable
)

type ErrEffortWithTools struct {
	Model  string
	Effort string
}

func (e *ErrEffortWithTools) Error() string {
	return fmt.Sprintf(
		"model %q rejects reasoning effort %q on a request carrying function tools; "+
			"the vendor serves this combination on the responses API only",
		e.Model, e.Effort)
}

// EffortWithTools reports the rule for a model. An unlisted generation stays free.
func EffortWithTools(model string) EffortToolsRule {
	for _, form := range modelSpellings(model) {
		if hasGeneration(form, "gpt-5.6") {
			return EffortToolsDisable
		}
		if hasGeneration(form, "gpt-5.4") || hasGeneration(form, "gpt-5.5") {
			return EffortToolsOmit
		}
	}
	return EffortToolsFree
}
