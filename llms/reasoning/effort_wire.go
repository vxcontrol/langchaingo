package reasoning

import (
	"fmt"
	"strings"
)

// RejectsPenalties reports whether frequency_penalty and presence_penalty
// must stay off the wire.
func RejectsPenalties(model string) bool {
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "grok") {
			return true
		}
	}
	return false
}

// RejectsTopK reports whether top_k must stay off the wire.
func RejectsTopK(model string) bool {
	return openAIProperName(model)
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

// RejectsMinP reports whether min_p must stay off the wire.
func RejectsMinP(model string) bool {
	return isClaudeModel(model) || openAIProperName(model)
}

// ReplaysReasoningOnEveryTurn reports whether reasoning_content goes back on every
// earlier assistant turn, not only on the turns that called a tool.
func ReplaysReasoningOnEveryTurn(model string) bool {
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "deepseek") {
			return true
		}
	}
	return false
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

// EffortToolsRule reports what to do with reasoning_effort when function tools
// ride on the same request.
type EffortToolsRule int

const (
	EffortToolsFree EffortToolsRule = iota
	EffortToolsOmit
	// EffortToolsDisable puts an explicit "none" on the wire; an omitted field is
	// not equivalent.
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
