package reasoning

import (
	"fmt"
	"slices"
	"strings"
)

// openAIEffortRank orders OpenAI reasoning efforts so a requested effort can be
// clamped to a model's ceiling. It does not include "none", which is the disable
// token handled by ResolveOff, not an effort level.
var openAIEffortRank = map[string]int{
	"minimal": 1,
	"low":     2,
	"medium":  3,
	"high":    4,
	"xhigh":   5,
	"max":     6,
}

// OpenAIReasoningCaps is a best-effort static projection of which reasoning
// efforts a model reachable on an OpenAI-compatible door accepts, so callers avoid
// sending a value the API would reject with a 400. It asserts only documented, non-default constraints; a
// model outside every listed line returns Known=false and is sent as requested.
type OpenAIReasoningCaps struct {
	// Known reports whether the model was explicitly classified.
	Known bool
	// CanDisable reports whether the model accepts reasoning_effort "none".
	CanDisable bool
	// Efforts are the accepted effort levels in ascending order, excluding "none";
	// nil when unknown.
	Efforts []string
}

// ClampEffort moves a requested effort to one the model accepts: above the
// ceiling drops to the ceiling, below the floor rises to the floor, and anything
// the set does not list steps down to the nearest level it does. Unknown models
// and efforts outside the scale are returned unchanged.
func (c OpenAIReasoningCaps) ClampEffort(effort string) string {
	want, ranked := openAIEffortRank[effort]
	if !c.Known || !ranked || len(c.Efforts) == 0 {
		return effort
	}
	if slices.Contains(c.Efforts, effort) {
		return effort
	}

	accepted := c.Efforts[0]
	for _, level := range c.Efforts {
		if openAIEffortRank[level] <= want {
			accepted = level
		}
	}
	return accepted
}

// OpenAIReasoningCapsFor classifies a reasoning model by the effort set its
// generation accepts on /chat/completions. An unlisted version of a listed line
// answers as the release it follows; a model outside every line returns
// Known=false and the API arbitrates.
func OpenAIReasoningCapsFor(model string) OpenAIReasoningCaps {
	for _, form := range modelSpellings(model) {
		if caps := openAICapsForForm(form); caps.Known {
			return caps
		}
	}
	return OpenAIReasoningCaps{Known: false}
}

func openAIChatVariant(m string) bool {
	return strings.HasPrefix(m, "gpt-") && strings.Contains(m, "-chat")
}

func openAICapsForForm(m string) OpenAIReasoningCaps {
	if openAIChatVariant(m) {
		return OpenAIReasoningCaps{Known: false}
	}
	switch {
	case strings.HasPrefix(m, "gpt-oss"):
		caps := gptOssCaps
		caps.Efforts = GptOssEfforts()
		return caps
	case openAIProVariant(m):
		if openAIXHighCeiling(m) {
			return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"medium", "high", "xhigh"}}
		}
		return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"high"}}
	case openAIMandatoryReasoning(m):
		return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"low", "medium", "high"}}
	case openAIGPT5Base(m):
		return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"minimal", "low", "medium", "high"}}
	case hasGeneration(m, "gpt-5.1"):
		return OpenAIReasoningCaps{Known: true, CanDisable: true, Efforts: []string{"low", "medium", "high"}}
	case hasGeneration(m, "gpt-6-astra"):
		return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"low", "medium", "high", "xhigh"}}
	case hasGeneration(m, "gpt-6.1-sol"):
		return OpenAIReasoningCaps{
			Known: true, CanDisable: false, Efforts: []string{"low", "medium", "high", "xhigh", "max"},
		}
	case hasGeneration(m, "gpt-6-sol") || hasGeneration(m, "gpt-6-luna"):
		return OpenAIReasoningCaps{Known: true, CanDisable: true, Efforts: []string{"low", "medium", "high", "xhigh"}}
	case openAIXHighCeiling(m):
		return OpenAIReasoningCaps{Known: true, CanDisable: true, Efforts: []string{"low", "medium", "high", "xhigh"}}
	case hasGeneration(m, "grok-4.6") || hasGeneration(m, "grok-4.7"):
		return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"low", "medium", "high", "xhigh"}}
	case hasGeneration(m, "grok-4.5"):
		return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"low", "medium", "high"}}
	case nonOpenAILowHighMax(m):
		return OpenAIReasoningCaps{Known: true, CanDisable: false, Efforts: []string{"low", "high", "max"}}
	case mistralReasons(m):
		return OpenAIReasoningCaps{Known: true, CanDisable: true, Efforts: []string{"high"}}
	default:
		return OpenAIReasoningCaps{Known: false}
	}
}

func openAIProVariant(m string) bool {
	return hasGeneration(m, "gpt-5") && strings.Contains(m, "-pro")
}

func openAIGPT5Base(m string) bool {
	return m == "gpt-5" || strings.HasPrefix(m, "gpt-5-20") ||
		hasGeneration(m, "gpt-5-mini") || hasGeneration(m, "gpt-5-nano")
}

func nonOpenAILowHighMax(m string) bool {
	return hasGeneration(m, "kimi-k3") || hasGeneration(m, "glm-5.3") || hasGeneration(m, "glm-5-3")
}

func openAIXHighCeiling(m string) bool {
	for _, generation := range []string{"gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5.6"} {
		if hasGeneration(m, generation) {
			return true
		}
	}
	return false
}

// OpenAIThinkingOptIn reports the generations that reason only when an effort
// asks them to.
func OpenAIThinkingOptIn(model string) bool {
	for _, form := range modelSpellings(model) {
		if openAIProVariant(form) {
			continue
		}
		for _, generation := range []string{"gpt-5.1", "gpt-5.2", "gpt-5.4"} {
			if hasGeneration(form, generation) {
				return true
			}
		}
	}
	return false
}

// OpenAIDisableEffort turns thinking off on the wire. It is not llms.ReasoningNone,
// which is the empty string and instead omits the field.
const OpenAIDisableEffort = "none"

// ChatToolsUnsupported reports whether OpenAI serves the model's function tools
// on the Responses API only.
func ChatToolsUnsupported(model string) bool {
	for _, form := range modelSpellings(model) {
		if hasGeneration(form, "gpt-6-astra") || hasGeneration(form, "gpt-6.1-sol") {
			return true
		}
	}
	return false
}

// OpenAITakesResponses reports whether a call to OpenAI's own API goes to the
// Responses API rather than Chat Completions.
func OpenAITakesResponses(model string, tools bool, mode ThinkingMode) bool {
	switch {
	case ChatCompletionsUnsupported(model):
		return true
	case !tools:
		return false
	case ChatToolsUnsupported(model):
		return true
	case mode == ThinkingOff, EffortWithTools(model) == EffortToolsFree:
		return false
	}
	return mode != ThinkingDefault || !OpenAIThinkingOptIn(model)
}

func TakesPromptCacheOptions(model string) bool {
	major, minor, ok := generationAfter("gpt-", routedName(model))
	return ok && (major > 5 || major == 5 && minor >= 6)
}

func ChatCompletionsUnsupported(model string) bool {
	for _, form := range modelSpellings(model) {
		for _, family := range []string{"gpt-5.6-cyber", "gpt-daybreak-red", "gpt-daybreak-blue"} {
			if hasGeneration(form, family) {
				return true
			}
		}
	}
	return false
}

// Deprecated: no door returns ErrChatCompletionsUnsupported.
type ErrChatCompletionsUnsupported struct {
	Model string
}

func (e *ErrChatCompletionsUnsupported) Error() string {
	return fmt.Sprintf("model %q is served only by the responses API, not by chat completions", e.Model)
}

// ErrChatToolsUnsupported reports a request that carries function tools for a
// model whose chat completions endpoint does not serve them.
type ErrChatToolsUnsupported struct {
	Model string
}

func (e *ErrChatToolsUnsupported) Error() string {
	return fmt.Sprintf(
		"model %q carries function tools on chat completions, where the vendor does not serve them; "+
			"the responses API is the only door for tools on this model",
		e.Model)
}
