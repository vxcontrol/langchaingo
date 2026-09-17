package llms

import (
	"strings"
	"sync"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

// ReasoningSupport describes which reasoning controls a model accepts, as a HINT
// for building a UI. It is a best-effort static projection, never authoritative:
// an unrecognized model returns Known=false so the UI shows all controls and the
// provider API (an HTTP 400) is the ultimate arbiter. The table asserts only
// KNOWN facts; it never fabricates a capability it cannot back — unknown effort
// tiers and default state are left nil rather than guessed.
type ReasoningSupport struct {
	// Supported reports whether the model reasons at all.
	Supported bool
	// Known reports whether this model is explicitly classified. When false, the
	// remaining fields are best-effort and the UI should offer all controls.
	Known bool
	// CannotDisable is set only for models KNOWN to reject disabling
	// (always-on Claude such as Fable 5, and OpenAI o-series). When false,
	// disabling may still fail at the API for an unclassified model.
	CannotDisable bool
	// RejectsSampling is set for models that reject temperature/top_p while thinking.
	RejectsSampling bool
	// Efforts are the effort tiers worth offering; empty both when the model is
	// unknown and when it takes none, so read Mechanism before offering a control.
	Efforts []ReasoningEffort
	// Mechanism reports whether the model takes an effort level, a token budget
	// or either; ReasoningMechanismUnknown when unclassified.
	Mechanism ReasoningMechanism
	// DefaultOn reports whether thinking runs when reasoning is unset; nil when unknown.
	DefaultOn *bool
}

// ReasoningMechanism describes how a model takes its thinking instruction.
type ReasoningMechanism int

const (
	ReasoningMechanismUnknown ReasoningMechanism = iota
	ReasoningMechanismAdaptive
	ReasoningMechanismBudget
	ReasoningMechanismAdaptiveAndBudget
)

var (
	reasoningOverridesMu sync.RWMutex
	reasoningOverrides   = map[string]ReasoningSupport{}
)

// RegisterReasoningSupport registers a UI hint for models this build does not
// classify (a proxy alias, or a model newer than this build). A model whose
// string contains pattern reports info from ReasoningSupportFor.
//
// Scope: this affects ONLY the ReasoningSupportFor hint, not the wire path. The
// enable/disable resolvers (reasoning.ResolveOff, reasoning.ResolveClaudeAdaptive,
// effort clamping) live in the lower-level reasoning package, which cannot read
// this registry, so a registered model still travels the optimistic pass-through
// path on the wire and the provider API remains the final arbiter.
func RegisterReasoningSupport(pattern string, info ReasoningSupport) {
	reasoningOverridesMu.Lock()
	defer reasoningOverridesMu.Unlock()
	reasoningOverrides[strings.ToLower(pattern)] = info
}

func lookupReasoningOverride(model string) (ReasoningSupport, bool) {
	reasoningOverridesMu.RLock()
	defer reasoningOverridesMu.RUnlock()
	m := strings.ToLower(model)
	for pattern, info := range reasoningOverrides {
		if strings.Contains(m, pattern) {
			return info, true
		}
	}
	return ReasoningSupport{}, false
}

func boolPtr(b bool) *bool { return &b }

// ReasoningSupportFor returns the reasoning-control hint for a model on a
// provider. Registered overrides win; then the models the shared tables classify
// for that provider; everything else returns Known=false
// (optimistic — the UI shows all controls and the API rejects what it cannot do).
func ReasoningSupportFor(model string, p reasoning.Provider) ReasoningSupport {
	if s, ok := lookupReasoningOverride(model); ok {
		return s
	}

	if reasoning.ClaudeSupportsThinking(model) {
		return ReasoningSupport{
			Supported:       true,
			Known:           true,
			CannotDisable:   reasoning.ResolveOff(model, p) == reasoning.OffUnsupported,
			RejectsSampling: reasoning.ClaudeRejectsSampling(model),
			Efforts:         toReasoningEfforts(reasoning.ClaudeEffortsFor(model, p)),
			Mechanism:       claudeMechanism(reasoning.ClaudeReasoningKindFor(model)),
			DefaultOn:       boolPtr(reasoning.ClaudeThinkingDefaultsOn(model)),
		}
	}

	if p == reasoning.ProviderBedrock && reasoning.IsGptOssModel(model) {
		return levelOnlySupport(model, p, reasoning.GptOssEfforts(), true)
	}

	if p == reasoning.ProviderBedrock && reasoning.IsNovaReasoningModel(model) {
		return levelOnlySupport(model, p, reasoning.NovaEfforts(), false)
	}

	if p == reasoning.ProviderBedrock && reasoning.IsBedrockNonReasoningModel(model) {
		return ReasoningSupport{Known: true, DefaultOn: boolPtr(false)}
	}

	if caps := reasoning.OpenAIReasoningCapsFor(model); caps.Known {
		return ReasoningSupport{
			Supported:       true,
			Known:           true,
			CannotDisable:   !caps.CanDisable,
			RejectsSampling: reasoning.RejectsSamplingWhileThinking(model),
			Efforts:         toReasoningEfforts(caps.Efforts),
			Mechanism:       ReasoningMechanismAdaptive,
			// DefaultOn is per-model on the GPT-5.x line (some default off) — leave unknown.
		}
	}

	if p == reasoning.ProviderOpenAI && reasoning.IsReasoningModel(model) {
		var defaultOn *bool
		if reasoning.QwenThinkingEnabledByFlag(model) {
			defaultOn = boolPtr(false)
		}
		return ReasoningSupport{
			Supported:       true,
			CannotDisable:   reasoning.ResolveOff(model, p) == reasoning.OffUnsupported,
			RejectsSampling: reasoning.RejectsSamplingWhileThinking(model),
			DefaultOn:       defaultOn,
		}
	}

	if p == reasoning.ProviderGoogleAI && reasoning.GeminiSupportsThinking(model) {
		// Efforts stays unset: this package does not classify the Google level names.
		mechanism := ReasoningMechanismBudget
		switch {
		case reasoning.GeminiUsesThinkingLevel(model), reasoning.GeminiTogglesThinkingByLevel(model):
			mechanism = ReasoningMechanismAdaptive
		}
		return ReasoningSupport{
			CannotDisable: reasoning.ResolveOff(model, p) == reasoning.OffUnsupported,
			Supported:     true,
			Known:         true,
			Mechanism:     mechanism,
			DefaultOn:     boolPtr(!reasoning.GeminiThinkingOffByDefault(model)),
		}
	}

	if efforts := reasoning.OllamaEffortsFor(model); p == reasoning.ProviderOllama && len(efforts) > 0 {
		return levelOnlySupport(model, p, efforts, true)
	}

	return ReasoningSupport{
		Supported:     reasoning.LikelyReasoningModel(model),
		Known:         false,
		CannotDisable: reasoning.ResolveOff(model, p) == reasoning.OffUnsupported,
	}
}

func levelOnlySupport(model string, p reasoning.Provider, efforts []string, defaultOn bool) ReasoningSupport {
	return ReasoningSupport{
		Supported:     true,
		Known:         true,
		CannotDisable: reasoning.ResolveOff(model, p) == reasoning.OffUnsupported,
		Efforts:       toReasoningEfforts(efforts),
		Mechanism:     ReasoningMechanismAdaptive,
		DefaultOn:     boolPtr(defaultOn),
	}
}

// toReasoningEfforts converts the resolver's raw effort strings to the public
// ReasoningEffort type for UI hints.
func toReasoningEfforts(efforts []string) []ReasoningEffort {
	if len(efforts) == 0 {
		return nil
	}
	out := make([]ReasoningEffort, len(efforts))
	for i, e := range efforts {
		out[i] = ReasoningEffort(e)
	}
	return out
}

func claudeMechanism(kind reasoning.ClaudeReasoningKind) ReasoningMechanism {
	switch kind {
	case reasoning.ClaudeReasoningAdaptiveOnly:
		return ReasoningMechanismAdaptive
	case reasoning.ClaudeReasoningBudgetOnly:
		return ReasoningMechanismBudget
	case reasoning.ClaudeReasoningAdaptiveAndBudget:
		return ReasoningMechanismAdaptiveAndBudget
	default:
		return ReasoningMechanismUnknown
	}
}
