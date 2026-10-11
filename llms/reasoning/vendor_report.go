package reasoning

// ClaudeReport is what Anthropic's model API reports about one model.
type ClaudeReport struct {
	Reported bool
	Thinking bool
	Adaptive bool
	Budget   bool
	// Disable is whether the API accepts thinking.type disabled, nil where the
	// response does not say.
	Disable           *bool
	Effort            bool
	Efforts           []string
	StructuredOutputs bool
}

func (r ClaudeReport) Kind() ClaudeReasoningKind {
	switch {
	case r.Adaptive && r.Budget:
		return ClaudeReasoningAdaptiveAndBudget
	case r.Adaptive:
		return ClaudeReasoningAdaptiveOnly
	case r.Budget:
		return ClaudeReasoningBudgetOnly
	}
	return ClaudeReasoningUnknown
}

func ClaudeDocumented(model string) bool {
	if _, inherited := InheritedModel(model); inherited {
		return false
	}
	return ClaudeReasoningKindFor(model) != ClaudeReasoningUnknown || ClaudePredatesAdaptive(model)
}

func GeminiDocumented(model string) bool {
	if _, inherited := InheritedModel(model); inherited {
		return false
	}
	return GeminiSupportsThinking(model) || geminiKnownNonThinking(model) || geminiNonChatSurface(baseModelName(model))
}

// HostVendor reports the vendor that owns host, leaving out the LiteLLM route
// prefixes ServedBy reads off other hosts.
func HostVendor(host string) Vendor {
	return vendorOfHost(host)
}

// DocumentedOn reports whether the tables answer for model on vendor's own API,
// so that what the API reports about the model does not decide.
func DocumentedOn(model string, vendor Vendor) bool {
	_, inherited := InheritedModel(model)
	switch vendor {
	case VendorXAI:
		return !inherited && (OpenAIReasoningCapsFor(model).Known || !AcceptsEffortWire(model))
	case VendorMoonshot:
		return !inherited && (listedRelease(model) || IsReasoningModel(model))
	case VendorMistral:
		for _, form := range modelSpellings(model) {
			if mistralWithoutReasoning(form) {
				return false
			}
		}
		return true
	default:
		return true
	}
}
