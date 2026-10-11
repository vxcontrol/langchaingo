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
