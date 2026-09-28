package ollama

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestADelegationThisDoorCannotExpressIsReported(t *testing.T) {
	t.Parallel()

	warn := &llms.Warnings{}
	opts := llms.CallOptions{Reasoning: &llms.ReasoningConfig{Adaptive: true}}
	reportOllamaOptions(warn, "gpt-oss:120b", opts)

	var found *llms.Warning
	for _, w := range warn.List() {
		if w.Option == "WithAdaptiveReasoning" {
			found = &w
			break
		}
	}
	require.NotNil(t, found, "the door sends no think field and must say so: %v", warn.List())
	assert.Equal(t, llms.WarningDrop, found.Kind)
}

func TestAnEffortTheCallerNamedIsNotReportedAsDropped(t *testing.T) {
	t.Parallel()

	warn := &llms.Warnings{}
	opts := llms.CallOptions{Reasoning: &llms.ReasoningConfig{Adaptive: true, Effort: llms.ReasoningLow}}
	reportOllamaOptions(warn, "gpt-oss:120b", opts)

	for _, w := range warn.List() {
		assert.NotEqual(t, "WithAdaptiveReasoning", w.Option,
			"the effort reaches the wire, so nothing was dropped")
	}
}
