package openai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func droppedDelegationFor(t *testing.T, model string) bool {
	t.Helper()

	warn := &llms.Warnings{}
	req := &openaiclient.ChatRequest{}
	llm := newUnitLLM(t, WithModel(model))

	_, err := llm.setReasoning(req, llms.CallOptions{
		Model:     &model,
		Reasoning: &llms.ReasoningConfig{Adaptive: true},
	}, warn)
	require.NoError(t, err)

	for _, w := range warn.List() {
		if w.Option == "WithAdaptiveReasoning" {
			return true
		}
	}
	return false
}

func TestEveryOptInModelReportsTheDelegationItCannotCarry(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5.1", "deepseek-v3.1", "magistral-medium-2509"} {
		assert.True(t, droppedDelegationFor(t, model),
			"%s does not reason unless asked, so a dropped delegation must be reported", model)
	}
}

func TestAModelThatAlreadyReasonsIsNotReportedAsDropped(t *testing.T) {
	t.Parallel()

	assert.False(t, droppedDelegationFor(t, "o3-mini"),
		"a model that reasons by default loses nothing when the depth is left to it")
}
