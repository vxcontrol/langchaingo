package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestABudgetAboveTheAnswerLimitNeedsRealInterleaving(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "f", Parameters: map[string]any{"type": "object"},
	}}})
	budget := llms.WithReasoning("", 1024)

	for _, tc := range []struct {
		model string
		want  float64
	}{
		{"claude-haiku-4-5", 2048},
		{"claude-opus-4-6", 2048},
		{"claude-sonnet-4-5", 1000},
		{"claude-opus-4-5", 1000},
		{"claude-sonnet-4-6", 1000},
	} {
		body, header := captureMessagesRequestModel(t, tc.model, llms.WithMaxTokens(1000), budget, tools)
		assert.Equal(t, tc.want, body["max_tokens"], tc.model)
		assert.Contains(t, header.Get("Anthropic-Beta"), "interleaved-thinking-2025-05-14", tc.model)
	}

	body, _ := captureMessagesRequestModel(t, "claude-haiku-4-5", llms.WithMaxTokens(1000), budget)
	assert.Equal(t, float64(2048), body["max_tokens"], "without tools the budget rule was already kept")

	for _, model := range []string{"claude-opus-4", "claude-opus-4-0", "claude-sonnet-4-20250514", "anthropic.claude-opus-4-1-20250805-v1:0"} {
		assert.True(t, reasoning.ClaudeInterleavesOnBudget(model), model)
	}
	for _, model := range []string{"claude-haiku-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-3-7-sonnet", "claude-mythos-preview"} {
		assert.False(t, reasoning.ClaudeInterleavesOnBudget(model), model)
	}
}
