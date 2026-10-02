package openai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAToolChoiceStaysOffTheWireWhenTheCallOffersNoTools(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5.6-sol", "claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"} {
		for _, choice := range []any{"required", "auto", "none", llms.ToolChoice{Type: "any"}} {
			body, err := wireBodyOf(t, model, nil, llms.WithToolChoice(choice), llms.WithReasoning(llms.ReasoningNone, 2048))
			require.NoError(t, err, "%s %v", model, choice)
			require.NotContains(t, body, "tool_choice", "%s %v", model, choice)
		}
	}

	body, err := wireBodyOf(t, "gpt-5.6-sol", nil, llms.WithToolChoice("required"), turnLimitTools)
	require.NoError(t, err)
	require.Equal(t, "required", body["tool_choice"])
}
