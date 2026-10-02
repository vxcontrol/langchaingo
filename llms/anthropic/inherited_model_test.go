package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAnUnlistedClaudeVersionIsSentTheShapeOfTheReleaseItFollows(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-opus-6", "claude-opus-5-6", "claude-opus-4-10"} {
		body, _ := captureMessagesRequestModel(t, model,
			llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithTemperature(0.7))
		thinking, _ := body["thinking"].(map[string]any)
		require.Equal(t, "adaptive", thinking["type"], "%s: %v", model, body)
		require.NotContains(t, thinking, "budget_tokens", model)
		require.NotContains(t, body, "temperature", model)
		config, _ := body["output_config"].(map[string]any)
		require.Equal(t, "high", config["effort"], "%s: %v", model, body)
	}
}
