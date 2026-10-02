package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestConverseReportsAnAdaptiveCallItCannotHonourAsTheLegacyDoorDoes(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"us.amazon.nova-pro-v1:0", "us.amazon.nova-micro-v1:0", "meta.llama3-70b-instruct-v1:0", "zai.glm-4.7",
	} {
		resp, body := bedrockWarningsSending(t, converseAnswer,
			[]bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()}, llms.WithAdaptiveReasoning(""))
		require.NotContains(t, body, "additionalModelRequestFields", model)
		dropped, reported := bedrockWarningsByOption(resp.Warnings)["WithReasoning"]
		require.True(t, reported, "%s: %v", model, resp.Warnings)
		require.Equal(t, llms.WarningDrop, dropped.Kind, model)
		require.Equal(t, "thinking", dropped.Asked, model)
	}

	for _, model := range []string{
		"openai.gpt-oss-120b-1:0", "deepseek.r1-v1:0", "amazon.nova-2-lite-v1:0", "anthropic.claude-opus-6-0-v1:0",
	} {
		resp, _ := bedrockWarningsSending(t, converseAnswer,
			[]bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()}, llms.WithAdaptiveReasoning(""))
		require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithReasoning",
			"%s thinks on its own default depth", model)
	}

	reasoningWarnings := func(resp *llms.ContentResponse) int {
		n := 0
		for _, w := range resp.Warnings {
			if w.Option == "WithReasoning" {
				n++
			}
		}
		return n
	}
	opts := []bedrock.Option{bedrock.WithModel("us.amazon.nova-pro-v1:0"), bedrock.WithConverseAPI()}
	resp, _ := bedrockWarningsSending(t, converseAnswer, opts, llms.WithReasoning(llms.ReasoningLow, 0))
	require.Equal(t, 1, reasoningWarnings(resp), "an asked effort is reported once: %v", resp.Warnings)
	resp, _ = bedrockWarningsSending(t, converseAnswer, opts, llms.WithReasoningDisabled())
	require.Zero(t, reasoningWarnings(resp), "thinking asked off is what goes out: %v", resp.Warnings)
}
