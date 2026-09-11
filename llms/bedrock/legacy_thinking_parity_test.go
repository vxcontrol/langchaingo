package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAClaudeThatCannotThinkSaysSoOnTheLegacyDoor(t *testing.T) {
	t.Parallel()

	resp := bedrockWarningsFor(t, legacyAnswer,
		[]bedrock.Option{bedrock.WithModel("anthropic.claude-instant-v1")},
		llms.WithAdaptiveReasoning(""))

	w, ok := bedrockWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "the ask to think went nowhere unreported: %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "thinking", w.Asked)
}

func TestBothNovaDoorsReportTheMechanismTheyPicked(t *testing.T) {
	t.Parallel()

	const novaAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

	for _, door := range []struct {
		name string
		opts []bedrock.Option
	}{
		{"legacy", []bedrock.Option{bedrock.WithModel("amazon.nova-2-lite-v1:0")}},
		{"converse", []bedrock.Option{
			bedrock.WithModel("amazon.nova-2-lite-v1:0"), bedrock.WithConverseAPI(),
		}},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			answer := novaAnswer
			if door.name == "converse" {
				answer = converseAnswer
			}
			resp := bedrockWarningsFor(t, answer, door.opts,
				llms.WithAdaptiveReasoning(llms.ReasoningHigh))

			w, ok := bedrockWarningsByOption(resp.Warnings)["WithAdaptiveReasoning"]
			require.True(t, ok, "the door picked the mechanism unreported: %v", resp.Warnings)
			require.Equal(t, llms.WarningSubstitute, w.Kind)
			require.Equal(t, "adaptive", w.Asked)
		})
	}
}
