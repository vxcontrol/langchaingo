package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAMechanismTheConverseDoorPickedInsteadOfAdaptiveIsReported(t *testing.T) {
	t.Parallel()

	for _, family := range []struct {
		name  string
		model string
		sent  string
	}{
		{"nova", "amazon.nova-2-lite-v1:0", "effort"},
		{"grok", "xai.grok-4-v1:0", "effort"},
		{"gpt-oss", "openai.gpt-oss-120b-1:0", "effort"},
		{"claude on a budget", "anthropic.claude-sonnet-4-5-v1:0", "enabled"},
	} {
		t.Run(family.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, converseAnswer,
				[]bedrock.Option{bedrock.WithModel(family.model), bedrock.WithConverseAPI()},
				llms.WithAdaptiveReasoning(llms.ReasoningHigh))

			w, ok := bedrockWarningsByOption(resp.Warnings)["WithAdaptiveReasoning"]
			require.True(t, ok, "the door picked the mechanism unreported: %v", resp.Warnings)
			require.Equal(t, llms.WarningSubstitute, w.Kind)
			require.Equal(t, "adaptive", w.Asked)
			require.Equal(t, family.sent, w.Sent)
		})
	}
}

func TestDelegatedDepthOnConverseIsNotReportedAsAPickedMechanism(t *testing.T) {
	t.Parallel()

	for _, family := range []struct {
		name  string
		model string
	}{
		{"nova", "amazon.nova-2-lite-v1:0"},
		{"grok", "xai.grok-4-v1:0"},
		{"gpt-oss", "openai.gpt-oss-120b-1:0"},
	} {
		t.Run(family.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, converseAnswer,
				[]bedrock.Option{bedrock.WithModel(family.model), bedrock.WithConverseAPI()},
				llms.WithAdaptiveReasoning(""))

			require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithAdaptiveReasoning",
				"the wire settles no depth, so the vendor picks it: %v", resp.Warnings)
		})
	}
}
