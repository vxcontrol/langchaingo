package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func bedrockWarningsFound(warnings []llms.Warning, option string) []llms.Warning {
	var found []llms.Warning
	for _, w := range warnings {
		if w.Option == option {
			found = append(found, w)
		}
	}
	return found
}

func TestBothBedrockDoorsCallTheEffortTurnedIntoABudgetASubstitution(t *testing.T) {
	t.Parallel()

	for _, door := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
	}{
		{"legacy", legacyAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"),
		}},
		{"converse", converseAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"), bedrock.WithConverseAPI(),
		}},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, door.answer, door.opts,
				llms.WithMaxTokens(8192), llms.WithReasoning(llms.ReasoningMedium, 0))

			found := bedrockWarningsFound(resp.Warnings, "WithReasoning")
			require.Len(t, found, 1, "one warning for one ask: %v", resp.Warnings)
			require.Equal(t, llms.WarningSubstitute, found[0].Kind)
			require.Equal(t, "medium", found[0].Asked)
			require.Contains(t, found[0].Sent, "tokens")
		})
	}
}

func TestBothBedrockDoorsCallTheEffortAnExplicitBudgetOverrodeALoss(t *testing.T) {
	t.Parallel()

	for _, door := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
	}{
		{"legacy", legacyAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"),
		}},
		{"converse", converseAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"), bedrock.WithConverseAPI(),
		}},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, door.answer, door.opts,
				llms.WithMaxTokens(8192), llms.WithReasoning(llms.ReasoningMedium, 2048))

			var dropped *llms.Warning
			for _, w := range bedrockWarningsFound(resp.Warnings, "WithReasoning") {
				if w.Kind == llms.WarningDrop {
					dropped = &w
				}
			}
			require.NotNil(t, dropped, "the effort went nowhere unreported: %v", resp.Warnings)
			require.Equal(t, "medium", dropped.Asked)
		})
	}
}
