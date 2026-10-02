package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestTurningThinkingOffOnClaudeSonnet55OnBedrockSendsItsLowestSetting(t *testing.T) {
	t.Parallel()

	const model = "anthropic.claude-sonnet-5-5"
	for _, tc := range []struct {
		name     string
		answer   string
		opts     []bedrock.Option
		thinking func(body map[string]any) any
	}{
		{"legacy", legacyAnswer, []bedrock.Option{bedrock.WithModel(model)},
			func(body map[string]any) any { return body["thinking"] }},
		{"converse", converseAnswer, []bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()},
			func(body map[string]any) any {
				fields, _ := body["additionalModelRequestFields"].(map[string]any)
				return fields["thinking"]
			}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, body := bedrockWarningsSending(t, tc.answer, tc.opts, llms.WithReasoningDisabled())

			require.Equal(t, map[string]any{"type": "between_tools"}, tc.thinking(body))
			w, ok := bedrockWarningsByOption(resp.Warnings)["WithReasoningDisabled"]
			require.True(t, ok, "the substituted off went unreported: %v", resp.Warnings)
			require.Equal(t, llms.WarningSubstitute, w.Kind)
			require.Equal(t, "between_tools", w.Sent)
		})
	}
}

func TestSonnet55OnBedrockReportsNoFloorWhileItThinksAndSendsNoEffortWithItsFloor(t *testing.T) {
	t.Parallel()

	const model = "anthropic.claude-sonnet-5-5"
	offAtXHigh := func(o *llms.CallOptions) {
		o.Reasoning = &llms.ReasoningConfig{Mode: llms.ReasoningOff, Effort: llms.ReasoningXHigh}
	}
	for _, converse := range []bool{false, true} {
		opts := []bedrock.Option{bedrock.WithModel(model)}
		answer := legacyAnswer
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
			answer = converseAnswer
		}

		resp, _ := bedrockWarningsSending(t, answer, opts, llms.WithAdaptiveReasoning(llms.ReasoningHigh))
		_, reported := bedrockWarningsByOption(resp.Warnings)["WithReasoningDisabled"]
		require.False(t, reported, "converse=%v: thinking was on: %v", converse, resp.Warnings)

		_, body := bedrockWarningsSending(t, answer, opts, offAtXHigh)
		fields := body
		if converse {
			fields, _ = body["additionalModelRequestFields"].(map[string]any)
		}
		require.Equal(t, map[string]any{"type": "between_tools"}, fields["thinking"], "converse=%v", converse)
		require.NotContains(t, fields, "output_config", "converse=%v: between_tools is refused above effort high", converse)
	}
}
