package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestClaudeOnBedrockTakesATemperatureFromZeroToOne(t *testing.T) {
	t.Parallel()

	const model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	for _, tc := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
		sent   func(body map[string]any) any
	}{
		{"legacy", legacyAnswer, []bedrock.Option{bedrock.WithModel(model)},
			func(body map[string]any) any { return body["temperature"] }},
		{"converse", converseAnswer, []bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()},
			func(body map[string]any) any {
				cfg, _ := body["inferenceConfig"].(map[string]any)
				return cfg["temperature"]
			}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, body := bedrockWarningsSending(t, tc.answer, tc.opts, llms.WithTemperature(1.5))

			require.InDelta(t, 1.0, tc.sent(body), 1e-9)
			w, ok := bedrockWarningsByOption(resp.Warnings)["WithTemperature"]
			require.True(t, ok, "the lowered temperature went unreported: %v", resp.Warnings)
			require.Equal(t, "1.5", w.Asked)
			require.Equal(t, "1", w.Sent)
		})
	}
}
