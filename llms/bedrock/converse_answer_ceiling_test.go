package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestConverseKeepsNovasAnswerLimitWithinItsSchema(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model       string
		asked, sent int
	}{
		{"us.amazon.nova-pro-v1:0", 16384, 5000},
		{"eu.amazon.nova-lite-v1:0", 8000, 5000},
		{"amazon.nova-2-lite-v1:0", 70000, 64000},
		{"us.amazon.nova-pro-v1:0", 5000, 5000},
		{"us.meta.llama3-3-70b-instruct-v1:0", 4000, 4000},
	} {
		resp, body := bedrockWarningsSending(t, converseAnswer,
			[]bedrock.Option{bedrock.WithModel(tc.model), bedrock.WithConverseAPI()}, llms.WithMaxTokens(tc.asked))
		config, _ := body["inferenceConfig"].(map[string]any)
		require.InDelta(t, tc.sent, config["maxTokens"], 0, tc.model)
		clamped, reported := bedrockWarningsByOption(resp.Warnings)["WithMaxTokens"]
		if tc.asked == tc.sent {
			require.False(t, reported, "%s: %v", tc.model, resp.Warnings)
			continue
		}
		require.True(t, reported, "%s: %v", tc.model, resp.Warnings)
		require.Equal(t, llms.WarningClamp, clamped.Kind, tc.model)
	}
}
