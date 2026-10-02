package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestLegacyNovaReportsTheAnswerLimitItSends(t *testing.T) {
	t.Parallel()

	limitWarnings := func(resp *llms.ContentResponse) []llms.Warning {
		var found []llms.Warning
		for _, w := range resp.Warnings {
			if w.Option == "WithMaxTokens" {
				found = append(found, w)
			}
		}
		return found
	}
	sent := func(body map[string]any) (any, bool) {
		config, _ := body["inferenceConfig"].(map[string]any)
		value, ok := config["maxTokens"]
		return value, ok
	}

	resp, body := bedrockWarningsSending(t, novaAnswer, []bedrock.Option{bedrock.WithModel("amazon.nova-2-lite-v1:0")},
		llms.WithMaxTokens(70000), llms.WithReasoning(llms.ReasoningHigh, 0))
	_, carried := sent(body)
	require.False(t, carried, "the top effort clears the limit")
	warnings := limitWarnings(resp)
	require.Len(t, warnings, 1, "one report for one limit: %v", warnings)
	require.Equal(t, llms.WarningDrop, warnings[0].Kind)

	resp, body = bedrockWarningsSending(t, novaAnswer, []bedrock.Option{bedrock.WithModel("us.amazon.nova-pro-v1:0")},
		llms.WithMaxTokens(-1))
	_, carried = sent(body)
	require.False(t, carried, "Nova's limit is optional, so a non-positive one is left out")
	warnings = limitWarnings(resp)
	require.Len(t, warnings, 1, "%v", warnings)
	require.Equal(t, llms.WarningDrop, warnings[0].Kind)
	require.Equal(t, "-1", warnings[0].Asked)

	resp, body = bedrockWarningsSending(t, novaAnswer, []bedrock.Option{bedrock.WithModel("amazon.nova-2-lite-v1:0")},
		llms.WithMaxTokens(70000), llms.WithReasoning(llms.ReasoningLow, 0))
	value, _ := sent(body)
	require.InDelta(t, 64000, value, 0)
	warnings = limitWarnings(resp)
	require.Len(t, warnings, 1, "%v", warnings)
	require.Equal(t, llms.WarningClamp, warnings[0].Kind)
	require.Equal(t, "64000", warnings[0].Sent)
}
