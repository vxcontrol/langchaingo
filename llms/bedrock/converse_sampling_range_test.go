package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestConverseSendsSamplingInsideTheRangeTheAPITakes(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name, model, option, field string
		call                       llms.CallOption
		want                       float64
	}{
		{"temperature above one", "openai.gpt-oss-120b-1:0", "WithTemperature", "temperature", llms.WithTemperature(1.5), 1},
		{"temperature below zero", "qwen.qwen3-32b-v1:0", "WithTemperature", "temperature", llms.WithTemperature(-0.5), 0},
		{"top_p above one", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "WithTopP", "topP", llms.WithTopP(1.3), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, body := bedrockWarningsSending(t, converseAnswer,
				[]bedrock.Option{bedrock.WithModel(tc.model), bedrock.WithConverseAPI()}, tc.call)
			config, _ := body["inferenceConfig"].(map[string]any)
			require.InDelta(t, tc.want, config[tc.field], 1e-6, "%v", body)
			w := bedrockWarningsByOption(resp.Warnings)[tc.option]
			require.Equal(t, llms.WarningClamp, w.Kind, "%v", resp.Warnings)
		})
	}

	resp, body := bedrockWarningsSending(t, converseAnswer,
		[]bedrock.Option{bedrock.WithModel("openai.gpt-oss-120b-1:0"), bedrock.WithConverseAPI()},
		llms.WithTemperature(0.7), llms.WithTopP(0.9))
	config, _ := body["inferenceConfig"].(map[string]any)
	require.InDelta(t, 0.7, config["temperature"], 1e-6)
	require.InDelta(t, 0.9, config["topP"], 1e-6)
	require.Empty(t, resp.Warnings, "values inside the range travel unchanged")
}

func TestClaudeTopKStaysWithinTheLimitBedrockDocuments(t *testing.T) {
	t.Parallel()

	const model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
	for _, tc := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
		topK   func(map[string]any) any
	}{
		{"converse", converseAnswer, []bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()},
			func(body map[string]any) any {
				extra, _ := body["additionalModelRequestFields"].(map[string]any)
				return extra["top_k"]
			}},
		{"legacy", legacyAnswer, []bedrock.Option{bedrock.WithModel(model)},
			func(body map[string]any) any { return body["top_k"] }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, body := bedrockWarningsSending(t, tc.answer, tc.opts, llms.WithTopK(600))
			require.InDelta(t, 500, tc.topK(body), 0, "%v", body)
			w := bedrockWarningsByOption(resp.Warnings)["WithTopK"]
			require.Equal(t, llms.WarningClamp, w.Kind, "%v", resp.Warnings)
			require.Equal(t, "500", w.Sent)

			_, body = bedrockWarningsSending(t, tc.answer, tc.opts, llms.WithTopK(40))
			require.InDelta(t, 40, tc.topK(body), 0, "%v", body)
		})
	}
}
