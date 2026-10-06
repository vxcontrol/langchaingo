package bedrock_test

import (
	"strconv"
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
			require.Equal(t, "Claude on Amazon Bedrock takes a top_k from 0 to 500", w.Reason)

			_, body = bedrockWarningsSending(t, tc.answer, tc.opts, llms.WithTopK(40))
			require.InDelta(t, 40, tc.topK(body), 0, "%v", body)
		})
	}
}

func TestLegacyClaudeSendsATopPInsideTheRangeBedrockDocuments(t *testing.T) {
	t.Parallel()

	opts := []bedrock.Option{bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0")}

	resp, body := bedrockWarningsSending(t, legacyAnswer, opts, llms.WithTopP(1.3))
	require.InDelta(t, 1, body["top_p"], 1e-9, "%v", body)
	w := bedrockWarningsByOption(resp.Warnings)["WithTopP"]
	require.Equal(t, llms.WarningClamp, w.Kind, "%v", resp.Warnings)
	require.Equal(t, "1.3", w.Asked)
	require.Equal(t, "1", w.Sent)
	require.Equal(t, "Claude on Amazon Bedrock takes a top_p from 0 to 1", w.Reason)

	resp, body = bedrockWarningsSending(t, legacyAnswer, opts, llms.WithTopP(-0.2))
	require.NotContains(t, body, "top_p", "a negative top_p does not go out")
	require.Equal(t, llms.WarningDrop, bedrockWarningsByOption(resp.Warnings)["WithTopP"].Kind, "%v", resp.Warnings)

	resp, body = bedrockWarningsSending(t, legacyAnswer, opts, llms.WithTopP(0.9))
	require.InDelta(t, 0.9, body["top_p"], 1e-9, "%v", body)
	require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithTopP")
}

func TestConverseSendsOnlyTheStopSequencesTheAPITakes(t *testing.T) {
	t.Parallel()

	opts := []bedrock.Option{bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI()}
	stops := func(body map[string]any) []any {
		config, _ := body["inferenceConfig"].(map[string]any)
		sent, _ := config["stopSequences"].([]any)
		return sent
	}

	resp, body := bedrockWarningsSending(t, converseAnswer, opts, llms.WithStopWords([]string{"", "END", ""}))
	require.Equal(t, []any{"END"}, stops(body), "an empty stop sequence stays off the wire")
	require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithStopWords")

	resp, body = bedrockWarningsSending(t, converseAnswer, opts, llms.WithStopWords([]string{""}))
	require.Empty(t, stops(body))
	require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithStopWords", "an empty stop sequence asks for nothing")

	many := make([]string, 2600)
	for i := range many {
		many[i] = "stop" + strconv.Itoa(i)
	}
	resp, body = bedrockWarningsSending(t, converseAnswer, opts, llms.WithStopWords(many))
	require.Len(t, stops(body), 2500)
	w := bedrockWarningsByOption(resp.Warnings)["WithStopWords"]
	require.Equal(t, llms.WarningClamp, w.Kind, "%v", resp.Warnings)
	require.Equal(t, "2600 words", w.Asked)
	require.Equal(t, "2500 words", w.Sent)
}
