package bedrock_test

import (
	"strconv"
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

			for _, c := range []struct {
				asked, sent float64
				reported    bool
			}{
				{asked: 1.5, sent: 1, reported: true},
				{asked: -0.5, sent: 0, reported: true},
				{asked: 0, sent: 0},
				{asked: 1, sent: 1},
			} {
				resp, body := bedrockWarningsSending(t, tc.answer, tc.opts, llms.WithTemperature(c.asked))

				sent := tc.sent(body)
				require.NotNil(t, sent, "asked %v: no temperature on the wire", c.asked)
				require.InDelta(t, c.sent, sent, 1e-6, "asked %v", c.asked)
				w, ok := bedrockWarningsByOption(resp.Warnings)["WithTemperature"]
				if !c.reported {
					require.False(t, ok, "asked %v: %v", c.asked, resp.Warnings)
					continue
				}
				require.True(t, ok, "asked %v: the clamp went unreported: %v", c.asked, resp.Warnings)
				require.Equal(t, llms.WarningClamp, w.Kind)
				require.Equal(t, strconv.FormatFloat(c.asked, 'g', -1, 64), w.Asked)
				require.Equal(t, strconv.FormatFloat(c.sent, 'g', -1, 64), w.Sent)
			}
		})
	}
}

func TestConverseSendsGptOssTheTemperatureAsAsked(t *testing.T) {
	t.Parallel()

	resp, body := bedrockWarningsSending(t, converseAnswer,
		[]bedrock.Option{bedrock.WithModel("openai.gpt-oss-120b-1:0"), bedrock.WithConverseAPI()},
		llms.WithTemperature(1.5))

	cfg, _ := body["inferenceConfig"].(map[string]any)
	require.InDelta(t, 1.5, cfg["temperature"], 1e-6)
	_, reported := bedrockWarningsByOption(resp.Warnings)["WithTemperature"]
	require.False(t, reported, "%v", resp.Warnings)
}

func TestConverseKeepsATopPOfExactlyTheThinkingFloor(t *testing.T) {
	t.Parallel()

	resp, body := bedrockWarningsSending(t, converseAnswer,
		[]bedrock.Option{bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI()},
		llms.WithReasoning(llms.ReasoningMedium, 2048), llms.WithTopP(0.95))

	cfg, _ := body["inferenceConfig"].(map[string]any)
	require.InDelta(t, 0.95, cfg["topP"], 1e-6)
	require.NotContains(t, cfg, "temperature")
	_, dropped := bedrockWarningsByOption(resp.Warnings)["WithTopP"]
	require.False(t, dropped, "%v", resp.Warnings)
}

func TestClaudeOpus41OnBedrockIsSentAZeroTemperatureWithoutTopP(t *testing.T) {
	t.Parallel()

	const model = "us.anthropic.claude-opus-4-1-20250805-v1:0"
	for _, converse := range []bool{false, true} {
		opts := []bedrock.Option{bedrock.WithModel(model)}
		answer := legacyAnswer
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
			answer = converseAnswer
		}
		resp, body := bedrockWarningsSending(t, answer, opts, llms.WithTemperature(0), llms.WithTopP(0.9))
		sampling := body
		if converse {
			sampling, _ = body["inferenceConfig"].(map[string]any)
		}
		require.Contains(t, sampling, "temperature", "converse=%v", converse)
		require.InDelta(t, 0, sampling["temperature"], 0, "converse=%v", converse)
		require.NotContains(t, sampling, "top_p", "converse=%v", converse)
		require.NotContains(t, sampling, "topP", "converse=%v", converse)
		_, reported := bedrockWarningsByOption(resp.Warnings)["WithTopP"]
		require.True(t, reported, "converse=%v: %v", converse, resp.Warnings)
	}
}

func TestAClaudeTemperatureBudgetThinkingReplacesOnBedrockIsASubstitute(t *testing.T) {
	t.Parallel()

	const model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	for _, converse := range []bool{false, true} {
		opts := []bedrock.Option{bedrock.WithModel(model)}
		answer := legacyAnswer
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
			answer = converseAnswer
		}
		for _, asked := range []float64{1.5, 0.5, -0.5} {
			resp, body := bedrockWarningsSending(t, answer, opts,
				llms.WithTemperature(asked), llms.WithReasoning(llms.ReasoningMedium, 2048))

			sampling := body
			if converse {
				sampling, _ = body["inferenceConfig"].(map[string]any)
			}
			require.InDelta(t, 1, sampling["temperature"], 0, "converse=%v asked %v", converse, asked)
			var reported []llms.Warning
			for _, w := range resp.Warnings {
				if w.Option == "WithTemperature" {
					reported = append(reported, w)
				}
			}
			require.Len(t, reported, 1, "converse=%v asked %v: %v", converse, asked, resp.Warnings)
			require.Equal(t, llms.WarningSubstitute, reported[0].Kind, "converse=%v asked %v", converse, asked)
			require.Equal(t, strconv.FormatFloat(asked, 'g', -1, 64), reported[0].Asked)
			require.Equal(t, "1", reported[0].Sent)
		}
	}
}
