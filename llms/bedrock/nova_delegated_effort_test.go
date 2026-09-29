package bedrock_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

const novaLegacyAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"no"}]}},` +
	`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

func TestADelegatedNovaDepthGoesOutAsAnEffortTheVendorAccepts(t *testing.T) {
	t.Parallel()

	for _, door := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
		config func(body map[string]any) map[string]any
	}{
		{
			name:   "invoke model",
			answer: novaLegacyAnswer,
			config: func(body map[string]any) map[string]any {
				inference, _ := body["inferenceConfig"].(map[string]any)
				config, _ := inference["reasoningConfig"].(map[string]any)
				return config
			},
		},
		{
			name:   "converse",
			answer: converseAnswer,
			opts:   []bedrock.Option{bedrock.WithConverseAPI()},
			config: func(body map[string]any) map[string]any {
				fields, _ := body["additionalModelRequestFields"].(map[string]any)
				config, _ := fields["reasoningConfig"].(map[string]any)
				return config
			},
		},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			llm, sent := legacyLLMCapturing(t, door.answer,
				append([]bedrock.Option{bedrock.WithModel("us.amazon.nova-2-lite-v1:0")}, door.opts...)...)

			resp, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "is 1001 prime?")},
				llms.WithAdaptiveReasoning(""), llms.WithMaxTokens(2000), llms.WithTemperature(0.5))
			require.NoError(t, err)

			var body map[string]any
			require.NoError(t, json.Unmarshal([]byte(*sent), &body))
			assert.Equal(t, map[string]any{"type": "enabled", "maxReasoningEffort": "medium"}, door.config(body),
				"nova refuses reasoningConfig type enabled without maxReasoningEffort")
			assert.Contains(t, *sent, `"maxTokens":2000`, "medium keeps the caller's limit")
			assert.Contains(t, *sent, `"temperature":0.5`, "medium keeps the caller's sampling")

			var adaptive []llms.Warning
			for _, w := range resp.Warnings {
				if w.Option == "WithAdaptiveReasoning" {
					adaptive = append(adaptive, w)
				}
			}
			assert.Equal(t, []llms.Warning{{
				Kind: llms.WarningSubstitute, Option: "WithAdaptiveReasoning", Model: "us.amazon.nova-2-lite-v1:0",
				Asked: "adaptive", Sent: "medium",
				Reason: "nova reasons only at a named effort, and none was named",
			}}, adaptive)
		})
	}
}
