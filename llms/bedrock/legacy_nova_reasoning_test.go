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

func novaLegacyInferenceConfig(t *testing.T, opts ...llms.CallOption) map[string]any {
	t.Helper()
	return novaLegacyInferenceConfigOf(t, "us.amazon.nova-2-lite-v1:0", opts...)
}

func novaLegacyInferenceConfigOf(t *testing.T, model string, opts ...llms.CallOption) map[string]any {
	t.Helper()

	const answer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

	llm, sent := legacyLLMCapturing(t, answer, bedrock.WithModel(model))

	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")}, opts...)
	require.NoError(t, err)

	var payload struct {
		InferenceConfig map[string]any `json:"inferenceConfig"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &payload))
	return payload.InferenceConfig
}

func TestTheLegacyNovaPathCarriesTheThinkingItWasAskedFor(t *testing.T) {
	t.Parallel()

	t.Run("an asked-for effort reaches the wire", func(t *testing.T) {
		t.Parallel()

		config := novaLegacyInferenceConfig(t,
			llms.WithMaxTokens(512), llms.WithReasoning(llms.ReasoningMedium, 0))

		assert.Equal(t, map[string]any{"type": "enabled", "maxReasoningEffort": "medium"},
			config["reasoningConfig"],
			"the Invoke schema carries reasoningConfig inside inferenceConfig")
	})

	t.Run("the top effort clears the sampling Nova refuses beside it", func(t *testing.T) {
		t.Parallel()

		config := novaLegacyInferenceConfig(t,
			llms.WithMaxTokens(512), llms.WithTemperature(0.4),
			llms.WithReasoning(llms.ReasoningHigh, 0))

		assert.Equal(t, map[string]any{"type": "enabled", "maxReasoningEffort": "high"},
			config["reasoningConfig"])
		assert.NotContains(t, config, "maxTokens")
		assert.NotContains(t, config, "temperature")
		assert.NotContains(t, config, "topP")
	})

	t.Run("a call that asked for nothing carries no reasoning", func(t *testing.T) {
		t.Parallel()

		config := novaLegacyInferenceConfig(t, llms.WithMaxTokens(512))

		assert.NotContains(t, config, "reasoningConfig")
	})

	t.Run("a nova outside the reasoning family is not told to think", func(t *testing.T) {
		t.Parallel()

		config := novaLegacyInferenceConfigOf(t, "us.amazon.nova-lite-v1:0",
			llms.WithMaxTokens(512), llms.WithReasoning(llms.ReasoningMedium, 0))

		assert.NotContains(t, config, "reasoningConfig",
			"the door that carries the field on Converse is the same one that carries it here")
	})
}
