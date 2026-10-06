package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestThinkingSwitchedOnInTheExtraBodyDropsTheSamplingItRefuses(t *testing.T) {
	t.Parallel()

	sampling := []llms.CallOption{llms.WithTemperature(0.7), llms.WithTopP(0.4)}
	for name, tc := range map[string]struct {
		model string
		extra map[string]any
	}{
		"an OpenAI effort":               {"gpt-5.4", map[string]any{"reasoning_effort": "high"}},
		"a nested OpenAI effort":         {"gpt-5.2", map[string]any{"reasoning": map[string]any{"effort": "medium"}}},
		"a Claude thinking object":       {"anthropic/claude-sonnet-4-5", map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 2048}}},
		"Claude adaptive on a gateway":   {"anthropic/claude-sonnet-4-6", map[string]any{"thinking": map[string]any{"type": "adaptive"}}},
		"an OpenRouter reasoning budget": {"anthropic/claude-sonnet-4-5", map[string]any{"reasoning": map[string]any{"max_tokens": 2048}}},
		"a budget decoded from JSON":     {"anthropic/claude-sonnet-4-5", map[string]any{"reasoning": map[string]any{"max_tokens": float64(2048)}}},
		"an OpenRouter reasoning switch": {"gpt-5.4", map[string]any{"reasoning": map[string]any{"enabled": true}}},
	} {
		body, err := wireBodyOf(t, tc.model, nil, append(sampling, llms.WithExtraBody(tc.extra))...)
		require.NoError(t, err, name)
		assert.NotContains(t, body, "top_p", name)
		if temperature, sent := body["temperature"]; sent {
			assert.InDelta(t, 1, temperature, 0, "%s: a thinking Claude takes only temperature 1", name)
		}
	}

	body, err := wireBodyOf(t, "gpt-5.4", nil, append(sampling,
		llms.WithExtraBody(map[string]any{"reasoning_effort": "none"}))...)
	require.NoError(t, err)
	assert.InDelta(t, 0.7, body["temperature"], 1e-9, "thinking switched off keeps the caller's sampling")
	assert.InDelta(t, 0.4, body["top_p"], 1e-9)
}

func TestTheExtraBodyDecidesThinkingBecauseItWinsOnTheWire(t *testing.T) {
	t.Parallel()

	sampling := []llms.CallOption{llms.WithTemperature(0.7), llms.WithTopP(0.4)}
	body, err := wireBodyOf(t, "gpt-5.4", nil, append(sampling, llms.WithReasoningDisabled(),
		llms.WithExtraBody(map[string]any{"reasoning_effort": "high"}))...)
	require.NoError(t, err)
	assert.NotContains(t, body, "temperature", "the extra body's effort reaches the wire over the door's none")
	assert.NotContains(t, body, "top_p")

	const deepSeek = "https://api.deepseek.com"
	body, _ = hostCall(t, deepSeek, "deepseek-v4-pro", llms.WithTemperature(0.7))
	assert.NotContains(t, body, "temperature", "DeepSeek V4 thinks unless told otherwise")

	for name, extra := range map[string]map[string]any{
		"thinking disabled": {"thinking": map[string]any{"type": "disabled"}},
		"effort none":       {"reasoning_effort": "none"},
	} {
		body, _ = hostCall(t, deepSeek, "deepseek-v4-pro", llms.WithTemperature(0.7), llms.WithExtraBody(extra))
		assert.InDelta(t, 0.7, body["temperature"], 1e-9, name)
	}
}
