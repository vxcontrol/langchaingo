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
		"an OpenAI effort":             {"gpt-5.4", map[string]any{"reasoning_effort": "high"}},
		"a nested OpenAI effort":       {"gpt-5.2", map[string]any{"reasoning": map[string]any{"effort": "medium"}}},
		"a Claude thinking object":     {"anthropic/claude-sonnet-4-5", map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 2048}}},
		"Claude adaptive on a gateway": {"anthropic/claude-sonnet-4-6", map[string]any{"thinking": map[string]any{"type": "adaptive"}}},
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
