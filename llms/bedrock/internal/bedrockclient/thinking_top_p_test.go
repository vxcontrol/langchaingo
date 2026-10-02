package bedrockclient

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
)

func TestLegacyThinkingKeepsTopPAboveTheFloor(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		model string
		topP  float64
		temp  *float64
		want  float64
	}{
		{"above the floor reaches the wire", "anthropic.claude-sonnet-4-5-v1:0", 0.97, nil, 0.97},
		{"caller set both — top_p is dropped", "anthropic.claude-sonnet-4-5-v1:0", 0.97, ptr(0.3), 0},
		{"a caller's temperature of 0 is set too", "anthropic.claude-sonnet-4-5-v1:0", 0.97, ptr(0.0), 0},
		{"exactly at the floor reaches the wire", "anthropic.claude-sonnet-4-5-v1:0", 0.95, nil, 0.95},
		{"below the floor is stripped", "anthropic.claude-sonnet-4-5-v1:0", 0.5, nil, 0},
		{"model without sampling does not get it", "anthropic.claude-sonnet-5-v1:0", 0.97, nil, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			input := anthropicTextGenerationInput{MaxTokens: 4096, TopP: tc.topP, Temperature: tc.temp}
			require.NoError(t, applyAnthropicReasoning(&input,
				&llms.ReasoningConfig{Tokens: 1024}, tc.model, 4096, nil))
			assert.InDelta(t, tc.want, input.TopP, 1e-9)
		})
	}
}

func TestConverseThinkingKeepsTopPAboveTheFloor(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		model string
		topP  float64
		temp  *float64
		want  *float32
	}{
		{"above the floor reaches the wire", "anthropic.claude-sonnet-4-5-v1:0", 0.97, nil, ptr(float32(0.97))},
		{"caller set both — top_p is dropped", "anthropic.claude-sonnet-4-5-v1:0", 0.97, ptr(0.3), nil},
		{"below the floor is stripped", "anthropic.claude-sonnet-4-5-v1:0", 0.5, nil, nil},
		{
			"below the floor is stripped where both params may travel together",
			"anthropic.claude-sonnet-4-20250514-v1:0", 0.5, nil, nil,
		},
		{"model without sampling does not get it", "anthropic.claude-sonnet-5-v1:0", 0.97, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			client := NewConverseClient(&MockBedrockRuntimeClient{})
			built, err := client.buildConverseInput(&ConverseInput{
				ModelID:         tc.model,
				Messages:        []Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
				MaxTokens:       ptr(4096),
				TopP:            ptr(tc.topP),
				Temperature:     tc.temp,
				ReasoningConfig: &llms.ReasoningConfig{Tokens: 1024},
			})
			require.NoError(t, err)
			if tc.want == nil {
				assert.Nil(t, built.InferenceConfig.TopP)
				return
			}
			require.NotNil(t, built.InferenceConfig.TopP)
			assert.InDelta(t, *tc.want, *built.InferenceConfig.TopP, 1e-6)
		})
	}
}

func TestLegacyClaudeSendsNoToolChoiceWithoutTools(t *testing.T) {
	t.Parallel()

	named := map[string]any{"type": "function", "function": map[string]any{"name": "echo"}}
	for _, choice := range []any{"none", "auto", "required", named} {
		assert.Nil(t, anthropicToolChoiceOnWire(choice, false), "%v with no tools", choice)
		assert.NotNil(t, anthropicToolChoiceOnWire(choice, true), "%v with tools", choice)
	}
}

func TestLegacyClaudeKeepsOneCallPerTurn(t *testing.T) {
	t.Parallel()

	got := anthropicToolChoiceOnWire(map[string]any{"type": "auto", "disable_parallel_tool_use": true}, true)
	assert.Equal(t, map[string]any{"type": "auto", "disable_parallel_tool_use": true}, got)
}
