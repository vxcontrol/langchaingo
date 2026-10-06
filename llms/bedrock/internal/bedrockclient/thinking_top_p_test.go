package bedrockclient

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
)

func TestLegacyThinkingSendsNoTopP(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		model string
		topP  float64
		temp  *float64
	}{
		{"above Anthropic's floor", "anthropic.claude-sonnet-4-5-v1:0", 0.97, nil},
		{"with a caller's temperature", "anthropic.claude-sonnet-4-5-v1:0", 0.97, ptr(0.3)},
		{"below Anthropic's floor", "anthropic.claude-sonnet-4-5-v1:0", 0.5, nil},
		{"a model without sampling", "anthropic.claude-sonnet-5-v1:0", 0.97, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			input := anthropicTextGenerationInput{MaxTokens: 4096, TopP: tc.topP, Temperature: tc.temp}
			require.NoError(t, applyAnthropicReasoning(&input,
				&llms.ReasoningConfig{Tokens: 1024}, tc.model, 4096, nil))
			assert.Zero(t, input.TopP)
		})
	}
}

func TestConverseThinkingSendsNoTopP(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		model string
		topP  float64
		temp  *float64
	}{
		{"above Anthropic's floor", "anthropic.claude-sonnet-4-5-v1:0", 0.97, nil},
		{"with a caller's temperature", "anthropic.claude-sonnet-4-5-v1:0", 0.97, ptr(0.3)},
		{"below Anthropic's floor", "anthropic.claude-sonnet-4-5-v1:0", 0.5, nil},
		{"where both params may travel together", "anthropic.claude-sonnet-4-20250514-v1:0", 0.5, nil},
		{"a model without sampling", "anthropic.claude-sonnet-5-v1:0", 0.97, nil},
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
			assert.Nil(t, built.InferenceConfig.TopP)
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

	for _, choice := range []any{
		map[string]any{"type": "auto", "disable_parallel_tool_use": true},
		json.RawMessage(`{"type":"auto","disable_parallel_tool_use":true}`),
	} {
		got := anthropicToolChoiceOnWire(choice, true)
		assert.Equal(t, map[string]any{"type": "auto", "disable_parallel_tool_use": true}, got, "%v", choice)
	}
}
