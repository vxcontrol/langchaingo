package bedrockclient

import (
	"encoding/json"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConverseSendsGptOssTheReasoningEffortItDocuments(t *testing.T) {
	t.Parallel()

	on := func(effort llms.ReasoningEffort) *llms.ReasoningConfig {
		return &llms.ReasoningConfig{Mode: llms.ReasoningOn, Effort: effort}
	}
	for _, model := range []string{"openai.gpt-oss-120b-1:0", "openai.gpt-oss-20b-1:0"} {
		for _, tc := range []struct {
			name string
			cfg  *llms.ReasoningConfig
			want map[string]any
		}{
			{"low", on(llms.ReasoningLow), map[string]any{"reasoning_effort": "low"}},
			{"medium", on(llms.ReasoningMedium), map[string]any{"reasoning_effort": "medium"}},
			{"high", on(llms.ReasoningHigh), map[string]any{"reasoning_effort": "high"}},
			{"minimal rises to the lowest level", on(llms.ReasoningMinimal), map[string]any{"reasoning_effort": "low"}},
			{"xhigh falls to the highest level", on(llms.ReasoningXHigh), map[string]any{"reasoning_effort": "high"}},
			{"max falls to the highest level", on(llms.ReasoningMax), map[string]any{"reasoning_effort": "high"}},
			{"a budget picks the level", &llms.ReasoningConfig{Tokens: 3000}, map[string]any{"reasoning_effort": "high"}},
			{"delegated depth leaves the level to the model", &llms.ReasoningConfig{Adaptive: true}, nil},
			{"unset reasoning sends nothing", nil, nil},
		} {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				t.Parallel()

				temperature, maxTokens := 0.3, 4096
				got, err := NewConverseClient(&MockBedrockRuntimeClient{}).buildConverseInput(&ConverseInput{
					ModelID:         model,
					Messages:        humanTurn(),
					Temperature:     &temperature,
					MaxTokens:       &maxTokens,
					ReasoningConfig: tc.cfg,
				})
				require.NoError(t, err)

				var fields map[string]any
				if got.AdditionalModelRequestFields != nil {
					raw, err := got.AdditionalModelRequestFields.MarshalSmithyDocument()
					require.NoError(t, err)
					require.NoError(t, json.Unmarshal(raw, &fields))
				}
				assert.Equal(t, tc.want, fields)

				require.NotNil(t, got.InferenceConfig.Temperature, "gpt-oss takes sampling while it reasons")
				assert.InDelta(t, temperature, *got.InferenceConfig.Temperature, 0.0001)
				require.NotNil(t, got.InferenceConfig.MaxTokens)
				assert.EqualValues(t, maxTokens, *got.InferenceConfig.MaxTokens)
			})
		}
	}
}
