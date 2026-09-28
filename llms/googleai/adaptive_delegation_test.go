package googleai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestNoFamilyInventsADepthTheCallerHandedToTheVendor(t *testing.T) {
	t.Parallel()

	for model, carriesBudget := range map[string]bool{
		"gemma-4-27b-it":   false,
		"gemini-3.5-flash": false,
		"gemini-2.5-flash": true, // its dynamic budget of -1 IS "the vendor decides"
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			config, err := resolveThinkingConfig(model, &llms.ReasoningConfig{Adaptive: true}, 0)
			require.NoError(t, err)
			require.NotNil(t, config)
			assert.Empty(t, config.ThinkingLevel, "%s: the level is the vendor's to choose", model)
			assert.True(t, config.IncludeThoughts)

			if carriesBudget {
				require.NotNil(t, config.ThinkingBudget)
				assert.Equal(t, int32(-1), *config.ThinkingBudget)
				return
			}
			assert.Nil(t, config.ThinkingBudget,
				"%s takes a level, so a budget is a depth it never asked for", model)
		})
	}
}

func TestAnEffortTheCallerNamedStillPicksTheLevel(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model  string
		effort llms.ReasoningEffort
		want   genai.ThinkingLevel
		why    string
	}{
		{"gemini-3.5-flash", llms.ReasoningLow, genai.ThinkingLevelLow, "a named effort is a depth the caller chose"},
		{"gemma-4-27b-it", llms.ReasoningHigh, genai.ThinkingLevelHigh, "a named effort is a depth the caller chose"},
		{"gemma-4-27b-it", llms.ReasoningLow, genai.ThinkingLevelHigh, "this family only switches thinking on, and on is HIGH"},
	} {
		config, err := resolveThinkingConfig(tc.model,
			&llms.ReasoningConfig{Adaptive: true, Effort: tc.effort}, 0)
		require.NoError(t, err)
		require.NotNil(t, config)
		assert.Equal(t, tc.want, config.ThinkingLevel, "%s %s: %s", tc.model, tc.effort, tc.why)
		assert.Nil(t, config.ThinkingBudget, "%s takes a level, not a budget", tc.model)
	}
}
