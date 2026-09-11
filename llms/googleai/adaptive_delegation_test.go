package googleai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

	config, err := resolveThinkingConfig("gemma-4-27b-it",
		&llms.ReasoningConfig{Adaptive: true, Effort: llms.ReasoningHigh}, 0)
	require.NoError(t, err)
	require.NotNil(t, config)
	assert.NotEmpty(t, config.ThinkingLevel, "a named effort is a depth the caller chose")
}
