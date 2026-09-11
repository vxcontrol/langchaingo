package bedrockclient

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func converseFieldsFor(t *testing.T, model string, cfg *llms.ReasoningConfig) *ConverseInput {
	t.Helper()

	limit := 4096
	return &ConverseInput{
		ModelID:         model,
		Messages:        []Message{{Role: llms.ChatMessageTypeHuman, Type: "text", Content: "hi"}},
		MaxTokens:       &limit,
		ReasoningConfig: cfg,
	}
}

func TestNovaTakesTheDelegationInsteadOfTheTopEffort(t *testing.T) {
	t.Parallel()

	client := NewConverseClient(nil)
	built, err := client.buildConverseInput(
		converseFieldsFor(t, "amazon.nova-2-lite-v1:0", &llms.ReasoningConfig{Adaptive: true}))
	require.NoError(t, err)

	require.NotNil(t, built.AdditionalModelRequestFields)
	raw, err := built.AdditionalModelRequestFields.MarshalSmithyDocument()
	require.NoError(t, err)

	assert.Contains(t, string(raw), `"type":"enabled"`)
	assert.NotContains(t, string(raw), "maxReasoningEffort",
		"the caller left the depth to the vendor")
	require.NotNil(t, built.InferenceConfig.MaxTokens,
		"the top effort clears the sampling limits; a delegation must not")
}
