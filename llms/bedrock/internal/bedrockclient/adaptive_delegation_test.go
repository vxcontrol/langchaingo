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

func TestNovaTakesTheDelegationAtAnEffortThatKeepsTheLimits(t *testing.T) {
	t.Parallel()

	client := NewConverseClient(nil)
	built, err := client.buildConverseInput(
		converseFieldsFor(t, "amazon.nova-2-lite-v1:0", &llms.ReasoningConfig{Adaptive: true}))
	require.NoError(t, err)

	require.NotNil(t, built.AdditionalModelRequestFields)
	raw, err := built.AdditionalModelRequestFields.MarshalSmithyDocument()
	require.NoError(t, err)

	assert.Contains(t, string(raw), `"reasoningConfig":{"type":"enabled","maxReasoningEffort":"medium"}`)
	require.NotNil(t, built.InferenceConfig.MaxTokens,
		"the top effort clears the sampling limits; a delegation must not")
}

func TestTheLegacyNovaDoorTakesTheDelegationToo(t *testing.T) {
	t.Parallel()

	maxTokens, temperature, topP := 500, 0.3, 0.9
	body, err := novaInputToJSON(nil, "", "us.amazon.nova-2-lite-v1:0", llms.CallOptions{
		MaxTokens: &maxTokens, Temperature: &temperature, TopP: &topP,
		Reasoning: &llms.ReasoningConfig{Adaptive: true},
	}, &llms.Warnings{})
	require.NoError(t, err)

	assert.Contains(t, string(body), `"reasoningConfig":{"type":"enabled","maxReasoningEffort":"medium"}`)
	assert.Contains(t, string(body), `"maxTokens":500`)
	assert.Contains(t, string(body), `"temperature":0.3`)
	assert.Contains(t, string(body), `"topP":0.9`)
}

func TestTheLegacyNovaDelegationSendsNoLimitTheCallerDidNotSet(t *testing.T) {
	t.Parallel()

	body, err := novaInputToJSON(nil, "", "us.amazon.nova-2-lite-v1:0", llms.CallOptions{
		Reasoning: &llms.ReasoningConfig{Adaptive: true},
	}, &llms.Warnings{})
	require.NoError(t, err)
	assert.NotContains(t, string(body), "maxTokens", "Converse sends only the caller's limit")

	body, err = novaInputToJSON(nil, "", "us.amazon.nova-pro-v1:0", llms.CallOptions{
		Reasoning: &llms.ReasoningConfig{Adaptive: true},
	}, &llms.Warnings{})
	require.NoError(t, err)
	assert.NotContains(t, string(body), "reasoningConfig", "a Nova that does not reason gets no reasoning config")
}
