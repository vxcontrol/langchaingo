package openai

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func astraTool() llms.Tool {
	return llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name:       "lookup",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
}

func TestToolsAreRefusedOnOpenAIsChatCompletionsWhereTheModelPageSaysSo(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-6-astra", "gpt-6.1-sol"} {
		for _, baseURL := range []string{"", "https://api.openai.com/v1", "https://eu.api.openai.com/v1"} {
			body, err := callWithATool(t, baseURL, model)
			var unsupported *reasoning.ErrChatToolsUnsupported
			require.True(t, errors.As(err, &unsupported), "%s on %q: %v", model, baseURL, err)
			assert.Equal(t, model, unsupported.Model)
			assert.Nil(t, body, "%s on %q", model, baseURL)

			sent, err := callWithToolsOnlyInTheExtraBody(t, baseURL, model)
			require.ErrorAs(t, err, &unsupported, "tools in the extra body, %s on %q", model, baseURL)
			assert.False(t, sent, "%s on %q", model, baseURL)
		}
		for _, baseURL := range []string{"https://openrouter.ai/api/v1", "http://litellm.internal/v1"} {
			body, err := callWithATool(t, baseURL, "openai/"+model)
			require.NoError(t, err, "a gateway decides for itself: %s on %s", model, baseURL)
			assert.Equal(t, []any{map[string]any{"type": "function", "function": map[string]any{
				"name": "lookup", "parameters": map[string]any{"type": "object", "properties": map[string]any{}},
			}}}, body["tools"], "%s on %s", model, baseURL)
		}
	}
}

func TestAstraRefusesAnExplicitDisableBeforeTheNetwork(t *testing.T) {
	t.Parallel()

	_, err := wireBodyOf(t, "gpt-6-astra", nil, llms.WithReasoningDisabled())
	require.Error(t, err)

	var off *reasoning.ErrReasoningOffUnsupported
	require.True(t, errors.As(err, &off), "want ErrReasoningOffUnsupported, got %v", err)
}

func TestAstraKeepsTheCallersSamplingOffTheWire(t *testing.T) {
	t.Parallel()

	body, err := wireBodyOf(t, "gpt-6-astra", nil,
		llms.WithTemperature(0.7), llms.WithTopP(0.4), llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)

	assert.NotContains(t, body, "temperature")
	assert.NotContains(t, body, "top_p")
	assert.Equal(t, "high", body["reasoning_effort"])
}

func TestAstraCarriesTheTopEffortTheVendorAccepts(t *testing.T) {
	t.Parallel()

	body, err := wireBodyOf(t, "gpt-6-astra", nil, llms.WithReasoning(llms.ReasoningXHigh, 0))
	require.NoError(t, err)
	assert.Equal(t, "xhigh", body["reasoning_effort"])
}

func TestAstraClampsMaxTheVendorRefuses(t *testing.T) {
	t.Parallel()

	body, err := wireBodyOf(t, "gpt-6-astra", nil, llms.WithReasoning(llms.ReasoningMax, 0))
	require.NoError(t, err)
	assert.Equal(t, "xhigh", body["reasoning_effort"],
		"the vendor answers 400 to max with this model and names low, medium, high and xhigh")
}

func TestAstraClampsAnEffortItsCardDoesNotList(t *testing.T) {
	t.Parallel()

	body, err := wireBodyOf(t, "gpt-6-astra", nil, llms.WithReasoning(llms.ReasoningMinimal, 0))
	require.NoError(t, err)

	assert.Equal(t, "low", body["reasoning_effort"],
		"the card lists low as the floor, so a lower ask travels as low")
}

func TestAstraSendsTheOutputLimitAsMaxCompletionTokens(t *testing.T) {
	t.Parallel()

	body, err := wireBodyOf(t, "gpt-6-astra", nil, llms.WithMaxTokens(256))
	require.NoError(t, err)

	assert.Contains(t, body, "max_completion_tokens")
	assert.NotContains(t, body, "max_tokens")
}
