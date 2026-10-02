package openai

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestGPT61SolFollowsItsModelPage(t *testing.T) {
	t.Parallel()

	_, err := wireBodyOf(t, "gpt-6.1-sol", nil, llms.WithReasoningDisabled())
	var off *reasoning.ErrReasoningOffUnsupported
	require.True(t, errors.As(err, &off), "the page lists no none effort: %v", err)

	body, err := wireBodyOf(t, "gpt-6.1-sol", nil,
		llms.WithTemperature(0.7), llms.WithTopP(0.4), llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)
	assert.NotContains(t, body, "temperature")
	assert.NotContains(t, body, "top_p")
	assert.Equal(t, "high", body["reasoning_effort"])

	for asked, sent := range map[llms.ReasoningEffort]string{
		llms.ReasoningMinimal: "low", llms.ReasoningXHigh: "xhigh", llms.ReasoningMax: "max",
	} {
		body, err := wireBodyOf(t, "gpt-6.1-sol", nil, llms.WithReasoning(asked, 0))
		require.NoError(t, err, asked)
		assert.Equal(t, sent, body["reasoning_effort"], asked)
	}
}
