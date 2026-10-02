package openai

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestClaudeThroughAGatewayIsSentTheTemperatureRangeAnthropicDocuments(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"anthropic/claude-haiku-4-5", "claude-sonnet-4-5"} {
		for asked, sent := range map[float64]float64{1.5: 1, 1.0001: 1, -0.5: 0} {
			body, warnings := hostCall(t, "http://litellm.internal/v1", model, llms.WithTemperature(asked))
			assert.InDelta(t, sent, body["temperature"], 0, model)
			if assert.Contains(t, warnings, "WithTemperature", model) {
				assert.Equal(t, llms.WarningClamp, warnings["WithTemperature"].Kind, model)
				assert.Equal(t, strconv.FormatFloat(asked, 'g', -1, 64), warnings["WithTemperature"].Asked, model)
				assert.Equal(t, strconv.FormatFloat(sent, 'g', -1, 64), warnings["WithTemperature"].Sent, model)
			}
		}

		for _, asked := range []float64{0, 0.7, 1} {
			body, warnings := hostCall(t, "http://litellm.internal/v1", model, llms.WithTemperature(asked))
			assert.InDelta(t, asked, body["temperature"], 1e-9, model)
			assert.NotContains(t, warnings, "WithTemperature", model)
		}
	}

	body, warnings := hostCall(t, "http://litellm.internal/v1", "anthropic/claude-opus-4-7", llms.WithTemperature(1.5))
	assert.NotContains(t, body, "temperature", "a Claude that takes no sampling is sent none")
	if assert.Contains(t, warnings, "WithTemperature") {
		assert.Equal(t, llms.WarningDrop, warnings["WithTemperature"].Kind)
	}

	body, warnings = hostCall(t, "http://litellm.internal/v1", "openai/gpt-4.1", llms.WithTemperature(1.5))
	assert.InDelta(t, 1.5, body["temperature"], 1e-9)
	assert.NotContains(t, warnings, "WithTemperature")
}
