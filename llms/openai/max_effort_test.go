package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestOpenAITakesMaxOnTheResponsesAPIAndXHighOnChatCompletions(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			for _, baseURL := range []string{"", "https://llm.pentagi.net/openai_passthrough/v1"} {
				body, err := callWithATool(t, baseURL, model, llms.WithReasoning(llms.ReasoningMax, 0))
				require.NoError(t, err, baseURL)
				assert.Equal(t, map[string]any{"effort": "max"}, body["reasoning"],
					"the model page lists max and the Responses API takes it: %q", baseURL)
			}

			body, warnings := hostCall(t, "", model, llms.WithReasoning(llms.ReasoningMax, 0))
			assert.Equal(t, "xhigh", body["reasoning_effort"],
				"Chat Completions answers 400 to max and names the efforts up to xhigh")
			if assert.Contains(t, warnings, "WithReasoning") {
				assert.Equal(t, llms.WarningClamp, warnings["WithReasoning"].Kind)
				assert.Equal(t, "max", warnings["WithReasoning"].Asked)
				assert.Equal(t, "xhigh", warnings["WithReasoning"].Sent)
			}
		})
	}
}
