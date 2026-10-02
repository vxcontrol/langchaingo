package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestGrokIsSentOnlyTheEffortsXAIDocuments(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ baseURL, model string }{
		{"https://api.x.ai/v1", "grok-4.7"},
		{"https://api.x.ai/v1", "grok-4.6"},
		{"http://litellm.internal/v1", "xai/grok-4.7"},
	} {
		for asked, sent := range map[llms.ReasoningEffort]string{
			llms.ReasoningMax: "xhigh", llms.ReasoningMinimal: "low", llms.ReasoningHigh: "high",
		} {
			body, warnings := hostCall(t, tc.baseURL, tc.model, llms.WithReasoning(asked, 0))
			assert.Equal(t, sent, body["reasoning_effort"], "%s %s", tc.model, asked)
			if string(asked) != sent {
				assert.Contains(t, warnings, "WithReasoning", "%s %s: a changed effort is reported", tc.model, asked)
			}
		}
	}
}
