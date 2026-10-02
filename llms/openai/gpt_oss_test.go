package openai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestGptOssIsSentItsThreeLevelsOnEveryHost(t *testing.T) {
	t.Parallel()

	for _, route := range []struct{ baseURL, model string }{
		{"https://api.groq.com/openai/v1", "openai/gpt-oss-120b"},
		{"http://vllm.internal/v1", "gpt-oss-20b"},
		{"http://litellm.internal/v1", "groq/openai/gpt-oss-120b"},
	} {
		body, warnings := hostCall(t, route.baseURL, route.model, llms.WithReasoning(llms.ReasoningXHigh, 0))
		require.Equal(t, "high", body["reasoning_effort"], route.model)
		require.Equal(t, llms.WarningClamp, warnings["WithReasoning"].Kind, route.model)
	}
}
