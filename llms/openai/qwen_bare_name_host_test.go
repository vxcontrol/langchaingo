package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestDashScopeRulesForABareQwenNameHoldOnlyWhereDashScopeServesIt(t *testing.T) {
	t.Parallel()

	const dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	high := llms.WithReasoning(llms.ReasoningHigh, 0)
	for _, baseURL := range []string{"http://vllm.internal:8000/v1", "http://localhost:1234/v1"} {
		for _, tc := range []struct {
			model string
			opts  []llms.CallOption
		}{
			{"qwen3-32b", []llms.CallOption{high}},
			{"qwen3-32b", nil},
			{"qwen-plus", []llms.CallOption{high}},
		} {
			body, _ := hostCall(t, baseURL, tc.model, tc.opts...)
			require.NotContains(t, body, "enable_thinking", "%s on %s: %v", tc.model, baseURL, body)
		}
	}

	llm := newUnitLLM(t, WithBaseURL(dashScope), WithModel("qwen3-32b"), WithHTTPClient(&bodyDoer{}))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, high)
	var streamOnly *reasoning.ErrThinkingRequiresStream
	require.ErrorAs(t, err, &streamOnly, "DashScope serves qwen3-32b thinking only on a stream")

	body, _ := hostCall(t, dashScope, "qwen3-32b")
	require.Equal(t, false, body["enable_thinking"], "%v", body)
	body, _ = hostCall(t, dashScope, "qwen-plus", high)
	require.Equal(t, true, body["enable_thinking"], "%v", body)
}
