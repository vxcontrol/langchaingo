package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestADocumentedAliasIsSentWhatItsModelTakes(t *testing.T) {
	t.Parallel()

	const dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	for _, model := range []string{"qwen-plus-latest", "qwen-plus-2025-12-01", "qwen3-max-preview", "qwen3-vl-plus-2025-12-19"} {
		body, _ := hostCall(t, dashScope, model, llms.WithReasoning(llms.ReasoningHigh, 0))
		require.Equal(t, true, body["enable_thinking"], "%s: %v", model, body)
	}

	llm := newUnitLLM(t, WithBaseURL(dashScope), WithModel("qwen3.7-max-2026-05-17"), WithHTTPClient(&bodyDoer{}))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
	var off *reasoning.ErrReasoningOffUnsupported
	require.ErrorAs(t, err, &off, "the snapshot thinks only")

	body, _ := hostCall(t, "https://api.mistral.ai/v1", "zai-glm-latest", llms.WithReasoning(llms.ReasoningXHigh, 0))
	require.Equal(t, "high", body["reasoning_effort"], "%v", body)
}
