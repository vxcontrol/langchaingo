package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAnEmptyLegacyAnswerIsAnErrorRatherThanNoChoices(t *testing.T) {
	t.Parallel()

	for _, family := range []struct{ model, answer string }{
		{"ai21.j2-ultra-v1", `{"completions":[]}`},
		{"ai21.jamba-1-5-large-v1:0", `{"id":"x","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`},
		{"amazon.titan-text-express-v1", `{"inputTextTokenCount":1,"results":[]}`},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", `{"id":"x","type":"message","role":"assistant","model":"m",` +
			`"content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`},
		{"cohere.command-text-v14", `{"generations":[]}`},
		{"deepseek.r1-v1:0", `{"choices":[]}`},
		{"amazon.nova-lite-v1:0", `{"output":{"message":{"content":[]}},"stopReason":"end_turn",` +
			`"usage":{"inputTokens":1,"outputTokens":0,"totalTokens":1}}`},
	} {
		llm, _ := legacyLLMCapturing(t, family.answer, bedrock.WithModel(family.model))
		resp, err := llm.GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		require.Error(t, err, family.model)
		require.Contains(t, err.Error(), "no results", family.model)
		require.Nil(t, resp, family.model)
	}

	llm, _ := legacyLLMCapturing(t, `{}`, bedrock.WithModel("foo.bar-v1"))
	_, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.ErrorContains(t, err, "unsupported provider")
}
