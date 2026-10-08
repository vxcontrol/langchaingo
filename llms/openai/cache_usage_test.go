package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func cacheUsage(t *testing.T, model, usage string) map[string]any {
	t.Helper()

	llm, _ := gatewayAnswering(t, "application/json", `{"id":"c","object":"chat.completion","model":"`+model+`",`+
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":`+usage+`}`,
		WithModel(model))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	return resp.Choices[0].GenerationInfo
}

func TestTheCacheEveryHostReportsReachesTheUsage(t *testing.T) {
	t.Parallel()

	deepseek := cacheUsage(t, "deepseek/deepseek-v4-pro",
		`{"prompt_tokens":427,"completion_tokens":53,"total_tokens":480,"prompt_cache_hit_tokens":384,"prompt_cache_miss_tokens":43}`)
	require.Equal(t, 384, deepseek["CacheReadInputTokens"])

	recorded := cacheUsage(t, "deepseek/deepseek-v4-pro", `{"prompt_tokens":427,"completion_tokens":53,"total_tokens":480,`+
		`"prompt_tokens_details":{"cached_tokens":384},"prompt_cache_hit_tokens":384,"prompt_cache_miss_tokens":43}`)
	require.Equal(t, 384, recorded["CacheReadInputTokens"])

	claude := cacheUsage(t, "anthropic/claude-sonnet-5", `{"prompt_tokens":1250,"completion_tokens":20,"total_tokens":1270,`+
		`"prompt_tokens_details":{"cached_tokens":0},"cache_creation_input_tokens":1200,"cache_read_input_tokens":0}`)
	require.Equal(t, 1200, claude["CacheCreationInputTokens"])
	require.Equal(t, 0, claude["CacheReadInputTokens"])
}

func TestTheCacheAStreamReportsReachesTheUsage(t *testing.T) {
	t.Parallel()

	llm, _ := gatewayAnswering(t, "text/event-stream", `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m",`+
		`"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n"+
		`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":1250,`+
		`"completion_tokens":20,"total_tokens":1270,"prompt_cache_hit_tokens":900,"cache_creation_input_tokens":300}}`+"\n\n"+
		"data: [DONE]\n\n", WithModel("deepseek/deepseek-v4-pro"))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	require.NoError(t, err)
	require.Equal(t, 900, resp.Choices[0].GenerationInfo["CacheReadInputTokens"])
	require.Equal(t, 300, resp.Choices[0].GenerationInfo["CacheCreationInputTokens"])
}
