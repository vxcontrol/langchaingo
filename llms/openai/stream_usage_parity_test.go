package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const usagePayload = `"usage":{"prompt_tokens":21,"completion_tokens":349,"total_tokens":370,` +
	`"prompt_tokens_details":{"cached_tokens":7,"cache_write_tokens":13},` +
	`"cost_details":{"upstream_inference_prompt_cost":0.000063,` +
	`"upstream_inference_completions_cost":0.005235},` +
	`"completion_tokens_details":{"reasoning_tokens":5}}`

func usageOfAWholeAnswer(t *testing.T) map[string]any {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","model":"test",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"sixty rooms are free"},` +
			`"finish_reason":"stop"}],` + usagePayload + `}`))
	}))
	t.Cleanup(server.Close)

	return generationInfoAgainst(t, server, nil)
}

func usageOfAStreamedAnswer(t *testing.T) map[string]any {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range []string{
			`data: {"id":"1","object":"chat.completion.chunk","model":"test",` +
				`"choices":[{"index":0,"delta":{"role":"assistant","content":"sixty rooms are free"}}]}`,
			`data: {"id":"1","object":"chat.completion.chunk","model":"test",` +
				`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` + usagePayload + `}`,
			"data: [DONE]",
		} {
			_, _ = w.Write([]byte(line + "\n\n"))
		}
	}))
	t.Cleanup(server.Close)

	return generationInfoAgainst(t, server,
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
}

func generationInfoAgainst(t *testing.T, server *httptest.Server, opt llms.CallOption) map[string]any {
	t.Helper()

	llm, err := New(WithToken("unit-test-key"), WithBaseURL(server.URL), WithModel("test-model"))
	require.NoError(t, err)

	opts := []llms.CallOption{}
	if opt != nil {
		opts = append(opts, opt)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		opts...)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	return resp.Choices[0].GenerationInfo
}

func TestAStreamedAnswerReportsTheSameCountersAsAWholeOne(t *testing.T) {
	t.Parallel()

	whole := usageOfAWholeAnswer(t)
	streamed := usageOfAStreamedAnswer(t)

	for _, key := range []string{
		"PromptTokens", "CompletionTokens", "TotalTokens",
		"CacheReadInputTokens", "CacheCreationInputTokens", "ReasoningTokens",
		"UpstreamInferencePromptCost", "UpstreamInferenceCompletionsCost",
	} {
		assert.Equal(t, whole[key], streamed[key],
			"%s must not depend on whether the answer arrived in one piece", key)
	}

	assert.Equal(t, 13, streamed["CacheCreationInputTokens"],
		"the cache-creation counter the vendor sent must reach the caller")
	require.NotNil(t, streamed["UpstreamInferencePromptCost"])
	assert.InDelta(t, 0.000063, *streamed["UpstreamInferencePromptCost"].(*float64), 1e-9)
}
