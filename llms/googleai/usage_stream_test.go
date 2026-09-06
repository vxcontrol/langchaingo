package googleai

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type cachedUsageStream struct{}

func (cachedUsageStream) RoundTrip(r *http.Request) (*http.Response, error) {
	const body = `data: {"candidates":[{"content":{"parts":[{"text":"sixty rooms are free"}],"role":"model"},` +
		`"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":100,` +
		`"candidatesTokenCount":9,"totalTokenCount":109,"cachedContentTokenCount":80}}` + "\r\n\r\n"
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(body))),
		Request:    r,
	}, nil
}

func TestAStreamedAnswerReportsCountersThatAddUp(t *testing.T) {
	t.Parallel()

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithRest(),
		WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: cachedUsageStream{}}))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	info := resp.Choices[0].GenerationInfo
	assert.Equal(t, 100, info["PromptTokens"], "the prompt count is the whole prompt, cache included")
	assert.Equal(t, 80, info["CacheReadInputTokens"])
	assert.Equal(t, 20, info["CacheCreationInputTokens"], "the uncached remainder keeps its own field")
	assert.Equal(t, info["TotalTokens"],
		info["PromptTokens"].(int)+info["CompletionTokens"].(int),
		"prompt plus completion must equal the total, as every other door reports it")
}

type uncachedUsageStream struct{}

func (uncachedUsageStream) RoundTrip(r *http.Request) (*http.Response, error) {
	const body = `data: {"candidates":[{"content":{"parts":[{"text":"sixty rooms are free"}],"role":"model"},` +
		`"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":100,` +
		`"candidatesTokenCount":9,"totalTokenCount":109}}` + "\r\n\r\n"
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(body))),
		Request:    r,
	}, nil
}

func TestAnAnswerWithoutCachedContentReportsNoCacheCreation(t *testing.T) {
	t.Parallel()

	t.Run("streamed", func(t *testing.T) {
		t.Parallel()

		llm, err := New(context.Background(),
			WithAPIKey("unit-test-key"), WithRest(),
			WithDefaultModel("gemini-2.5-flash"),
			WithHTTPClient(&http.Client{Transport: uncachedUsageStream{}}))
		require.NoError(t, err)

		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
		require.NoError(t, err)
		require.Len(t, resp.Choices, 1)

		info := resp.Choices[0].GenerationInfo
		assert.Equal(t, 100, info["PromptTokens"])
		assert.Equal(t, 0, info["CacheReadInputTokens"])
		assert.Equal(t, 0, info["CacheCreationInputTokens"],
			"nothing was written to a cache on this request")
	})

	t.Run("whole answer", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"sixty rooms are free"}]},` +
				`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,` +
				`"candidatesTokenCount":9,"totalTokenCount":109}}`))
		}))
		defer server.Close()

		llm, err := New(context.Background(),
			WithAPIKey("unit-test-key"), WithEndpoint(server.URL),
			WithDefaultModel("gemini-2.5-flash"))
		require.NoError(t, err)

		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		require.NoError(t, err)
		require.Len(t, resp.Choices, 1)

		assert.Equal(t, 0, resp.Choices[0].GenerationInfo["CacheCreationInputTokens"],
			"the whole-answer path must agree with the streamed one on the same usage")
	})
}
