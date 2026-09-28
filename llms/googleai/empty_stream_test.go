package googleai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type emptyStream struct{}

func (emptyStream) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(bytes.NewReader(nil)),
		Request:    r,
	}, nil
}

func TestStreamWithoutCandidatesIsNotASilentSuccess(t *testing.T) {
	t.Parallel()

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithRest(),
		WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: emptyStream{}}))
	require.NoError(t, err)

	msgs := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	sink := func(context.Context, streaming.Chunk) error { return nil }

	resp, err := llm.GenerateContent(context.Background(), msgs,
		llms.WithMaxTokens(24), llms.WithStreamingFunc(sink))

	require.Error(t, err, "an empty answer with no stop reason must not look like a success")
	var apiErr *llms.Error
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, llms.ErrCodeTokenLimit, apiErr.Code)
	require.NotNil(t, resp)
	assert.Empty(t, resp.Choices[0].Content)
}

type failedStream struct{}

func (failedStream) RoundTrip(r *http.Request) (*http.Response, error) {
	body := `{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}`
	return &http.Response{
		StatusCode: http.StatusTooManyRequests,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(body))),
		Request:    r,
	}, nil
}

func TestARealStreamErrorIsNotBlamedOnTheBudget(t *testing.T) {
	t.Parallel()

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithRest(),
		WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: failedStream{}}))
	require.NoError(t, err)

	msgs := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	sink := func(context.Context, streaming.Chunk) error { return nil }

	_, err = llm.GenerateContent(context.Background(), msgs,
		llms.WithMaxTokens(24), llms.WithStreamingFunc(sink))

	require.Error(t, err)
	assert.Contains(t, err.Error(), "429",
		"a stream that failed HTTP must surface its own error, not a budget guess")
	var apiErr *llms.Error
	if errors.As(err, &apiErr) {
		assert.NotEqual(t, llms.ErrCodeTokenLimit, apiErr.Code,
			"a set max_tokens must not mask the real stream failure")
	}
}

type blockedPromptStream struct{}

func (blockedPromptStream) RoundTrip(r *http.Request) (*http.Response, error) {
	const event = "data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"," +
		"\"blockReasonMessage\":\"the prompt was rejected\"},\"candidates\":[]}\n\n"
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(event))),
		Request:    r,
	}, nil
}

func TestABlockedPromptIsNotBlamedOnTheBudget(t *testing.T) {
	t.Parallel()

	llm, err := New(context.Background(),
		WithRest(), WithAPIKey("test-key"), WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: blockedPromptStream{}}))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithMaxTokens(16384),
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	var apiErr *llms.Error
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, llms.ErrCodeContentFilter, apiErr.Code,
		"a blocked prompt must not be reported as a budget that is too small")
	assert.Contains(t, apiErr.Message, "SAFETY")
}
