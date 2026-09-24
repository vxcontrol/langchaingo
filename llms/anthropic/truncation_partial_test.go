package anthropic_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const truncatedThenBroken = deliveredThenBroken + `event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":4}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}
`

func truncatedStreamThenError(t *testing.T, call ...llms.CallOption) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, truncatedThenBroken)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-haiku-4-5"))
	require.NoError(t, err)

	options := append([]llms.CallOption{
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }),
	}, call...)

	return llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		options...)
}

func TestAPartialAnswerStillAnswersTheTruncationQuestion(t *testing.T) {
	t.Parallel()

	resp, err := truncatedStreamThenError(t, llms.WithFailOnTruncation())

	require.Error(t, err)
	require.NotNil(t, resp, "the text already delivered must travel with the error")

	var apiErr *llms.Error
	require.True(t, errors.As(err, &apiErr), "the truncation error is reachable: %v", err)
	assert.Equal(t, llms.ErrCodeTruncated, apiErr.Code)
	assert.Contains(t, err.Error(), "overloaded", "the vendor error is kept alongside it")
}

func TestAPartialAnswerIsSilentWhenTruncationWasNotAsked(t *testing.T) {
	t.Parallel()

	_, err := truncatedStreamThenError(t)

	require.Error(t, err)
	var apiErr *llms.Error
	if errors.As(err, &apiErr) {
		assert.NotEqual(t, llms.ErrCodeTruncated, apiErr.Code)
	}
}
