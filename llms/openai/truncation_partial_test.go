package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func truncatedStreamGivenUpOn(t *testing.T, call ...llms.CallOption) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,`+
			`"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"sixty "},"finish_reason":null}]}`+"\n\n")
		_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,`+
			`"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"rooms"},"finish_reason":"length"}]}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	llm := newUnitLLM(t, WithBaseURL(srv.URL), WithModel("gpt-4o"))

	delivered := 0
	options := append([]llms.CallOption{
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type != streaming.ChunkTypeText {
				return nil
			}
			delivered++
			if delivered == 2 {
				return errGaveUp
			}
			return nil
		}),
	}, call...)

	return llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		options...)
}

func TestAPartialAnswerStillAnswersTheTruncationQuestion(t *testing.T) {
	t.Parallel()

	resp, err := truncatedStreamGivenUpOn(t, llms.WithFailOnTruncation())

	require.Error(t, err)
	require.NotNil(t, resp, "the text already delivered must travel with the error")
	assert.ErrorIs(t, err, errGaveUp, "the error the consumer raised is kept")

	var apiErr *llms.Error
	require.True(t, errors.As(err, &apiErr), "the truncation error is reachable: %v", err)
	assert.Equal(t, llms.ErrCodeTruncated, apiErr.Code)
}

func TestAPartialAnswerIsSilentWhenTruncationWasNotAsked(t *testing.T) {
	t.Parallel()

	_, err := truncatedStreamGivenUpOn(t)

	require.Error(t, err)
	var apiErr *llms.Error
	if errors.As(err, &apiErr) {
		assert.NotEqual(t, llms.ErrCodeTruncated, apiErr.Code,
			"a caller who did not ask to fail on truncation gets only the stream error")
	}
}
