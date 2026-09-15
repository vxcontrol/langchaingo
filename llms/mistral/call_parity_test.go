package mistral

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func mistralStreaming(t *testing.T, chunks ...string) *Model {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)
	return m
}

func TestCallDeliversTheStreamTheCallerAskedFor(t *testing.T) {
	t.Parallel()

	m := mistralStreaming(t,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-small-latest",`+
			`"choices":[{"index":0,"delta":{"role":"assistant","content":"an "},"finish_reason":""}]}`,
		`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`)

	var chunks int32
	got, err := m.Call(context.Background(), "hi",
		llms.WithStreamingFunc(func(_ context.Context, c streaming.Chunk) error {
			if c.Type == streaming.ChunkTypeText {
				atomic.AddInt32(&chunks, 1)
			}
			return nil
		}))

	require.NoError(t, err)
	assert.Equal(t, "an answer", got)
	assert.Positive(t, atomic.LoadInt32(&chunks),
		"a caller that passed a streaming callback to Call must be called with the text")
}

func TestCallRefusesATruncatedAnswerWhenAskedTo(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"mistral-small-latest",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"half an ans"},"finish_reason":"length"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":16,"total_tokens":17}}`)
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)

	_, err = m.Call(context.Background(), "hi", llms.WithMaxTokens(16), llms.WithFailOnTruncation())

	require.Error(t, err, "Call must refuse a truncated answer when the caller forbade truncation")
	require.True(t, llms.IsTruncatedError(err), "the refusal must say the answer was cut, got %v", err)
}
