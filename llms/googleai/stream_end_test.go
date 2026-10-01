package googleai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const partialGeminiChunk = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}` + "\n\n"

func streamFrom(t *testing.T, ctx context.Context, handler http.HandlerFunc) (*llms.ContentResponse, error) {
	t.Helper()

	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	llm, err := New(t.Context(), WithAPIKey("unit-test-key"), WithEndpoint(server.URL),
		WithDefaultModel("gemini-2.5-flash"))
	require.NoError(t, err)

	return llm.GenerateContent(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
}

func TestAGeminiStreamWithoutAFinishReasonIsAnError(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk)
	})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	assert.Equal(t, "partial", resp.Choices[0].Content)
}

func TestAGeminiStreamCutByTheDeadlineKeepsItsCause(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), 200*time.Millisecond)
	defer cancel()

	resp, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.NotNil(t, resp)
	assert.Equal(t, "partial", resp.Choices[0].Content)
}

func TestAGeminiStreamWithAFinishReasonIsComplete(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk+
			`data: {"candidates":[{"content":{"role":"model","parts":[{"text":" answer"}]},"finishReason":"STOP"}]}`+"\n\n")
	})

	require.NoError(t, err)
	assert.Equal(t, "partial answer", resp.Choices[0].Content)
}
