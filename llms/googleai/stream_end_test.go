package googleai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/internal/testutil/cutctx"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const partialGeminiChunk = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}` + "\n\n"

func streamFrom(
	t *testing.T, ctx context.Context, handler http.HandlerFunc, onChunk func(),
) (*llms.ContentResponse, error) {
	t.Helper()

	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	llm, err := New(t.Context(), WithAPIKey("unit-test-key"), WithEndpoint(server.URL),
		WithDefaultModel("gemini-2.5-flash"))
	require.NoError(t, err)

	return llm.GenerateContent(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error {
			onChunk()
			return nil
		}))
}

func TestAGeminiStreamWithoutAFinishReasonIsAnError(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk)
	}, func() {})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	assert.Equal(t, "partial", resp.Choices[0].Content)
}

func TestAGeminiStreamCutByTheDeadlineKeepsItsCause(t *testing.T) {
	t.Parallel()

	ctx := cutctx.New(t.Context(), context.DeadlineExceeded)

	resp, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}, ctx.Cut)

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
	}, func() {})

	require.NoError(t, err)
	assert.Equal(t, "partial answer", resp.Choices[0].Content)
}

var errUserStop = errors.New("user pressed stop")

func TestAnErrorGeminiSendsInsideTheStreamIsAStreamFailure(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk+
			`{"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}`+"\n\n")
	}, func() {})

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	assert.Contains(t, err.Error(), "The model is overloaded.")
	require.NotNil(t, resp)
	assert.Equal(t, "partial", resp.Choices[0].Content)
}

func TestAGeminiStreamCutInsideAnEventIsIncomplete(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk+`data: {"candidates":[{"content":{"role":"model","parts":[{"text":" and mo`)
	}, func() {})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	assert.Equal(t, "partial", resp.Choices[0].Content)
}

func TestAGeminiStreamCutWithACauseCarriesBothTheCauseAndTheContextError(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancelCause(t.Context())
	_, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, partialGeminiChunk)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}, func() { cancel(errUserStop) })

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.ErrorIs(t, err, context.Canceled)
	require.ErrorIs(t, err, errUserStop)
}

func TestAGeminiStreamCutInsideItsFirstEventKeepsTheDeadline(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), 200*time.Millisecond)
	defer cancel()
	_, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"sixty`)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}, func() {})

	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.ErrorIs(t, err, llms.ErrIncompleteStream)
}

func TestADeadlineBeforeTheGeminiStreamStartsIsNotACutStream(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	_, err := streamFrom(t, ctx, func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}, func() {})

	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.NotErrorIs(t, err, llms.ErrIncompleteStream)
}
