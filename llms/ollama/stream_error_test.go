package ollama

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

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

var errUserStop = errors.New("user pressed stop")

func streamFrom(t *testing.T, ctx context.Context, write func(http.ResponseWriter, *http.Request), onChunk func()) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		write(w, r)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("llama3"))
	require.NoError(t, err)

	return llm.GenerateContent(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error {
			onChunk()
			return nil
		}))
}

func TestAnErrorOllamaSendsInsideTheStreamIsAStreamFailure(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, cutFrame("sixty rooms are")+"\n"+
			`{"error":"model runner has unexpectedly stopped"}`+"\n")
	}, func() {})

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	assert.Contains(t, err.Error(), "model runner has unexpectedly stopped")
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are", resp.Choices[0].Content)
}

func TestAnOllamaStreamCutInsideAFrameIsIncomplete(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, cutFrame("sixty rooms are")+"\n"+
			`{"model":"llama3","message":{"role":"assistant","content":" fr`)
	}, func() {})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotErrorIs(t, err, llms.ErrStreamFailed)
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are", resp.Choices[0].Content)
}

func TestAnOllamaStreamCutWithACauseCarriesBothTheCauseAndTheContextError(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancelCause(t.Context())
	_, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, cutFrame("sixty rooms are")+"\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}, func() { cancel(errUserStop) })

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.ErrorIs(t, err, context.Canceled)
	require.ErrorIs(t, err, errUserStop)
}

func TestADroppedConnectionAfterTheFinalFrameLeavesTheOllamaAnswerComplete(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, cutFrame("sixty rooms are free")+"\n"+
			`{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`+"\n")
		w.(http.Flusher).Flush()
		if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
			_ = conn.Close()
		}
	}, func() {})

	require.NoError(t, err)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}

func TestADeadlineBeforeTheFirstFrameIsNotReportedAsACutStream(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	_, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request) {
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}, func() {})

	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.NotErrorIs(t, err, llms.ErrIncompleteStream)
}

func TestAnOllamaStreamCutInsideItsFirstFrameKeepsTheDeadline(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), 200*time.Millisecond)
	defer cancel()
	_, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"model":"llama3","message":{"role":"assistant","content":"six`)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}, func() {})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestAnHTTP2ResetIsAReadFailureNotAServerError(t *testing.T) {
	t.Parallel()

	final := `{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`
	for name, tc := range map[string]struct {
		lines        string
		wantComplete bool
	}{
		"after the final frame": {lines: cutFrame("sixty rooms are free") + "\n" + final + "\n", wantComplete: true},
		"inside the stream":     {lines: cutFrame("sixty rooms are free") + "\n"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "application/x-ndjson")
				_, _ = io.WriteString(w, tc.lines)
				w.(http.Flusher).Flush()
				panic(http.ErrAbortHandler)
			}))
			srv.EnableHTTP2 = true
			srv.StartTLS()
			t.Cleanup(srv.Close)

			llm, err := New(WithServerURL(srv.URL), WithModel("llama3"), WithHTTPClient(srv.Client()))
			require.NoError(t, err)

			resp, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
				llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

			require.NotErrorIs(t, err, llms.ErrStreamFailed)
			if tc.wantComplete {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, llms.ErrIncompleteStream)
			}
			require.NotNil(t, resp)
			assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
		})
	}
}
