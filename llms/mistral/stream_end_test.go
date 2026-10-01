package mistral

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

var errUserStop = errors.New("user pressed stop")

const partialMistralChunk = `{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-small-latest",` +
	`"choices":[{"index":0,"delta":{"role":"assistant","content":"sixty rooms are"},"finish_reason":null}]}`

func TestAMistralStreamWithoutAFinishReasonIsIncomplete(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\n", partialMistralChunk)
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)

	resp, err := m.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are", resp.Choices[0].Content)
}

func TestAMistralStreamReturnsAsSoonAsItsContextEnds(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\n", partialMistralChunk)
		w.(http.Flusher).Flush()
		<-release
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)

	ctx, cancel := context.WithCancelCause(t.Context())
	start := time.Now()
	_, err = m.GenerateContent(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error {
			cancel(errUserStop)
			return nil
		}))

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.ErrorIs(t, err, context.Canceled)
	require.ErrorIs(t, err, errUserStop)
	assert.Less(t, time.Since(start), 5*time.Second)
}

func sdkReaderAlive() bool {
	buf := make([]byte, 1<<20)
	return strings.Contains(string(buf[:runtime.Stack(buf, true)]), "mistral-go.(*MistralClient).ChatStream")
}

func TestAnAbandonedMistralStreamDoesNotLeaveTheSDKReaderBlocked(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\n", partialMistralChunk)
		w.(http.Flusher).Flush()
		<-release
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", partialMistralChunk)
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)

	gaveUp := errors.New("consumer gave up")
	_, err = m.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return gaveUp }))
	require.ErrorIs(t, err, gaveUp)

	close(release)
	assert.Eventually(t, func() bool { return !sdkReaderAlive() }, 5*time.Second, 20*time.Millisecond)
}
