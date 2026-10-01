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

const finishedMistralChunk = `{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-small-latest",` +
	`"choices":[{"index":0,"delta":{"role":"assistant","content":"sixty rooms are free"},"finish_reason":"stop"}]}`

func mistralAgainst(t *testing.T, handler http.HandlerFunc) *Model {
	t.Helper()

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)
	return m
}

func TestAMistralCallReturnsWhenItsContextEndsBeforeTheServerAnswers(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	m := mistralAgainst(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		<-release
	})
	t.Cleanup(func() { close(release) })

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := m.GenerateContent(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.NotErrorIs(t, err, llms.ErrIncompleteStream)
	assert.Less(t, time.Since(start), 5*time.Second)
}

func TestAMistralAnswerThatFinishedStaysCompleteWhenTheContextEnds(t *testing.T) {
	t.Parallel()

	for range 50 {
		m := mistralAgainst(t, func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", finishedMistralChunk)
		})

		ctx, cancel := context.WithCancel(t.Context())
		resp, err := m.GenerateContent(ctx,
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error {
				cancel()
				return nil
			}))

		require.NoError(t, err)
		require.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	}
}

func TestAMistralStreamThatFinishesWithAnErrorIsAStreamFailure(t *testing.T) {
	t.Parallel()

	m := mistralAgainst(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\n", partialMistralChunk)
		_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,`+
			`"delta":{"content":""},"finish_reason":"error"}]}`+"\n\ndata: [DONE]\n\n")
	})

	resp, err := m.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are", resp.Choices[0].Content)
}

func TestAMistralAnswerThatFinishedWithAnErrorIsAStreamFailure(t *testing.T) {
	t.Parallel()

	m := mistralAgainst(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"mistral-small-latest",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"sixty rooms are"},"finish_reason":"error"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":3,"total_tokens":4}}`)
	})

	resp, err := m.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")})

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are", resp.Choices[0].Content)
}
