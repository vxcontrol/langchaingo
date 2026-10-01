package anthropic_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const overloadedEvent = `event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}
`

func TestAnAnthropicStreamWithoutMessageStopIsAnError(t *testing.T) {
	t.Parallel()

	resp, err := streamThenFail(t, "")

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}

func TestAnAnthropicErrorEventIsAStreamFailure(t *testing.T) {
	t.Parallel()

	_, err := streamThenFail(t, overloadedEvent)

	require.ErrorIs(t, err, llms.ErrStreamFailed)
}

func TestTheDoneChunkArrivesBeforeGenerateContentReturns(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, deliveredThenBroken+overloadedEvent)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-haiku-4-5"))
	require.NoError(t, err)

	var done atomic.Bool
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type == streaming.ChunkTypeDone {
				time.Sleep(50 * time.Millisecond)
				done.Store(true)
			}
			return nil
		}))

	require.Error(t, err)
	assert.True(t, done.Load(), "the stream's done chunk must reach the caller before GenerateContent returns")
}

const finishedTail = `event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`

var errUserStop = errors.New("user pressed stop")

func streamThenDrop(t *testing.T, ctx context.Context, tail string, onChunk func()) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, deliveredThenBroken+tail)
		w.(http.Flusher).Flush()
		if tail == "" {
			<-r.Context().Done()
			return
		}
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			_ = conn.Close()
		}
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-haiku-4-5"))
	require.NoError(t, err)

	return llm.GenerateContent(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error {
			onChunk()
			return nil
		}))
}

func TestADroppedConnectionAfterMessageStopLeavesTheAnswerComplete(t *testing.T) {
	t.Parallel()

	resp, err := streamThenDrop(t, t.Context(), finishedTail, func() {})

	require.NoError(t, err)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	assert.Equal(t, "end_turn", resp.Choices[0].StopReason)
}

func TestAStreamCutInsideAnEventIsIncomplete(t *testing.T) {
	t.Parallel()

	resp, err := streamThenDrop(t, t.Context(), "event: content_block_delta\n"+
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" and mo`, func() {})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}

func TestAStreamCutWithACauseCarriesBothTheCauseAndTheContextError(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancelCause(t.Context())
	_, err := streamThenDrop(t, ctx, "", func() { cancel(errUserStop) })

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.ErrorIs(t, err, context.Canceled)
	require.ErrorIs(t, err, errUserStop)
}
