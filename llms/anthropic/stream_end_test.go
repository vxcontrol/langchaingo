package anthropic_test

import (
	"context"
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
