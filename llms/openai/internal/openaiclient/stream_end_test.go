package openaiclient

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/internal/testutil/cutctx"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const partialChunk = `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"partial"}}]}` + "\n\n"

func parseStream(ctx context.Context, t *testing.T, body io.Reader) (*ChatCompletionResponse, error) {
	t.Helper()

	r := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(body)}
	req := &ChatRequest{StreamingFunc: func(context.Context, streaming.Chunk) error { return nil }}
	return parseStreamingChatResponse(ctx, r, req)
}

func TestAStreamThatEndsWithoutItsFinalEventIsAnError(t *testing.T) {
	t.Parallel()

	resp, err := parseStream(t.Context(), t, strings.NewReader(partialChunk))

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "partial", resp.Choices[0].Message.Content)
}

func TestAStreamThatFinishesItsChoicesIsComplete(t *testing.T) {
	t.Parallel()

	for name, tail := range map[string]string{
		"done marker":   "data: [DONE]\n\n",
		"finish reason": `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			resp, err := parseStream(t.Context(), t, strings.NewReader(partialChunk+tail))
			require.NoError(t, err)
			assert.Equal(t, "partial", resp.Choices[0].Message.Content)
		})
	}
}

func TestAnErrorTheProviderSendsInsideTheStreamIsAnError(t *testing.T) {
	t.Parallel()

	for name, tail := range map[string]string{
		"error event":  `data: {"error":{"message":"upstream overloaded","type":"server_error","code":529}}` + "\n\ndata: [DONE]\n\n",
		"error finish": `data: {"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}` + "\n\ndata: [DONE]\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			resp, err := parseStream(t.Context(), t, strings.NewReader(partialChunk+tail))
			require.ErrorIs(t, err, llms.ErrStreamFailed)
			require.NotNil(t, resp)
			assert.Equal(t, "partial", resp.Choices[0].Message.Content)
		})
	}
}

func TestAnErrorEventCarriesTheProvidersMessage(t *testing.T) {
	t.Parallel()

	_, err := parseStream(t.Context(), t, strings.NewReader(partialChunk+
		`data: {"error":{"message":"upstream overloaded","type":"server_error","code":529}}`+"\n\n"))

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	assert.Contains(t, err.Error(), "upstream overloaded")
}

type blockingBody struct {
	ctx        context.Context
	chunks     []string
	readsCause bool
}

func (b *blockingBody) Read(p []byte) (int, error) {
	if len(b.chunks) > 0 {
		n := copy(p, b.chunks[0])
		b.chunks[0] = b.chunks[0][n:]
		if b.chunks[0] == "" {
			b.chunks = b.chunks[1:]
		}
		return n, nil
	}
	<-b.ctx.Done()
	if b.readsCause {
		return 0, context.Cause(b.ctx)
	}
	return 0, b.ctx.Err()
}

func TestAStreamCutByTheContextIsAnErrorEveryTime(t *testing.T) {
	t.Parallel()

	for name, cause := range map[string]error{
		"deadline": context.DeadlineExceeded,
		"cancel":   context.Canceled,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			for range 100 {
				ctx := cutctx.New(t.Context(), cause)
				r := &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(&blockingBody{ctx: ctx, chunks: []string{partialChunk}}),
				}
				req := &ChatRequest{StreamingFunc: func(context.Context, streaming.Chunk) error {
					ctx.Cut()
					return nil
				}}

				resp, err := parseStreamingChatResponse(ctx, r, req)

				require.ErrorIs(t, err, llms.ErrIncompleteStream)
				require.ErrorIs(t, err, cause)
				require.NotNil(t, resp)
				require.Len(t, resp.Choices, 1)
				require.Equal(t, "partial", resp.Choices[0].Message.Content)
			}
		})
	}
}

const finishedChunk = `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"partial"},` +
	`"finish_reason":"stop"}]}` + "\n\n"

var errUserStop = errors.New("user pressed stop")

type failingBody struct {
	chunks []string
	err    error
}

func (b *failingBody) Read(p []byte) (int, error) {
	if len(b.chunks) > 0 {
		n := copy(p, b.chunks[0])
		b.chunks[0] = b.chunks[0][n:]
		if b.chunks[0] == "" {
			b.chunks = b.chunks[1:]
		}
		return n, nil
	}
	return 0, b.err
}

func TestAReadErrorAfterTheFinishReasonLeavesTheAnswerComplete(t *testing.T) {
	t.Parallel()

	resp, err := parseStream(t.Context(), t,
		&failingBody{chunks: []string{finishedChunk}, err: errors.New("connection reset by peer")})

	require.NoError(t, err)
	assert.Equal(t, "partial", resp.Choices[0].Message.Content)
}

func TestAContextCutAfterTheFinishReasonLeavesTheAnswerCompleteEveryTime(t *testing.T) {
	t.Parallel()

	for range 100 {
		ctx := cutctx.New(t.Context(), context.DeadlineExceeded)
		r := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(&blockingBody{ctx: ctx, chunks: []string{finishedChunk}}),
		}
		req := &ChatRequest{StreamingFunc: func(context.Context, streaming.Chunk) error {
			ctx.Cut()
			return nil
		}}

		resp, err := parseStreamingChatResponse(ctx, r, req)

		require.NoError(t, err)
		require.Equal(t, "partial", resp.Choices[0].Message.Content)
	}
}

func TestAStreamCutWithACauseCarriesBothTheCauseAndTheContextErrorEveryTime(t *testing.T) {
	t.Parallel()

	for range 100 {
		ctx, cancel := context.WithCancelCause(t.Context())
		r := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(&blockingBody{ctx: ctx, chunks: []string{partialChunk}, readsCause: true}),
		}
		req := &ChatRequest{StreamingFunc: func(context.Context, streaming.Chunk) error {
			cancel(errUserStop)
			return nil
		}}

		_, err := parseStreamingChatResponse(ctx, r, req)

		require.ErrorIs(t, err, llms.ErrIncompleteStream)
		require.ErrorIs(t, err, context.Canceled)
		require.ErrorIs(t, err, errUserStop)
	}
}

func TestAnErrorTheProviderSendsAsAStringIsAStreamFailure(t *testing.T) {
	t.Parallel()

	for name, tail := range map[string]string{
		"with done marker": `data: {"error":"Request failed during generation: CUDA out of memory",` +
			`"error_type":"generation"}` + "\n\ndata: [DONE]\n\n",
		"without done marker": `data: {"error":"Request failed during generation: CUDA out of memory"}` + "\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			resp, err := parseStream(t.Context(), t, strings.NewReader(partialChunk+tail))

			require.ErrorIs(t, err, llms.ErrStreamFailed)
			assert.Contains(t, err.Error(), "CUDA out of memory")
			require.NotNil(t, resp)
			assert.Equal(t, "partial", resp.Choices[0].Message.Content)
		})
	}
}
