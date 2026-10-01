package bedrock_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const legacyClaudeModel = "anthropic.claude-sonnet-4-5-20250929-v1:0"

var errUserStop = errors.New("user pressed stop")

func streamFrom(
	t *testing.T, ctx context.Context, write func(http.ResponseWriter, *http.Request, *eventstream.Encoder),
	onChunk func(), opts ...bedrock.Option,
) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		write(w, r, eventstream.NewEncoder())
	}))
	t.Cleanup(srv.Close)

	return bedrockLLMAgainst(t, srv, opts...).GenerateContent(ctx,
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error {
			onChunk()
			return nil
		}))
}

func writeModelStreamError(t *testing.T, w io.Writer, enc *eventstream.Encoder) {
	t.Helper()

	require.NoError(t, enc.Encode(w, eventstream.Message{
		Headers: eventstream.Headers{
			{Name: ":message-type", Value: eventstream.StringValue("exception")},
			{Name: ":exception-type", Value: eventstream.StringValue("modelStreamErrorException")},
			{Name: ":content-type", Value: eventstream.StringValue("application/json")},
		},
		Payload: []byte(`{"message":"the model stream failed"}`),
	}))
}

func dropConnection(w http.ResponseWriter) {
	w.(http.Flusher).Flush()
	if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
		_ = conn.Close()
	}
}

const legacyClaudeStart = `{"type":"message_start","message":{"id":"x","type":"message","role":"assistant",` +
	`"model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}`

func legacyClaudeText(text string) string {
	return `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + text + `"}}`
}

func TestALegacyStreamCutBeforeItsStopReasonIsIncomplete(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		model  string
		chunks []string
	}{"anthropic": {model: legacyClaudeModel, chunks: []string{legacyClaudeStart, legacyClaudeText("sixty ")}}}
	for _, family := range nonClaudeFamilies() {
		cases[family.name] = struct {
			model  string
			chunks []string
		}{model: family.model, chunks: family.chunks[:1]}
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request, enc *eventstream.Encoder) {
				for _, chunk := range tc.chunks {
					writeLegacyChunk(t, w, enc, chunk)
				}
			}, func() {}, bedrock.WithModel(tc.model))

			require.ErrorIs(t, err, llms.ErrIncompleteStream)
			require.NotNil(t, resp)
			assert.Equal(t, "sixty ", resp.Choices[0].Content)
		})
	}
}

func TestALegacyExceptionEventIsAStreamFailureThatKeepsWhatArrived(t *testing.T) {
	t.Parallel()

	for range 30 {
		resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request, enc *eventstream.Encoder) {
			writeLegacyChunk(t, w, enc, legacyClaudeStart)
			writeLegacyChunk(t, w, enc, legacyClaudeText("sixty rooms "))
			writeLegacyChunk(t, w, enc, legacyClaudeText("are free"))
			writeModelStreamError(t, w, enc)
		}, func() {}, bedrock.WithModel(legacyClaudeModel))

		require.ErrorIs(t, err, llms.ErrStreamFailed)
		require.NotNil(t, resp)
		require.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	}
}

func TestAStreamCutWithACauseCarriesBothTheCauseAndTheContextError(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		opts  []bedrock.Option
		write func(*testing.T, http.ResponseWriter, *eventstream.Encoder)
	}{
		"legacy": {
			opts: []bedrock.Option{bedrock.WithModel(legacyClaudeModel)},
			write: func(t *testing.T, w http.ResponseWriter, enc *eventstream.Encoder) {
				writeLegacyChunk(t, w, enc, legacyClaudeStart)
				writeLegacyChunk(t, w, enc, legacyClaudeText("sixty "))
			},
		},
		"converse": {
			opts: []bedrock.Option{bedrock.WithModel(legacyClaudeModel), bedrock.WithConverseAPI()},
			write: func(t *testing.T, w http.ResponseWriter, enc *eventstream.Encoder) {
				writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
				writeConverseEvent(t, w, enc, "contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"sixty "}}`)
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithCancelCause(t.Context())
			_, err := streamFrom(t, ctx, func(w http.ResponseWriter, r *http.Request, enc *eventstream.Encoder) {
				tc.write(t, w, enc)
				w.(http.Flusher).Flush()
				<-r.Context().Done()
			}, func() { cancel(errUserStop) }, tc.opts...)

			require.ErrorIs(t, err, llms.ErrIncompleteStream)
			require.ErrorIs(t, err, context.Canceled)
			require.ErrorIs(t, err, errUserStop)
		})
	}
}

func TestADroppedConnectionAfterMessageStopLeavesTheConverseAnswerComplete(t *testing.T) {
	t.Parallel()

	resp, err := streamFrom(t, t.Context(), func(w http.ResponseWriter, _ *http.Request, enc *eventstream.Encoder) {
		writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
		writeConverseEvent(t, w, enc, "contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"sixty rooms are free"}}`)
		writeConverseEvent(t, w, enc, "messageStop", `{"stopReason":"end_turn"}`)
		dropConnection(w)
	}, func() {}, bedrock.WithModel(legacyClaudeModel), bedrock.WithConverseAPI())

	require.NoError(t, err)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	assert.Equal(t, "end_turn", resp.Choices[0].StopReason)
}
