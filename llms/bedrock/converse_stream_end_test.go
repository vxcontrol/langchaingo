package bedrock_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func converseStreamEnding(t *testing.T, finish func(w io.Writer, enc *eventstream.Encoder)) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
		writeConverseEvent(t, w, enc, "contentBlockDelta",
			`{"contentBlockIndex":0,"delta":{"text":"sixty rooms are free"}}`)
		finish(w, enc)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())

	return llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
}

func TestAConverseStreamWithoutMessageStopIsAnError(t *testing.T) {
	t.Parallel()

	resp, err := converseStreamEnding(t, func(io.Writer, *eventstream.Encoder) {})

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	assert.False(t, slices.ContainsFunc(resp.Warnings, func(w llms.Warning) bool { return w.Option == "usage" }),
		"%v", resp.Warnings)
}

func TestAnExceptionEventInsideAConverseStreamIsAnError(t *testing.T) {
	t.Parallel()

	resp, err := converseStreamEnding(t, func(w io.Writer, enc *eventstream.Encoder) {
		require.NoError(t, enc.Encode(w, eventstream.Message{
			Headers: eventstream.Headers{
				{Name: ":message-type", Value: eventstream.StringValue("exception")},
				{Name: ":exception-type", Value: eventstream.StringValue("modelStreamErrorException")},
				{Name: ":content-type", Value: eventstream.StringValue("application/json")},
			},
			Payload: []byte(`{"message":"the model stream failed"}`),
		}))
	})

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	require.NotNil(t, resp)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}

func TestAConverseStreamCutBeforeItsMetadataWarnsThatItsUsageIsLost(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		metadata bool
		lost     bool
	}{
		"cut after messageStop":      {lost: true},
		"metadata after messageStop": {metadata: true},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			resp, err := converseStreamEnding(t, func(w io.Writer, enc *eventstream.Encoder) {
				writeConverseEvent(t, w, enc, "messageStop", `{"stopReason":"end_turn"}`)
				if tc.metadata {
					writeConverseEvent(t, w, enc, "metadata",
						`{"usage":{"inputTokens":7,"outputTokens":3,"totalTokens":10},"metrics":{"latencyMs":1}}`)
				}
			})

			require.NoError(t, err)
			assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
			lost := slices.ContainsFunc(resp.Warnings, func(w llms.Warning) bool {
				return w.Kind == llms.WarningDrop && w.Option == "usage"
			})
			assert.Equal(t, tc.lost, lost, "%v", resp.Warnings)
			if tc.metadata {
				assert.Equal(t, 3, resp.Choices[0].GenerationInfo["CompletionTokens"])
			}
		})
	}
}
