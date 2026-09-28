package anthropic_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const deliveredThenBroken = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"sixty rooms are free"}}

`

func streamThenFail(t *testing.T, tail string) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, deliveredThenBroken+tail)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-haiku-4-5"))
	require.NoError(t, err)

	return llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
}

func TestTextAlreadyDeliveredSurvivesAVendorErrorEvent(t *testing.T) {
	t.Parallel()

	resp, err := streamThenFail(t, `event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}
`)

	require.Error(t, err)
	require.NotNil(t, resp, "text already handed to the consumer must travel with the error")
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}

func TestTextAlreadyDeliveredSurvivesAMalformedEvent(t *testing.T) {
	t.Parallel()

	resp, err := streamThenFail(t, "data: {not json at all\n")

	require.Error(t, err)
	require.NotNil(t, resp, "one unreadable event must not discard the answer so far")
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}

func TestTextAlreadyDeliveredSurvivesAnOversizedLine(t *testing.T) {
	t.Parallel()

	resp, err := streamThenFail(t, "data: "+strings.Repeat("x", 9*1024*1024)+"\n")

	require.Error(t, err)
	require.NotNil(t, resp, "a line the scanner cannot hold must not discard the answer so far")
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}

func TestAToolCallCutOffMidStreamIsNotHandedBackAsIfItHadNoArguments(t *testing.T) {
	t.Parallel()

	resp, err := streamThenFail(t, `event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"delete_files","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"/tmp/bui"}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}
`)

	require.Error(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	assert.Empty(t, resp.Choices[0].ToolCalls, "the model never finished these arguments")
}

func TestAToolCallWhoseArgumentsDoNotParseIsNotHandedBackAsIfItHadNone(t *testing.T) {
	t.Parallel()

	resp, err := streamThenFail(t, `event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"delete_files","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"/tmp/build\""}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

`)

	require.Error(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	assert.Empty(t, resp.Choices[0].ToolCalls, "arguments that never parsed are not a call")
}
