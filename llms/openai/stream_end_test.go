package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestACutStreamReturnsThePartialAnswerWithAnError(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"partial"}}]}`+"\n\n")
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("gpt-4.1"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "partial", resp.Choices[0].Content)
}

func TestACutToolCallIsLeftOutOfThePartialAnswer(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"checking",`+
			`"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"weather",`+
			`"arguments":"{\"city\":\"Paris\"}"}},{"index":1,"id":"call_2","type":"function",`+
			`"function":{"name":"weather","arguments":"{\"city\":\"Ro"}}]}}]}`+"\n\n")
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("gpt-4.1"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "weather in Paris and Rome?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.NotNil(t, resp)
	assert.Equal(t, "checking", resp.Choices[0].Content)
	require.Len(t, resp.Choices[0].ToolCalls, 1)
	assert.JSONEq(t, `{"city":"Paris"}`, resp.Choices[0].ToolCalls[0].FunctionCall.Arguments)
}
