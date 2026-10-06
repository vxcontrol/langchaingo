package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
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
	assert.Empty(t, resp.Warnings)
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

func TestAStreamCutAfterEveryChoiceFinishedWarnsOnlyWhenItsUsageIsLost(t *testing.T) {
	t.Parallel()

	const (
		answer = `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"done"}}]}` + "\n\n" +
			`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
		usage = `data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}` + "\n\n"
		done  = "data: [DONE]\n\n"
	)
	for name, tc := range map[string]struct {
		stream     string
		lost       bool
		completion int
	}{
		"cut before the usage chunk": {stream: answer, lost: true},
		"cut after the usage chunk":  {stream: answer + usage, completion: 3},
		"a full stream":              {stream: answer + usage + done, completion: 3},
		"a host that sends no usage": {stream: answer + done},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, tc.stream)
			}))
			t.Cleanup(srv.Close)

			llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("gpt-4.1"))
			require.NoError(t, err)

			resp, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
				llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

			require.NoError(t, err)
			require.Len(t, resp.Choices, 1)
			assert.Equal(t, "done", resp.Choices[0].Content)
			assert.Equal(t, tc.completion, resp.Choices[0].GenerationInfo["CompletionTokens"])
			lost := slices.ContainsFunc(resp.Warnings, func(w llms.Warning) bool {
				return w.Kind == llms.WarningDrop && w.Option == "usage"
			})
			assert.Equal(t, tc.lost, lost, "warnings: %v", resp.Warnings)
		})
	}
}
