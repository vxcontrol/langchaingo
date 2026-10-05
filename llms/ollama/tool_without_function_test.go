package ollama

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAToolWithoutAFunctionIsDroppedAndReportedOnOllama(t *testing.T) {
	t.Parallel()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("gemma3:1b"))
	require.NoError(t, err)
	named := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "search", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithTools([]llms.Tool{{Type: "function"}, named}))
	require.NoError(t, err)
	tools, _ := body["tools"].([]any)
	require.Len(t, tools, 1, "%v", body)
	var dropped *llms.Warning
	for i, w := range resp.Warnings {
		if w.Option == "WithTools" {
			dropped = &resp.Warnings[i]
		}
	}
	require.NotNil(t, dropped, "%v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, dropped.Kind)
	require.Equal(t, "1 tools", dropped.Asked)
}

func TestAHistoryCallOrToolTheDoorCannotSendIsRefusedBeforeTheRequestOnOllama(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("gemma3:1b"))
	require.NoError(t, err)
	for name, call := range map[string]struct {
		messages []llms.MessageContent
		opts     []llms.CallOption
	}{
		"a history call without a function": {[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: "call_1", Name: "search", Content: "found"},
			}},
		}, nil},
		"a built-in tool the door cannot send": {
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			[]llms.CallOption{llms.WithTools([]llms.Tool{{Type: "web_search"}})},
		},
	} {
		_, err = llm.GenerateContent(t.Context(), call.messages, call.opts...)
		require.ErrorIs(t, err, llms.ErrInvalidRequest, name)
		require.Zero(t, requests.Load(), "%s: no request may go out", name)
	}
}
